/**
 * Guardrails a reader writes: rules over a specification export's per-configuration readouts,
 * edited in components/GuardrailEditor (react-querybuilder) and run here. Pure functions, so
 * the node tests can check them against the saved guardrails (lib/guardrails.test.mjs).
 *
 * - The rule is react-querybuilder's query tree: AND / OR groups, NOT, a comparison per rule,
 *   against a number, a list of categories, or another field.
 * - Calculated fields (lib/formula) are named formulas over the numeric fields, worked out per
 *   configuration first, so rules use them like any other field.
 * - Evaluation: formatQuery → JsonLogic → json-logic-engine, compiled once per rule. Comparisons
 *   are redefined so a missing value fails them (plain JsonLogic, like JavaScript, reads
 *   null <= 0.5 as true, which would let a configuration with no held-out split pass).
 */
import { formatQuery, type Field, type RuleGroupType } from '@react-querybuilder/core'
import { LogicEngine } from 'json-logic-engine'
import { evalFormula, parseFormula, type Logic } from './formula.ts'
import { CALIBRATION_LOT, COHORT, CV_PROTOCOL, PREPROCESSING, SELECTION_SPACE } from './labels.ts'

/** The columns of a specification export a rule can read (spec_curve.json and the meeting exports). */
export interface GuardrailSource {
  n: number
  ok: string
  intercept: (number | null)[]
  slope: (number | null)[]
  heldout: (number | null)[]
  cut: (number | null)[]
  k: (number | null)[]
  categories: { co: string[]; sp: string[]; sel: string[]; mo: string[]; lot: string[] }
  co: number[]; sp: number[]; sel: number[]; mo: number[]; lot: number[]
  /** extra readouts (gallery/data/export_spec_curve.py EXTRA_FIELDS); each export carries what its search recorded */
  extra?: Record<string, (number | null)[]>
}

export interface Formula { name: string; label: string; formula: string }
export interface Guardrail { query: RuleGroupType; formulas: Formula[] }

type Kind = 'number' | 'category' | 'boolean'
interface FieldInfo { label: string; unit?: string; group: string; help?: string }
/** every field a specification export may carry; an export offers the ones it has */
export const FIELD_INFO: Record<string, FieldInfo> = {
  // named by the fit, not the site: the test set is the page's (Addis), the rule is about the fit
  intercept: { label: 'Deming intercept', unit: 'µg/m³', group: 'Test-set fit', help: 'Deming intercept at MAC 10 (OLS when the page shows OLS)' },
  absIntercept: { label: '|Deming intercept|', unit: 'µg/m³', group: 'Test-set fit', help: 'distance of the intercept from zero' },
  slope: { label: 'Deming slope', group: 'Test-set fit', help: 'Deming slope at MAC 10; 1 = agreement' },
  slopeOffset: { label: '|Deming slope − 1|', group: 'Test-set fit', help: 'distance of the slope from agreement' },
  r2: { label: 'R²', group: 'Test-set fit' },
  rmse: { label: 'RMSE', unit: 'µg/m³', group: 'Test-set fit' },
  interceptMac6: { label: 'Deming intercept at MAC 6', unit: 'µg/m³', group: 'Test-set fit' },
  predMedian: { label: 'Median prediction', unit: 'µg/m³', group: 'Test-set predictions' },
  neg: { label: 'Negative predictions', unit: '%', group: 'Test-set predictions', help: 'share of test-set filters predicted below zero' },
  above8: { label: 'Predictions above 8 µg/m³', unit: '%', group: 'Test-set predictions' },
  seasonSpan: { label: 'Spread of season medians', unit: 'µg/m³', group: 'Test-set predictions' },
  qres: { label: 'Q-residual out of domain', unit: '%', group: 'Spectral domain', help: 'share of test-set filters with unmodelled spectral variation beyond the calibration set’s 95th percentile' },
  extrap: { label: 'Score extrapolation', unit: '%', group: 'Spectral domain', help: 'share of test-set filters beyond the calibration set’s 95th percentile in score space' },
  heldout: { label: 'IMPROVE cross-validation R²', group: 'Calibration set', help: 'held-out TOR R²; site-grouped cross-validation only' },
  k: { label: 'PLS factors', group: 'Calibration set' },
  cut: { label: 'Cohort size', unit: 'filters', group: 'Calibration set', help: 'requested cutoff; complete cohorts have none' },
  trainN: { label: 'Training filters', group: 'Calibration set' },
  trainSites: { label: 'Training sites', group: 'Calibration set' },
  drySlope: { label: 'Dry-season slope', group: 'Seasons' }, dryIntercept: { label: 'Dry-season intercept', unit: 'µg/m³', group: 'Seasons' },
  belgSlope: { label: 'Belg slope', group: 'Seasons' }, belgIntercept: { label: 'Belg intercept', unit: 'µg/m³', group: 'Seasons' },
  kiremtSlope: { label: 'Kiremt slope', group: 'Seasons' }, kiremtIntercept: { label: 'Kiremt intercept', unit: 'µg/m³', group: 'Seasons' },
  ok: { label: 'Passes the saved guardrails', group: 'Saved checks' },
  co: { label: 'Cohort', group: 'Choices' },
  sel: { label: 'Selection spectra', group: 'Choices' },
  sp: { label: 'Model spectra', group: 'Choices' },
  mo: { label: 'Cross-validation', group: 'Choices' },
  lot: { label: 'Calibration lot', group: 'Choices' },
}
const CATEGORY_LABELS: Record<string, Record<string, string>> = { co: COHORT, sel: SELECTION_SPACE, sp: PREPROCESSING, mo: CV_PROTOCOL, lot: CALIBRATION_LOT }
const CATEGORIES = ['co', 'sel', 'sp', 'mo', 'lot'] as const
/** the columns every specification export has (the rest come from its `extra`) */
const ALWAYS = ['intercept', 'absIntercept', 'slope', 'slopeOffset', 'heldout', 'k', 'cut']

export interface GuardField { name: string; label: string; kind: Kind; unit?: string; group: string; help?: string; values?: { name: string; label: string }[] }

/** The fields this export can be ruled on, in the editor's order; calculated fields last. */
export function guardFields(src: GuardrailSource, formulas: Formula[] = []): GuardField[] {
  const has = (key: string) => key in (src.extra ?? {})
  const numeric = ['intercept', 'absIntercept', 'slope', 'slopeOffset', 'r2', 'rmse', 'interceptMac6', 'predMedian', 'neg', 'above8', 'seasonSpan',
    'qres', 'extrap', 'heldout', 'k', 'cut', 'trainN', 'trainSites', 'drySlope', 'dryIntercept', 'belgSlope', 'belgIntercept', 'kiremtSlope', 'kiremtIntercept']
    .filter((key) => ALWAYS.includes(key) || has(key))
  const out: GuardField[] = numeric.map((name) => ({ name, kind: 'number', ...FIELD_INFO[name] }))
  out.push({ name: 'ok', kind: 'boolean', ...FIELD_INFO.ok })
  for (const key of CATEGORIES) {
    const values = src.categories[key]
    if (values.length < 2) continue
    out.push({ name: key, kind: 'category', ...FIELD_INFO[key], values: values.map((v) => ({ name: v, label: CATEGORY_LABELS[key][v] ?? v })) })
  }
  for (const f of formulas) out.push({ name: f.name, label: f.label || f.name, kind: 'number', group: 'Calculated', help: f.formula })
  return out
}

export type GuardRow = Record<string, number | string | boolean | null>

/** One plain object per configuration, in the export's order, with the derived fields filled in. */
export function guardRows(src: GuardrailSource): GuardRow[] {
  const extra = Object.entries(src.extra ?? {})
  const rows: GuardRow[] = new Array(src.n)
  for (let i = 0; i < src.n; i++) {
    const intercept = src.intercept[i], slope = src.slope[i]
    const row: GuardRow = {
      intercept, slope, heldout: src.heldout[i], k: src.k[i], cut: src.cut[i],
      absIntercept: intercept === null ? null : Math.abs(intercept),
      slopeOffset: slope === null ? null : Math.abs(slope - 1),
      ok: src.ok[i] === '1',
    }
    for (const key of CATEGORIES) row[key] = src.categories[key][src[key][i]]
    for (const [key, col] of extra) row[key] = col[i]
    rows[i] = row
  }
  return rows
}

/** A calculated field's name: a plain identifier, unused by the export's own fields. */
export const formulaName = (label: string, taken: readonly string[]) => {
  const base = (label.replace(/[^A-Za-z0-9]+(.)?/g, (_, c: string | undefined) => (c ? c.toUpperCase() : '')).replace(/^[^A-Za-z_]+/, '') || 'calc')
  const stem = base[0].toLowerCase() + base.slice(1)
  let name = stem, n = 2
  while (taken.includes(name)) name = `${stem}${n++}`
  return name
}

/** Parse every calculated field; the first error, if any, names the field it is in. */
export function compileFormulas(formulas: Formula[], numericNames: readonly string[]) {
  const parsed: { name: string; logic: Logic }[] = []
  const known = [...numericNames]
  for (const f of formulas) {
    try { parsed.push({ name: f.name, logic: parseFormula(f.formula, known) }) }
    catch (e) { return { parsed, error: `${f.label || f.name}: ${(e as Error).message}` } }
    known.push(f.name)   // a later formula may use an earlier one
  }
  return { parsed, error: null as string | null }
}

const missing = (v: unknown) => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))
const cmp = (test: (a: number, b: number) => boolean) => (args: unknown[]) => {
  const a = args as number[]
  if (a.some(missing)) return false
  // JsonLogic's three-argument form is a range: a <= b <= c
  return a.length === 3 ? test(a[0], a[1]) && test(a[1], a[2]) : test(a[0], a[1])
}
function engine() {
  const e = new LogicEngine()
  e.addMethod('<', cmp((a, b) => a < b))
  e.addMethod('<=', cmp((a, b) => a <= b))
  e.addMethod('>', cmp((a, b) => a > b))
  e.addMethod('>=', cmp((a, b) => a >= b))
  return e
}

/** Whether a query has at least one rule anywhere in it (empty groups do not count). */
export function hasConditions(query: RuleGroupType): boolean {
  return query.rules.some((r) => typeof r === 'object' && ('rules' in r ? hasConditions(r as RuleGroupType) : true))
}

export function toJsonLogic(query: RuleGroupType) {
  return formatQuery(query, { format: 'jsonlogic', parseNumbers: true })
}

/**
 * Which configurations a guardrail passes: one boolean per row (true for every row when the
 * rule has no conditions). Returns the error instead when a formula does not parse.
 */
export function evaluateGuardrail(rule: Guardrail, rows: GuardRow[], numericNames: readonly string[]): { pass: boolean[]; error: string | null } {
  const { parsed, error } = compileFormulas(rule.formulas, numericNames)
  if (error) return { pass: rows.map(() => false), error }
  // no conditions at all: every configuration passes (formatQuery writes an empty group as `false`)
  if (!hasConditions(rule.query)) return { pass: rows.map(() => true), error: null }
  const logic = toJsonLogic(rule.query)
  const run = engine().build(logic) as (row: GuardRow) => unknown
  const pass = rows.map((row) => {
    if (!parsed.length) return !!run(row)
    const r: GuardRow = { ...row }
    for (const f of parsed) r[f.name] = evalFormula(f.logic, r)
    return !!run(r)
  })
  return { pass, error: null }
}

/** The rule in words, with the editor's field names (for the chart's footnote and tooltips). */
export function describeGuardrail(rule: Guardrail, fields: GuardField[]) {
  const text = formatQuery(rule.query, {
    format: 'natural_language', parseNumbers: true,
    fields: fields.map((f) => ({ name: f.name, label: f.label, values: f.values })) as Field[],
  })
  return text || 'no conditions (every configuration passes)'
}

// ---------------------------------------------------------------- templates
const rule = (field: string, operator: string, value: unknown) => ({ field, operator, value: String(value) })
const group = (...rules: RuleGroupType['rules']): RuleGroupType => ({ combinator: 'and', rules })

/** The rules the chart's old fixed highlights used, as editable starting points. */
export const TEMPLATES: { key: string; label: string; needs?: string[]; rule: Guardrail }[] = [
  {
    key: 'guardrails', label: 'Saved guardrails', needs: ['qres', 'extrap', 'neg'],
    rule: { formulas: [], query: group(
      rule('heldout', '>=', 0.85), rule('slope', 'between', '0.7,1.3'),
      rule('qres', '<=', 30), rule('extrap', '<=', 30), rule('neg', '<=', 10)) },
  },
  { key: 'intercept', label: 'Near-zero intercept', rule: { formulas: [], query: group(rule('absIntercept', '<=', 0.5)) } },
  { key: 'cv-slope', label: 'Slope + cross-validation', rule: { formulas: [], query: group(rule('slope', 'between', '0.7,1.3'), rule('heldout', '>=', 0.85)) } },
  { key: 'custom', label: 'Intercept + slope + cross-validation', rule: { formulas: [], query: group(rule('absIntercept', '<=', 0.5), rule('slope', 'between', '0.7,1.3'), rule('heldout', '>=', 0.85)) } },
]
export const templatesFor = (src: GuardrailSource) => TEMPLATES.filter((t) => (t.needs ?? []).every((k) => k in (src.extra ?? {})))

// ---------------------------------------------------------------- URL
/** A rule as a URL-safe string (base64url of its compact JSON, without the editor's ids). */
export function encodeGuardrail(rule: Guardrail): string {
  const json = JSON.stringify({ q: JSON.parse(formatQuery(rule.query, 'json_without_ids')), f: rule.formulas.length ? rule.formulas : undefined })
  const bytes = new TextEncoder().encode(json)
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
export function decodeGuardrail(token: string): Guardrail | null {
  if (!token) return null
  try {
    const bin = atob(token.replace(/-/g, '+').replace(/_/g, '/'))
    const json = new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)))
    const d = JSON.parse(json) as { q?: RuleGroupType; f?: Formula[] }
    if (!d.q || !Array.isArray(d.q.rules)) return null
    return { query: d.q, formulas: Array.isArray(d.f) ? d.f : [] }
  } catch { return null }
}

// ---------------------------------------------------------------- one condition at a time
/** The rule cut at its first `upTo` top-level conditions (each a rule or a whole group), in order. */
export function ruleUpTo(rule: Guardrail, upTo: number): Guardrail {
  return { ...rule, query: { ...rule.query, rules: rule.query.rules.slice(0, upTo) } }
}

/** Each top-level condition on its own, as a rule of one condition (the root's NOT left aside). */
export function ruleSteps(rule: Guardrail): Guardrail[] {
  return rule.query.rules.filter((r) => typeof r === 'object').map((r) => ({ ...rule, query: { combinator: 'and', rules: [r] } }))
}

export interface StepStats {
  /** configurations in scope */
  n: number
  /** per condition: how many fail it on its own */
  fails: number[]
  /** per condition, applied in order: how many it removes from what the earlier ones left (AND), or adds (OR) */
  atStep: number[]
  /** configurations passing after each step */
  left: number[]
  /** per condition: how many only it removes (they pass every other condition) */
  only: number[]
  /** fails[i][j]: how many fail both i and j (the diagonal is `fails`) */
  shared: number[][]
}

/**
 * How much each condition filters, given each one's pass/fail per configuration (`pass[step][row]`)
 * and the rows in scope. With AND, steps remove; with OR, a configuration passes once any step
 * passes it, so steps add. "Only" is the removals no other condition would make (AND).
 */
export function stepStats(pass: boolean[][], scope: number[], combinator = 'and'): StepStats {
  const k = pass.length, or = combinator === 'or'
  const fails = pass.map((p) => scope.reduce((s, i) => s + (p[i] ? 0 : 1), 0))
  const atStep: number[] = [], left: number[] = []
  let current = scope.map(() => !or)
  for (let s = 0; s < k; s++) {
    const next = scope.map((i, j) => (or ? current[j] || pass[s][i] : current[j] && pass[s][i]))
    atStep.push(next.reduce((a, v, j) => a + (v !== current[j] ? 1 : 0), 0))
    current = next
    left.push(current.filter(Boolean).length)
  }
  const only = pass.map((p, s) => scope.reduce((a, i) => a + (!p[i] && pass.every((q, t) => t === s || q[i]) ? 1 : 0), 0))
  const shared = pass.map((p) => pass.map((q) => scope.reduce((a, i) => a + (!p[i] && !q[i] ? 1 : 0), 0)))
  return { n: scope.length, fails, atStep, left, only, shared }
}
