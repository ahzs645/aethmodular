import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { QueryBuilder, ValueEditor, formatQuery, isOptionGroupArray, type Field, type OptionGroup, type RuleGroupType, type ValueEditorProps, type ValueSelectorProps } from 'react-querybuilder'
import { Combobox } from '@/components/Combobox'
import { MultiCombobox } from '@/components/MultiCombobox'
import { Toggle } from '@/components/ChartFrame'
import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import 'react-querybuilder/dist/query-builder-layout.css'
import { FORMULA_FUNCTIONS } from '@/lib/formula'
import {
  describeGuardrail, evaluateGuardrail, formulaName, guardFields, guardRows, ruleSteps, ruleUpTo, stepStats, templatesFor,
  type Formula, type Guardrail, type GuardField, type GuardrailSource,
} from '@/lib/guardrails'

/**
 * The guardrail editor: a popup where a reader writes the rule that picks a specification
 * curve's red dots, instead of choosing from fixed presets. The chart's curve sits at the top
 * of the popup (`preview`, kept in view while the options scroll) and every edit is sent back
 * (`onPreview`), so the red dots move while the rule is written; Apply keeps it, Cancel puts
 * back what was there. react-querybuilder draws the rule
 * (AND / OR groups, NOT, a comparison per rule, against a value or another field); calculated
 * fields add formulas over the numeric fields (lib/formula); lib/guardrails runs it. The count
 * of passing configurations updates as the rule changes, before anything is applied.
 *
 * Named rule sets are kept in this browser (localStorage); the applied rule travels in the URL.
 */
const STORE = 'gallery.guardrails.v1'
interface SavedSet { name: string; rule: Guardrail }
const readSaved = (): SavedSet[] => { try { const v = JSON.parse(localStorage.getItem(STORE) ?? '[]'); return Array.isArray(v) ? v : [] } catch { return [] } }
const writeSaved = (sets: SavedSet[]) => { try { localStorage.setItem(STORE, JSON.stringify(sets)) } catch { /* private window: sets last this visit only */ } }

const NUMBER_OPS = ['<=', '<', '>=', '>', 'between', 'notBetween', '=', '!=', 'null', 'notNull']
const CATEGORY_OPS = ['in', 'notIn', '=', '!=']
const OP_LABELS: Record<string, string> = {
  '<=': '≤', '<': '<', '>=': '≥', '>': '>', between: 'between', notBetween: 'not between', '=': '=', '!=': '≠',
  null: 'is missing', notNull: 'is present', in: 'is one of', notIn: 'is not one of',
}

/** the editor's field list: grouped, with each kind's operators and value editor */
function builderFields(fields: GuardField[]): OptionGroup<Field>[] {
  const groups: OptionGroup<Field>[] = []
  for (const f of fields) {
    const field: Field = {
      name: f.name, value: f.name, label: f.unit ? `${f.label} (${f.unit})` : f.label, kind: f.kind, title: f.help,
      ...(f.kind === 'number' ? {
        inputType: 'number', defaultOperator: '<=', operators: NUMBER_OPS.map((name) => ({ name, value: name, label: OP_LABELS[name] })),
        valueSources: ['value', 'field'], comparator: 'kind',
      } : f.kind === 'category' ? {
        defaultOperator: 'in', operators: CATEGORY_OPS.map((name) => ({ name, value: name, label: OP_LABELS[name] })),
        valueEditorType: (op: string) => (op === 'in' || op === 'notIn' ? 'multiselect' : 'select'),
        values: f.values?.map((v) => ({ name: v.name, value: v.name, label: v.label })),
      } : {
        defaultOperator: '=', defaultValue: true, operators: [{ name: '=', value: '=', label: 'is' }],
        valueEditorType: 'checkbox',
      }),
    }
    const g = groups.find((x) => x.label === f.group)
    if (g) g.options.push(field)
    else groups.push({ label: f.group, options: [field] })
  }
  return groups
}

const CLASSES = {
  combinators: 'select', fields: 'select', operators: 'select', value: 'select', valueSource: 'select',
  addRule: 'btn quiet', addGroup: 'btn quiet', removeRule: 'btn quiet', removeGroup: 'btn quiet', notToggle: 'toggle',
}

/**
 * The gallery's own dropdowns in place of the builder's native selects: Combobox for one choice
 * (option groups become its headings; 'Deming' follows the page's estimator, as everywhere), and a
 * checklist for "is one of" lists.
 */
type Opt = { name?: string; value?: string; label: string }
const flatOptions = (options: ValueSelectorProps['options']) => (isOptionGroupArray(options)
  ? (options as unknown as { label: string; options: Opt[] }[]).flatMap((g) => g.options.map((o) => ({ ...o, group: g.label })))
  : (options as unknown as Opt[]).map((o) => ({ ...o, group: undefined as string | undefined })))
function GuardSelect(props: ValueSelectorProps) {
  const method = useEstimator()
  const opts = flatOptions(props.options).map((o) => ({ key: String(o.value ?? o.name), label: estimatorText(o.label, method), group: o.group }))
  if (props.multiple) return <GuardMultiSelect {...props} opts={opts} />
  // the builder's class names, without the native-select styling (the combobox draws its own box)
  return <span className={(props.className ?? '').replace(/\bselect\b/g, '').trim()}><Combobox label="" value={String(props.value ?? '')} options={opts} title={props.title}
    onChange={(v) => props.handleOnChange(v)} /></span>
}
function GuardMultiSelect({ value, handleOnChange, title, opts }: ValueSelectorProps & { opts: { key: string; label: string }[] }) {
  const chosen = String(value ?? '').split(',').map((v) => v.trim()).filter(Boolean)
  return <MultiCombobox label="" title={title} value={chosen} options={opts} onChange={(keys) => handleOnChange(keys.join(','))} />
}
const SELECTORS = { combinatorSelector: GuardSelect, fieldSelector: GuardSelect, operatorSelector: GuardSelect, valueSourceSelector: GuardSelect, valueSelector: GuardSelect }

/** a range reads "0.7 and 1.3" */
const RangeValueEditor = (props: ValueEditorProps) => <ValueEditor {...props} separator={<span className="guard-and">and</span>} />

export function GuardrailEditor({ title, src, scope, initial, onApply, onPreview, preview, onClose }: {
  title: string
  src: GuardrailSource
  /** the rows the counts cover (the chart's configurations in view); every row when absent */
  scope?: number[]
  initial: Guardrail
  onApply: (rule: Guardrail) => void
  /** each edit, as it is made (null while a calculated field does not parse) */
  onPreview?: (rule: Guardrail | null) => void
  /** the live chart drawn above the options */
  preview?: ReactNode
  onClose: () => void
}) {
  const [query, setQuery] = useState<RuleGroupType>(initial.query)
  const [formulas, setFormulas] = useState<Formula[]>(initial.formulas)
  const [saved, setSaved] = useState<SavedSet[]>(readSaved)
  const [saveName, setSaveName] = useState('')
  const dialog = useRef<HTMLDivElement>(null)
  // focus the popup once when it opens; Esc cancels (via a ref, so typing never re-runs the focus)
  // closing with changes asks first: apply them, discard them, or keep editing
  const [confirming, setConfirming] = useState(false)
  const snapshot = (q: RuleGroupType, fs: Formula[]) => formatQuery(q, 'json_without_ids') + JSON.stringify(fs.filter((f) => f.formula.trim()))
  const initialSnapshot = useMemo(() => snapshot(initial.query, initial.formulas), [initial]) // eslint-disable-line react-hooks/exhaustive-deps
  const dirty = snapshot(query, formulas) !== initialSnapshot
  const requestClose = () => (dirty ? setConfirming(true) : onClose())
  const close = useRef(requestClose)
  close.current = confirming ? () => setConfirming(false) : requestClose
  useEffect(() => {
    dialog.current?.focus()
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') close.current() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [])

  const rows = useMemo(() => guardRows(src), [src])
  const baseFields = useMemo(() => guardFields(src), [src])
  const fields = useMemo(() => guardFields(src, formulas.filter((f) => f.formula.trim())), [src, formulas])
  const numericNames = baseFields.filter((f) => f.kind === 'number').map((f) => f.name)
  const rule: Guardrail = { query, formulas: formulas.filter((f) => f.formula.trim()) }
  const result = useMemo(() => evaluateGuardrail(rule, rows, numericNames), [query, formulas, rows]) // eslint-disable-line react-hooks/exhaustive-deps
  const inScope = useMemo(() => scope ?? rows.map((_, i) => i), [scope, rows])
  // apply in order: each top-level condition (a rule or a whole group) as a step, with what it
  // removes on its own, in sequence, alone among the others, and together with each other one
  const [stepping, setStepping] = useState(false)
  const [upTo, setUpTo] = useState<number | null>(null)
  const steps = useMemo(() => ruleSteps(rule), [query, formulas]) // eslint-disable-line react-hooks/exhaustive-deps
  const stats = useMemo(() => {
    if (!stepping || result.error || !steps.length) return null
    const pass = steps.map((s) => evaluateGuardrail(s, rows, numericNames).pass)
    return stepStats(pass, inScope, query.combinator)
  }, [stepping, steps, rows, inScope, result.error]) // eslint-disable-line react-hooks/exhaustive-deps
  const shownUpTo = stepping && upTo !== null && upTo < steps.length ? upTo : null
  // the chart previews the rule being written (cut at a step when one is picked)
  const previewRule = shownUpTo === null ? rule : ruleUpTo(rule, shownUpTo)
  // the chart previews the rule being written
  const sendPreview = useRef(onPreview)
  sendPreview.current = onPreview
  useEffect(() => { sendPreview.current?.(result.error ? null : previewRule) }, [result, shownUpTo]) // eslint-disable-line react-hooks/exhaustive-deps
  const groups = useMemo(() => builderFields(fields), [fields])
  const templates = templatesFor(src)

  const load = (r: Guardrail) => { setQuery(structuredClone(r.query)); setFormulas(structuredClone(r.formulas)) }
  const addFormula = () => {
    const label = `Calculated ${formulas.length + 1}`
    setFormulas((fs) => [...fs, { name: formulaName(label, [...baseFields.map((f) => f.name), ...fs.map((f) => f.name)]), label, formula: '' }])
  }
  const editFormula = (i: number, patch: Partial<Formula>) => setFormulas((fs) => fs.map((f, j) => {
    if (j !== i) return f
    const next = { ...f, ...patch }
    // the name follows the label (rules refer to the name); renaming a field in use leaves its rules empty
    if (patch.label !== undefined) next.name = formulaName(patch.label || 'calc', [...baseFields.map((x) => x.name), ...fs.filter((_, k) => k !== i).map((x) => x.name)])
    return next
  }))
  const saveSet = () => {
    const name = saveName.trim()
    if (!name) return
    const next = [...saved.filter((s) => s.name !== name), { name, rule }]
    setSaved(next); writeSaved(next); setSaveName('')
  }
  const removeSet = (name: string) => { const next = saved.filter((s) => s.name !== name); setSaved(next); writeSaved(next) }

  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) requestClose() }}>
      <div ref={dialog} tabIndex={-1} className="export-dialog guard-dialog" role="dialog" aria-modal="true" aria-label={`Guardrails for ${title}`}>
        <div className="guard-top">
          <header className="guard-head">
            <div><strong>Guardrails</strong><span className="export-title">{title}</span></div>
            <button type="button" className="btn quiet" onClick={requestClose} aria-label="Cancel" title="Cancel (Esc): put the red dots back">✕</button>
          </header>
          {preview && <div className="guard-preview">{preview}</div>}
        </div>

        <div className="guard-start">
          <span className="control">Start from</span>
          {templates.map((t) => <button key={t.key} type="button" className="btn quiet" onClick={() => load(t.rule)}>{t.label}</button>)}
          <button type="button" className="btn quiet" onClick={() => load({ query: { combinator: 'and', rules: [] }, formulas: [] })}>Empty</button>
          {saved.length > 0 && <span className="guard-saved">
            <span className="control">Saved</span>
            {saved.map((s) => <span key={s.name} className="guard-chip">
              <button type="button" className="btn quiet" onClick={() => load(s.rule)} title="Load this rule set">{s.name}</button>
              <button type="button" className="btn quiet" onClick={() => removeSet(s.name)} aria-label={`Delete ${s.name}`} title="Delete">×</button>
            </span>)}
          </span>}
        </div>

        <section className="guard-builder">
          <QueryBuilder fields={groups} query={query} onQueryChange={setQuery} controlClassnames={CLASSES} controlElements={{ ...SELECTORS, valueEditor: RangeValueEditor }}
            showNotToggle resetOnFieldChange addRuleToNewGroups
            translations={{ addRule: { label: '+ Condition', title: 'Add a condition' }, addGroup: { label: '+ Group', title: 'Add a group of conditions' },
              removeRule: { label: '×', title: 'Remove this condition' }, removeGroup: { label: '×', title: 'Remove this group' }, notToggle: { label: 'Not', title: 'Pass when this group does NOT hold' } }}
            combinators={[{ name: 'and', value: 'and', label: 'All of' }, { name: 'or', value: 'or', label: 'Any of' }]} />
        </section>

        <section className="guard-formulas">
          <div className="guard-formulas-head">
            <strong>Calculated fields</strong>
            <span className="control">a formula over the fields below, usable in any condition (Calculated group)</span>
            <button type="button" className="btn quiet" onClick={addFormula}>+ Calculated field</button>
          </div>
          {formulas.map((f, i) => <div key={i} className="guard-formula">
            <input className="select" aria-label="Calculated field name" value={f.label} onChange={(e) => editFormula(i, { label: e.target.value })} style={{ width: 180 }} />
            <span>=</span>
            <input className="select guard-formula-text" aria-label={`Formula for ${f.label}`} value={f.formula} placeholder="e.g. abs(slope - 1) * 10"
              onChange={(e) => editFormula(i, { formula: e.target.value })} spellCheck={false} />
            <code className="guard-name" title="Its name in other formulas">{f.name}</code>
            <button type="button" className="btn quiet" onClick={() => setFormulas((fs) => fs.filter((_, j) => j !== i))} aria-label={`Remove ${f.label}`}>×</button>
          </div>)}
          {(formulas.length > 0) && <p className="guard-help">
            Fields: {baseFields.filter((f) => f.kind === 'number').map((f) => <code key={f.name} title={f.label}>{f.name}</code>)}
            {' '}· Functions: {Object.entries(FORMULA_FUNCTIONS).map(([k, v]) => <code key={k} title={v.help}>{k}()</code>)} · Operators: + − * / ^ ( )
          </p>}
        </section>

        {stats && <StepTable stats={stats} labels={steps.map((st) => describeGuardrail(st, fields))} combinator={query.combinator}
          upTo={shownUpTo} onUpTo={setUpTo} />}

        <footer className="guard-foot">
          <div className="guard-result" aria-live="polite">
            {/* no overall count here: what passes depends on the panel being looked at, which the preview shows */}
            {result.error && <span className="guard-error">{result.error}</span>}
            {steps.length > 1 && <span className="guard-step-toggle"><Toggle label="Apply in order" checked={stepping} onChange={(v) => { setStepping(v); setUpTo(null) }}
              title="Apply the conditions one at a time, in the order listed: what each removes, and what they share" /></span>}
          </div>
          <div className="guard-actions">
            <input className="select" placeholder="Name this rule set" value={saveName} onChange={(e) => setSaveName(e.target.value)} aria-label="Rule set name"
              onKeyDown={(e) => { if (e.key === 'Enter') saveSet() }} style={{ width: 170 }} />
            <button type="button" className="btn quiet" onClick={saveSet} disabled={!saveName.trim()}>Save</button>
            <span style={{ flex: 1 }} />
            <button type="button" className="btn quiet" onClick={requestClose}>Cancel</button>
            <button type="button" className="btn" onClick={() => { onApply(rule); onClose() }} disabled={!!result.error}>Apply</button>
          </div>
        </footer>
        {confirming && <div className="guard-confirm-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) setConfirming(false) }}>
          <div className="guard-confirm" role="alertdialog" aria-modal="true" aria-labelledby="guard-confirm-title">
            <strong id="guard-confirm-title">Keep these guardrail changes?</strong>
            <p>You changed the conditions. Apply them to the red dots, or discard them and go back to what was there before.</p>
            <div className="guard-actions">
              <button type="button" className="btn quiet" onClick={() => setConfirming(false)} autoFocus>Keep editing</button>
              <span style={{ flex: 1 }} />
              <button type="button" className="btn quiet danger" onClick={onClose}>Discard</button>
              <button type="button" className="btn" onClick={() => { onApply(rule); onClose() }} disabled={!!result.error}
                title={result.error ? 'A calculated field does not parse' : undefined}>Apply</button>
            </div>
          </div>
        </div>}
      </div>
    </div>,
    document.body,
  )
}

/**
 * The conditions applied one at a time, in order. Each row: the condition, how many configurations
 * fail it on its own, how many it removes from what the earlier rows left (or, joined by Any of,
 * adds), how many are left, and how many only it removes. Picking a row previews the rule up to
 * there on the chart. Beside it, how many removals each pair of conditions share.
 */
function StepTable({ stats, labels, combinator, upTo, onUpTo }: {
  stats: ReturnType<typeof stepStats>; labels: string[]; combinator: string; upTo: number | null; onUpTo: (k: number | null) => void
}) {
  const or = combinator === 'or'
  const pct = (v: number) => (stats.n ? `${Math.round((100 * v) / stats.n)} %` : '')
  const max = Math.max(1, ...stats.fails)
  const heat = (v: number) => `rgba(178, 24, 43, ${(0.06 + 0.5 * (v / max)).toFixed(2)})`
  const last = labels.length
  const current = upTo ?? last
  return <section className="guard-steps">
    <table className="census-table guard-step-table">
      <thead><tr>
        <th title="Preview the rule up to this row on the chart">Preview</th><th>#</th><th>Condition</th>
        <th title="Configurations that fail this condition, ignoring the others">Fail it alone</th>
        <th title={or ? 'Configurations this step adds to what passed before it' : 'Configurations this step removes from what the earlier steps left'}>{or ? 'Added here' : 'Removed here'}</th>
        <th>{or ? 'Passing' : 'Left'}</th>
        {!or && <th title="Configurations only this condition removes: they pass every other one">Only this one</th>}
      </tr></thead>
      <tbody>
        <tr className={current === 0 ? 'on' : ''}>
          <td><input type="radio" name="guard-upto" checked={current === 0} onChange={() => onUpTo(0)} aria-label="Preview with no conditions" /></td>
          <td></td><td className="guard-step-label">{or ? 'None passing yet' : 'All configurations in view'}</td><td></td><td></td><td>{(or ? 0 : stats.n).toLocaleString()}</td>{!or && <td></td>}
        </tr>
        {labels.map((label, s) => <tr key={s} className={current === s + 1 ? 'on' : ''}>
          <td><input type="radio" name="guard-upto" checked={current === s + 1} onChange={() => onUpTo(s + 1 === last ? null : s + 1)} aria-label={`Preview up to condition ${s + 1}`} /></td>
          <td>{s + 1}</td>
          <td className="guard-step-label" title={label}>{label}</td>
          <td>{stats.fails[s].toLocaleString()} <span className="guard-pct">{pct(stats.fails[s])}</span></td>
          <td>{stats.atStep[s] ? `${or ? '+' : '−'}${stats.atStep[s].toLocaleString()}` : '0'}</td>
          <td><strong>{stats.left[s].toLocaleString()}</strong></td>
          {!or && <td>{stats.only[s].toLocaleString()}</td>}
        </tr>)}
      </tbody>
    </table>
    {labels.length > 1 && <div className="guard-shared">
      <span className="control">Removed by both conditions (row × column); the diagonal is each alone</span>
      <table className="guard-shared-table">
        <thead><tr><th></th>{labels.map((l, j) => <th key={j} title={l}>{j + 1}</th>)}</tr></thead>
        <tbody>{stats.shared.map((row, i) => <tr key={i}><th title={labels[i]}>{i + 1}</th>{row.map((v, j) =>
          <td key={j} style={{ background: heat(v), fontWeight: i === j ? 700 : 400 }} title={`${i + 1}: ${labels[i]}\n${i === j ? 'fail it' : `${j + 1}: ${labels[j]}`}\n${v.toLocaleString()} configurations`}>{v.toLocaleString()}</td>)}</tr>)}</tbody>
      </table>
    </div>}
    <p className="guard-help">Apply keeps the whole rule; picking a row only previews the rule up to there on the chart.</p>
  </section>
}
