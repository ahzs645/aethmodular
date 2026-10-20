export type HighlightRule = 'guardrails' | 'intercept' | 'cv-slope' | 'custom'
export interface HighlightLimits { intercept: number; slopeMin: number; slopeMax: number; cvMin: number }
export const DEFAULT_HIGHLIGHT_LIMITS: HighlightLimits = { intercept: 0.5, slopeMin: 0.7, slopeMax: 1.3, cvMin: 0.85 }
export interface HighlightMetrics { intercept: number | null; slope: number | null; cv: number | null; guardrails: boolean }

/** Missing measurements cannot satisfy a criterion that requires them. */
export function highlightMatches(metrics: HighlightMetrics, rule: HighlightRule, limits: HighlightLimits): boolean {
  if (rule === 'guardrails') return metrics.guardrails
  const nearZero = metrics.intercept !== null && Number.isFinite(metrics.intercept) && Math.abs(metrics.intercept) <= limits.intercept
  if (rule === 'intercept') return nearZero
  const cvSlope = metrics.slope !== null && Number.isFinite(metrics.slope) && metrics.slope >= limits.slopeMin && metrics.slope <= limits.slopeMax
    && metrics.cv !== null && Number.isFinite(metrics.cv) && metrics.cv >= limits.cvMin
  return cvSlope && (rule !== 'custom' || nearZero)
}

/** Preserve intercept ordering within each selection population; never pair unlike configurations. */
export function selectionPanels(shown: number[], codes: number[], categories: string[]) {
  return categories.map((key, code) => ({ key, indices: shown.filter(i => codes[i] === code) })).filter(p => p.indices.length > 0)
}

type ChoiceDimension = 'co' | 'sp' | 'sel' | 'mo' | 'lot' | 'cut' | 'k'
interface ChoiceFile extends Record<ChoiceDimension, (number | null)[]> { n: number; masks: string[] }

/** Identify the setting each saved mask represents without parsing its display label.
 * Cohort takes precedence over cutoff when a complete cohort has a unique size.
 * Unknown compound choices retain their own independent requirement.
 */
export function choiceFilterDimensions(file: ChoiceFile): string[] {
  const keys: ChoiceDimension[] = ['co', 'sp', 'sel', 'mo', 'lot', 'k', 'cut']
  return file.masks.map((mask, c) => keys.find(key => {
    const bits = new Map<number | null, string>()
    for (let i = 0; i < file.n; i++) {
      const value = file[key][i], bit = mask[i]
      if (bits.has(value) && bits.get(value) !== bit) return false
      bits.set(value, bit)
    }
    return new Set(bits.values()).size > 1
  }) ?? `choice-${c}`)
}

/** Alternatives within one setting are combined; different settings narrow the search. */
export function choiceFilterSets(required: ReadonlySet<number>, dimensions: readonly string[]): number[][] {
  const groups = new Map<string, number[]>()
  for (const c of required) {
    const dimension = dimensions[c] ?? `choice-${c}`
    const choices = groups.get(dimension) ?? []
    choices.push(c)
    groups.set(dimension, choices)
  }
  return [...groups.values()]
}

export function matchesChoiceFilters(masks: readonly string[], i: number, groups: readonly number[][]): boolean {
  return groups.every(choices => choices.some(c => masks[c][i] === '1'))
}

/** The columns of a specification export that a merge carries (spec_curve.json's schema). */
export interface SpecColumns {
  n: number; n_pass: number; guardrails: string
  choices: string[]; groups?: { title: string; start: number; end: number }[]; masks: string[]; ok: string
  intercept: (number | null)[]; slope: (number | null)[]; heldout: (number | null)[]
  olsIntercept?: (number | null)[]; olsSlope?: (number | null)[]
  cut: (number | null)[]; k: (number | null)[]
  categories: { co: string[]; sp: string[]; sel: string[]; mo: string[]; lot: string[] }
  co: number[]; sp: number[]; sel: number[]; mo: number[]; lot: number[]
  extra?: Record<string, (number | null)[]>
}

/**
 * Several specification searches as one curve: rows appended in order, categories unioned (the
 * same key, e.g. 'airspec', is one category), each analytic choice kept once by its label (rows
 * of a search without it are marked unused), choice rows regrouped under their headings, and a
 * "Search" group whose rows say which search each configuration came from. Columns only one
 * search has are missing (null) for the others' rows, so rules and fits on them leave those
 * rows out rather than guess. Rows are re-sorted by intercept, as every export is.
 */
export function mergeSpecFiles<T extends SpecColumns>(files: { file: T; name: string }[]): T {
  if (files.length === 1) return files[0].file
  const n = files.reduce((s, f) => s + f.file.n, 0)
  const keys = ['co', 'sp', 'sel', 'mo', 'lot'] as const
  const categories = Object.fromEntries(keys.map(k => [k, [...new Set(files.flatMap(f => f.file.categories[k]))]])) as T['categories']
  const codes = Object.fromEntries(keys.map(k => [k, files.flatMap(f => f.file[k].map(c => categories[k].indexOf(f.file.categories[k][c])))])) as Record<typeof keys[number], number[]>
  const col = <V,>(get: (f: T) => V[] | undefined, fill: V) => files.flatMap(f => get(f.file) ?? new Array<V>(f.file.n).fill(fill))
  // each choice once, under its heading; heading order follows the first search that has it
  const headingOf = (f: T, c: number) => f.groups?.find(g => c >= g.start && c < g.end)?.title ?? ''
  const rows = new Map<string, { title: string; bits: string[] }>()
  files.forEach((f, fi) => f.file.choices.forEach((label, c) => {
    const row = rows.get(label) ?? { title: headingOf(f.file, c), bits: files.map(g => '0'.repeat(g.file.n)) }
    row.bits[fi] = f.file.masks[c]
    rows.set(label, row)
  }))
  files.forEach((f, fi) => rows.set(`From the ${f.name} search`, { title: 'Search', bits: files.map((g, gi) => (gi === fi ? '1' : '0').repeat(g.file.n)) }))
  const titles = [...new Set([...rows.values()].map(r => r.title))]
  const ordered = titles.flatMap(t => [...rows.entries()].filter(([, r]) => r.title === t))
  let at = 0
  const groups = titles.filter(Boolean).map(title => { const len = ordered.filter(([, r]) => r.title === title).length; const g = { title, start: at, end: at + len }; at += len; return g })
  const extraKeys = [...new Set(files.flatMap(f => Object.keys(f.file.extra ?? {})))]
  const merged = {
    ...files[0].file,
    n, n_pass: files.reduce((s, f) => s + f.file.n_pass, 0),
    choices: ordered.map(([label]) => label), groups, masks: ordered.map(([, r]) => r.bits.join('')),
    ok: files.map(f => f.file.ok).join(''),
    intercept: col(f => f.intercept, null), slope: col(f => f.slope, null), heldout: col(f => f.heldout, null),
    olsIntercept: col(f => f.olsIntercept, null), olsSlope: col(f => f.olsSlope, null),
    cut: col(f => f.cut, null), k: col(f => f.k, null),
    categories, ...codes,
    extra: Object.fromEntries(extraKeys.map(key => [key, col(f => f.extra?.[key], null)])),
  } as T
  // back into intercept order, every column together
  const order = merged.intercept.map((_, i) => i).sort((a, b) => (merged.intercept[a] ?? Infinity) - (merged.intercept[b] ?? Infinity))
  const pick = <V,>(a: V[]) => order.map(i => a[i])
  const bits = (s: string) => order.map(i => s[i]).join('')
  return {
    ...merged,
    masks: merged.masks.map(bits), ok: bits(merged.ok),
    intercept: pick(merged.intercept), slope: pick(merged.slope), heldout: pick(merged.heldout),
    olsIntercept: pick(merged.olsIntercept!), olsSlope: pick(merged.olsSlope!), cut: pick(merged.cut), k: pick(merged.k),
    co: pick(merged.co), sp: pick(merged.sp), sel: pick(merged.sel), mo: pick(merged.mo), lot: pick(merged.lot),
    extra: Object.fromEntries(Object.entries(merged.extra!).map(([k, v]) => [k, pick(v)])),
  }
}
