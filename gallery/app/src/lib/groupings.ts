/**
 * The reader's own groupings: one measurement split into labelled bands ("Iron: Low / High"), usable in every
 * chart as panels or as a colour. The notebooks did this case by case (pd.cut on fixed cutoffs, e.g. rain as
 * No rain / Light / Moderate / Heavy or FTIR EC in 2 µg/m³ steps; pd.qcut on quantiles, e.g. low / mid / high OP
 * tertiles, OC/EC deciles); here it is one definition, kept in the URL (`grp`) so a shared link carries it.
 *
 * Bands are half-open, [lower, upper), like pd.cut(right=False): a value equal to a cutoff goes to the band above.
 * Quantile cutoffs are computed from the values in view (every site pooled), so they follow the subset.
 */
export type SplitKind = 'cutoffs' | 'median' | 'tertiles' | 'quartiles' | 'quintiles' | 'deciles'

export interface GroupingDef {
  /** stable key, also the chart option's value */
  id: string
  name: string
  field: string
  kind: SplitKind
  /** the cutoffs, ascending, when kind is 'cutoffs' */
  cuts: number[]
  /** one per band; empty or blank entries fall back to an automatic label */
  labels: string[]
}

export const SPLITS: { kind: SplitKind; label: string; bands?: number }[] = [
  { kind: 'cutoffs', label: 'Cutoffs' },
  { kind: 'median', label: 'Median split', bands: 2 },
  { kind: 'tertiles', label: 'Tertiles', bands: 3 },
  { kind: 'quartiles', label: 'Quartiles', bands: 4 },
  { kind: 'quintiles', label: 'Quintiles', bands: 5 },
  { kind: 'deciles', label: 'Deciles', bands: 10 },
]

const QUANTILE_NAMES: Record<number, string[]> = {
  2: ['Low', 'High'], 3: ['Low', 'Mid', 'High'], 4: ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'], 5: ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)'],
  10: Array.from({ length: 10 }, (_, i) => `D${i + 1}${i === 0 ? ' (lowest)' : i === 9 ? ' (highest)' : ''}`),
}

/** the quantile of sorted values (linear between order statistics, as numpy's default) */
function quantileSorted(sorted: number[], p: number): number {
  if (!sorted.length) return NaN
  const h = (sorted.length - 1) * p
  const lo = Math.floor(h)
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (h - lo)
}

/** The cutoffs a grouping uses on these values: its own, or the quantiles of the values. */
export function cutsFor(def: GroupingDef, values: number[]): number[] {
  if (def.kind === 'cutoffs') return [...def.cuts].filter(Number.isFinite).sort((a, b) => a - b)
  const bands = SPLITS.find((s) => s.kind === def.kind)?.bands ?? 2
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b)
  return Array.from({ length: bands - 1 }, (_, i) => quantileSorted(sorted, (i + 1) / bands))
}

/** the band of a value: 0 below the first cutoff, …, cuts.length at or above the last; null when there is no value */
export function bandOf(cuts: number[], v: unknown): number | null {
  if (typeof v !== 'number' || !Number.isFinite(v)) return null
  let i = 0
  while (i < cuts.length && v >= cuts[i]) i++
  return i
}

const short = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2))

/** "< 2", "2 to 4", "≥ 4": what a band covers, for the editor and tooltips */
export function rangeText(cuts: number[], i: number): string {
  if (!cuts.length) return 'all values'
  if (i === 0) return `< ${short(cuts[0])}`
  if (i === cuts.length) return `≥ ${short(cuts[cuts.length - 1])}`
  return `${short(cuts[i - 1])} to < ${short(cuts[i])}`
}

/** each band's label: the reader's, else Low / Mid / High for quantiles, else its range */
export function labelsFor(def: GroupingDef, cuts: number[]): string[] {
  const n = cuts.length + 1
  const auto = def.kind === 'cutoffs' ? null : QUANTILE_NAMES[n]
  return Array.from({ length: n }, (_, i) => def.labels[i]?.trim() || auto?.[i] || rangeText(cuts, i))
}

/** Ordered bands get an ordered palette: blue (low) to red (high), through amber for three or more. */
export function colorsFor(n: number): string[] {
  const ramps: Record<number, string[]> = {
    1: ['#4e79a7'], 2: ['#4e79a7', '#e15759'], 3: ['#4e79a7', '#edc948', '#e15759'],
    4: ['#4e79a7', '#76b7b2', '#f28e2b', '#e15759'], 5: ['#4e79a7', '#76b7b2', '#edc948', '#f28e2b', '#e15759'],
  }
  if (ramps[n]) return ramps[n]
  // more bands: interpolate along the five-colour ramp, so neighbours differ and none repeats
  const base = ramps[5].map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)))
  return Array.from({ length: n }, (_, i) => {
    const t = (i / (n - 1)) * (base.length - 1), k = Math.min(base.length - 2, Math.floor(t)), f = t - k
    return '#' + base[k].map((c, j) => Math.round(c + (base[k + 1][j] - c) * f).toString(16).padStart(2, '0')).join('')
  })
}

/** the colour of filters without the grouping's measurement (drawn, in no band) */
export const NO_BAND_COLOR = '#9aa5b1'
/** the legend name of those filters (shown only when there are any) */
export const noBandLabel = (_d: GroupingDef) => 'No data'

/** A grouping applied to rows: cutoffs, labels, colours, and each row's band label (null: no value). */
export function applyGrouping<R extends Record<string, unknown>>(def: GroupingDef, rows: R[]) {
  const values = rows.map((r) => r[def.field]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  const cuts = cutsFor(def, values)
  const labels = labelsFor(def, cuts)
  const colors = colorsFor(labels.length)
  const labelOf = (r: R) => { const b = bandOf(cuts, r[def.field]); return b === null ? null : labels[b] }
  return { cuts, labels, colors, labelOf, colorOf: (label: string) => colors[labels.indexOf(label)] ?? NO_BAND_COLOR }
}

// ---------------------------------------------------------------- the URL (`grp`)
const toB64 = (s: string) => btoa(String.fromCharCode(...new TextEncoder().encode(s))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
const fromB64 = (t: string) => new TextDecoder().decode(Uint8Array.from(atob(t.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0)))

export function groupingsParam(defs: GroupingDef[]): string | null {
  if (!defs.length) return null
  return toB64(JSON.stringify(defs.map((d) => [d.id, d.name, d.field, d.kind, d.cuts, d.labels])))
}

export function parseGroupings(param: string | null): GroupingDef[] {
  if (!param) return []
  try {
    const raw = JSON.parse(fromB64(param)) as unknown[]
    return raw.flatMap((x) => {
      if (!Array.isArray(x) || x.length < 6) return []
      const [id, name, field, kind, cuts, labels] = x as [string, string, string, SplitKind, number[], string[]]
      if (typeof id !== 'string' || typeof field !== 'string' || !SPLITS.some((s) => s.kind === kind)) return []
      return [{ id, name: String(name ?? ''), field, kind, cuts: (cuts ?? []).filter((v) => typeof v === 'number'), labels: (labels ?? []).map(String) }]
    })
  } catch {
    return []
  }
}

export const newGroupingId = () => `g${Date.now().toString(36)}${Math.floor(Math.random() * 1296).toString(36)}`

// ---------------------------------------------------------------- the browser (localStorage)
/**
 * The reader's groupings also live in this browser, so they survive closing the tab; a link's groupings (`grp`)
 * join them when it is opened. Same id = same grouping: the link's copy wins, nothing is duplicated.
 */
const SAVED_KEY = 'aeth-gallery:groupings'
export function readSavedGroupings(): GroupingDef[] {
  try { return parseGroupings(localStorage.getItem(SAVED_KEY)) } catch { return [] }
}
export function writeSavedGroupings(defs: GroupingDef[]) {
  try {
    const v = groupingsParam(defs)
    if (v) localStorage.setItem(SAVED_KEY, v)
    else localStorage.removeItem(SAVED_KEY)
  } catch { /* private window or blocked storage: the link still carries them */ }
}
/** `first` wins on a shared id; order: `first`, then the rest of `second` */
export function mergeGroupings(first: GroupingDef[], second: GroupingDef[]): GroupingDef[] {
  const ids = new Set(first.map((d) => d.id))
  return [...first, ...second.filter((d) => !ids.has(d.id))]
}

// ---------------------------------------------------------------- from the notebooks
/** a grouping an earlier analysis used, offered ready-made; `source` names the notebook */
export interface BuiltInGrouping extends GroupingDef { source: string; note?: string }
const nb = (id: string, name: string, field: string, kind: SplitKind, cuts: number[], labels: string[], source: string, note?: string): BuiltInGrouping =>
  ({ id: `nb-${id}`, name, field, kind, cuts, labels, source, note })
/**
 * Splits earlier analyses used, on a measurement the gallery has (surveyed 22 Oct 2026 from research/ and notebooks/).
 * Cutoffs keep the notebooks' values; where the notebook binned something else (hourly rain, IMPROVE TOR OC/EC, a
 * per-site median) the note says so. Shown only when the view has the field.
 */
export const BUILT_IN_GROUPINGS: BuiltInGrouping[] = [
  // weather (means over each filter's sampling window)
  nb('rain-station', 'Rain (station)', 'Precipitation (station)', 'cutoffs', [0.1, 1, 5],
    ['No rain', 'Light (0.1 to 1 mm/day)', 'Moderate (1 to 5 mm/day)', 'Heavy (≥ 5 mm/day)'],
    'research/ftir_hips_chem/addis_03_meteorology.ipynb (analyze_bc_precipitation_relationship)', 'the notebook cut hourly rain at 0, 1, 5 mm; here the window mean in mm/day'),
  nb('rain-era5', 'Rain (ERA5)', 'Precipitation (ERA5)', 'cutoffs', [0.1, 1, 5],
    ['No rain', 'Light (0.1 to 1 mm/day)', 'Moderate (1 to 5 mm/day)', 'Heavy (≥ 5 mm/day)'],
    'research/ftir_hips_chem/addis_03_meteorology.ipynb', 'as the station version; ERA5 rains about 1.8× the Addis station'),
  nb('rain-meteostat', 'Rain, wider classes (station)', 'Precipitation (station)', 'cutoffs', [0.1, 2, 10],
    ['Dry', 'Light (0.1 to 2 mm)', 'Moderate (2 to 10 mm)', 'Heavy (≥ 10 mm)'],
    'notebooks/analysis/meteorology/meteorology_source_interaction.ipynb (prcp_bins)'),
  nb('rh-55', 'Dry vs humid days (RH 55 %)', 'Relative humidity (station)', 'cutoffs', [55], ['Dry (RH < 55 %)', 'Humid (RH ≥ 55 %)'],
    'notebooks/analysis/meteorology/meteorology_source_interaction.ipynb (RH_THRESHOLD, the sweep inflection)'),
  nb('rh-median', 'Dry vs humid days (median RH)', 'Relative humidity (station)', 'median', [], ['Dry', 'Humid'],
    'notebooks/analysis/meteorology/hips_ftir_intercept_rh_recursive_diagnostics.ipynb', 'HIPS vs FTIR regressions split by RH'),
  nb('trange-quartiles', 'Temperature range quartiles', 'Temperature range (station)', 'quartiles', [], [],
    'notebooks/analysis/meteorology/meteorology_source_interaction.ipynb (pd.qcut temp_range, q=4)'),
  nb('temp-quartiles', 'Cold to warm quartiles', 'Temperature (station)', 'quartiles', [], ['Q1 (coldest)', 'Q2', 'Q3', 'Q4 (warmest)'],
    'notebooks/analysis/meteorology/temperature_gradient_analysis.ipynb', 'the notebook used daily minimum temperature; here the window mean'),
  nb('wind-classes', 'Wind classes', 'Wind speed (station)', 'cutoffs', [2, 4, 6, 8],
    ['Calm (< 2 m/s)', '2 to 4 m/s', '4 to 6 m/s', '6 to 8 m/s', '≥ 8 m/s'],
    'notebooks/analysis/meteorology/meteorology_revisions.ipynb (SPEED_BINS_MS)'),
  nb('blh-classes', 'Boundary-layer height classes', 'Boundary-layer height (ERA5)', 'cutoffs', [500, 1000],
    ['< 500 m', '500 to 1000 m', '≥ 1000 m'], 'notebooks/analysis/surface_columnar_linkage.ipynb', 'drafted on daily maximum BLH and never run; here the window mean'),
  // composition
  nb('ocec-addis', 'Addis-like OC/EC', 'OC (FTIR) / EC (FTIR)', 'cutoffs', [2.27], ['Addis-like (OC/EC < 2.27)', 'Higher OC/EC'],
    'research/ftir_ec_phase3 (run_ftir_11 lowest-OC/EC 800 cohort; OCEC_ADDIS_LIKE in run_ftir_26/30)', 'the cutoff came from IMPROVE TOR OC/EC; applied here to FTIR OC/EC'),
  nb('ocec-smoke', 'Organic-rich OC/EC', 'OC (FTIR) / EC (FTIR)', 'cutoffs', [5], ['OC/EC < 5', 'Organic-rich (OC/EC ≥ 5)'],
    'research/improve_hips_offset/improve_smoke_event_qc.ipynb (OC_EC_SMOKE_MIN)', 'set on IMPROVE TOR OC/EC'),
  nb('ocec-deciles', 'OC/EC deciles', 'OC (FTIR) / EC (FTIR)', 'deciles', [], [], 'research/ftir_ec_phase3/scripts/run_ftir_16.py, run_ftir_31.py (implied MAC per OC/EC decile)'),
  nb('iron-median', 'Iron high vs low', 'Iron', 'median', [], ['Low Fe', 'High Fe'],
    'research/ftir_hips_chem/FTIR_Group_Talk_Apr2026_Draft.ipynb (Iron median split, HIPS vs FTIR EC)', 'the notebook split at each site\'s own median; here every site pooled'),
  // loading
  nb('ec-bands', 'EC (FTIR) bands', 'EC (FTIR)', 'cutoffs', [2, 4, 6, 8, 10, 15], [],
    'research/ftir_hips_chem/hips_offset_narrative.ipynb (Addis vs Delhi HIPS at matched EC)'),
  nb('hipsbc-tertiles', 'HIPS BC thirds', 'HIPS BC', 'tertiles', [], ['Lower third', 'Middle third', 'Upper third'],
    'research/ftir_hips_chem/workflows/raw_vs_spline_20261020.py (HIPS gap, high vs low third)'),
  nb('fabs-tail', 'HIPS Fabs high tail', 'HIPS Fabs', 'cutoffs', [50, 60, 70], ['< 50', '50 to 60', '60 to 70', 'High tail (≥ 70 Mm⁻¹)'],
    'research/ftir_hips_chem/improve_high_fabs_comparison.ipynb, research/improve_hips_offset (HIGH_TAIL_THRESHOLD = 70)', 'set on IMPROVE Fabs'),
  nb('fabs-quintiles', 'HIPS Fabs quintiles', 'HIPS Fabs', 'quintiles', [], [], 'research/ftir_ec_phase3/scripts/theory_test_suite.py (loading_bins, run_ftir_37)'),
  nb('fabs-deciles', 'HIPS Fabs deciles', 'HIPS Fabs', 'deciles', [], [], 'research/ftir_hips_chem/improve_high_fabs_comparison.ipynb (fit per Fabs decile)'),
  nb('pm-deciles', 'PM2.5 mass deciles', 'PM2.5 mass', 'deciles', [], [], 'research/ftir_hips_chem/decision_workbook_20261022.ipynb (loading deciles)',
    'the workbook used µg per filter; here µg/m³ (switch Filters → units to mass per filter)'),
]
export const isBuiltIn = (id: string) => BUILT_IN_GROUPINGS.some((d) => d.id === id)

/**
 * What a grouping consists of, for the picker's second line: the measurement, the split, and the band names when
 * they are short ("Precipitation (station) · cutoffs 0.1, 1, 5 · No rain / Light / Moderate / Heavy").
 */
export function describeGrouping(d: GroupingDef): string {
  const split = d.kind === 'cutoffs' ? `cutoffs ${d.cuts.join(', ')}` : (SPLITS.find((x) => x.kind === d.kind)?.label ?? d.kind).toLowerCase()
  // band names without the ranges in brackets, when the reader or the notebook named them
  const named = d.labels.map((l) => l.replace(/\s*\(.*\)\s*$/, '').trim()).filter(Boolean)
  const bands = named.length && named.join(' / ').length <= 60 ? ` · ${named.join(' / ')}` : ''
  return `${d.field} · ${split}${bands}`
}
