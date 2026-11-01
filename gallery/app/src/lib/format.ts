/**
 * Number formatting for every chart, table and tooltip, so a value reads the
 * same on every tab. Missing or non-finite values print as an em dash; inputs
 * may be numbers or numeric strings (export columns are loosely typed).
 *
 *   fmt(0.12345)      -> '0.123'   (3 digits, the library default)
 *   f2(1.5)           -> '1.50'
 *   signed(-0.4)      -> '−0.40'   (true minus sign, for intercepts)
 *   pct(0.256)        -> '26 %'
 *   pct(0.256, 1)     -> '25.6 %'
 *   percent(25.6)     -> '26 %'     (a value already in percent)
 *   fitSummary({ slope: 0.95, intercept: -1.23, r2: 0.9, n: 40 })
 *                     -> 'slope 0.950 · intercept -1.230 · R² 0.900 · n 40'
 */
const num = (v: unknown): number | null => {
  if (v === null || v === undefined || v === '') return null
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n : null
}

/** Fixed digits; the shared default for charts. */
export const fmt = (v: unknown, digits = 3): string => {
  const n = num(v)
  return n === null ? '—' : n.toFixed(digits)
}

/** Fixed digits with a 2-digit default (tables of slopes and intercepts). */
export const fixed = (v: unknown, digits = 2): string => fmt(v, digits)
export const f2 = (v: unknown): string => fmt(v, 2)
export const f3 = (v: unknown): string => fmt(v, 3)

/** Explicit sign with a true minus, e.g. a fit intercept: "+0.12", "−1.08". */
export const signed = (v: unknown, digits = 2): string => {
  const n = num(v)
  return n === null ? '—' : `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(digits)}`
}

/** A fraction as a percentage: 0.256 -> "26 %", pct(0.256, 1) -> "25.6 %". */
export const pct = (v: unknown, digits = 0): string => {
  const n = num(v)
  return n === null ? '—' : `${digits ? (100 * n).toFixed(digits) : Math.round(100 * n)} %`
}

/** A value already in percent: percent(25.6) -> "26 %", percent(25.64, 1) -> "25.6 %". */
export const percent = (v: unknown, digits = 0): string => {
  const n = num(v)
  return n === null ? '—' : `${n.toFixed(digits)} %`
}

/** At most `digits` decimals, trailing zeros dropped (axis ticks): trim(0.50) -> "0.5", trim(2) -> "2". */
export const trim = (v: unknown, digits = 2): string => {
  const n = num(v)
  return n === null ? '—' : String(+n.toFixed(digits))
}

/** Decimals by magnitude (ramp legends): 0 from 100, 1 from 10, else 2: 123.4 -> "123", 12.34 -> "12.3", 1.234 -> "1.23". */
export const adaptive = (v: unknown): string => {
  const n = num(v)
  return n === null ? '—' : n.toFixed(Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 10 ? 1 : 2)
}

/** Scientific notation with one decimal: 0.000123 -> "1.2e-4". */
export const sci = (v: unknown): string => {
  const n = num(v)
  return n === null ? '—' : n.toExponential(1)
}

/**
 * A fitted line as "slope x ± intercept": fitLine(0.82, -1.06) -> "0.82x − 1.06"; a saved
 * calibration at the coefficient precision is fitLine(m, b, 3). "—" when either is missing.
 */
export const fitLine = (slope: unknown, intercept: unknown, digits = 2): string => {
  const m = num(slope), b = num(intercept)
  if (m === null || b === null) return '—'
  return `${m.toFixed(digits)}x ${b >= 0 ? '+ ' + b.toFixed(digits) : '− ' + Math.abs(b).toFixed(digits)}`
}

export interface FitParts { slope?: unknown; intercept?: unknown; r2?: unknown; n?: unknown }
export interface FitSummaryOptions {
  /** slope digits (and intercept digits unless `interceptDigits`); 3, the coefficient precision */
  digits?: number
  interceptDigits?: number
  r2Digits?: number
  /** the intercept's unit, printed after it ("µg/m³") */
  unit?: string
  /** print the intercept with its sign ("+0.120", "−1.230"); plain fmt otherwise */
  signedIntercept?: boolean
  /** between the parts; " · " by default, ", " in running text */
  sep?: string
  /** a word before the parts ("Deming", "OLS:") */
  label?: string
}

/**
 * One fit's coefficients as text: "slope 0.950 · intercept -1.230 · R² 0.900 · n 40". Parts that
 * are absent from `f` (not merely null) are left out, so a fit without n or R² reads naturally;
 * a missing fit prints "—". The one wording for fit summaries in notes, subtitles and tooltips.
 */
export const fitSummary = (f: FitParts | null | undefined, { digits = 3, interceptDigits = digits, r2Digits = 3, unit, signedIntercept = false, sep = ' · ', label }: FitSummaryOptions = {}): string => {
  if (!f) return '—'
  const parts: string[] = []
  if ('slope' in f) parts.push(`slope ${fmt(f.slope, digits)}`)
  if ('intercept' in f) parts.push(`intercept ${signedIntercept ? signed(f.intercept, interceptDigits) : fmt(f.intercept, interceptDigits)}${unit ? ` ${unit}` : ''}`)
  if ('r2' in f) parts.push(`R² ${fmt(f.r2, r2Digits)}`)
  if ('n' in f && f.n !== undefined && f.n !== null) parts.push(`n ${f.n}`)
  return `${label ? `${label} ` : ''}${parts.join(sep)}`
}

/**
 * Fit coefficients (slope, intercept, R²) at 3 dp, the agreed precision
 * (docs/gallery-chart-standards.md §5); concentrations and loadings stay at 2 dp.
 */
export const coef = (v: unknown): string => fmt(v, 3)
/** An intercept or a difference with its sign, at the coefficient precision. */
export const coefSigned = (v: unknown): string => signed(v, 3)

/**
 * Chemical subscripts for display: "PM2.5 mass" reads PM₂.₅ mass, "SO4" SO₄. Field names stay
 * plain in data, URLs and lookups; only rendered text goes through this (via estimatorText,
 * which every chart title, axis, legend and tooltip already uses, and the pickers).
 */
const SUBSCRIPTS: [RegExp, string][] = [
  [/\bPM2\.5\b/g, 'PM₂.₅'], [/\bPM10\b/g, 'PM₁₀'], [/\bPM1\b/g, 'PM₁'],
  [/\bSO4\b/g, 'SO₄'], [/\bNO3\b/g, 'NO₃'], [/\bNH4\b/g, 'NH₄'], [/\bCO2\b/g, 'CO₂'], [/\bNO2\b/g, 'NO₂'], [/\bSO2\b/g, 'SO₂'],
]
export const chemText = (text: string): string => {
  let out = text
  for (const [re, sub] of SUBSCRIPTS) out = out.replace(re, sub)
  return out
}
