/**
 * Number formatting for every chart, table and tooltip, so a value reads the
 * same on every tab. Missing or non-finite values print as an em dash; inputs
 * may be numbers or numeric strings (export columns are loosely typed).
 *
 *   fmt(0.12345)      -> '0.123'   (3 digits, the library default)
 *   f2(1.5)           -> '1.50'
 *   signed(-0.4)      -> '−0.40'   (true minus sign, for intercepts)
 *   pct(0.256)        -> '26 %'
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

/** A fraction as a whole percentage: 0.256 -> "26 %". */
export const pct = (v: unknown): string => {
  const n = num(v)
  return n === null ? '—' : `${Math.round(100 * n)} %`
}

/** Scientific notation with one decimal: 0.000123 -> "1.2e-4". */
export const sci = (v: unknown): string => {
  const n = num(v)
  return n === null ? '—' : n.toExponential(1)
}

/** A fitted line as "slope x ± intercept": fitLine(0.82, -1.06) -> "0.82x − 1.06". */
export const fitLine = (slope: unknown, intercept: unknown, digits = 2): string => {
  const b = num(intercept)
  return `${fmt(slope, digits)}x ${b === null ? '' : b >= 0 ? '+ ' + b.toFixed(digits) : '− ' + Math.abs(b).toFixed(digits)}`.trim()
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
