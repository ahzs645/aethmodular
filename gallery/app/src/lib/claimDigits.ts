/**
 * How many digits a value with a stated range can carry. The convention is the usual one for a measurement
 * (JCGM 100 "GUM" §7.2.6; Taylor, An Introduction to Error Analysis, §2.2): the last digit written is the one the
 * uncertainty starts in. An uncertainty of 1.3 starts in the units, so 8.04 is written 8; one of 0.4 starts in the
 * tenths, so 8.03 is written 8.0. A "place" is the power of ten of that digit: 0 for whole numbers, −1 for tenths.
 *
 * Used by the "Linearising HIPS at Addis" page (pages/HipsDecimals.tsx), where the uncertainty is half the range
 * the ceiling's interval gives.
 */

/** half the width of a range, the "±" a symmetric statement of it would carry */
export const halfRange = (lo: number, hi: number): number => Math.abs(hi - lo) / 2

/** the power of ten of the uncertainty's leading digit (1.3 → 0, 0.4 → −1, 0.03 → −2); NaN for no uncertainty */
export const claimPlace = (uncertainty: number): number =>
  (uncertainty > 0 && Number.isFinite(uncertainty) ? Math.floor(Math.log10(uncertainty) + 1e-12) : NaN)

/**
 * The place a value is written to: the uncertainty's place, but never finer than the value was reported
 * (`finest`, −2 for a quantity given to two decimals). No uncertainty means the reported precision stands.
 */
export const writtenPlace = (uncertainty: number, finest = -2): number => {
  const place = claimPlace(uncertainty)
  return Number.isNaN(place) ? finest : Math.max(finest, place)
}

/** a value rounded to a place and written with exactly the decimals that place has: (8.04, 0) → "8", (8.03, −1) → "8.0" */
export const writeAt = (value: number, place: number): string => {
  if (!Number.isFinite(value) || !Number.isFinite(place)) return '—'
  if (place < 0) return value.toFixed(-place)
  const unit = 10 ** place
  return String(Math.round(value / unit) * unit)
}

/**
 * A computed value split into the digits its uncertainty supports and the digits it does not:
 * (8.04, 0) → ["8", ".04"], (8.03, −1) → ["8.0", "3"]. `shown` is the number of decimals the value is printed with.
 */
export const digitSplit = (value: number, place: number, shown = 2): [string, string] => {
  if (!Number.isFinite(value) || !Number.isFinite(place)) return ['—', '']
  const text = Math.abs(value).toFixed(shown), sign = value < 0 ? '−' : ''
  const whole = shown > 0 ? text.indexOf('.') : text.length
  const cut = Math.max(0, Math.min(text.length, place >= 0 ? whole - place : whole + 1 - place))
  return [sign + text.slice(0, cut), text.slice(cut)]
}

/** what a value with a range can be written as */
export interface Claim {
  /** half the width of the range */
  uncertainty: number
  /** the power of ten of the last digit written */
  place: number
  /** "8 (7 to 10)": the value and both ends of its range, rounded to that place */
  text: string
}

/** A value and its range, rounded to the place the range supports: (8.04, 7.19, 9.76) → "8 (7 to 10)". */
export const claimOf = (value: number, lo: number, hi: number, finest = -2): Claim => {
  const uncertainty = halfRange(lo, hi), place = writtenPlace(uncertainty, finest)
  return { uncertainty, place, text: `${writeAt(value, place)} (${writeAt(lo, place)} to ${writeAt(hi, place)})` }
}

/** what a place is called on an axis band or in a sentence */
export const placeName = (place: number): string =>
  place >= 1 ? (place === 1 ? 'tens' : `10^${place}`) : place === 0 ? 'whole numbers'
    : place === -1 ? 'one decimal' : place === -2 ? 'two decimals' : place === -3 ? 'three decimals' : `${-place} decimals`
