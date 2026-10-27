/**
 * The AERONET tab's arithmetic, kept free of React so node can test it (aeronetMath.test.mjs): the filter
 * chemistry the surface–column notebooks combine with AERONET, the Eom et al. (2025) exclusion curve, the
 * Ångström-matrix classes of Cazorla et al. (2013), and monthly means of AERONET's daily series.
 */
type Row = Record<string, unknown>
const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/**
 * Fine soil (µg/m³), Malm et al. (1994): 2.20 Al + 2.49 Si + 1.63 Ca + 2.42 Fe + 1.94 Ti, as
 * notebooks/analysis/surface_columnar_linkage.ipynb computes it. Elemental (XRF) calcium where the filter
 * has it, else the ion. Null unless all five are measured.
 */
export function fineSoil(r: Row): number | null {
  const ca = num(r['Calcium (element)']) ? r['Calcium (element)'] : r['Calcium']
  const parts = [r['Aluminum'], r['Silicon'], ca, r['Iron'], r['Titanium']]
  if (!parts.every(num)) return null
  const [al, si, c, fe, ti] = parts as number[]
  return 2.2 * al + 2.49 * si + 1.63 * c + 2.42 * fe + 1.94 * ti
}

/** ammonium sulfate and ammonium nitrate from the ions: (NH4)2SO4 / SO4 = 1.375, NH4NO3 / NO3 = 1.290 */
export const AS_PER_SULFATE = 1.375
export const AN_PER_NITRATE = 1.29

/**
 * Eom et al.'s non-absorbing fraction w = (AS + AN) / (AS + AN + BC): the share of the fine mass that
 * scatters without absorbing (surface_columnar_linkage.ipynb section 2). Null without sulfate, nitrate or
 * BC, or a non-positive total.
 */
export function nonAbsorbingFraction(r: Row, bcField: string): number | null {
  const so4 = r['Sulfate'], no3 = r['Nitrate'], bc = r[bcField]
  if (!num(so4) || !num(no3) || !num(bc)) return null
  const scatter = AS_PER_SULFATE * so4 + AN_PER_NITRATE * no3
  const total = scatter + bc
  return total > 0 ? scatter / total : null
}

/** Pearson r of paired values (non-finite pairs dropped); null under three pairs or with no spread */
export function pearson(xs: readonly (number | null)[], ys: readonly (number | null)[]): number | null {
  const p: [number, number][] = []
  xs.forEach((x, i) => { const y = ys[i]; if (num(x) && num(y)) p.push([x, y]) })
  if (p.length < 3) return null
  const mx = p.reduce((s, [x]) => s + x, 0) / p.length, my = p.reduce((s, [, y]) => s + y, 0) / p.length
  let sxy = 0, sxx = 0, syy = 0
  for (const [x, y] of p) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2 }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null
}

/** the p-quantile of sorted values, linear between order statistics (numpy's and pandas' default) */
export function quantileSorted(sorted: readonly number[], p: number): number {
  if (!sorted.length) return NaN
  const h = (sorted.length - 1) * p, lo = Math.floor(h)
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (h - lo)
}

export interface ExclusionPoint { ratio: number; x: number; ys: Record<string, number | null> }
export interface ExclusionStep { excluded: number; n: number; r: Record<string, { r: number | null; n: number }> }

/**
 * Eom et al. (2025) sequential exclusion, as surface_columnar_linkage.ipynb runs it: drop the filters whose
 * surface/column ratio (BC / AOD 500) is below its `excluded`-th percentile, then correlate surface BC with
 * each column quantity again. If the low-ratio days are the ones with aerosol aloft, r rises as they go.
 * A step needs 5 filters, a correlation 5 pairs (the notebook's floors).
 */
export function exclusionCurve(points: readonly ExclusionPoint[], keys: readonly string[], steps: readonly number[] = [0, 20, 40, 60, 80]): ExclusionStep[] {
  const ok = points.filter((p) => num(p.ratio) && num(p.x))
  const sorted = ok.map((p) => p.ratio).sort((a, b) => a - b)
  return steps.flatMap((ex) => {
    const cut = ex === 0 ? -Infinity : quantileSorted(sorted, ex / 100)
    const kept = ok.filter((p) => p.ratio >= cut)
    if (kept.length < 5) return []
    const r: ExclusionStep['r'] = {}
    for (const k of keys) {
      const pairs = kept.filter((p) => num(p.ys[k]))
      r[k] = { r: pairs.length >= 5 ? pearson(pairs.map((p) => p.x), pairs.map((p) => p.ys[k])) : null, n: pairs.length }
    }
    return [{ excluded: ex, n: kept.length, r }]
  })
}

/**
 * Cazorla et al. (2013, ACP 13, 9337) Fig. 2, the Ångström matrix: absorption Ångström exponent against
 * scattering Ångström exponent, both 440–675 nm, cut at 1.0 and 1.5 on each axis. The coated-large-particle
 * class spans SAE < 1.5 under AAE 1; the middle cell (1 to 1.5 on both) is "Mix" (the figure draws a
 * diagonal through it that its classes do not use).
 */
export const CAZORLA_CLASSES = ['EC dominated', 'EC/OC mixture', 'OC dominated', 'OC/Dust mix', 'Dust dominated', 'Dust/EC mix', 'Coated large particles', 'Mix'] as const
export type CazorlaClass = (typeof CAZORLA_CLASSES)[number]
export function cazorlaClass(sae: number, aae: number): CazorlaClass | null {
  if (!num(sae) || !num(aae)) return null
  if (aae > 1.5) return sae < 1 ? 'Dust dominated' : sae < 1.5 ? 'OC/Dust mix' : 'OC dominated'
  if (aae >= 1) return sae < 1 ? 'Dust/EC mix' : sae < 1.5 ? 'Mix' : 'EC/OC mixture'
  return sae < 1.5 ? 'Coated large particles' : 'EC dominated'
}
/** each class's rectangle in (SAE, AAE), for shading: [sae0, sae1, aae0, aae1]; open ends run to the plot edge */
export const CAZORLA_REGIONS: Record<CazorlaClass, [number, number, number, number]> = {
  'Dust dominated': [-Infinity, 1, 1.5, Infinity],
  'OC/Dust mix': [1, 1.5, 1.5, Infinity],
  'OC dominated': [1.5, Infinity, 1.5, Infinity],
  'Dust/EC mix': [-Infinity, 1, 1, 1.5],
  'Mix': [1, 1.5, 1, 1.5],
  'EC/OC mixture': [1.5, Infinity, 1, 1.5],
  'Coated large particles': [-Infinity, 1.5, -Infinity, 1],
  'EC dominated': [1.5, Infinity, -Infinity, 1],
}

/**
 * The window's aerosol type (gallery/data/export_aeronet.py aerosol_type, after Giles et al. 2012): the class
 * number the export writes in "Aerosol type class (AERONET)", and its name.
 */
export const AEROSOL_TYPES: Record<number, string> = {
  1: 'Coarse-dominated (dust-like)',
  2: 'Mixed size',
  3: 'Fine, weakly absorbing (U/I-like)',
  4: 'Fine, moderately absorbing',
  5: 'Fine, strongly absorbing (smoke/BC-like)',
  6: 'Fine, no inversion',
}

/** month (1–12) → mean of the values on dates in that month, with the count of days; null where none */
export function monthlyMeans(dates: readonly string[], values: readonly (number | null)[]): { mean: number | null; n: number }[] {
  const sum = new Array(12).fill(0), n = new Array(12).fill(0)
  dates.forEach((d, i) => {
    const v = values[i]
    if (!num(v)) return
    const m = Number(d.slice(5, 7)) - 1
    if (m >= 0 && m < 12) { sum[m] += v; n[m]++ }
  })
  return sum.map((s, m) => ({ mean: n[m] ? s / n[m] : null, n: n[m] }))
}

/** mean and standard error of the finite values; null mean when there are none */
export function meanSe(values: readonly (number | null | undefined)[]): { mean: number | null; se: number | null; n: number } {
  const v = values.filter(num)
  if (!v.length) return { mean: null, se: null, n: 0 }
  const m = v.reduce((a, b) => a + b, 0) / v.length
  if (v.length < 2) return { mean: m, se: null, n: 1 }
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1))
  return { mean: m, se: sd / Math.sqrt(v.length), n: v.length }
}
