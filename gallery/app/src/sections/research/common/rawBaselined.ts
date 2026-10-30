/**
 * ftir_52 §7 as pure functions (node-tested in rawBaselined.test.mjs): what baselining does to who looks like whom.
 * Two site × site matrices of median spectra over the same sites: RAW (the pool's absorbance on the corrected
 * grid's 1425–3999 cm⁻¹ window, exported as raw_sites + raw_pairs) and AIRSpec-baselined (site_correlation, the
 * notebook's own CSV). Everything is read as dissimilarity d = 1 − r, as the notebook plots it on log axes:
 * raw pairs sit at r ≈ 0.999, so a linear r axis would put the whole raw network on one tick.
 *
 * The notebook's network numbers (Spearman ρ between the two matrices, the median pair r in each space, how many
 * sites change their nearest site) are over the IMPROVE sites only; the SPARTAN targets (leading '*') are in
 * both matrices here as well, so a target can be put in focus too.
 */
import { median, spearman } from './stats.ts'

export type Cells = Record<string, unknown[]>

export interface Spaces {
  /** site names in the raw export's order; a SPARTAN target starts with '*' */
  names: string[]
  spartan: boolean[]
  /** raw spectra behind each site's median (library spectra, or a target's filters) */
  n: number[]
  /** dissimilarity 1 − r, symmetric, NaN on the diagonal and where a space lacks the pair */
  raw: Float64Array[]
  baselined: Float64Array[]
}

export const isTarget = (name: string) => name.startsWith('*')
export const siteLabel = (name: string) => (isTarget(name) ? name.slice(1) : name)

/**
 * The two spaces on the raw export's site list. `rawSites` is {site, spartan, n_spectra}; `rawPairs` is {d_raw_ppm},
 * one value per pair in upper-triangle order (0–1, 0–2, …, 1–2, …), 1 − r in parts per million; `corr` is
 * site_correlation ('Unnamed: 0' names the rows; columns by position, since a column may still say Pasadena where
 * its row says JPL). Null when a table is missing or the shapes disagree.
 */
export function spaces(rawSites: Cells | undefined, rawPairs: Cells | undefined, corr: Cells | undefined): Spaces | null {
  if (!rawSites || !rawPairs || !corr) return null
  const names = (rawSites.site ?? []).map(String)
  const m = names.length
  const ppm = (rawPairs.d_raw_ppm ?? []) as (number | null)[]
  if (m < 3 || ppm.length !== (m * (m - 1)) / 2) return null
  const rowNames = (corr['Unnamed: 0'] ?? []).map(String)
  const keys = Object.keys(corr).filter((k) => k !== 'Unnamed: 0')
  if (keys.length !== rowNames.length) return null
  const at = new Map(rowNames.map((s, i) => [s, i]))
  const grid = () => names.map(() => new Float64Array(m).fill(NaN))
  const raw = grid(), baselined = grid()
  let k = 0
  for (let a = 0; a < m; a++) {
    for (let b = a + 1; b < m; b++, k++) {
      const v = ppm[k]
      if (v != null && Number.isFinite(v)) raw[a][b] = raw[b][a] = v / 1e6
    }
  }
  for (let a = 0; a < m; a++) {
    const i = at.get(names[a])
    if (i == null) continue
    for (let b = 0; b < m; b++) {
      const j = at.get(names[b])
      const r = a === b || j == null ? null : (corr[keys[j]] as (number | null)[])[i]
      if (r != null && Number.isFinite(r)) baselined[a][b] = 1 - r
    }
  }
  return {
    names, raw, baselined,
    spartan: names.map((s, i) => Boolean((rawSites.spartan ?? [])[i]) || isTarget(s)),
    n: names.map((_, i) => Number((rawSites.n_spectra ?? [])[i] ?? NaN)),
  }
}

/** the sites in play: IMPROVE only (as ftir_52), or with the SPARTAN targets */
export function members(s: Spaces, withTargets: boolean): number[] {
  return s.names.map((_, i) => i).filter((i) => withTargets || !s.spartan[i])
}

export interface Pair { a: number; b: number; raw: number; baselined: number }

/** every pair among `sites` once, with both dissimilarities (pairs a space lacks are left out) */
export function pairs(s: Spaces, sites: readonly number[]): Pair[] {
  const out: Pair[] = []
  for (let x = 0; x < sites.length; x++) {
    for (let y = x + 1; y < sites.length; y++) {
      const a = sites[x], b = sites[y], raw = s.raw[a][b], baselined = s.baselined[a][b]
      if (Number.isFinite(raw) && Number.isFinite(baselined)) out.push({ a, b, raw, baselined })
    }
  }
  return out
}

/** the nearest site to `i` among `sites` in one space (smallest 1 − r), −1 when none */
export function nearest(d: Float64Array, i: number, sites: readonly number[]): number {
  let best = -1, bestD = Infinity
  for (const j of sites) if (j !== i && d[j] < bestD) { bestD = d[j]; best = j }
  return best
}

export interface NetworkSummary {
  sites: number
  pairs: number
  /** Spearman ρ between the two spaces' pair dissimilarities (= between the r matrices) */
  rho: number | null
  /** median pair r in each space */
  medianRaw: number
  medianBaselined: number
  /** pairs that baselining separates (further apart after: above the 1:1 line) */
  separated: number
  /** sites whose nearest site is a different one after baselining */
  changed: number
}

/** ftir_52 cell [21]'s printed numbers, over `sites` */
export function networkSummary(s: Spaces, sites: readonly number[]): NetworkSummary {
  const ps = pairs(s, sites)
  const changed = sites.filter((i) => nearest(s.raw[i], i, sites) !== nearest(s.baselined[i], i, sites)).length
  return {
    sites: sites.length,
    pairs: ps.length,
    rho: spearman(ps.map((p) => p.raw), ps.map((p) => p.baselined)),
    medianRaw: 1 - median(ps.map((p) => p.raw)),
    medianBaselined: 1 - median(ps.map((p) => p.baselined)),
    separated: ps.filter((p) => p.baselined > p.raw).length,
    changed,
  }
}

export interface Neighbour { j: number; raw: number; baselined: number; rankRaw: number; rankBaselined: number }

/**
 * One site against every other site in `sites`: both dissimilarities and the rank of each partner in each space
 * (1 = nearest; ties share the better rank), sorted by raw rank. ftir_52's rank_of_raw_nn_after is the baselined
 * rank of the raw rank-1 partner.
 */
export function neighbours(s: Spaces, i: number, sites: readonly number[]): Neighbour[] {
  const others = sites.filter((j) => j !== i && Number.isFinite(s.raw[i][j]) && Number.isFinite(s.baselined[i][j]))
  const rankIn = (d: Float64Array) => {
    const values = others.map((j) => d[j])
    return new Map(others.map((j) => [j, 1 + values.filter((v) => v < d[j]).length]))
  }
  const rr = rankIn(s.raw[i]), rb = rankIn(s.baselined[i])
  return others.map((j) => ({ j, raw: s.raw[i][j], baselined: s.baselined[i][j], rankRaw: rr.get(j)!, rankBaselined: rb.get(j)! }))
    .sort((a, b) => a.rankRaw - b.rankRaw || a.rankBaselined - b.rankBaselined)
}

export interface SiteSummary {
  partners: number
  nearestRaw: Neighbour | null
  nearestBaselined: Neighbour | null
  /** Spearman ρ between the site's raw and baselined dissimilarities to every partner */
  rho: number | null
  /** of the k nearest in raw space, how many are still among the k nearest after baselining */
  keptOfTop: number
  /** pairs with this site that baselining separates */
  separated: number
}

export function siteSummary(list: readonly Neighbour[], k: number): SiteSummary {
  const byRaw = list.find((n) => n.rankRaw === 1) ?? null
  const byBaselined = list.slice().sort((a, b) => a.rankBaselined - b.rankBaselined)[0] ?? null
  return {
    partners: list.length,
    nearestRaw: byRaw,
    nearestBaselined: byBaselined,
    rho: spearman(list.map((n) => n.raw), list.map((n) => n.baselined)),
    keptOfTop: list.filter((n) => n.rankRaw <= k && n.rankBaselined <= k).length,
    separated: list.filter((n) => n.baselined > n.raw).length,
  }
}

/** `bins` log-spaced bin edges over [lo, hi] and the counts of each set of values in them (ftir_52: 60 bins) */
export function logHistogram(sets: readonly (readonly number[])[], lo: number, hi: number, bins: number): { edges: number[]; counts: number[][] } {
  const l0 = Math.log10(lo), l1 = Math.log10(hi)
  const edges = Array.from({ length: bins + 1 }, (_, i) => 10 ** (l0 + ((l1 - l0) * i) / bins))
  const counts = sets.map((values) => {
    const c = new Array<number>(bins).fill(0)
    for (const v of values) {
      if (!(v > 0)) continue
      const at = Math.floor(((Math.log10(v) - l0) / (l1 - l0)) * bins)
      if (at >= 0 && at <= bins) c[Math.min(at, bins - 1)]++
    }
    return c
  })
  return { edges, counts }
}
