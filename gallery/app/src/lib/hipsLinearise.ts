/**
 * Linearising HIPS on dark filters (the Meeting page "Linearising HIPS at Addis" (pages/HipsLinearisePage.tsx)). The IMPROVE curve of HIPS optical
 * depth against thermal EC on the filter is saturating, τ = aL / (1 + aL/τ∞); inverting it gives the optical depth a
 * proportional HIPS would have read, τ_lin = τ / (1 − τ/τ∞), which is research/ftir_hips_chem/scripts/
 * reference_closure.linearise. Because volume, deposit area and MAC cancel in the ratio, BC (Fabs ÷ MAC) is
 * linearised by the same factor. Everything here is a pure function of the per-filter optical depth and the ceiling,
 * so the page can follow any ceiling the reader picks.
 */
import { bisectRight, group } from 'd3'
import SITE_READINGS from './hipsSites.json' with { type: 'json' }
import { deming, median, medianSorted } from './stats.ts'

/** deposit area of the 25 mm PTFE filter HIPS reads, cm² (hips_loading.DEPOSIT_AREA_CM2) */
export const DEPOSIT_AREA_CM2 = 3.53

/**
 * The IMPROVE ceilings the gallery offers (hips_loading.IMPROVE_CEILING and the 10 Nov closure): the central fit of
 * all sites 2005–2025, both ends of its 90 % interval, and the 2015–2025 fit. hipsLinearise.test.mjs fails if they
 * move away from the committed summary (hips_linearise/linearise.json).
 */
export const IMPROVE_CEILINGS = [
  { key: 'central', tau_inf: 2.2596, label: '2.26 (central)' },
  { key: 'lo', tau_inf: 1.8212, label: '1.82 (90 % low)' },
  { key: 'hi', tau_inf: 2.7014, label: '2.70 (90 % high)' },
  { key: 'era', tau_inf: 1.7092, label: '1.71 (2015–2025)' },
] as const
export const CENTRAL_CEILING: number = IMPROVE_CEILINGS[0].tau_inf

/** HIPS optical depth of a filter from Fabs (Mm⁻¹) and its sampled volume (m³): the laboratory's own HIPS τ */
export const opticalDepth = (fabs: number, volume: number): number => (fabs * volume) / (100 * DEPOSIT_AREA_CM2)

/** the share of the absorption on the filter HIPS does not report, τ/τ∞ (hips_loading.under_read): 0 to 1 */
export const underRead = (tau: number, ceiling: number = CENTRAL_CEILING): number => Math.min(1, Math.max(0, tau / ceiling))

/**
 * The 11 Nov protocol's marks (hips_loading.py, docs/hips-loading-protocol.md), all on the expected under-read at
 * the central ceiling: a filter is used linearised up to 20 %, with caution to 30 %, and for shape only beyond; the
 * bend is expected at a site whose median filter under-reads by more than 10 %.
 */
export const USE_UP_TO = 0.20
export const SHAPE_ONLY_FROM = 0.30
export const EXPECTED_FROM = 0.10
export const HIPS_STATUSES = ['use, linearised', 'caution', 'shape only'] as const
export type HipsStatus = typeof HIPS_STATUSES[number]
export const statusOf = (share: number): HipsStatus => (share <= USE_UP_TO ? 'use, linearised' : share <= SHAPE_ONLY_FROM ? 'caution' : 'shape only')

/** what a filter of each status may be used for (docs/hips-loading-protocol.md, step 1) */
export const STATUS_USE: Record<HipsStatus, string> = {
  'use, linearised': 'level and shape',
  caution: 'shape; quote a level only with its range',
  'shape only': 'rank order, timing, season; not a level',
}

/** a filter's factor at the lowest and highest of the IMPROVE ceilings, for a filter at this expected under-read */
export const factorRange = (share: number): [number, number] => {
  const ks = IMPROVE_CEILINGS.map((c) => correction(share * CENTRAL_CEILING, c.tau_inf))
  return [Math.min(...ks), Math.max(...ks)]
}

/**
 * What a status means, for a hover: its marks, the factor at the central ceiling, and how far the IMPROVE ceilings
 * (1.71 to 2.70) move that factor. The status says how well the correction is known, not how large it is.
 */
export function statusMeaning(s: HipsStatus): string {
  const f = (v: number) => v.toFixed(2)
  const tau = (share: number) => f(share * CENTRAL_CEILING)
  const k = (share: number) => f(correction(share * CENTRAL_CEILING, CENTRAL_CEILING))
  const span = (share: number) => { const [a, b] = factorRange(share); return `×${f(a)} to ×${f(b)}` }
  const use = USE_UP_TO * 100, shape = SHAPE_ONLY_FROM * 100
  const body = s === 'use, linearised'
    ? `Expected under-read up to ${use} % (τ up to ${tau(USE_UP_TO)}). Factor up to ×${k(USE_UP_TO)} at the central ceiling; at ${use} % the IMPROVE ceilings give ${span(USE_UP_TO)}.`
    : s === 'caution'
      ? `Expected under-read ${use} to ${shape} % (τ ${tau(USE_UP_TO)} to ${tau(SHAPE_ONLY_FROM)}). Factor ×${k(USE_UP_TO)} to ×${k(SHAPE_ONLY_FROM)} at the central ceiling; at ${shape} % the IMPROVE ceilings give ${span(SHAPE_ONLY_FROM)}.`
      : `Expected under-read over ${shape} % (τ over ${tau(SHAPE_ONLY_FROM)}). Factor over ×${k(SHAPE_ONLY_FROM)}, and the ceilings part fast: ${span(SHAPE_ONLY_FROM)} at ${shape} %, ${span(0.5)} at 50 %.`
  return `${s}: ${body} Use for ${STATUS_USE[s]}.`
}

/**
 * Which sites the gallery's linearised view corrects (lib/viewFields, `hipsat` in a link):
 *   all        every filter, each by its own factor (a light filter's factor is close to 1)
 *   expected   the sites where the bend is expected: the median filter under-reads by more than 10 %
 *   seen       the sites whose own filters show it as well ("expected and seen" in the 11 Nov screen)
 *   past       every site, but only the filters at or past an optical depth (`hipscut`, default 0.45: a 20 % under-read)
 * The other sites (or, under `past`, the lighter filters) keep HIPS as measured. A cut on filters puts a step in the
 * response at the cut (× 1.25 at 0.45); it is offered because Ann asked for it (19 Nov 2026), not as the default.
 */
export const HIPS_SCOPES = [
  { key: 'all', label: 'All sites', mark: '' },
  { key: 'expected', label: 'Where expected', mark: ' where expected' },
  { key: 'seen', label: 'Where seen', mark: ' where seen' },
  { key: 'past', label: 'Filters past a τ', mark: ' past τ' },
] as const
export type HipsScope = typeof HIPS_SCOPES[number]['key']
export const parseHipsScope = (v: string | null | undefined): HipsScope => (v === 'expected' || v === 'seen' || v === 'past' ? v : 'all')
/** the optical depth the `past` scope linearises from (`hipscut` in a link) */
export const DEFAULT_PAST_TAU = 0.45
export const parseHipsCut = (v: string | null | undefined): number => { const n = Number(v); return v && Number.isFinite(n) && n > 0 && n < 3 ? n : DEFAULT_PAST_TAU }
/** a scope's mark on a unit: " where seen", " past τ 0.45" */
export const scopeMark = (scope: HipsScope, cut: number = DEFAULT_PAST_TAU): string =>
  scope === 'past' ? ` past τ ${cut.toFixed(2)}` : HIPS_SCOPES.find((x) => x.key === scope)?.mark ?? ''
/** whether a filter of optical depth τ is linearised under a scope's filter rule (every filter except under `past`) */
export const filterInScope = (tau: number, scope: HipsScope, cut: number = DEFAULT_PAST_TAU): boolean => scope !== 'past' || tau >= cut

/** a SPARTAN site as the 11 Nov screen read it (hips_loading.identify and check; hipsSites.json, written by gallery/data/export_hips_linearise.py) */
export interface SiteReading {
  code: string; site: string; filters: number; median_tau: number; expected_under_read: number; tested_filters: number
  /** b in Fabs ∝ (FTIR EC)^b on the site's own filters, with its 95 % interval, and what the curve predicts; null where too few filters */
  exponent: number | null; exponent_lo: number | null; exponent_hi: number | null; exponent_predicted: number | null
  reading: string
  /** the same exponent after linearising, and the check's verdict */
  exponent_after: number | null; verdict: string | null
}
export const HIPS_SITE_READINGS: readonly SiteReading[] = SITE_READINGS.sites
export const SEEN_READING = 'expected and seen'
/** the SPARTAN sites whose own filters show the bend */
export const SEEN_SITES: ReadonlySet<string> = new Set(HIPS_SITE_READINGS.filter((s) => s.reading === SEEN_READING).map((s) => s.code))
const READING_OF = new Map(HIPS_SITE_READINGS.map((s) => [s.code, s]))
export const siteReading = (code: string): SiteReading | undefined => READING_OF.get(code)

export interface SiteExposure { code: string; n: number; medianTau: number; underRead: number; expected: boolean }

/**
 * How dark each site's filters are, from the filters themselves: the median optical depth and the under-read the
 * curve expects of it at the central ceiling. `expected` is the protocol's rule (hips_loading.identify), so on the
 * SPARTAN table it gives the sites the 11 Nov screen calls expected (hipsLinearise.test.mjs), and it reads an
 * IMPROVE site, or a site added since, the same way.
 */
export function siteExposure(filters: Iterable<{ code: string; tau: number }>): Map<string, SiteExposure> {
  const by = new Map<string, number[]>()
  for (const f of filters) if (Number.isFinite(f.tau) && f.tau > 0) { const own = by.get(f.code); if (own) own.push(f.tau); else by.set(f.code, [f.tau]) }
  const out = new Map<string, SiteExposure>()
  for (const [code, taus] of by) {
    const medianTau = median(taus), share = underRead(medianTau)
    out.set(code, { code, n: taus.length, medianTau, underRead: share, expected: share > EXPECTED_FROM })
  }
  return out
}

/** the site codes a scope linearises, of the sites on file; null for every site */
export function scopeSites(scope: HipsScope, exposure: ReadonlyMap<string, SiteExposure>): Set<string> | null {
  if (scope === 'all' || scope === 'past') return null
  return new Set([...exposure.values()].filter((s) => (scope === 'seen' ? SEEN_SITES.has(s.code) : s.expected)).map((s) => s.code))
}

interface LoadingSample { id: string; site: string; date: string; loading: number; tau: number }
export type LoadingGroupSample = [site: string, date: string, previousLoading: number, loading: number, tau: number, id?: string]
export interface LoadingGroup { lower: number; upper: number | null; rows: LoadingGroupSample[] }

/** SPARTAN's nominal sampling cadence. Use the same loading edges as IMPROVE, with the prior sampled date at each site. */
export const SPARTAN_LOADING_MAX_GAP_DAYS = 9

export function previousLoadingGroups(samples: readonly LoadingSample[], edges: readonly number[]): LoadingGroup[] {
  const bins = new Map<number, LoadingGroupSample[]>()
  const eligible = samples.filter((s) => s.loading > 0 && Number.isFinite(s.loading) && Number.isFinite(s.tau) && Number.isFinite(Date.parse(s.date)))
  const sites = [...group(eligible, (s) => s.site)].sort(([a], [b]) => a.localeCompare(b))
  for (const [site, samplesAtSite] of sites) {
    const dates = [...group(samplesAtSite, (s) => s.date)].sort(([a], [b]) => a.localeCompare(b))
    for (let i = 1; i < dates.length; i++) {
      const [date, rows] = dates[i], [previousDate, previousRows] = dates[i - 1]
      const gapDays = (Date.parse(date) - Date.parse(previousDate)) / 86_400_000
      if (gapDays < 1 || gapDays > SPARTAN_LOADING_MAX_GAP_DAYS) continue
      // Co-located filters on one date contribute a single lag value, with no arbitrary order within that date.
      const lag = median(previousRows.map((s) => s.loading))
      const index = bisectRight(edges, lag) - 1
      if (index < 0) continue
      const members = bins.get(index) ?? []
      for (const row of rows) members.push([site, date, lag, row.loading, row.tau, row.id])
      bins.set(index, members)
    }
  }
  return [...bins].sort(([a], [b]) => a - b).map(([i, rows]) => ({ lower: edges[i], upper: edges[i + 1] ?? null, rows }))
}

/** the saturating IMPROVE response: optical depth for a proportional optical depth u = aL */
export const saturate = (u: number, ceiling: number): number => u / (1 + u / ceiling)

/** τ_lin = τ / (1 − τ/τ∞); NaN at or above the ceiling, where there is no finite answer */
export const linearise = (tau: number, ceiling: number): number => (tau < ceiling ? tau / (1 - tau / ceiling) : NaN)

/** τ_lin / τ = 1 / (1 − τ/τ∞): the factor that multiplies BC (and Fabs) */
export const correction = (tau: number, ceiling: number): number => (tau < ceiling ? 1 / (1 - tau / ceiling) : NaN)

/** dτ_lin/dτ = 1 / (1 − τ/τ∞)²: how much an absolute error in τ grows */
export const noiseGain = (tau: number, ceiling: number): number => (tau < ceiling ? 1 / (1 - tau / ceiling) ** 2 : NaN)

/** the exponent b in HIPS ∝ EC^b that the curve predicts on these filters: the mean of d ln τ / d ln u = 1 − τ/τ∞ */
export const predictedExponent = (taus: readonly number[], ceiling: number): number =>
  taus.length ? taus.reduce((s, t) => s + Math.min(1, Math.max(0, 1 - t / ceiling)), 0) / taus.length : NaN

export interface LinFilter { id: string; season: string; lot: string | null; tau: number; volume_m3: number; volume_known: boolean; hips_bc: number; ftir: number | null; pm25: number | null }

/** a filter's HIPS BC once linearised at this ceiling (µg/m³), NaN past the ceiling */
export const linearBc = (f: Pick<LinFilter, 'tau' | 'hips_bc'>, ceiling: number): number => f.hips_bc * correction(f.tau, ceiling)

/**
 * The Deming fit of FTIR EC on HIPS BC, as measured (ceiling null) or linearised at a ceiling, with the error ratio
 * λ the closure workflow uses (closeout LAMBDA_ADDIS). Filters past the ceiling drop out, as in the workflow.
 */
export function fitAgainstHips(filters: readonly LinFilter[], lambda: number, ceiling: number | null): { slope: number; intercept: number; n: number } | null {
  const xs: number[] = [], ys: number[] = []
  for (const f of filters) {
    if (f.ftir == null) continue
    const x = ceiling == null ? f.hips_bc : linearBc(f, ceiling)
    if (Number.isFinite(x)) { xs.push(x); ys.push(f.ftir) }
  }
  const d = deming(xs, ys, lambda)
  return d ? { ...d, n: xs.length } : null
}

/**
 * The MAC at which the Deming fit of FTIR EC on HIPS BC has slope 1 (as measured, or linearised at a ceiling). HIPS BC
 * is Fabs ÷ MAC, so another MAC rescales the x axis, and its errors with it: the error ratio λ moves with MAC², the
 * convention under which a MAC pivot survives a Deming fit exactly (pls_transfer.deming_regression, ftir_31). The
 * slope then scales with the MAC and the intercept is unchanged, so the MAC for slope 1 is MAC ÷ slope.
 */
export function macForSlopeOne(filters: readonly LinFilter[], lambda: number, ceiling: number | null, mac: number): number {
  const slope = fitAgainstHips(filters, lambda, ceiling)?.slope ?? NaN
  return slope > 0 ? mac / slope : NaN
}

/** median FTIR EC ÷ HIPS BC over a group of filters, as measured and linearised at a ceiling */
export function ratioMedians(filters: readonly LinFilter[], ceiling: number): { n: number; tau: number; measured: number; linearised: number } {
  const own = filters.filter((f) => f.ftir != null)
  return {
    n: own.length,
    tau: median(own.map((f) => f.tau)),
    measured: median(own.map((f) => f.ftir! / f.hips_bc)),
    linearised: median(own.map((f) => f.ftir! / linearBc(f, ceiling)).filter(Number.isFinite)),
  }
}

/**
 * A site's median or mean HIPS BC once linearised at a ceiling (µg/m³ at the export's MAC). A filter at or past the
 * ceiling has no finite linearised value and counts as +∞ instead of dropping out: the median stays finite while
 * fewer than half the filters are past the ceiling, and the mean has no finite value as soon as one is.
 */
export function linearLevel(filters: readonly Pick<LinFilter, 'tau' | 'hips_bc'>[], ceiling: number | null, stat: 'median' | 'mean'): number {
  const values = filters.map((f) => { const v = ceiling == null ? f.hips_bc : linearBc(f, ceiling); return Number.isNaN(v) ? Infinity : v })
  if (!values.length) return NaN
  return stat === 'median' ? medianSorted(values.sort((a, b) => a - b)) : values.reduce((a, b) => a + b, 0) / values.length
}
