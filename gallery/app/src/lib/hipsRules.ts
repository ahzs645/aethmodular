/**
 * The rules by which HIPS can be linearised across a network, and what each does to the same filters (the Meeting
 * page "Linearising HIPS across SPARTAN", pages/HipsStoryPage.tsx). The curve is one (lib/hipsLinearise); a rule
 * decides which filters it is applied to, or replaces a filter's own factor with its site's:
 *
 *   filter     every filter by its own factor, 1 / (1 − τ/τ∞)
 *   expected   the same, only at sites where the bend is expected (median filter under-reads by more than 10 %)
 *   seen       the same, only at sites whose own filters also show the bend (the 11 Nov screen)
 *   cut        the same, only on filters whose expected under-read is past a cut-off
 *   site       one factor per site, its median filter's, on every filter of the site
 *
 * Everything is a pure function of the filters (Fabs, sampled volume, FTIR EC, PM2.5), the ceiling and the rule, so
 * the page recomputes every chart as the reader changes either. hipsRules.test.mjs holds it to the 11 Nov screen:
 * the filters counted, the exponent of each site as measured, and after every filter is linearised.
 */
import { CENTRAL_CEILING, correction, opticalDepth, scopeSites, siteExposure, underRead, type SiteExposure } from './hipsLinearise.ts'
import { isFiniteNumber, median } from './stats.ts'

export const HIPS_RULES = [
  { key: 'filter', label: 'Every filter', short: 'every filter', by: 'filter' },
  { key: 'expected', label: 'Sites where expected', short: 'where expected', by: 'site' },
  { key: 'seen', label: 'Sites where seen', short: 'where seen', by: 'site' },
  { key: 'cut', label: 'Filters past a cut-off', short: 'past a cut-off', by: 'filter' },
  { key: 'site', label: 'One factor per site', short: 'one factor per site', by: 'site' },
] as const
export type HipsRule = typeof HIPS_RULES[number]['key']
export const RULE_KEYS: readonly HipsRule[] = HIPS_RULES.map((r) => r.key)
export const ruleLabel = (rule: HipsRule): string => HIPS_RULES.find((r) => r.key === rule)!.label

/** filters a site needs, with FTIR EC and PM2.5, for its own exponent (hips_loading.MIN_TEST) */
export const MIN_TEST = 40

export interface NetFilter { id: string; code: string; date: string; fabs: number; volume: number; tau: number; ec: number | null; pm: number | null }

type Row = { id: string; code: string; date: string; excluded?: boolean } & Record<string, unknown>

/**
 * The filters the 11 Nov screen counts (run_hips_network_20261111.load): the export's exclusions out, HIPS Fabs
 * (else the public BC × 10) and a sampled volume both above zero.
 */
export function networkFilters(rows: readonly Row[], exportMac = 10): NetFilter[] {
  const out: NetFilter[] = []
  for (const r of rows) {
    if (r.excluded) continue
    const fabs = isFiniteNumber(r['HIPS Fabs']) ? r['HIPS Fabs'] : isFiniteNumber(r['BC (public HIPS)']) ? r['BC (public HIPS)'] * exportMac : NaN
    const volume = r['Sampled volume']
    if (!(fabs > 0) || !isFiniteNumber(volume) || !(volume > 0)) continue
    out.push({ id: r.id, code: r.code, date: r.date, fabs, volume, tau: opticalDepth(fabs, volume),
      ec: isFiniteNumber(r['EC (FTIR)']) ? r['EC (FTIR)'] : null, pm: isFiniteNumber(r['PM2.5 mass']) ? r['PM2.5 mass'] : null })
  }
  return out
}

export interface RuleOptions {
  rule: HipsRule
  /** the ceiling a corrected filter is linearised with */
  ceiling: number
  /** the cut-off of the `cut` rule, as a share (0.2 = a fifth): the expected under-read at the central ceiling */
  cut?: number
}

/** a filter's own factor at a ceiling; +∞ at or past it, where there is no finite value */
const own = (tau: number, ceiling: number): number => { const k = correction(tau, ceiling); return Number.isNaN(k) ? Infinity : k }

/**
 * The factor each filter's HIPS is multiplied by under a rule, in the filters' order: 1 where the rule leaves the
 * filter as measured, +∞ where it corrects a filter at or past the ceiling.
 */
export function ruleFactors(filters: readonly NetFilter[], o: RuleOptions, exposure: ReadonlyMap<string, SiteExposure> = siteExposure(filters)): number[] {
  const c = o.ceiling
  if (o.rule === 'filter') return filters.map((f) => own(f.tau, c))
  if (o.rule === 'cut') { const cut = o.cut ?? 0.2; return filters.map((f) => (underRead(f.tau, CENTRAL_CEILING) > cut ? own(f.tau, c) : 1)) }
  if (o.rule === 'site') {
    const by = new Map<string, number[]>()
    for (const f of filters) { const k = own(f.tau, c), list = by.get(f.code); if (list) list.push(k); else by.set(f.code, [k]) }
    const factor = new Map([...by].map(([code, ks]) => [code, median(ks)]))
    return filters.map((f) => factor.get(f.code)!)
  }
  const sites = scopeSites(o.rule, exposure)!
  return filters.map((f) => (sites.has(f.code) ? own(f.tau, c) : 1))
}

/**
 * The exponent b in HIPS ∝ (FTIR EC)^b on a set of filters, with weighed PM2.5 as the instrument so that error in
 * FTIR EC does not flatten it: cov(ln HIPS, ln PM) ÷ cov(ln EC, ln PM) (hips_loading._point). `factors` multiply
 * HIPS; a filter without FTIR EC, PM2.5 or a finite factor is left out. NaN under `min` filters.
 */
export function exponent(filters: readonly NetFilter[], factors?: readonly number[], min = MIN_TEST): { b: number; n: number } {
  const y: number[] = [], x: number[] = [], z: number[] = []
  filters.forEach((f, i) => {
    const k = factors ? factors[i] : 1
    if (f.ec == null || f.pm == null || !(f.ec > 0) || !(f.pm > 0) || !Number.isFinite(k)) return
    y.push(Math.log(f.fabs * k)); x.push(Math.log(f.ec)); z.push(Math.log(f.pm))
  })
  const n = y.length
  if (n < min) return { b: NaN, n }
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / n
  const my = mean(y), mx = mean(x), mz = mean(z)
  let syz = 0, sxz = 0
  for (let i = 0; i < n; i++) { syz += (y[i] - my) * (z[i] - mz); sxz += (x[i] - mx) * (z[i] - mz) }
  return { b: syz / sxz, n }
}

export interface SiteOutcome {
  code: string; n: number
  /** the under-read expected of the site's median filter at the central ceiling */
  underRead: number
  /** whether the rule changes any of the site's filters */
  touched: boolean
  /** median HIPS Fabs (Mm⁻¹) as measured, under the rule, and with every filter by its own factor */
  fabs: number; fabsAfter: number; fabsOwn: number
  /** the exponent on the site's own filters as measured, under the rule, and with every filter by its own factor; NaN under 40 tested filters */
  b: number; bAfter: number; bOwn: number; tested: number
  /** the typical filter's distance from its own factor (median of |applied ÷ own − 1|) and the share of filters more than a tenth away */
  off: number; offTenth: number
}

/** each site under a rule, darkest first */
export function siteOutcomes(filters: readonly NetFilter[], o: RuleOptions): SiteOutcome[] {
  const exposure = siteExposure(filters)
  const applied = ruleFactors(filters, o, exposure), ownK = ruleFactors(filters, { ...o, rule: 'filter' }, exposure)
  const index = new Map<string, number[]>()
  filters.forEach((f, i) => { const list = index.get(f.code); if (list) list.push(i); else index.set(f.code, [i]) })
  const out: SiteOutcome[] = []
  for (const [code, ids] of index) {
    const mine = ids.map((i) => filters[i]), k = ids.map((i) => applied[i]), ko = ids.map((i) => ownK[i])
    const distance = k.map((v, j) => Math.abs(v / ko[j] - 1)).filter(Number.isFinite)
    const before = exponent(mine)
    out.push({
      code, n: mine.length, underRead: exposure.get(code)!.underRead, touched: k.some((v) => v !== 1),
      fabs: median(mine.map((f) => f.fabs)), fabsAfter: median(mine.map((f, j) => f.fabs * k[j])), fabsOwn: median(mine.map((f, j) => f.fabs * ko[j])),
      b: before.b, bAfter: exponent(mine, k).b, bOwn: exponent(mine, ko).b, tested: before.n,
      off: distance.length ? median(distance) : NaN, offTenth: distance.length ? distance.filter((d) => d > 0.1).length / distance.length : NaN,
    })
  }
  return out.sort((a, b) => b.underRead - a.underRead)
}

export interface RuleScore {
  rule: HipsRule; filters: number
  /** filters the rule changes at all, and by more than 5 % */
  corrected: number; changed: number
  /** filters more than a tenth away from their own factor */
  offTenth: number
  /** sites with an exponent of their own, and how many of them sit within 0.1 of proportional (b = 1) before and after */
  tested: number; nearBefore: number; nearAfter: number
  /** tested sites the rule moves more than 0.02 closer to b = 1, and further from it */
  closer: number; further: number
}

/** one line per rule on the same filters: what it changes, how far it is from each filter's own factor, and what it does to the bend */
export function scoreRule(filters: readonly NetFilter[], o: RuleOptions): RuleScore {
  const exposure = siteExposure(filters)
  const applied = ruleFactors(filters, o, exposure), ownK = ruleFactors(filters, { ...o, rule: 'filter' }, exposure)
  const sites = siteOutcomes(filters, o).filter((s) => Number.isFinite(s.b))
  const near = (b: number) => Math.abs(b - 1) <= 0.1
  return {
    rule: o.rule, filters: filters.length,
    corrected: applied.filter((k) => k !== 1).length, changed: applied.filter((k) => !(Math.abs(k - 1) <= 0.05)).length,
    offTenth: applied.filter((k, i) => Math.abs(k / ownK[i] - 1) > 0.1).length,
    tested: sites.length, nearBefore: sites.filter((s) => near(s.b)).length, nearAfter: sites.filter((s) => near(s.bAfter)).length,
    closer: sites.filter((s) => Math.abs(s.bAfter - 1) < Math.abs(s.b - 1) - 0.02).length,
    further: sites.filter((s) => Math.abs(s.bAfter - 1) > Math.abs(s.b - 1) + 0.02).length,
  }
}

/** the jump the `cut` rule puts in the response at its cut-off: a filter just past it is multiplied by its own factor, one just short of it by 1 */
export const stepAt = (cut: number, ceiling: number = CENTRAL_CEILING): number => correction(cut * CENTRAL_CEILING, ceiling)
