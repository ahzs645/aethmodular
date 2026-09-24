import { regression } from '@/lib/stats'
import { localSeasonsFor } from '@/siteSeasons'
import type { CalibRun, MetaFile, MetricRow, RunEval, SeasonMeta } from '@/lib/types'

/**
 * Season filter for the per-filter calibration section. The explorer labels
 * every evaluation filter with the Ethiopian dry_feb season; the gallery's
 * default is each site's own calendar (docs/site-seasonality.md), so the
 * labels are recomputed from the sampling date under whichever calendar is
 * picked, and a season (and/or PMF-source) subset is refitted with the
 * explorer's estimators.
 */
export const RUN_CALENDARS = ['Site local calendar', 'Ethiopian · Feb in dry (explorer)'] as const
export type RunCalendar = (typeof RUN_CALENDARS)[number]

/** Deming λ for an Fabs/MAC x-axis; mirrors calibration_modes.deming_lambda (λ ∝ MAC², 2.96 at MAC 10). */
export const demingLambda = (mac: number | null, lambda10: number) => (mac ? lambda10 * (mac / 10) ** 2 : 1)

export function runCalendar(site: string | undefined, meta: MetaFile, cal: RunCalendar): SeasonMeta[] {
  if (cal !== 'Site local calendar') return meta.season_conventions.dry_feb ?? []
  // Bishoftu has no entry of its own; it shares the Ethiopian (Addis) calendar
  return localSeasonsFor(site ?? null, meta) ?? localSeasonsFor('Addis Ababa', meta) ?? meta.season_conventions.dry_feb ?? []
}

/** One readout row refitted on a subset, in the explorer's convention: y = prediction, x = ref ÷ MAC. */
function refit(e: RunEval, idx: number[], row: MetricRow, lambda10: number): MetricRow {
  const set = row.evaluation_set === 'fixed' ? idx.filter((i) => e.fixed[i]) : idx
  const mac = row.MAC
  const xs = set.map((i) => (mac ? e.ref[i] / mac : e.ref[i]))
  const ys = set.map((i) => e.pred[i])
  const st = xs.length >= 3 ? regression(xs, ys, { errorsInVariables: true, lambda: demingLambda(mac, lambda10) }) : null
  const rmse = xs.length ? Math.sqrt(xs.reduce((a, x, k) => a + (ys[k] - x) ** 2, 0) / xs.length) : null
  return {
    ...row,
    n: set.length,
    R2: st?.r2 ?? null,
    RMSE: rmse,
    ols_slope: st?.slope ?? null,
    ols_intercept: st?.intercept ?? null,
    deming_slope: st?.demingSlope ?? null,
    deming_intercept: st?.demingIntercept ?? null,
  }
}

/**
 * The run as the charts should see it: groups relabelled to the chosen
 * calendar and, when a season subset is picked, only those filters, with every
 * metrics row refitted on them. With every season kept the explorer's own
 * metrics are passed through untouched.
 */
/** The PMF-source filter, when the tab offers it: a filter's PMF group from its date, and which groups are kept. */
export interface PmfKeep { labelOf: (date: string | null | undefined) => string; keep: Set<string> | null }

export function seasonalRun(run: CalibRun, seasons: SeasonMeta[], keep: Set<string> | null, lambda10: number, pmf?: PmfKeep): CalibRun {
  const e = run.eval
  const label = (i: number) => {
    const d = e.date?.[i]
    const m = d ? Number(d.slice(5, 7)) : NaN
    return seasons.find((s) => s.months.includes(m))?.name ?? e.group[i]
  }
  const labels = e.pred.map((_, i) => label(i))
  const pmfKeep = pmf?.keep ?? null
  const idx = labels.map((_, i) => i).filter((i) =>
    (!keep || keep.has(labels[i])) && (!pmfKeep || pmfKeep.has(pmf!.labelOf(e.date?.[i]))))
  const subset = !!keep || !!pmfKeep
  const pick = <T,>(a: T[]) => idx.map((i) => a[i])
  const ev: RunEval = {
    id: pick(e.id),
    ref: pick(e.ref),
    pred: pick(e.pred),
    group: pick(labels),
    date: e.date ? pick(e.date) : null,
    deployed: e.deployed ? pick(e.deployed) : null,
    fixed: pick(e.fixed),
  }
  return {
    ...run,
    eval: ev,
    metrics: subset ? run.metrics.map((row) => refit(e, idx, row, lambda10)) : run.metrics,
  }
}
