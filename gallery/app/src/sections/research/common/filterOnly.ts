/**
 * The filter-only release's temporal-transfer models as pure functions (node-tested in filterOnly.test.mjs against
 * the release's own tables). They follow research/ftir_hips_chem/scripts/filter_relationship_stability.py and
 * filter_proportionality.py line for line:
 *
 * - blocks are calendar quarters (pandas Q-DEC) from a site's first to last diagnostic reported date, empty ones
 *   included;
 * - leave-quarter-out trains on every other quarter, later-period on filters reported before the quarter starts;
 * - a fold is fitted only with ≥ `minTrain` (10) training filters and some spread in EC; the three models are
 *   HIPS Fabs on FTIR EC by OLS with an intercept, the training median of Fabs, and the through-origin slope
 *   Σxy / Σx² (the proportional benchmark), all scored on the same withheld filters;
 * - summaries weight filters equally, or (proportionality v2) each evaluated quarter equally.
 *
 * Fabs is in Mm⁻¹ and EC in µg/m³; every model is linear in Fabs, so dividing Fabs by a MAC divides every
 * prediction and error by it and leaves the percentages alone.
 */
import { median, quantile } from './stats.ts'

/** one diagnostic filter (gallery/data/export_research_filter_only.py) */
export interface FPoint {
  id: string
  site: string
  /** reported date, YYYY-MM-DD */
  date: string
  /** FTIR-predicted EC (µg/m³), HIPS Fabs (Mm⁻¹), the EC's reported MDL (µg/m³) */
  ec: number
  fabs: number
  mdl: number
  /** the largest saved MDL multiple whose denominator population holds the filter; 0 = diagnostic only */
  mult: number
  /** the EC_ftir source rows' CalibrationSetId ('11', '17' or 'unreported') */
  cal: string
}

export const POPULATIONS = [
  { key: 'diagnostic', label: 'Diagnostic', short: 'Diag.', min: null },
  { key: 'ratio_baseline', label: 'Ratio / 1× MDL', short: 'Ratio', min: 1 },
  { key: 'mdl_1_5x', label: '1.5× MDL', short: '1.5×', min: 1.5 },
  { key: 'mdl_2x', label: '2× MDL', short: '2×', min: 2 },
  { key: 'mdl_3x', label: '3× MDL', short: '3×', min: 3 },
  { key: 'mdl_5x', label: '5× MDL', short: '5×', min: 5 },
] as const
export type Population = (typeof POPULATIONS)[number]['key']
export const MDL_MULTIPLES = [1, 1.5, 2, 3, 5]
export const MIN_TRAIN = 10

export function inPopulation(p: FPoint, population: Population): boolean {
  const min = POPULATIONS.find((q) => q.key === population)?.min ?? null
  return min === null || p.mult >= min
}

/** the pandas Q-DEC period of a YYYY-MM-DD date: an index that sorts, and its label (2023Q1) */
export const quarterIndex = (date: string) => Number(date.slice(0, 4)) * 4 + Math.floor((Number(date.slice(5, 7)) - 1) / 3)
export const quarterLabel = (q: number) => `${Math.floor(q / 4)}Q${(q % 4) + 1}`

export type Scheme = 'leave_quarter_out' | 'later_period'
export type Status = 'evaluated' | 'insufficient_training_n' | 'constant_training_ec' | 'empty_test_block'
export const MODELS = ['median', 'proportional', 'ols'] as const
export type Model = (typeof MODELS)[number]

export interface TrainFit { status: Exclude<Status, 'empty_test_block'>; slope: number; intercept: number; median: number; k: number }

const mean = (v: readonly number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN)

/** fit_training + proportional_fit: OLS (scipy linregress), the training median of Fabs, Σxy / Σx² */
export function fitTraining(train: readonly FPoint[], minTrain = MIN_TRAIN): TrainFit {
  const none = { slope: NaN, intercept: NaN, median: NaN, k: NaN }
  if (train.length < minTrain) return { status: 'insufficient_training_n', ...none }
  if (new Set(train.map((p) => p.ec)).size < 2) return { status: 'constant_training_ec', ...none }
  const x = train.map((p) => p.ec), y = train.map((p) => p.fabs)
  const mx = mean(x), my = mean(y)
  let sxx = 0, sxy = 0, xx = 0, xy = 0
  for (let i = 0; i < x.length; i++) { sxx += (x[i] - mx) ** 2; sxy += (x[i] - mx) * (y[i] - my); xx += x[i] * x[i]; xy += x[i] * y[i] }
  const slope = sxy / sxx
  return { status: 'evaluated', slope, intercept: my - slope * mx, median: median(y), k: xy / xx }
}

export interface Pred { p: FPoint; block: string; pred: Record<Model, number>; err: Record<Model, number>; outside: boolean | null }
export interface Metrics { mse: Record<Model, number>; mae: Record<Model, number>; rmse: Record<Model, number>; n: number }

/** error_stats / metrics: mean signed error, MAE and RMSE per model over the predictions that have one */
export function metrics(preds: readonly Pred[]): Metrics {
  const ok = preds.filter((q) => Number.isFinite(q.err.ols))
  const out = { mse: {}, mae: {}, rmse: {}, n: ok.length } as Metrics
  for (const m of MODELS) {
    const e = ok.map((q) => q.err[m])
    out.mse[m] = mean(e); out.mae[m] = mean(e.map(Math.abs)); out.rmse[m] = Math.sqrt(mean(e.map((v) => v * v)))
  }
  return out
}

export interface Block extends Metrics {
  block: string
  /** quarterIndex of the block */
  q: number
  status: Status
  trainN: number
  testN: number
  fit: TrainFit
  trainEc: [number, number] | null
  testEc: [number, number] | null
  outsideN: number | null
  preds: Pred[]
}

/**
 * Every quarter of one cohort under one scheme. `siteDiagnostic` sets the calendar (the site's whole diagnostic
 * population, as the release does for every population and variant); `cohort` is what is split and fitted.
 */
export function evaluate(siteDiagnostic: readonly FPoint[], cohort: readonly FPoint[], scheme: Scheme, minTrain = MIN_TRAIN): Block[] {
  if (!siteDiagnostic.length) return []
  const qs = siteDiagnostic.map((p) => quarterIndex(p.date))
  const first = Math.min(...qs), last = Math.max(...qs)
  const tagged = cohort.map((p) => ({ p, q: quarterIndex(p.date) }))
  const out: Block[] = []
  for (let q = first; q <= last; q++) {
    const test = tagged.filter((t) => t.q === q).map((t) => t.p)
    const train = tagged.filter((t) => (scheme === 'leave_quarter_out' ? t.q !== q : t.q < q)).map((t) => t.p)
    const fit = fitTraining(train, minTrain)
    const ok = fit.status === 'evaluated'
    const lo = train.length ? Math.min(...train.map((p) => p.ec)) : NaN, hi = train.length ? Math.max(...train.map((p) => p.ec)) : NaN
    const preds: Pred[] = test.map((p) => {
      const pred = { median: ok ? fit.median : NaN, proportional: ok ? fit.k * p.ec : NaN, ols: ok ? fit.intercept + fit.slope * p.ec : NaN }
      return { p, block: quarterLabel(q), pred, err: { median: pred.median - p.fabs, proportional: pred.proportional - p.fabs, ols: pred.ols - p.fabs }, outside: train.length ? p.ec < lo || p.ec > hi : null }
    })
    out.push({
      block: quarterLabel(q), q, status: test.length ? fit.status : 'empty_test_block', trainN: train.length, testN: test.length, fit,
      trainEc: train.length ? [lo, hi] : null, testEc: test.length ? [Math.min(...test.map((p) => p.ec)), Math.max(...test.map((p) => p.ec))] : null,
      outsideN: train.length ? preds.filter((x) => x.outside).length : null, preds, ...metrics(preds),
    })
  }
  return out
}

/**
 * The held-out OLS prediction for a filter left out of the cohort (an excluded filter, drawn as × under Filters →
 * excluded: Mark): the fit of the block its quarter falls in, which never sees that quarter under leave-quarter-out
 * (so it is the prediction the filter would get in the cohort) and sees only earlier filters under later-period.
 * Null where the quarter is off the calendar or its fit was not evaluated. In no metric.
 */
export function heldOutPrediction(blocks: readonly Block[], p: Pick<FPoint, 'date' | 'ec'>): number | null {
  const b = blocks.find((x) => x.q === quarterIndex(p.date))
  return b && b.fit.status === 'evaluated' ? b.fit.intercept + b.fit.slope * p.ec : null
}

export interface Summary {
  /** every evaluated filter weighted equally (the release's primary estimand) */
  filters: Metrics
  /** each evaluated quarter weighted equally: means of the block MAE and signed error, RMSE as √(mean RMSE²) */
  quarters: Metrics
  evaluatedBlocks: number
  nonemptyBlocks: number
  /** blocks where OLS beats the training median, and where it beats the proportional fit */
  improvedBlocks: number
  interceptBetterBlocks: number
  /** 100 × (median MAE − OLS MAE) / median MAE, equal filters */
  improvementPct: number
}

export function summarize(blocks: readonly Block[]): Summary {
  const valid = blocks.filter((b) => b.status === 'evaluated')
  const filters = metrics(blocks.flatMap((b) => b.preds))
  const quarters = { mse: {}, mae: {}, rmse: {}, n: filters.n } as Metrics
  for (const m of MODELS) {
    quarters.mse[m] = mean(valid.map((b) => b.mse[m])); quarters.mae[m] = mean(valid.map((b) => b.mae[m]))
    quarters.rmse[m] = Math.sqrt(mean(valid.map((b) => b.rmse[m] ** 2)))
  }
  return {
    filters, quarters, evaluatedBlocks: valid.length, nonemptyBlocks: blocks.filter((b) => b.testN > 0).length,
    improvedBlocks: valid.filter((b) => b.mae.median - b.mae.ols > 0).length,
    interceptBetterBlocks: valid.filter((b) => b.mae.proportional - b.mae.ols > 0).length,
    improvementPct: filters.mae.median > 0 ? (100 * (filters.mae.median - filters.mae.ols)) / filters.mae.median : NaN,
  }
}

/** one quarter of the omission comparison: the same withheld filters fitted with and without the omitted one */
export interface InfluenceRow { block: string; n: number; baseMae: number; withoutMae: number; change: number }

/** paired_influence: MAE on filters predicted under both variants, over all of them and per quarter */
export function pairedInfluence(base: readonly Block[], without: readonly Block[]): InfluenceRow[] {
  const omit = new Map(without.flatMap((b) => b.preds.map((q) => [`${b.block}|${q.p.id}`, q] as const)))
  const pairs = base.flatMap((b) => b.preds.flatMap((q) => {
    const o = omit.get(`${b.block}|${q.p.id}`)
    return o && Number.isFinite(q.err.ols) && Number.isFinite(o.err.ols) ? [{ block: b.block, a: Math.abs(q.err.ols), b: Math.abs(o.err.ols) }] : []
  }))
  const row = (block: string, g: typeof pairs): InfluenceRow => {
    const a = mean(g.map((x) => x.a)), w = mean(g.map((x) => x.b))
    return { block, n: g.length, baseMae: a, withoutMae: w, change: w - a }
  }
  const blocks = [...new Set(pairs.map((x) => x.block))]
  return pairs.length ? [row('All common filters', pairs), ...blocks.map((k) => row(k, pairs.filter((x) => x.block === k)))] : []
}

export interface Id11Row {
  block: string
  /** the release's status: 'evaluated' when both training choices are fitted and there are ID-11 test filters */
  common: boolean
  status: string
  testN: number
  allEarlierN: number
  id11EarlierN: number
  all: Metrics
  id11: Metrics
}

/**
 * id11_sensitivity: later-period folds trained on every earlier filter, or only on earlier filters whose EC came
 * from CalibrationSetId 11, both tested on the same later ID-11 filters, and compared only where both fit.
 */
export function id11Sensitivity(later: readonly Block[], cohort: readonly FPoint[], minTrain = MIN_TRAIN, id = '11'): Id11Row[] {
  return later.map((b) => {
    const q = b.q
    const earlier = cohort.filter((p) => quarterIndex(p.date) < q)
    const restricted = earlier.filter((p) => p.cal === id)
    const test = cohort.filter((p) => quarterIndex(p.date) === q && p.cal === id)
    const fits = { all: fitTraining(earlier, minTrain), id11: fitTraining(restricted, minTrain) }
    const common = test.length > 0 && fits.all.status === 'evaluated' && fits.id11.status === 'evaluated'
    const status = common ? 'evaluated' : !test.length ? 'empty_id11_test_block'
      : (['all', 'id11'] as const).filter((k) => fits[k].status !== 'evaluated').map((k) => `${k === 'all' ? 'all_earlier' : 'id11_earlier'}:${fits[k].status}`).join(';')
    const score = (fit: TrainFit) => metrics(test.map((p) => {
      const pred = { median: common ? fit.median : NaN, proportional: common ? fit.k * p.ec : NaN, ols: common ? fit.intercept + fit.slope * p.ec : NaN }
      return { p, block: b.block, pred, err: { median: pred.median - p.fabs, proportional: pred.proportional - p.fabs, ols: pred.ols - p.fabs }, outside: null }
    }))
    return { block: b.block, common, status, testN: test.length, allEarlierN: earlier.length, id11EarlierN: restricted.length, all: score(fits.all), id11: score(fits.id11) }
  })
}

export interface DenominatorRow { multiple: number; n: number; median: number; q25: number; q75: number }

/** filter_only_diagnostics 05: HIPS / FTIR EC over the ratio population at each stricter EC ≥ k × MDL rule */
export function denominatorSensitivity(site: readonly FPoint[]): DenominatorRow[] {
  return MDL_MULTIPLES.map((m) => {
    const r = site.filter((p) => p.mult >= m).map((p) => p.fabs / p.ec)
    return { multiple: m, n: r.length, median: quantile(r, 0.5), q25: quantile(r, 0.25), q75: quantile(r, 0.75) }
  })
}
