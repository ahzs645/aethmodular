/** Saved, physical-filter ML evaluation. The browser selects results; it never trains or refits a model. */
import { finite } from './baselineLab.ts'

export const SEASONAL_ML_FILE = 'research/baseline_lab/seasonal_ml.json'
export const ML_TASKS = ['season', 'disagreement'] as const
export const ML_CALENDARS = ['belg_feb', 'dry_feb'] as const
export type MLTask = typeof ML_TASKS[number]
export type MLCalendar = typeof ML_CALENDARS[number]
export type MLFamily = 'rf' | 'linear' | 'pls' | 'dummy'
export type MLMetric = 'balanced_accuracy' | 'accuracy' | 'macro_f1' | 'rmse' | 'mae' | 'r2'
export type MLImportanceKind = 'permutation' | 'ablation'
export type MLPermutationScheme = 'across_filters' | 'within_season'
export type MLRegionSet = 'bins' | 'named' | 'whole'

export interface MLModel {
  id: string
  label: string
  family: MLFamily
  representation: string
  label_note?: string | null
  covariates: string[] | string
}
export interface MLRepresentation { id: string; label: string; n_channels: number }
export interface MLFold {
  id: string
  label: string
  n_train: number
  n_test: number
  complete_test: boolean
  missing_test_months: (number | string)[]
  note?: string | null
}
export interface MLKey {
  task: MLTask
  calendar: MLCalendar
  cohort: string
  model_id: string
  fold_id: string
}
export interface MLScore extends MLKey {
  n_train: number
  n_test: number
  balanced_accuracy?: number | null
  accuracy?: number | null
  macro_f1?: number | null
  rmse?: number | null
  mae?: number | null
  r2?: number | null
}
export interface MLConfusion extends MLKey { actual: string; predicted: string; n: number }
export interface MLImportance extends MLKey {
  kind: MLImportanceKind
  scheme: MLPermutationScheme
  region_id: string
  region_label: string
  low: number | null
  high: number | null
  n_channels: number
  delta_mean: number | null
  delta_sd?: number | null
  q05?: number | null
  q95?: number | null
  repeats: number
  baseline_loss: number | null
  metric: 'balanced_error' | 'rmse'
  /** Missing/null means all held-out seasons. A named season is a subset of the same pooled model's test fold. */
  season?: string | null
  n_test?: number
}
export interface MLPrediction extends MLKey {
  id: string
  date: string
  season: string
  actual: number | string
  predicted: number | string
}
export interface MLSeasonalSpectrum {
  representation: string
  season: string
  n: number
  wavenumbers: number[]
  median: (number | null)[]
  q25: (number | null)[]
  q75: (number | null)[]
}
export interface SeasonalMLData {
  schema_version: 1
  generated_at: string
  source_revision?: string
  counts: Record<string, unknown>
  models: MLModel[]
  representations: MLRepresentation[]
  folds: MLFold[]
  scores: MLScore[]
  confusion: MLConfusion[]
  importance: MLImportance[]
  predictions: MLPrediction[]
  /** Full descriptive cohort, not the held-out ML fold selected above. Signed per-channel medians and IQRs. */
  seasonal_spectra?: MLSeasonalSpectrum[]
  limitations: string[]
  provenance: { inputs: unknown; settings: unknown }
}

export const ML_TASK_LABEL: Record<MLTask, string> = {
  season: 'Season classification', disagreement: 'FTIR − optical prediction',
}
export const ML_FAMILY_LABEL: Record<MLFamily, string> = {
  rf: 'Random forest', linear: 'Linear model', pls: 'PLS', dummy: 'Reference model',
}
export const ML_METRIC_LABEL: Record<MLMetric, string> = {
  balanced_accuracy: 'Balanced accuracy', accuracy: 'Accuracy', macro_f1: 'Macro F1',
  rmse: 'RMSE (µg/m³)', mae: 'MAE (µg/m³)', r2: 'Prediction R²',
}
export const ML_COHORT_LABEL: Record<string, string> = {
  direct232: 'Direct HIPS · 232 filters', weather188: 'Weather overlap · 188 filters',
  scan_qc229: 'Prior scan-QC sensitivity · 229 filters',
}
export const mlMetricChoices = (task: MLTask): MLMetric[] => task === 'season'
  ? ['balanced_accuracy', 'macro_f1', 'accuracy'] : ['rmse', 'mae', 'r2']

/** Exact membership: do not pool folds, calendars, targets or source cohorts in the browser. */
export function selectMLRows<T extends MLKey>(rows: readonly T[], selection: Partial<MLKey>): T[] {
  return rows.filter(row => (Object.keys(selection) as (keyof MLKey)[])
    .every(key => selection[key] === undefined || row[key] === selection[key]))
}

/** A saved sparse-cycle choice is respected; the first/default view prefers a fully sampled test cycle. */
export function chooseMLFold(folds: readonly MLFold[], available: readonly string[], wanted: string): MLFold | null {
  const eligible = folds.filter(fold => available.includes(fold.id))
  return eligible.find(fold => fold.id === wanted) ?? eligible.find(fold => fold.complete_test) ?? eligible[0] ?? null
}

/** A task-incompatible or missing score is unavailable, never silently zero. */
export function mlScoreValue(row: MLScore, metric: MLMetric): number | null {
  return mlMetricChoices(row.task).includes(metric) && finite(row[metric]) ? row[metric] as number : null
}

/** Classification counts are normalized within the actual class, not across the entire matrix. */
export function mlConfusionMatrix(rows: readonly MLConfusion[]) {
  const order = (name: string) => /^(Bega|Dry)/i.test(name) ? 0 : /^Belg/i.test(name) ? 1 : /^Kiremt/i.test(name) ? 2 : 3
  const classes = [...new Set(rows.flatMap(row => [row.actual, row.predicted]))]
    .sort((a, b) => order(a) - order(b) || a.localeCompare(b))
  const counts = new Map<string, number>()
  const totals = new Map<string, number>()
  for (const row of rows) {
    const key = JSON.stringify([row.actual, row.predicted])
    counts.set(key, (counts.get(key) ?? 0) + row.n)
    totals.set(row.actual, (totals.get(row.actual) ?? 0) + row.n)
  }
  return {
    classes,
    cells: classes.map(actual => classes.map(predicted => {
      const n = counts.get(JSON.stringify([actual, predicted])) ?? 0
      const total = totals.get(actual) ?? 0
      return { n, total, fraction: total > 0 ? n / total : null }
    })),
  }
}

/** Sign and repeat spread are retained. Spread describes repeated perturbations, not a confidence interval. */
export function selectMLImportance(rows: readonly MLImportance[], selection: Partial<MLKey>, kind: MLImportanceKind, scheme: MLPermutationScheme, season: string | null = null): MLImportance[] {
  return selectMLRows(rows, selection).filter(row => row.kind === kind && row.scheme === scheme && finite(row.delta_mean)
    && (season === null ? !row.season : row.season === season))
    .sort((a, b) => (b.high ?? -Infinity) - (a.high ?? -Infinity) || a.region_id.localeCompare(b.region_id))
}

/** Whole-spectrum shuffling is an overall control, never a localized point at the spectrum's midpoint. */
export const mlRegionSet = (row: Pick<MLImportance, 'region_id'>): MLRegionSet => row.region_id === 'whole_spectrum' ? 'whole'
  : /^bin_[\d.]+_[\d.]+$/.test(row.region_id) ? 'bins' : 'named'

/** Profiles retain their actual spectral representation, season counts and signed values without a browser refit. */
export const selectMLProfiles = (rows: readonly MLSeasonalSpectrum[], representation: string) => rows.filter(row => row.representation === representation)

const evaluationKey = (row: MLKey) => [row.task, row.calendar, row.cohort, row.model_id, row.fold_id]
const count = (v: unknown) => finite(v) && Number.isInteger(v) && v >= 0
const optionalFinite = (v: unknown) => v == null || finite(v)

/** Refuse identity inflation and unit/target mismatches before a misleading chart can be drawn. */
export function validateSeasonalMLData(data: SeasonalMLData): string | null {
  if (!data || data.schema_version !== 1 || !['models', 'representations', 'folds', 'scores', 'confusion', 'importance', 'predictions', 'limitations']
    .every(key => Array.isArray(data[key as keyof SeasonalMLData]))) return 'unsupported seasonal ML export'
  for (const rows of [data.models, data.representations, data.folds]) {
    if (rows.some(row => !row.id) || new Set(rows.map(row => row.id)).size !== rows.length) return 'duplicate or empty ML definitions'
  }
  const modelIds = new Set(data.models.map(row => row.id))
  const foldIds = new Set(data.folds.map(row => row.id))
  const validKey = (row: MLKey) => ML_TASKS.includes(row.task) && ML_CALENDARS.includes(row.calendar)
    && !!row.cohort && modelIds.has(row.model_id) && foldIds.has(row.fold_id)
  const seen = new Set<string>()
  for (const row of data.scores) {
    const key = JSON.stringify(evaluationKey(row))
    if (!validKey(row) || !count(row.n_train) || !count(row.n_test) || seen.has(key)) return 'invalid or duplicate model score'
    seen.add(key)
    for (const metric of ['balanced_accuracy', 'accuracy', 'macro_f1', 'rmse', 'mae', 'r2'] as const) {
      const value = row[metric]
      if (!optionalFinite(value)) return 'nonfinite model score'
      if (finite(value) && ((['balanced_accuracy', 'accuracy', 'macro_f1'].includes(metric) && (value < 0 || value > 1))
        || (['rmse', 'mae'].includes(metric) && value < 0))) return 'model score outside its metric range'
    }
  }
  const predictionIds = new Set<string>()
  for (const row of data.predictions) {
    const key = JSON.stringify([...evaluationKey(row), row.id])
    if (!validKey(row) || !row.id || predictionIds.has(key)) return 'duplicate or invalid physical-filter prediction'
    predictionIds.add(key)
    if (row.task === 'disagreement' ? !finite(row.actual) || !finite(row.predicted)
      : typeof row.actual !== 'string' || typeof row.predicted !== 'string') return 'prediction values do not match their task'
  }
  const confusionIds = new Set<string>()
  for (const row of data.confusion) {
    const key = JSON.stringify([...evaluationKey(row), row.actual, row.predicted])
    if (!validKey(row) || row.task !== 'season' || !row.actual || !row.predicted || !count(row.n) || confusionIds.has(key)) return 'invalid or duplicate confusion cell'
    confusionIds.add(key)
  }
  const importanceIds = new Set<string>()
  for (const row of data.importance) {
    const key = JSON.stringify([...evaluationKey(row), row.kind, row.scheme, row.season ?? '', row.region_id])
    if (!validKey(row) || !['permutation', 'ablation'].includes(row.kind) || !['across_filters', 'within_season'].includes(row.scheme)
      || !row.region_id || !count(row.n_channels) || !count(row.repeats)
      || (row.season != null && typeof row.season !== 'string') || (row.n_test != null && !count(row.n_test)) || importanceIds.has(key)
      || ![row.low, row.high, row.delta_mean, row.delta_sd, row.q05, row.q95, row.baseline_loss].every(optionalFinite)
      || (finite(row.low) && finite(row.high) && row.low > row.high)
      || (finite(row.q05) && finite(row.q95) && row.q05 > row.q95)
      || row.metric !== (row.task === 'season' ? 'balanced_error' : 'rmse')) return 'invalid spectral importance, duplicate region or metric units'
    importanceIds.add(key)
  }
  const profileIds = new Set<string>()
  if (data.seasonal_spectra != null && !Array.isArray(data.seasonal_spectra)) return 'invalid seasonal spectrum collection'
  for (const row of data.seasonal_spectra ?? []) {
    const key = JSON.stringify([row.representation, row.season])
    if (!row.representation || !row.season || !count(row.n) || row.n < 1 || profileIds.has(key)
      || !Array.isArray(row.wavenumbers) || row.wavenumbers.length < 2 || !row.wavenumbers.every(finite)
      || ![row.median, row.q25, row.q75].every(values => Array.isArray(values) && values.length === row.wavenumbers.length && values.every(value => value === null || finite(value)))) return 'invalid or duplicate seasonal spectrum'
    const ascending = row.wavenumbers[1] > row.wavenumbers[0]
    if (row.wavenumbers.some((value, i) => i > 0 && (ascending ? value <= row.wavenumbers[i - 1] : value >= row.wavenumbers[i - 1]))) return 'seasonal spectrum grid is not ordered'
    profileIds.add(key)
  }
  return null
}
