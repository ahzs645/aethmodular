/**
 * Saved lot and FTIR-EC controls for the seasonal ML (run_seasonal_ml_checks.py).
 * D = FTIR EC(X) − HIPS BC, so every disagreement score is read against a straight line on FTIR EC,
 * and region tests are repeated on the part of D that line does not explain. The browser never refits.
 */
import { finite } from './baselineLab.ts'

export const SEASONAL_CHECKS_FILE = 'research/baseline_lab/seasonal_checks.json'
export const CHECK_TARGETS = ['residual', 'disagreement'] as const
export type CheckTarget = typeof CHECK_TARGETS[number]
export type CheckFamily = 'rf' | 'linear' | 'pls'

export interface ECLineScore {
  task: 'disagreement'
  calendar: string
  cohort: string
  fold_id: string
  model_id: 'ec_line'
  n_train: number
  n_test: number
  rmse: number
  mae: number
  r2: number
  bias: number
  slope: number
  intercept: number
  dummy_rmse?: number | null
}
export interface ResidualScore {
  fold_id: string
  model_id: string
  family: CheckFamily
  representation: string
  target: CheckTarget
  n_train: number
  n_test: number
  rmse: number
  mae: number
  r2: number
  bias: number
  saved_rmse?: number | null
}
export interface ResidualImportance {
  fold_id: string
  model_id: string
  family: CheckFamily
  representation: string
  target: CheckTarget
  season: string | null
  region_id: string
  region_label: string
  low: number | null
  high: number | null
  n_channels: number
  n_test: number
  delta_mean: number | null
  delta_sd?: number | null
  q05?: number | null
  q95?: number | null
  repeats: number
  baseline_loss: number | null
  metric: 'rmse'
}
export interface LotControl {
  test_id: string
  test_label: string
  family: 'rf' | 'linear'
  representation: string
  model_id: string
  n_train: number
  n_test: number
  lots_train: string
  lots_test: string
  accuracy: number
  balanced_accuracy: number | null
  [recall: `recall__${string}`]: number
}
export interface LotSeparability { family: string; representation: string; model_id: string; n: number; lots: string; balanced_accuracy: number; date_range_248: string[]; date_range_251: string[] }
export interface LotCount { cycle_end: number; season: string; lot_group: string; n: number }
export interface SavedRecall { model_id: string; actual: string; recall: number }
export interface SeasonalChecksData {
  schema_version: 1
  generated_at: string
  lots: { counts: LotCount[]; sources: Record<string, number>; note: string }
  ec_line: ECLineScore[]
  disagreement_ec_correlation: Record<string, number>
  disagreement_by_season: { cycle_end: number; season: string; n: number; mean: number; sd: number | null }[]
  residual_scores: ResidualScore[]
  residual_importance: ResidualImportance[]
  lot_control: LotControl[]
  lot_separability: LotSeparability[]
  saved_forward2024_recall: SavedRecall[]
  labels: { family: Record<string, string>; representation: Record<string, string>; named_regions: string[] }
  reproduction: { compared_models: number; max_abs_rmse_difference: number | null; common_channels: number }
  limitations: string[]
  provenance: { spectra?: { kind: string }; [key: string]: unknown }
}

export const CHECK_TARGET_LABEL: Record<CheckTarget, string> = {
  residual: 'What the EC line misses', disagreement: 'FTIR − HIPS directly',
}

/** The EC-only comparator for one saved disagreement evaluation; null when none was scored (never a guess). */
export function ecLineFor(rows: readonly ECLineScore[] | undefined, key: { calendar: string; cohort: string; fold_id: string }): ECLineScore | null {
  return rows?.find(row => row.calendar === key.calendar && row.cohort === key.cohort && row.fold_id === key.fold_id) ?? null
}

/** Direct and EC-line + residual scores for one fold, paired by model so the two read side by side. */
export function pairResidualScores(rows: readonly ResidualScore[], fold: string, representation: string | 'all') {
  const pick = rows.filter(row => row.fold_id === fold && (representation === 'all' || row.representation === representation))
  const ids = [...new Set(pick.map(row => row.model_id))]
  return ids.map(id => ({
    model_id: id,
    family: pick.find(row => row.model_id === id)!.family,
    representation: pick.find(row => row.model_id === id)!.representation,
    direct: pick.find(row => row.model_id === id && row.target === 'disagreement') ?? null,
    residual: pick.find(row => row.model_id === id && row.target === 'residual') ?? null,
  }))
}

export type CheckRegionSet = 'bins' | 'named' | 'whole'
export const checkRegionSet = (row: Pick<ResidualImportance, 'region_id'>): CheckRegionSet => row.region_id === 'whole_spectrum' ? 'whole'
  : /^bin_[\d.]+_[\d.]+$/.test(row.region_id) ? 'bins' : 'named'

/** One target, fold, representation and season; a null season is the whole held-out cycle. */
export function selectResidualImportance(rows: readonly ResidualImportance[], sel: {
  fold_id: string; target: CheckTarget; representation: string; season: string | null; set: CheckRegionSet; family?: string
}) {
  return rows.filter(row => row.fold_id === sel.fold_id && row.target === sel.target && row.representation === sel.representation
    && (sel.season === null ? row.season == null : row.season === sel.season) && checkRegionSet(row) === sel.set
    && (!sel.family || row.family === sel.family) && finite(row.delta_mean))
    .sort((a, b) => (b.high ?? -Infinity) - (a.high ?? -Infinity) || a.region_id.localeCompare(b.region_id))
}

const count = (v: unknown) => finite(v) && Number.isInteger(v) && v >= 0
const optionalFinite = (v: unknown) => v == null || finite(v)

export function validateSeasonalChecks(data: SeasonalChecksData): string | null {
  if (!data || data.schema_version !== 1 || !['ec_line', 'residual_scores', 'residual_importance', 'lot_control', 'lot_separability', 'limitations']
    .every(key => Array.isArray(data[key as keyof SeasonalChecksData])) || !data.lots || !Array.isArray(data.lots.counts)) return 'unsupported seasonal checks export'
  const lineKeys = new Set<string>()
  for (const row of data.ec_line) {
    const key = JSON.stringify([row.calendar, row.cohort, row.fold_id])
    if (row.model_id !== 'ec_line' || !count(row.n_train) || !count(row.n_test) || lineKeys.has(key)
      || ![row.rmse, row.mae, row.slope, row.intercept].every(finite) || row.rmse < 0) return 'invalid or duplicate EC-line score'
    lineKeys.add(key)
  }
  const scoreKeys = new Set<string>()
  for (const row of data.residual_scores) {
    const key = JSON.stringify([row.fold_id, row.model_id, row.target])
    if (!CHECK_TARGETS.includes(row.target) || !count(row.n_test) || !finite(row.rmse) || row.rmse < 0 || scoreKeys.has(key)) return 'invalid or duplicate residual score'
    scoreKeys.add(key)
  }
  const impKeys = new Set<string>()
  for (const row of data.residual_importance) {
    const key = JSON.stringify([row.fold_id, row.model_id, row.target, row.season ?? '', row.region_id])
    if (!CHECK_TARGETS.includes(row.target) || !row.region_id || !count(row.n_channels) || !count(row.repeats) || impKeys.has(key)
      || ![row.low, row.high, row.delta_mean, row.delta_sd, row.q05, row.q95, row.baseline_loss].every(optionalFinite)
      || (finite(row.q05) && finite(row.q95) && row.q05 > row.q95) || row.metric !== 'rmse') return 'invalid or duplicate residual region test'
    impKeys.add(key)
  }
  for (const row of data.lot_control) {
    if (!row.test_id || !count(row.n_test) || !finite(row.accuracy) || row.accuracy < 0 || row.accuracy > 1) return 'invalid lot-controlled score'
  }
  if (data.lots.counts.some(row => !count(row.n) || !row.season || !row.lot_group)) return 'invalid lot counts'
  return null
}
