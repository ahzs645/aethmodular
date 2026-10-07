import type { CalibRow } from './types.ts'

export type CutoffAxis = 'fitted' | 'cutoff'
export type CalibrationReadout = 'cv_rmse' | 'ho' | 'rmsecv'

/** Missing export fields stay missing; target-site RMSE is never a substitute. */
export function calibrationReadout(row: CalibRow, metric: CalibrationReadout): number | null {
  const value = row[metric]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function cutoffPosition(row: CalibRow, axis: CutoffAxis): number | null {
  const value = axis === 'fitted' ? row.n_fit : row.cut
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

export function fittingCounts(row: CalibRow) {
  const count = (value: number | null | undefined) => typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
  return { cohort: count(row.n_cohort), fitted: count(row.n_fit), heldout: count(row.n_heldout) }
}

/** The plotted x value may be a fit count; selection always uses the original cutoff. */
export function cutoffConfig(row: CalibRow) {
  return { co: row.co, cut: row.cut, sel: row.sel, sp: row.sp }
}
