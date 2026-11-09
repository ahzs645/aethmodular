/** A legacy export must never acquire a corrected EPO score just by opening it with newer code. */
export interface SourceEvaluation {
  status?: string
  protocol?: string
  reason?: string
  population?: Record<string, unknown>
  provenance?: Record<string, unknown>
}

export interface SourceScore extends Record<string, unknown> {
  model: string
  evaluation_status?: string
}

export function epoSourceVerified(evaluation?: SourceEvaluation): boolean {
  return evaluation?.status === 'corrected' && evaluation.protocol === 'fixed_k_common_physical_filter_metrics_v1'
}

/** Full-source target predictions are separate; only source-evaluation quantities are withheld here. */
export function sourceScoreValue(row: SourceScore | undefined, key: string, evaluation?: SourceEvaluation): number | null {
  if (!row || (row.model === 'epo' && (!epoSourceVerified(evaluation) || row.evaluation_status !== 'corrected'))) return null
  const value = row[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function medianAbsoluteError(row: SourceScore | undefined, evaluation?: SourceEvaluation): number | null {
  return sourceScoreValue(row, row && 'improve_median_absolute_error' in row ? 'improve_median_absolute_error' : 'improve_error', evaluation)
}

/** The old sweep used improve_mae for a median. A new schema's true MAE must never be read as that median. */
export function sweepMedianAbsoluteError(row: SourceScore, evaluation?: SourceEvaluation): number | null {
  const key = 'improve_median_absolute_error' in row ? 'improve_median_absolute_error' : 'improve_mae'
  return sourceScoreValue(row, key, evaluation)
}
