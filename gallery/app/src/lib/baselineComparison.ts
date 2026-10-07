export type BaselineMethod = 'AIRSpec' | 'VIBES'
export type BaselineCohort = 'full_pool' | 'locked800'
export interface BaselinePair {
  sample_id: string; Site: string; lot: number; date: string; y: number
  AIRSpec: number; VIBES: number; cohort: BaselineCohort; loading_band: string
}
export interface BaselineGroup {
  cohort: BaselineCohort; dimension: string; group: string; n: number; n_sites: number
  delta_RMSE: number; delta_RMSE_ci_low: number | null; delta_RMSE_ci_high: number | null
}
export interface BaselineSpectrum {
  sample_id: string; site: string; date: string; direction: string; truth: number
  AIRSpec_prediction: number; VIBES_prediction: number; raw: number[]; airspec: number[]; vibes: number[]
}
export interface HistoricalMask {
  mask: string; co2_low: number | null; co2_high: number | null; upper: number | null
  n_channels: number; changed_vs_full: number; changed_vs_no_co2: number
}
/** Historical selection boundaries, reused only to control spectral display. */
export function retainedWavenumber(wn: number, mask: HistoricalMask) {
  return !(mask.co2_low !== null && mask.co2_high !== null && wn >= mask.co2_low && wn <= mask.co2_high)
    && (mask.upper === null || wn <= mask.upper)
}
/** Negative means VIBES is closer to the same physical filter's TOR reference. */
export function pairedAbsoluteErrorDifference(row: BaselinePair) {
  return Math.abs(row.VIBES - row.y) - Math.abs(row.AIRSpec - row.y)
}
export interface BaselineData {
  schema_version: number; evidence_date: string
  run: { n_cases: number; n_failed: number; n_retried: number; n_background_blanks: number; background_rank: number; signature: string }
  scores: { cohort: BaselineCohort; method: BaselineMethod; k: number; n_train: number; n_test: number }[]
  groups: BaselineGroup[]; pairs: BaselinePair[]; wn: number[]; spectra: BaselineSpectrum[]
  historical_masks: HistoricalMask[]
  contributions: BaselineContribution[]
  blank_details: { sample_id: string; source: string; method: BaselineMethod; rms_from_zero: number }[]
  injection_details: { sample_id: string; parent_id: string; amplitude: number; method: BaselineMethod; recovery_rmse: number }[]
  addis_predictions: { sample_id: string; method: BaselineMethod; cohort: BaselineCohort; prediction_ugm3: number; HIPS_EC_equivalent: number }[]
  blanks: { source: string; method: BaselineMethod; n: number; median_rms: number }[]
  injections: { amplitude: number; method: BaselineMethod; n: number; median_recovery_rmse: number }[]
  exclusions: {
    pool_total: number; pool_eligible: number; pool_reasons: { reason: string; n: number }[]
    locked_eligible: number; locked_omitted: { reason: string; n: number }[]
    addis_total: number; addis_registry_excluded: number; addis_roles: { role: string; n: number }[]; registry_unchanged: boolean
  }
  source_sha256: Record<string, string>
}
export interface BaselineContribution {
  sample_id: string; cohort: BaselineCohort; prediction_delta: number
  intercept_and_centering_delta: number; centering_precision_delta: number
  spectra_1425_1799: number; coefficients_1425_1799: number
  spectra_1800_2499: number; coefficients_1800_2499: number
  spectra_2500_2999: number; coefficients_2500_2999: number
  spectra_3000_4000: number; coefficients_3000_4000: number
}
export interface ApplicabilityRow {
  sample_id: string; site: string; kind: 'test' | 'addis'; method: BaselineMethod
  t2: number; q_rms: number; t2_train_percentile: number; q_train_percentile: number
  abs_error: number | null
}
export interface ApplicabilityData {
  background_rank?: number
  schema_version: number; scope: 'full_pool'; n_components: number
  method: string; counts: { train: number; test: number; addis: number }
  rows: ApplicabilityRow[]; source_sha256: Record<string,string>
}
/** Predictions remain unclipped and every finite pair contributes to both methods. */
export function baselineMetrics(rows: BaselinePair[], method: BaselineMethod) {
  if (!rows.length) return null
  const mean = rows.reduce((s, r) => s + r.y, 0) / rows.length
  const sse = rows.reduce((s, r) => s + (r[method] - r.y) ** 2, 0)
  const sst = rows.reduce((s, r) => s + (r.y - mean) ** 2, 0)
  return {
    rmse: Math.sqrt(sse / rows.length),
    mae: rows.reduce((s, r) => s + Math.abs(r[method] - r.y), 0) / rows.length,
    bias: rows.reduce((s, r) => s + r[method] - r.y, 0) / rows.length,
    r2: sst > 0 ? 1 - sse / sst : null,
  }
}
