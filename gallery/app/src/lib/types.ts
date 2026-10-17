export interface FilterRow {
  id: string
  site: string
  code: string
  color: string
  date: string
  year: number | null
  month: number | null
  season: string
  lat: number | null
  lon: number | null
  /** from outliers.EXCLUDED_SAMPLES via apply_exclusion_flags */
  excluded: boolean
  exclusion_reason: string | null
  [field: string]: string | number | boolean | null
}

export interface FiltersFile {
  generated: string
  source: string
  n: number
  n_excluded: number
  fields: string[]
  coverage: Record<string, number>
  rows: FilterRow[]
}

export interface SiteMeta {
  code: string
  name: string
  location: string
  color: string
  timezone: string
  lat: number
  lon: number
  n_filters: number
  date_min: string
  date_max: string
}

export interface SeasonMeta {
  name: string
  months: number[]
  color: string
  /** set under the per-site calendar: the site whose local calendar this season belongs to */
  site?: string
}

export interface MetaFile {
  generated: string
  mac_value: number
  /** the gallery's units view (lib/viewFields): concentration, or mass per filter */
  basis?: 'conc' | 'mass'
  /** the PMF groups under the chosen scheme, with 'No PMF day', where a site in view has PMF (lib/pmfGroups) */
  pmf_groups?: { name: string; color: string }[]
  season_convention: string
  seasons: SeasonMeta[]
  /** every registered Ethiopian calendar, keyed by convention name */
  season_conventions: Record<string, SeasonMeta[]>
  wavelengths_nm: Record<string, number>
  /** declared unit per field, e.g. 'ug/m3'; metals arrive ng/m3 and are converted on export */
  field_units: Record<string, string>
  /** measurement families for the grouped pickers */
  field_groups: { label: string; fields: string[] }[]
  sites: SiteMeta[]
  fields: string[]
  coverage: Record<string, number>
}

export interface CorrBlock {
  code: string
  name: string
  fields: string[]
  matrix: (number | null)[][]
  n: Record<string, number>
}

export interface CorrelationFile {
  generated: string
  method: string
  sites: CorrBlock[]
}

export interface CompositionChild {
  name: string
  value: number
  median: number
  n: number
}

export interface CompositionFile {
  generated: string
  units: string
  sites: {
    code: string
    name: string
    color: string
    pm25_mean: number | null
    children: CompositionChild[]
  }[]
}

export interface CensusExample {
  notebook: string
  title: string
  x: string
  y: string
}

export interface CensusFile {
  generated: string
  n_notebooks: number
  n_figures: number
  by_category: [string, number][]
  by_chart: [string, number][]
  top_notebooks: [string, number][]
  examples: Record<string, CensusExample[]>
}

export interface PmfRow {
  id: string
  filter_id: string
  date: string
  year: number
  month: number
  season: string
  fraction: Record<string, number | null>
  ugm3: Record<string, number | null>
  total_ugm3: number | null
  dominant_source: string | null
  dominant_fraction: number | null
}

export interface PmfFile {
  generated: string
  site: string
  site_name: string
  note: string
  sources: { key: string; label: string; color: string }[]
  n: number
  rows: PmfRow[]
}

/** One configuration × evaluation target at the explorer's rule k (compact keys, see gallery/data/export_calibration.py). */
export interface CalibRow {
  guardDm?: number | null; guardDb?: number | null
  co: string; cut: number | null; sel: string | null; sp: string; mo: string; tg: string
  lot: string; el: string; k: number | null; ak: number | null
  om: number | null; ob: number | null; dm: number | null; db: number | null
  ob6: number | null; db6: number | null; r2: number | null; rmse: number | null
  aom: number | null; aob: number | null; adm: number | null; adb: number | null; ar2: number | null
  ho: number | null; n: number | null
  /** IMPROVE TOR loading scores (µg/filter), never evaluation-site concentration RMSE. */
  cv_rmse?: number | null; rmsecv?: number | null
  /** Full selected cohort, final fitted filters, and whole-site holdout filters. */
  n_cohort?: number | null; n_fit?: number | null; n_heldout?: number | null
}

export interface SweepRow {
  co: string; cut: number | null; sel: string | null; sp: string; k: number; ak: number | null
  om: number | null; ob: number | null; dm: number | null; db: number | null; r2: number | null; ho: number | null
}

export interface CalibTarget {
  id: string; label: string; site: string; code: string; color: string; role: string | null; provisional: boolean
}

export interface CalibFile {
  generated: string
  source: string
  n_source_rows: number
  cohorts: Record<string, string>
  spectra: Record<string, string>
  modes: Record<string, string>
  default_cutoff: Record<string, number>
  mac_value: number
  deming_lambda_mac10: number
  slope_box: [number, number]
  heldout_floor: number
  targets: Record<string, CalibTarget>
  grid: CalibRow[]
  sweeps: SweepRow[]
  docs: { file: string; title: string; date: string | null; summary: string }[]
}

// ---- calibration_runs.json: per-filter readouts from the explorer (export_calibration_runs.py)

export interface MetricRow {
  evaluation_set: string
  MAC: number | null
  n: number
  ols_slope: number | null
  ols_intercept: number | null
  deming_slope: number | null
  deming_intercept: number | null
  R2?: number | null
  RMSE?: number | null
  [k: string]: number | string | null | undefined
}

export interface RunEval {
  id: (string | null)[]
  ref: number[]
  pred: number[]
  group: string[]
  date: (string | null)[] | null
  deployed: (number | null)[] | null
  fixed: boolean[]
}

export interface CalibRun {
  preset: string
  target: string
  k: number
  auto_k: number
  n_cohort: number
  n_train: number
  n_train_sites: number
  rmsecv_floor: number
  pct_rmsecv_floor: number
  curve: { k: number; rmsecv: number | null; se: number | null }[]
  heldout: { slope: number; intercept: number; R2: number; RMSE: number } | null
  metrics: MetricRow[]
  split_check: { split: string; n: number; metrics: MetricRow[] }[]
  plausibility: {
    negative_pct: number | null; above_8_pct: number | null; median: number | null
    group_medians: Record<string, number>; group_median_span: number | null
  } | null
  extrap_pct: number | null
  q_residual_pct: number | null
  ref_kind: 'fabs' | 'ec'
  eval: RunEval
}

export interface CohortInfo {
  composition: { centers: number[]; pool: number[]; cohort: number[]; pool_median: number; addis_marker: number }
  label: string
  n: number
  n_sites: number
  top_sites: string[]
  lots: Record<string, number>
  [k: string]: unknown
}

export interface RankingInfo {
  cohort: string
  selection_space: string
  rank: number[]
  metric: number[]
  hist: { centers: number[]; counts: number[]; clipped_at_p99: number }
  cutoff_metric: number
  label: string
  n_total: number
  default_cutoff: number
}

export interface OverlapRow { a: string; b: string; n_a: number; n_b: number; overlap: number }

export interface SpectraBand { q25: number[]; median: number[]; q75: number[] }
export interface CohortSpectra {
  wn: number[]
  series: (SpectraBand & { label: string })[]
  reference: SpectraBand
  reference_label: string
  spectra: string
}
export interface SiteSpectraSeries extends SpectraBand { name: string; label: string; n: number; wn: number[] }

export interface AnalogLab {
  space: string
  n: number
  metrics: string[]
  labels: Record<string, string>
  agreement: Record<string, number>
  explained: number[]
  rank_stride: number
  ranks: Record<string, number[]>
  sample_idx: number[]
  pool_xy: [number, number][]
  addis_xy: [number, number][]
}

export interface YorkFit { slope: number; slope_se: number; intercept: number; intercept_se: number; kappa: number; median_fabs?: number }
export interface HipsYorkRow {
  site: string; code?: string; k?: number; label?: string; n?: number; n_matched?: number
  frac_below_blank_r1?: number
  fits?: Record<string, YorkFit | { error: string }>
  error?: string
}

export interface CalibRunsFile {
  generated: string
  source: string
  provenance: Record<string, unknown> | null
  mode: string
  mac_value: number
  presets: { key: string; label: string; cohort: string; cutoff: number | null; spectra: string }[]
  targets: Record<string, { id: string; label: string; site: string; code: string; color: string }>
  runs: CalibRun[]
  cohorts: Record<string, CohortInfo>
  rankings: RankingInfo[]
  overlap: OverlapRow[]
  cohort_spectra: Record<string, CohortSpectra>
  site_spectra: Record<string, { space: string; series: SiteSpectraSeries[] }>
  analog: Record<string, AnalogLab>
  hips: { blanks?: Record<string, Record<string, unknown>> | null; york?: { rows: HipsYorkRow[]; mac: number }; config?: string }
}
