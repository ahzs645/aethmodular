/**
 * Baseline lab export contract. Scalars retain the portable ledger's names; arrays of curves carry
 * their own spectral grids. Current EC products are never recalculated in the browser.
 * Main file: research/baseline_lab/baseline_lab.json
 * Lazy spectra: `${spectra_base}${physical_filter_id}.json`
 */
export interface LabSite { code: string; name: string; color: string }
export interface LabMethod { id: string; label: string; description?: string }
export interface LabDownload { label: string; path: string }

export interface LabFilter {
  id: string
  code: string
  date: string | null
  type: string
  lot_group: string
  ethiopia_season_dry_feb?: string | null
  is_excluded: boolean
  exclusion_reason?: string | null
  spectral_pair_eligible: boolean
  spectral_pair_reason?: string | null
  ec_ftir_ugm3: number | null
  optical_bc_ugm3: number | null
  optical_disagreement_ugm3: number | null
  optical_source: string
  pm25_ugm3: number | null
  volume_m3: number | null
  baseline_sensitivity_rms: number | null
  escape_recomputed: number | null
  contrast_1580_1650?: number | null
  contrast_1600_1635?: number | null
  contrast_1700_1750?: number | null
  contrast_2850_2950?: number | null
  contrast_3200_3400?: number | null
  review_note?: string | null
  [field: string]: string | number | boolean | null | undefined
}

export interface LabCurve {
  id: string
  label: string
  method: string
  kind: 'raw' | 'corrected' | 'removed' | 'derivative' | 'difference' | 'injected'
  group?: string
  scan_id?: string
  wn: number[]
  /** Null is an unsupported spectral channel, not an observed zero. Preserve its grid position. */
  y: (number | null)[]
  lo?: (number | null)[]
  hi?: (number | null)[]
  n?: number
}

export interface LabSpectra {
  schema_version: 1
  id: string
  source: 'native' | 'packed'
  /** Raw/corrected/removed profiles are on physical absorbance scales; never normalized display traces. */
  curves: LabCurve[]
  notes?: string[]
}

export interface LabAssociation {
  /** Distinguish portable, current native direct-HIPS and current native combined cohorts. */
  cohort: string
  code: string
  feature: string
  n: number
  partial_rank_r: number | null
  partial_rank_r_squared: number | null
  controls?: string
}

export interface LabChronology {
  population: string
  quarter: string
  n: number
  metadata_rmse: number | null
  baseline_indicators_rmse: number | null
  band_contrasts_rmse: number | null
  spectra_rmse: number | null
  units?: string
}

export interface LabCohort {
  /** Separate availability, overlapping dates, source restrictions, lot and season contrasts. */
  comparison: string
  group: string
  code: string
  n: number
  median_disagreement: number | null
  median_abs_disagreement: number | null
  season?: string | null
  lot?: string | null
  date_start?: string | null
  date_end?: string | null
  note?: string
}

export interface LabBlank {
  id: string
  method: string
  lot?: string | null
  corrected_1617: number | null
  corrected_rms: number | null
  negative_point_fraction?: number | null
  split?: string
  converged?: boolean | null
  convergence_note?: string
}

export interface LabInjection {
  method: string
  control: string
  amplitude: number
  recovery_fraction: number | null
  n_parents: number
  n_finite?: number
  n_converged?: number
  recovery_converged?: number | null
  recovery_q25?: number | null
  recovery_q75?: number | null
  units?: string
}

export interface LabRepeat {
  id: string
  method: string
  raw_relative_difference_rms: number | null
  corrected_relative_difference_rms: number | null
  filter_type?: string
  raw_difference_rms?: number | null
  corrected_difference_rms?: number | null
  n_pairs?: number
  analysis_a?: string | number
  analysis_b?: string | number
}

export interface LabCalibration {
  cohort: string
  method: string
  split: string
  n: number
  rmse: number | null
  mae?: number | null
  r_squared?: number | null
  units: string
  reference: string
  calibration_matched: boolean
  note?: string
}

export interface LabCalibrationBlank {
  method: string
  n: number
  median: number | null
  mean: number | null
  q25: number | null
  q75: number | null
  rms_from_nominal_zero: number | null
  units: string
}

export interface BaselineLabData {
  schema_version: 1
  generated: string
  source_revision?: string
  mac_m2_g: number
  sites: LabSite[]
  methods: LabMethod[]
  filters: LabFilter[]
  associations: LabAssociation[]
  chronology: LabChronology[]
  cohorts: LabCohort[]
  blanks: LabBlank[]
  injections: LabInjection[]
  repeats?: LabRepeat[]
  /** Saved aggregate signed spectra, e.g. field-blank medians/IQR; not individual filter traces. */
  curves?: LabCurve[]
  calibration?: LabCalibration[]
  calibration_note?: string
  calibration_blanks?: LabCalibrationBlank[]
  calibration_blanks_cohort?: string
  spectra_base: string
  spectra_ids: string[]
  native_spectra_ids?: string[]
  notes?: string[]
  downloads?: LabDownload[]
}

export const LAB_FILE = 'research/baseline_lab/baseline_lab.json'
export const LAB_SOURCE = 'gallery/app/src/sections/research/pages/baselineLab'
export const LAB_UNITS = 'µg/m³'
export const LAB_RESPONSE = 'FTIR EC − optical BC (µg/m³)'

/** No implicit NaN-to-zero coercion in exported diagnostics. */
export const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

export type LabOpticalSource = 'direct' | 'combined'
/** The combined view extends direct Fabs with public HIPS BC only; an SSR-derived product cannot enter. */
export function labOpticalSourceIncluded(row: LabFilter, source: LabOpticalSource): boolean {
  return row.optical_source === 'direct_hips_fabs' || (source === 'combined' && row.optical_source === 'public_hips_extension')
}

export const FEATURE_LABELS: Record<string, string> = {
  baseline_sensitivity_rms: 'Baseline sensitivity RMS',
  protected_1617_change: 'Protected 1617 change',
  escape_recomputed: 'Outside reference envelope',
  contrast_1580_1650: 'Broad 1580–1650 contrast',
  contrast_1600_1635: 'Narrow 1600–1635 contrast',
  contrast_1700_1750: 'Carbonyl 1700–1750',
  contrast_2850_2950: 'C–H 2850–2950',
  contrast_3200_3400: 'O–H 3200–3400',
  removed_1617: 'Removed component at 1617',
  corrected_1617: 'Corrected absorbance at 1617',
  raw_1617: 'Raw absorbance at 1617',
  segment1_sensitivity_rms: 'Segment-one sensitivity',
  segment2_sensitivity_rms: 'Segment-two sensitivity',
  negative_fraction_soft: 'Negative corrected fraction',
}
export const featureLabel = (key: string) => FEATURE_LABELS[key] ?? key.replace(/_/g, ' ')

export function validateLabData(data: BaselineLabData): string | null {
  if (data.schema_version !== 1 || !Array.isArray(data.filters) || !Array.isArray(data.spectra_ids)) return 'unsupported baseline lab export'
  if (data.filters.some(r => !r.id) || new Set(data.filters.map(r => r.id)).size !== data.filters.length) return 'duplicate or empty physical filters in baseline lab export'
  if (new Set(data.spectra_ids).size !== data.spectra_ids.length) return 'duplicate physical-filter spectrum identities'
  for (const curve of data.curves ?? []) {
    const problem = validateLabCurve(curve)
    if (problem) return problem
  }
  return null
}

function validateLabCurve(curve: LabCurve): string | null {
  const validSignal = (values: readonly (number | null)[]) => Array.isArray(values) && values.length === curve.wn.length && values.every(v => v === null || finite(v))
  if (!Array.isArray(curve.wn) || curve.wn.length < 2 || curve.wn.some(v => !finite(v)) || !validSignal(curve.y)
    || (curve.lo && !validSignal(curve.lo)) || (curve.hi && !validSignal(curve.hi))) return 'invalid spectrum grid or values'
  const ascending = curve.wn[1] > curve.wn[0]
  for (let i = 1; i < curve.wn.length; i++) if (ascending ? curve.wn[i] <= curve.wn[i - 1] : curve.wn[i] >= curve.wn[i - 1]) return 'spectrum grid is not strictly ordered'
  return null
}

export function validateLabSpectrum(data: LabSpectra, expectedId: string): string | null {
  if (data.schema_version !== 1 || data.id !== expectedId || !Array.isArray(data.curves)) return 'spectrum identity does not match the selected filter'
  for (const curve of data.curves) {
    const problem = validateLabCurve(curve)
    if (problem) return problem
  }
  return null
}

/**
 * SpectrumPlot's finite-value rule breaks lines and bands at NaN. Keep null channels in place:
 * filtering them out would bridge an interior mask; coercing them to zero would invent absorbance.
 */
export const labSpectrumValues = (values: readonly (number | null)[]): number[] => values.map(v => v === null ? Number.NaN : v)

export type LabSignalView = 'raw' | 'corrected' | 'removed'
export type LabScanView = 'mean' | 'scans' | 'both'

/** A raw-only fallback is visible as Raw; it never acquires an invented correction or normalized trace. */
export function selectLabCurves(data: LabSpectra | null, view: LabSignalView, method: string, scans: LabScanView): {
  view: LabSignalView; curves: LabCurve[]
} {
  const selectedView = data?.source === 'packed' && view === 'corrected' && !data.curves.some(c => c.kind === 'corrected') ? 'raw' : view
  return { view: selectedView, curves: data?.curves.filter(c => c.kind === selectedView
    && (selectedView === 'raw' || method === 'all' || c.method === method)
    && (scans === 'both' || (scans === 'scans' ? !!c.scan_id : !c.scan_id))) ?? [] }
}

/** Exact control membership keeps held-out-blank and leave-lot-out injections separate, even at equal amplitudes. */
export function labInjectionRows(rows: readonly LabInjection[], control: string, method: string): LabInjection[] {
  return rows.filter(r => r.control === control && r.method === method).sort((a, b) => a.amplitude - b.amplitude)
}

/** Saved null outcomes and empty matched populations are unavailable, never zero-valued estimates. */
export function labCohortRows(rows: readonly LabCohort[], comparison: string, absolute = false) {
  const selected = rows.filter(r => r.comparison === comparison)
  return {
    available: selected.filter(r => r.n > 0 && finite(absolute ? r.median_abs_disagreement : r.median_disagreement)),
    unavailable: selected.filter(r => r.n <= 0 || !finite(absolute ? r.median_abs_disagreement : r.median_disagreement)),
  }
}

/** Quarter points and pooled scores are mutually exclusive; the input table is never sorted in place. */
export function labChronologyRows(rows: readonly LabChronology[], population: string, pooled: boolean) {
  return rows.filter(r => r.population === population && (pooled ? /^(all|all_folds|all_test_quarters|pooled|overall)$/i.test(r.quarter) : /^\d{4}Q[1-4]$/.test(r.quarter)))
    .sort((a, b) => a.quarter.localeCompare(b.quarter))
}

/** One cohort and validation split at a time; unmatched coefficient applications cannot enter an accuracy chart. */
export function labCalibrationRows(rows: readonly LabCalibration[], cohort: string, split: string) {
  const selected = rows.filter(r => r.cohort === cohort && r.split === split && r.calibration_matched && finite(r.rmse))
  return { rows: selected, units: selected[0]?.units ?? 'µg/filter', sameUnits: new Set(selected.map(r => r.units)).size <= 1 }
}

/** Field blanks are shown on an absolute scale; a near-zero blank cannot define a stable relative denominator. */
export function labRepeatValues(row: LabRepeat, relative: boolean): { x: number; y: number } | null {
  const useRelative = relative && (row.filter_type ?? 'PM2.5') !== 'FB'
  const x = useRelative ? row.raw_relative_difference_rms : row.raw_difference_rms
  const y = useRelative ? row.corrected_relative_difference_rms : row.corrected_difference_rms
  return finite(x) && finite(y) ? { x, y } : null
}
