/**
 * Reader-facing names for the calibration work: one place, used by the gallery,
 * mirrored in docs/naming-conventions.md for the notebooks, poster and paper.
 *
 * Agreed with Ann, 23 Sep 2026: "calibration set" = IMPROVE filters the PLS model
 * is built on; "cross-validation" = holding IMPROVE sites (or every 10th filter)
 * out of the calibration set and refitting; "test set" = data the model never saw,
 * here the Addis filters. Internal codes (k, B, AIRSpec, held-out TOR) are not shown
 * to readers without one of these names.
 */

/** Spectral preprocessing (the `spectra` / method field). */
export const PREPROCESSING: Record<string, string> = {
  raw: 'Raw spectra',
  airspec: 'Spline baseline',
  AIRSpec: 'Spline baseline',
  deriv2: 'Second derivative',
  neutral: 'Neutral baseline (pspline-arPLS)',
  vibes: 'Archived VIBES (fit 4000–1425)',
  VIBES: 'Archived VIBES (fit 4000–1425)',
  'VIBES-full': 'VIBES baseline (4000–500, rank cap 30)',
  'VIBES-full-cut': 'Sensitivity: VIBES fit 4000–500, predict 4000–1425 (rank cap 30)',
  'vibes-full': 'VIBES baseline (4000–500, rank cap 30)',
  'vibes-full-cut': 'Sensitivity: VIBES fit 4000–500, predict 4000–1425 (rank cap 30)',
  'VIBES-full-r70': 'VIBES baseline (4000–500, completed rank 70)',
  'VIBES-full-cut-r70': 'Sensitivity: VIBES fit 4000–500, predict 4000–1425 (completed rank 70)',
  'vibes-full-r70': 'VIBES baseline (4000–500, completed rank 70)',
  'vibes-full-cut-r70': 'Sensitivity: VIBES fit 4000–500, predict 4000–1425 (completed rank 70)',
}

/** Short names for controls, legends and table rows where the long name does not fit. */
export const PREPROCESSING_SHORT: Record<string, string> = {
  raw: 'Raw',
  airspec: 'Spline',
  AIRSpec: 'Spline',
  deriv2: '2nd derivative',
  vibes: 'Archived VIBES',
  VIBES: 'Archived VIBES',
  'VIBES-full': 'VIBES (rank 30)',
  'vibes-full': 'VIBES (rank 30)',
  'VIBES-full-cut': 'VIBES (rank 30), 4000–1425',
  'vibes-full-cut': 'VIBES (rank 30), 4000–1425',
  'VIBES-full-r70': 'VIBES',
  'vibes-full-r70': 'VIBES',
  'VIBES-full-cut-r70': 'VIBES, 4000–1425',
  'vibes-full-cut-r70': 'VIBES, 4000–1425',
}

/**
 * Reader-facing name of a baseline. Older exports call every VIBES run "VIBES";
 * pass the run's background rank so a completed rank-70 run gets the current name.
 */
export function baselineName(method: string, { short = false, backgroundRank }: { short?: boolean; backgroundRank?: number | null } = {}): string {
  const key = (method === 'VIBES' || method === 'vibes') && backgroundRank === 70 ? 'VIBES-full-r70' : method
  return (short ? PREPROCESSING_SHORT : PREPROCESSING)[key] ?? method
}

/** Axis titles for the EC crossplots, in either concentration or per-filter mass. */
export const ecAxis = (what: 'hips' | 'tor' | 'ftir', unit = 'µg/m³') =>
  `${{ hips: 'HIPS Fabs / MAC 10', tor: 'TOR EC', ftir: 'FTIR EC' }[what]} (${unit})`

/** One-line definitions, for tooltips and the explain text. */
export const PREPROCESSING_NOTE: Record<string, string> = {
  airspec: 'Smoothing-spline baseline correction (AIRSpec; Kuzmiakova, Dillner and Takahama 2016), 4000–1425 cm⁻¹',
  AIRSpec: 'Smoothing-spline baseline correction (AIRSpec; Kuzmiakova, Dillner and Takahama 2016), 4000–1425 cm⁻¹',
  vibes: 'Probabilistic background removal learned from field blanks (pyvibes; Kamper, Krymova and Takahama)',
  VIBES: 'Archived truncated-grid experiment: sample and blank spectra were restricted to 4000–1425 cm⁻¹ before VIBES fitting. Superseded by the full-range fit; retained for reproducibility.',
  raw: 'Spectra as measured, no baseline correction',
  deriv2: 'Savitzky–Golay second derivative of the raw spectra',
}

/** How the IMPROVE filters in a calibration set were chosen. */
export const COHORT: Record<string, string> = {
  pool: 'All IMPROVE filters',
  smoke: 'Biomass-smoke filters (906)',
  ocec: 'Lowest OC/EC',
  analogs: 'Spectral analogs (VIP-weighted distance)',
  vip_analogs: 'Spectral analogs (VIP-weighted distance)',
  eth_shaped: 'Ethiopia-shaped smoke',
  corr_all: 'Spectral analogs (correlation), all Addis',
  corr_dry: 'Spectral analogs (correlation), Dry season',
  corr_belg: 'Spectral analogs (correlation), Belg season',
  corr_kiremt: 'Spectral analogs (correlation), Kiremt season',
}

/** Where the analog/cohort similarity was measured. */
export const SELECTION_SPACE: Record<string, string> = {
  raw: 'selected on raw spectra',
  airspec: 'selected on spline-baselined spectra',
  vibes: 'selected on VIBES spectra (4000–1425)',
  'vibes-full': 'selected on VIBES spectra (4000–500, rank cap 30)',
  'vibes-full-cut': 'selected on VIBES spectra (fit 4000–500, used 4000–1425, rank cap 30)',
  'vibes-full-r70': 'selected on VIBES spectra (4000–500, completed rank 70)',
  'vibes-full-cut-r70': 'selected on VIBES spectra (fit 4000–500, used 4000–1425, completed rank 70)',
}

/** Cross-validation protocol (the `mode` field). */
export const CV_PROTOCOL: Record<string, string> = {
  site_heldout: 'Site-grouped CV (5-fold, whole sites held out)',
  app: 'Interleaved CV (10-fold, every 10th filter)',
  app_fmm: 'Interleaved CV, first-minimum rule',
}
export const CV_PROTOCOL_SHORT: Record<string, string> = {
  site_heldout: 'site-grouped CV',
  app: 'interleaved CV',
  app_fmm: 'interleaved CV (first minimum)',
}

/** Metrics. Everything evaluated on Addis is the test set. */
export const METRIC = {
  cvR2: 'IMPROVE cross-validation R² (calibration set)',
  cvRmse: 'IMPROVE cross-validation RMSE (calibration set)',
  testR2: 'Test set R² (Addis)',
  testSlope: 'Test set Deming slope (Addis)',
  testIntercept: 'Test set Deming intercept (Addis, µg/m³)',
  k: 'Number of PLS factors',
  cohortSize: 'Calibration cohort size (IMPROVE filters)',
  hips: 'HIPS Fabs / MAC 10 (µg/m³)',
} as const

export const label = (map: Record<string, string>, key: string | null | undefined) =>
  key == null ? '—' : map[key] ?? key
