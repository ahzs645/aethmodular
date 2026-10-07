interface CalibrationFit {
  train_n?: number | null
  test_n?: number | null
  cohort_n?: number | null
  k?: number | null
  TOR_test_R2?: number | null
  TOR_test_RMSE_ug?: number | null
}

const count = (v: number | null | undefined) => typeof v === 'number' && Number.isInteger(v) && v >= 0 ? v.toLocaleString() : '—'
const score = (v: number | null | undefined) => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '—'

/** Scores from the saved calibration's IMPROVE holdout, independent of Addis. */
export function MeetingCalibrationDetails({ fit }: { fit?: CalibrationFit }) {
  const cohort = fit?.cohort_n ?? (fit?.train_n != null && fit?.test_n != null ? fit.train_n + fit.test_n : null)
  return <div className="meeting-calibration-details" style={{ marginBottom: 16, overflowX: 'auto' }}>
    <table className="placement" aria-label="Selected calibration fitting counts and IMPROVE scores">
      <thead><tr><th>IMPROVE cohort</th><th>Used in fitting</th><th>Held out</th><th>PLS factors</th><th>Held-out IMPROVE R²</th><th>Held-out IMPROVE RMSE (µg/filter)</th></tr></thead>
      <tbody><tr><td>{count(cohort)}</td><td>{count(fit?.train_n)}</td><td>{count(fit?.test_n)}</td><td>{count(fit?.k)}</td><td>{score(fit?.TOR_test_R2)}</td><td>{score(fit?.TOR_test_RMSE_ug)}</td></tr></tbody>
    </table>
    <p className="chart-note">Counts describe the selected saved calibration, before the Addis test set is evaluated. These scores compare predicted EC with IMPROVE TOR loading on held-out filters; RMSE is in µg/filter. Factor-selection RMSECV is a separate measure. <a href="#tab=calibration">Explore cohort size, RMSECV and fitting counts in the calibration cutoff sweep.</a></p>
  </div>
}
