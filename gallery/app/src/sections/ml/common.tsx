/**
 * What the ML pages share. Their exports have the Research pages' shape (gallery/data/export_research_spectral.py
 * Page, written by gallery/data/export_ml.py to public/data/ml/), so the Research helpers draw their intro,
 * provenance and missing-table list.
 */
import { useSpectral, Waiting } from '../research/pages/spectralCommon'

export { PageIntro, provenance, rows } from '../research/pages/spectralCommon'
export const ML_EXPORTER = 'gallery/data/export_ml.py'

/** one ML page's export (public/data/ml/<key>.json) */
export const useMl = (key: string) => useSpectral(key, 'ml')

/** an ML page's loading and error panel (spectralCommon Waiting, i.e. components/ExportState, with export_ml.py) */
export function MlWaiting({ error, what }: { error?: string; what?: string }) {
  return <Waiting error={error} what={what} exporter={ML_EXPORTER} />
}
