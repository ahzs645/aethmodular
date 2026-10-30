/**
 * What the ML pages share. Their exports have the Research pages' shape (gallery/data/export_research_spectral.py
 * Page, written by gallery/data/export_ml.py to public/data/ml/), so the Research helpers draw their intro,
 * provenance and missing-table list.
 */
import { ChartFrame, Note } from '@/components/ChartFrame'
import { useSpectral, Waiting } from '../research/pages/spectralCommon'

export { PageIntro, provenance, rows } from '../research/pages/spectralCommon'
export const ML_EXPORTER = 'gallery/data/export_ml.py'

/** one ML page's export (public/data/ml/<key>.json) */
export const useMl = (key: string) => useSpectral(key, 'ml')

export function MlWaiting({ error, what }: { error?: string; what?: string }) {
  return <Waiting error={error} what={what} exporter={ML_EXPORTER} />
}

/** A page whose charts are not built yet: what it will hold and which notebooks it draws. */
export function PagePending({ id, title, notebooks, plan }: { id: string; title: string; notebooks: string; plan: string }) {
  return (
    <ChartFrame id={id} title={title} subtitle={plan} exportable={false}>
      <Note>Notebooks: {notebooks}. Static: precomputed by <code>{ML_EXPORTER}</code>.</Note>
    </ChartFrame>
  )
}
