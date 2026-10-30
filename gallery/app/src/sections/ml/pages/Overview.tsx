import { ChartFrame, Segmented } from '@/components/ChartFrame'
import { useExport } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { INK } from '@/lib/theme'
import type { SectionPageProps } from '../../types'
import type { SpectralExport } from '../../research/pages/spectralCommon'
import { MlWaiting, PageIntro, provenance, rows } from '../common'

const STATUS = {
  here: { label: 'this section', color: () => INK.good },
  elsewhere: { label: 'another page', color: () => INK.accent },
  partial: { label: 'partly', color: () => INK.warn },
  missing: { label: 'not yet', color: () => INK.negative },
} as const
type Status = keyof typeof STATUS
const SHOW = ['all', 'here', 'elsewhere', 'partial', 'missing'] as const

/**
 * The section's map (gallery/data/ml_pages/overview.py, from the 30 Oct 2026 notebook audit): every group of
 * model-fitting results in the notebooks, the gallery page that draws it, and what a partial or missing one needs.
 */
export function Overview(_props: SectionPageProps) {
  const { data, error } = useExport<SpectralExport>('ml/overview.json', (d) => (d.schema_version === 1 ? null : 'unsupported data version'))
  const [show, setShow] = useSearchParam<(typeof SHOW)[number]>('ml-overview.show', 'all', SHOW)
  if (!data) return <MlWaiting error={error} what="the section map" />
  const map = rows(data, 'map').filter((r) => show === 'all' || r.status === show)
  const groups = [...new Set(map.map((r) => String(r.group)))]
  return <>
    <PageIntro id="ml-intro" title="Machine learning — the notebooks’ model-fitting results, as they ran" data={data}
      notebooks="every notebook that fits a model on spectra (the 30 Oct 2026 audit)">
      <p>Everything here needs a model fitted on spectra (PLS calibrations, component selection, cross-validation,
        residual learners, other model forms, bootstraps of refits), so it is precomputed: the pages draw the
        notebooks’ numbers and refit nothing. Where a result is against HIPS Fabs/MAC and the export carries its
        inputs, it still follows the MAC set under Filters → units. Analyses the browser can recompute from the
        filters stay on the chart tabs and under Research.</p>
    </PageIntro>
    <ChartFrame id="ml-map" title="Where each model-fitting result lives"
      subtitle="One row per group of figures. A link opens the gallery page that draws it; a row that is partly there or not yet says what would complete it."
      provenance={provenance(data, ['map'])} exportable={false}
      controls={<Segmented label="Show" value={show} options={SHOW} onChange={setShow}
        optionLabel={(o) => (o === 'all' ? 'All' : STATUS[o].label)} />}>
      <table className="census-table">
        <thead><tr><th>Result</th><th>Notebooks</th><th>Where</th><th>Needs</th></tr></thead>
        <tbody>{groups.flatMap((g) => [
          <tr key={`g-${g}`}><td colSpan={4} style={{ fontWeight: 600, paddingTop: 14 }}>{g}</td></tr>,
          ...map.filter((r) => r.group === g).map((r, i) => {
            const s = STATUS[r.status as Status]
            return <tr key={`${g}-${i}`}>
              <td>{String(r.result)}</td>
              <td><code>{String(r.notebooks)}</code></td>
              <td><span style={{ color: s.color(), fontWeight: 600 }}>{s.label}</span>
                {r.link ? <> · <a href={String(r.link)}>{String(r.where)}</a></> : null}</td>
              <td style={{ color: INK.muted }}>{String(r.needs ?? '')}</td>
            </tr>
          }),
        ])}</tbody>
      </table>
    </ChartFrame>
  </>
}
