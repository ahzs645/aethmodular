import { useMemo } from 'react'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Figure } from '@/components/Figure'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useFitHeight } from '@/lib/chartFit'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt } from '@/lib/format'
import { useSearchParam } from '@/lib/url'
import { StudyLoading, useAddisStudy } from './data'

const PROVENANCE = 'EC_BC.pptx · JPL-main.zip · attributed source-material review; historical summaries and analytical illustrations'
const STATUS: Record<string, string> = {
  ready: 'Displayed in gallery', existing: 'Already available', sensitivity_only: 'Sensitivity candidate',
  requires_inputs: 'Needs measurement inputs', requires_verification: 'Needs method verification',
}

export function AddisStudySourceMaterials({ onPage }: { onPage?: (page: string) => void }) {
  const { data, error } = useAddisStudy()
  const review = data?.source_review
  const rows = useMemo(() => review?.transfer_items.map((r) => ({ ...r, status: STATUS[r.status] ?? r.status })) ?? [], [review])
  if (!data) return <StudyLoading id="aeth-addis-study-materials" title="What the supplied materials add" error={error} />
  return <Figure kind="table" id="aeth-addis-study-materials" title="What the supplied materials add" data={rows}
    provenance={PROVENANCE}
    columns={[{ key: 'item', label: 'Material or method', format: 'text' }, { key: 'status', label: 'Use in the gallery', format: 'text' },
      { key: 'detail', label: 'What comes through', format: 'text' }, { key: 'source_note', label: 'Evidence', format: 'text' }]}
    noteAlways note="The supplied presentation and archive add method evidence and saved historical results. These are kept separate from the Jacros observations above. Their correction candidates have not been applied to that record."
    empty="Regenerate the Addis study export to include the supplied-materials review."
    aside={review && <div className="legend-definition" data-export="text">
      {onPage && <p style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        <button type="button" className="btn" onClick={() => onPage('quality')}>Inspect the existing data-quality checks ›</button>
        <button type="button" className="btn" onClick={() => onPage('sources')}>Compare the absorption and AAE methods ›</button>
      </p>}
      <details><summary>Source identity and interpretation limits</summary>
        {review.sources.map((source) => <section key={source.id}>
          <p><b>{source.title}</b> · {source.kind}</p>
          <ul>{source.evidence.map((item) => <li key={item}>{item}</li>)}</ul>
          <p>SHA-256: <code style={{ overflowWrap: 'anywhere' }}>{source.sha256}</code></p>
        </section>)}
        <ul>{review.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
      </details>
    </div>} />
}

export function AddisStudyHistoricalFits() {
  const { data, error } = useAddisStudy()
  const review = data?.source_review
  const rows = useMemo(() => review?.historical_regressions.map((r) => ({ ...r })) ?? [], [review])
  if (!data) return <StudyLoading id="aeth-addis-study-historical-fits" title="EC/BC fits reported in the presentation" error={error} />
  return <Figure kind="table" id="aeth-addis-study-historical-fits" title="EC/BC fits reported in the presentation" data={rows}
    provenance="EC_BC.pptx · slide 9 · transcribed historical fit summaries"
    columns={[{ key: 'variant', label: 'BC variant (x)', format: 'text' }, { key: 'slope', label: 'EC/BC slope', format: 'f2' },
      { key: 'r_squared', label: 'Reported R²', format: 'f2' }, { key: 'intercept_ug_m3', label: 'Intercept (µg/m³)', format: 'f2' },
      { key: 'residual_r_atn1', label: 'Residual r with ATN1', format: 'f2' }]}
    noteAlways note={review?.regression_context ?? 'Historical fit summaries are unavailable in this export.'}
    aside={<p className="legend-definition" data-export="text">
      The response is FTIR-derived EC (y). These reported relationships are optical/FTIR comparisons; they do not establish thermal-EC accuracy.
      {' '}A higher R² does not select a correction for the current record. The source fit population and evaluation split must be recovered before adoption.
    </p>} />
}

export function AddisStudyOpticalSensitivity() {
  const { data, error } = useAddisStudy()
  const optical = data?.source_review?.optical_sensitivity
  const rows = useMemo(() => optical?.rows.map((r) => ({ ...r,
    biomass_display: fmt(r.biomass_percent, 1),
    interval_display: `${fmt(r.ci_low_percent, 1)}–${fmt(r.ci_high_percent, 1)}`,
  })) ?? [], [optical])
  if (!data) return <StudyLoading id="aeth-addis-study-optical-sensitivity" title="Source-model sensitivity saved in the archive" error={error} />
  return <Figure kind="table" id="aeth-addis-study-optical-sensitivity" title="Source-model sensitivity saved in the archive" data={rows}
    provenance={optical?.source_note ?? PROVENANCE}
    choose={[{ column: 'alpha_fossil', label: 'α fossil', default: '1' }, { column: 'alpha_biomass', label: 'α biomass', default: '2' }]}
    columns={[{ key: 'site', label: 'Archived site', format: 'text' }, { key: 'biomass_display', label: 'Reported biomass BC (%)', format: 'text' },
      { key: 'interval_display', label: 'Archived 95% bootstrap interval (%)', format: 'text' }]}
    noteAlways note={optical?.context ?? 'The archived sensitivity summary is unavailable in this export.'}
    aside={optical && <div className="legend-definition" data-export="text">
      <p>{optical.aggregation}</p><p>{optical.uncertainty}</p>
      <p>The controls select saved table entries. They do not calculate source fractions from the gallery’s IR-only hourly record.
        {' '}The published study uses α<sub>fossil</sub> = 1 and α<sub>biomass</sub> = 2; the archive’s cleaning-pipeline call uses 1 and 1.8.
        {' '}Its separate saved sensitivity grid includes both settings.</p>
    </div>} />
}

const LOADING_ID = 'aeth-addis-study-loading-sensitivity'
const LOADING_TITLE = 'Loading-correction sensitivity from the presentation'

/** Analytical candidate response only; no fitted k or corrected observations are fabricated. */
export function AddisStudyLoadingSensitivity() {
  const { data, error } = useAddisStudy()
  const review = data?.source_review
  const options = useMemo(() => ['all', ...(review?.loading_candidates.map((c) => c.id) ?? [])], [review])
  const [candidate, setCandidate] = useSearchParam<string>(`${LOADING_ID}.candidate`, 'all', options)
  const palette = useColourScheme(LOADING_ID, 'categorical')
  const fit = useFitHeight()
  const series = useMemo<SweepSeries[]>(() => review?.loading_candidates.flatMap((c, i) => {
    const response = review.loading_sensitivity.series.find((s) => s.candidate_id === c.id)
    if (!response || candidate !== 'all' && candidate !== c.id) return []
    return [{ id: c.id, label: `${c.label} · k = ${c.k}`, color: palette.stops[i % palette.stops.length], marks: false, gaps: true,
      dash: i === 1 ? '5 3' : undefined,
      points: response.points.map((p) => ({ x: p.atn1, y: p.multiplier,
        tip: [`${c.label} · k = ${c.k}`, `ATN1 = ${p.atn1} instrument units`,
          p.denominator_valid ? `Analytical BCc/BC1 = ${fmt(p.multiplier, 3)}` : 'Nonpositive denominator: unavailable',
          `Zero denominator at ATN1 = ${fmt(response.singular_atn1, 2)}`, 'Candidate model response; not measured BC'] })) }]
  }) ?? [], [review, candidate, palette.stops.join('|')])
  const key = useMemo(() => series.map((s) => ({ label: s.label, color: s.color, kind: 'line' as const })), [series])
  if (!data) return <StudyLoading id={LOADING_ID} title={LOADING_TITLE} error={error} />
  return <ChartFrame id={LOADING_ID} title={LOADING_TITLE} provenance="EC_BC.pptx · slides 2–5 · analytical constant-k response"
    evidence={<><b>Analytical illustration.</b> BCc / BC1 = 1 / (1 − k × ATN1). The three k values are separately reported candidates.
      {' '}This display does not refit k or modify any Jacros concentration. Nonpositive denominators are unavailable.</>}
    controls={<><Select label="Candidate" value={candidate} options={options} onChange={setCandidate}
      optionLabel={(id) => id === 'all' ? 'Compare all candidates' : review?.loading_candidates.find((c) => c.id === id)?.label ?? id} />{palette.picker}</>}>
    {!review ? <Empty>The supplied loading candidates are unavailable in this export.</Empty> : <>
      <SweepPlot series={series} xScale="linear" xDomain={[review.loading_sensitivity.atn_min, review.loading_sensitivity.atn_max]}
        xLabel="ATN1 (instrument units)" yLabel="Analytical BCc / BC1" reference={1} include={[1]}
        yFormat={(value) => `${fmt(value, 1)}×`} height={Math.max(220, fit ?? 280)} />
      <Key items={key} />
      <div className="legend-definition" data-export="text">
        <p>{review.loading_sensitivity.scope}</p>
        <ul>{review.loading_candidates.filter((c) => candidate === 'all' || candidate === c.id).map((c) => <li key={c.id}>
          <b>{c.label}: k = {c.k}.</b> {c.objective} {c.source_note}
        </li>)}</ul>
        <p>The existing minute-level spot-closure checks test internal algebra. A correction still needs flow/QC review and evaluation on separate time periods before it can support an accuracy claim.</p>
      </div>
    </>}
  </ChartFrame>
}
