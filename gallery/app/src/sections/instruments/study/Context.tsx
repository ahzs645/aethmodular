import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { fmt } from '@/lib/format'
import { hoursLabel, localRangeLabel, STUDY_PROVENANCE, StudyLoading, useAddisStudy } from './data'

/** Published table entries only: no synthetic Central or full-period Jacros observations. */
export function AddisStudyBenchmarks() {
  const { data, error } = useAddisStudy()
  const rows = useMemo(() => data ? [data.paper.reported.central, data.paper.reported.jacros].map((site) => ({
    site: site.label, mean: site.bc_mean_ug_m3, sd: site.bc_sd_ug_m3, max: site.bc_max_ug_m3, completeness: site.completeness_percent,
  })) : [], [data])
  if (!data) return <StudyLoading id="aeth-addis-study-benchmarks" title="Published study benchmarks" error={error} />
  const paper = data.paper
  return <Figure kind="table" id="aeth-addis-study-benchmarks" title="Published study benchmarks" data={rows}
    provenance={STUDY_PROVENANCE}
    columns={[
      { key: 'site', label: 'Paper site', format: 'text' },
      { key: 'mean', label: 'Mean BC (µg/m³)', format: 'f2' },
      { key: 'sd', label: 'SD (µg/m³)', format: 'f2' },
      { key: 'max', label: 'Maximum (µg/m³)', format: 'f2' },
      { key: 'completeness', label: 'Reported completeness (%)', format: 'f2' },
    ]}
    noteAlways note={`Reported by Shlipak et al. (2026) for ${localRangeLabel(paper.start_local, paper.end_local_exclusive)}. These are published summaries, not observations loaded by the gallery. The paper’s completeness definition has not been reproduced; its 93.9% is not the denominator used for the saved-record coverage below.`}
    aside={<div className="legend-definition" data-export="text">
      <a href={paper.citation.url} target="_blank" rel="noreferrer">Read the study</a>. The study’s MA350 had no size-selective inlet.
      {' '}Its discussion reports {fmt(paper.reported.biomass_bc_percent_discussion, 1)}% biomass BC with α<sub>fossil</sub> = {fmt(paper.source_model.alpha_fossil, 0)} and α<sub>biomass</sub> = {fmt(paper.source_model.alpha_biomass, 0)}.
      {' '}This modelled BC fraction differs from the percentage of observations in an AAE class. Local source fractions are pending.
    </div>} />
}

export function AddisStudyReadiness() {
  const { data, error } = useAddisStudy()
  const rows = useMemo(() => data ? [
    { item: 'Jacros hourly IR BC', status: 'Available · partial record', detail: `${hoursLabel(data.coverage.valid_hours)} valid hours from ${data.deployment.serial}; ${localRangeLabel(data.source.first_local, data.source.end_local_exclusive)}.` },
    { item: 'Named study-comparison workflow', status: 'Implemented · ready for hourly inputs', detail: 'Checks deployment identity and unique UTC hours, then reports named-calendar coverage and BC summaries. Central–Jacros comparisons use their common valid hours when both identified records are supplied.' },
    { item: 'Central observed series', status: 'Pending', detail: data.capabilities.central.reason },
    { item: 'Central–Jacros common hours', status: 'Pending', detail: 'Requires a separately identified Central deployment. ETAD and ETAD-JACROS are overlapping records from the same Jacros instrument.' },
    { item: 'Two-source optical calculation', status: 'Implemented · awaiting optical inputs', detail: 'Uses explicit wavelengths and signal basis for nonlinear mixing, preserving unbounded fractions and invalid/clipping flags. Outputs are long-wavelength optical fractions; a BC mass partition also requires a declared common reference MAC.' },
    { item: 'Local source fractions', status: 'Pending', detail: data.capabilities.source_fractions.reason },
    { item: 'Exact paper reproduction', status: 'Pending', detail: 'The complete deployment records, supporting-information QC and source-model settings, and exact climatology exclusion windows are still needed.' },
    { item: 'Independent validation', status: 'Not established', detail: 'Overlapping Jacros observations provide descriptive context, not an independent validation dataset.' },
  ] : [], [data])
  if (!data) return <StudyLoading id="aeth-addis-study-readiness" title="What is available and what is pending" error={error} />
  return <Figure kind="table" id="aeth-addis-study-readiness" title="What is available and what is pending" data={rows}
    provenance={STUDY_PROVENANCE}
    columns={[{ key: 'item', label: 'Comparison', format: 'text' }, { key: 'status', label: 'Status', format: 'text' }, { key: 'detail', label: 'Basis', format: 'text' }]}
    noteAlways note="This page describes a saved hourly export. It does not rerun raw-minute QC, infer Central from a Jacros alias, or estimate source fractions from IR concentration alone."
    aside={<details className="legend-definition" style={{ marginTop: 8 }}>
      <summary>Input provenance and remaining method details</summary>
      <p>Input: <code>{data.source.input_file}</code>. Upstream record: {typeof data.source.upstream_record === 'string' ? data.source.upstream_record : 'identified in the export'}.</p>
      <p>{data.source.timestamp_basis}. {data.source.finite_handling}</p>
      <p>Input SHA-256: <code style={{ overflowWrap: 'anywhere' }}>{data.source.input_sha256}</code></p>
      <ul>{data.paper.unresolved_for_exact_reproduction.map((reason) => <li key={reason}>{reason}</li>)}</ul>
    </details>} />
}
