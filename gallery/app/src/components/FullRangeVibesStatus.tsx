import { useExport } from '@/lib/data'
import { ChartFrame, Note } from '@/components/ChartFrame'

interface FullRangeEvidence {
  rank_cap: number
  background_rank: number
  n_recovered: number
  n_cases: number
  n_corrected: number
  n_failed: number
  channels: number
  wavenumber_range: [number, number]
  case_counts: Record<string, number>
  source: string
  signature: string
  presentation_status: 'completed_uncapped'
}

/** Completed correction with explicitly recorded numerical recovery. */
export function FullRangeVibesStatus({ linkToMeeting = false }: { linkToMeeting?: boolean }) {
  // shown only for a complete run; a missing file is reported by the spectra panel, not here
  const { data: evidence } = useExport<FullRangeEvidence>('similarity/traces_VIBES-full-r70.json',
    (d) => d.presentation_status === 'completed_uncapped' && d.n_corrected === d.n_cases && d.n_failed === 0 ? null : 'run not complete')
  if (!evidence) return null
  const e = evidence
  return <ChartFrame id="full-range-vibes-status" title={`Completed full-range VIBES · background rank ${e.background_rank}`} exportable={false}
    provenance={`${e.source} · run signature ${e.signature}`}>
    <p><strong>Complete recovered calculation:</strong> {e.n_corrected.toLocaleString()} corrected spectra, {e.n_failed} unresolved failures,
      {' '}{e.channels.toLocaleString()} channels from {Math.round(e.wavenumber_range[0])} to {Math.round(e.wavenumber_range[1])} cm⁻¹.</p>
    <p>{e.case_counts.calibration.toLocaleString()} IMPROVE filters and {e.case_counts.target.toLocaleString()} Addis filters,
      plus {e.case_counts.blank} blank cases and {e.case_counts.injection} injection cases. Addis plot counts depend on matched measurement coverage.</p>
    <p>{e.n_recovered} held-out blank fits were recovered with a larger optimizer budget and the same frozen background and model settings. Original outputs are preserved.</p>
    <Note>The completed source-domain benchmark and current per-filter spectra use rank {e.background_rank}.
      Seasonal models and the specification grid now use the completed rank-70 correction; earlier analyses remain available as archives. IMPROVE TOR results assess source-domain prediction;
      Addis HIPS comparisons assess optical agreement.</Note>
    {linkToMeeting && <p><a href="#tab=meeting">Open full-range spectra and seasonal comparisons</a></p>}
  </ChartFrame>
}
