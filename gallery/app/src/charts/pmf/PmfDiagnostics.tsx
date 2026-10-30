import type { MetaFile, PmfFile, PmfRow } from '@/lib/types'
import { DominanceHistogram } from './DominanceHistogram'
import { PmfCoverage } from './PmfCoverage'
import { ResidualVsBiomass } from './ResidualVsBiomass'

/**
 * The PMF sources tab's charts after the source stack: what the PMF solution covers (ETAD_Factor_Analysis c11–13),
 * how dominant its dominant factor is (dominant_source_threshold_analysis c12), and the method difference against the
 * biomass share by season (ETAD_Factor_Analysis c37–38). One component, so the tab mounts them with one line.
 */
export function PmfDiagnostics({ pmf, rows, meta }: { pmf: PmfFile; rows: PmfRow[]; meta: MetaFile }) {
  return <>
    <PmfCoverage pmf={pmf} rows={rows} meta={meta} />
    <DominanceHistogram pmf={pmf} rows={rows} />
    <ResidualVsBiomass pmf={pmf} rows={rows} meta={meta} />
  </>
}
