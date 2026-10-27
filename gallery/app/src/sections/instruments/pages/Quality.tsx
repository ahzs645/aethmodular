import { Placeholder } from './Placeholder'
import type { SectionPageProps } from '../../types'

/** Sub-page 'quality' of the Aethalometer tab (placeholder until built). */
export function Quality(_props: SectionPageProps) {
  return <Placeholder title="Data quality — flow fix, smoothing, coverage" plan="Before and after the flow fix, raw against smoothed BC, attenuation and dual-spot diagnostics, and daily coverage."
    notebooks={['research/ftir_hips_chem/FlowFix_BeforeAfter_Analysis.ipynb','research/ftir_hips_chem/flow_fix_explorer.ipynb','notebooks/qc/flow_qc_analysis.ipynb','research/ftir_hips_chem/HIPS_Aeth_SmoothRaw_Analysis.ipynb','research/ftir_hips_chem/smoothing_comparison.ipynb','research/ftir_hips_chem/Raw_ATN_Correlation_Analysis.ipynb','notebooks/plotting_gaps_scenarios.ipynb','research/ftir_ec_phase3/ftir_49_ma350_raw_chain_diagnostics.ipynb']} />
}
