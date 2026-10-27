import { Placeholder } from './Placeholder'
import type { SectionPageProps } from '../../types'

/** Sub-page 'sources' of the Aethalometer tab (placeholder until built). */
export function Sources(_props: SectionPageProps) {
  return <Placeholder title="Sources — AAE and source classes" plan="Absorption Ångström exponent from absorption, fossil/mixed/biomass classes, and BC/EC by source against the PMF factors."
    notebooks={['research/ftir_hips_chem/dominant_source_comparison.ipynb','research/ftir_hips_chem/dominant_source_threshold_analysis.ipynb','research/ftir_hips_chem/threshold_by_source_analysis.ipynb','research/ftir_hips_chem/source_apportionment_regression.ipynb','research/ftir_hips_chem/addis_ababa_source_analysis.ipynb','research/ftir_hips_chem/addis_01_source_apportionment.ipynb']} />
}
