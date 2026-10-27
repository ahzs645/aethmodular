import { Placeholder } from './Placeholder'
import type { SectionPageProps } from '../../types'

/** Sub-page 'overview' of the Aethalometer tab (placeholder until built). */
export function Overview(_props: SectionPageProps) {
  return <Placeholder title="Overview — black carbon at five wavelengths" plan="Daily BCc by channel and site, seasonal and monthly patterns, and the per-filter values against the filters' own dates."
    notebooks={['research/ftir_hips_chem/Multi_Site_Analysis.ipynb','research/ftir_hips_chem/Multi_Site_Analysis_Fixed.ipynb','research/ftir_hips_chem/Multi_Site_Analysis_FollowUp.ipynb','research/ftir_hips_chem/addis_02_temporal_patterns.ipynb','research/ftir_hips_chem/addis_02.5_temporal_patterns_filter.ipynb']} />
}
