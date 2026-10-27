import { Placeholder } from './Placeholder'
import type { SectionPageProps } from '../../types'

/** Sub-page 'filters' of the Aethalometer tab (placeholder until built). */
export function FilterComparison(_props: SectionPageProps) {
  return <Placeholder title="Against the filters — HIPS, FTIR EC, MAC and iron" plan="The aethalometer against HIPS Fabs and FTIR EC, the MAC each implies, and the iron interference at Addis."
    notebooks={['research/ftir_hips_chem/HIPS_vs_Aethalometer_Optical_Comparison.ipynb','notebooks/analysis/absorption/AAARpos.ipynb','notebooks/analysis/absorption/ETAD_Aug13.ipynb','notebooks/analysis/absorption/ETAD_comprehensive_absorption_analysis.ipynb','notebooks/analysis/iron_correction_test.ipynb']} />
}
