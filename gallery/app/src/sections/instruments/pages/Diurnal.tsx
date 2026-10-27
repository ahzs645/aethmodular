import { Placeholder } from './Placeholder'
import type { SectionPageProps } from '../../types'

/** Sub-page 'diurnal' of the Aethalometer tab (placeholder until built). */
export function Diurnal(_props: SectionPageProps) {
  return <Placeholder title="Diurnal — the day inside the filter" plan="Hour-of-day cycles by channel, season and site, the brown-carbon test, and what AERONET's daylight hours see of a 24 h filter. Needs hourly data from the raw MA350 files."
    notebooks={['research/ftir_hips_chem/addis_05_diurnal_wavelength_analysis.ipynb','research/ftir_hips_chem/multisite_diurnal_wavelength_analysis.ipynb','notebooks/analysis/diurnal_pub_figures.ipynb','research/ftir_ec_phase3/ftir_28_ma350_brc_falsification.ipynb','research/ftir_ec_phase3/scripts/ma350_diurnal_coverage.py']} />
}
