/**
 * The Research diagnostics tab's sub-pages, in menu order. Each is its own file under ./pages and takes
 * SectionPageProps; list it here. The index page stays first.
 */
import type { ComponentType } from 'react'
import type { SectionPage, SectionPageProps } from '../types'
import { Index } from './pages/Index'
// research-spectra: FTIR spectral and calibration diagnostics (gallery/data/export_research_spectral.py)
import { Band1617 } from './pages/Band1617'
import { MacChoice } from './pages/MacChoice'
import { ProvisionalSeries } from './pages/ProvisionalSeries'
import { Applicability } from './pages/Applicability'
import { NetworkMap } from './pages/NetworkMap'
import { EthiopiaCalibrations } from './pages/EthiopiaCalibrations'
import { Generalise } from './pages/Generalise'

export const RESEARCH_PAGES: (SectionPage & { Page: ComponentType<SectionPageProps> })[] = [
  { key: 'index', label: 'Index', title: 'What each page here reproduces', Page: Index },
  // research-spectra
  { key: 'band-1617', label: '1617 cm⁻¹ band', title: 'ftir_12, ftir_38: what the Addis ~1600 cm⁻¹ band is, by peak position, companion bands and PMF', Page: Band1617 },
  { key: 'mac-choice', label: 'MAC 6 vs 10', title: 'ftir_16, ftir_19: MAC 6 and 10 applied across every calibration, and the IMPROVE implied-MAC bridge', Page: MacChoice },
  { key: 'provisional-ec', label: 'Provisional Addis EC', title: 'ftir_29, ftir_30: the corrected calibration’s Addis EC series as a product, and where its validation stops', Page: ProvisionalSeries },
  { key: 'applicability', label: 'Applicability domain', title: 'ftir_36, ftir_39, ftir_50, char_12: in-domain at Addis, and why that is not validity', Page: Applicability },
  { key: 'network-map', label: 'Network spectral map', title: 'ftir_52: Ward classes over the IMPROVE network, site similarity, a validated match threshold, seasonal analogs', Page: NetworkMap },
  { key: 'ethiopia-june', label: 'Ethiopia calibrations (Jun 2026)', title: 'ftir_ec_calibration_2026_06_25 07/09/10/12/15, spartan_ec 08: ETAD EC by sample choice, components and lots', Page: EthiopiaCalibrations },
  { key: 'generalise', label: 'Does it generalise', title: 'ftir_53: five-site intercepts and slopes, Bishoftu replication, the deck lessons restated', Page: Generalise },
]
