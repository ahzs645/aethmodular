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

// research-hips-met: the HIPS offset and meteorology pages
import { RhSweep } from './pages/RhSweep'
import { OffsetDrivers } from './pages/OffsetDrivers'
import { TwoProblem } from './pages/TwoProblem'
import { OffsetShift } from './pages/OffsetShift'
import { MoistureMac } from './pages/MoistureMac'
import { HipsEnvelope } from './pages/HipsEnvelope'
import { ImproveEnvelope } from './pages/ImproveEnvelope'
import { ImproveOrigin } from './pages/ImproveOrigin'
import { SpartanImprove } from './pages/SpartanImprove'
import { SourceWeather } from './pages/SourceWeather'
import { BlankLines } from './pages/BlankLines'
import { MassLead } from './pages/MassLead'
import { FilterOnly } from './pages/FilterOnly'
import { CharcoalScores } from './pages/CharcoalScores'
import { LotSpectra } from './pages/LotSpectra'

export const RESEARCH_PAGES: (SectionPage & { Page: ComponentType<SectionPageProps> })[] = [
  { key: 'index', label: 'Index', title: 'What each page here reproduces', Page: Index, hidden: true },
  // research-spectra
  { key: 'band-1617', label: '1617 cm⁻¹ band', title: 'ftir_12, ftir_38: what the Addis ~1600 cm⁻¹ band is, by peak position, companion bands and PMF', Page: Band1617 },
  { key: 'mac-choice', label: 'MAC 6 vs 10', title: 'ftir_16, ftir_19: MAC 6 and 10 applied across every calibration, and the IMPROVE implied-MAC bridge', Page: MacChoice },
  { key: 'provisional-ec', label: 'Provisional Addis EC', title: 'ftir_29, ftir_30: the corrected calibration’s Addis EC series as a product, and where its validation stops', Page: ProvisionalSeries },
  { key: 'applicability', label: 'Applicability domain', title: 'ftir_36, ftir_39, ftir_50, char_12: in-domain at Addis, and why that is not validity', Page: Applicability },
  { key: 'network-map', label: 'Network spectral map', title: 'ftir_52: Ward classes over the IMPROVE network, site similarity, a validated match threshold, seasonal analogs', Page: NetworkMap },
  { key: 'ethiopia-june', label: 'Ethiopia calibrations (Jun 2026)', title: 'ftir_ec_calibration_2026_06_25 02/03/07/08/09/10/12/15, spartan_ec 02/03/05/07/08, July07: ETAD EC by sample choice, components and lots; the lot-251 smoke tool; the Adama TOR set', Page: EthiopiaCalibrations },
  { key: 'generalise', label: 'Does it generalise', title: 'ftir_53: five-site intercepts and slopes, Bishoftu replication, the deck lessons restated', Page: Generalise },
  // research-hips-met
  { key: 'rh-sweep', label: 'RH threshold sweep', title: 'Method agreement split at every humidity threshold, by season', Page: RhSweep },
  { key: 'offset-drivers', label: 'What tracks the offset', title: 'Residuals, nested models, the recursive split scan, Bland–Altman and influential filters', Page: OffsetDrivers },
  { key: 'two-problem', label: 'Two-problem decomposition', title: 'The HIPS offset as a baseline plus a compressed sensitivity, by site', Page: TwoProblem },
  { key: 'offset-shift', label: 'EC shift and the offset', title: 'Deming against the error ratio, the EC shift that zeros the intercept, and the implied missing EC by season', Page: OffsetShift },
  { key: 'moisture-mac', label: 'Moisture and MAC', title: 'Effective MAC against precipitable water and humidity; does moisture move the slope or the intercept?', Page: MoistureMac },
  { key: 'hips-envelope', label: 'HIPS operating envelope', title: 'Raw HIPS optics, mass per area, lot screen, blank domain and the Addis windows', Page: HipsEnvelope },
  { key: 'improve-envelope', label: 'Addis vs IMPROVE loading', title: 'The IMPROVE network in loading space: Addis windows, matched subsets, the fAbs sweep, smoke and R/T proxies', Page: ImproveEnvelope },
  { key: 'improve-origin', label: 'IMPROVE HIPS at EC = 0', title: 'Does IMPROVE HIPS read a nonzero Fabs at zero EC, and is its response curved? (ftir_26, ftir_37)', Page: ImproveOrigin },
  { key: 'spartan-improve', label: 'SPARTAN vs IMPROVE', title: 'The shared parameters as one set: distributions, ratios, fits and Addis-like IMPROVE sites', Page: SpartanImprove },
  { key: 'pmf-weather', label: 'PMF sources and weather', title: 'Factor × meteorology, rain washout, temperature range, and method agreement by dominant source (Addis)', Page: SourceWeather },
  { key: 'blank-lines', label: 'Blank lines per deployed line', title: 'ftir_47, ftir_53 01: field-blank R1–T1 lines per deployed HIPS calibration line, and the Fabs each implies', Page: BlankLines },
  { key: 'mass-lead', label: 'The mass lead', title: 'ftir_46: does PM2.5 mass explain the calibration residual at fixed prediction? Live, at every site', Page: MassLead },
  { key: 'filter-only', label: 'Filter-only transfer limits', title: 'filter_only_diagnostics, filter_proportionality, filter_relationship_stability: how far the filter relationships carry over time', Page: FilterOnly },
  { key: 'charcoal', label: 'Charcoal similarity', title: 'char_05–char_09: how closely each Addis spectrum matches the charcoal reference classes, by season, lot and loading', Page: CharcoalScores },
  { key: 'lot-spectra', label: 'PTFE lot comparison', title: 'char_12 §4 for any site or sites: spectra by filter lot, all filters and season-controlled, raw, 2nd derivative and AIRSpec', Page: LotSpectra },
]
