/**
 * The Research diagnostics tab's sub-pages, in menu order. Each is its own file under ./pages and takes
 * SectionPageProps; list it here. The index page stays first.
 */
import type { ComponentType } from 'react'
import type { SectionPage, SectionPageProps } from '../types'
import { Index } from './pages/Index'
// research-hips-met: the HIPS offset and meteorology pages
import { RhSweep } from './pages/RhSweep'
import { OffsetDrivers } from './pages/OffsetDrivers'
import { TwoProblem } from './pages/TwoProblem'
import { OffsetShift } from './pages/OffsetShift'
import { MoistureMac } from './pages/MoistureMac'
import { HipsEnvelope } from './pages/HipsEnvelope'
import { ImproveEnvelope } from './pages/ImproveEnvelope'
import { ImproveOrigin } from './pages/ImproveOrigin'

export const RESEARCH_PAGES: (SectionPage & { Page: ComponentType<SectionPageProps> })[] = [
  { key: 'index', label: 'Index', title: 'What each page here reproduces', Page: Index },
  // research-hips-met
  { key: 'rh-sweep', label: 'RH threshold sweep', title: 'Method agreement split at every humidity threshold, by season', Page: RhSweep },
  { key: 'offset-drivers', label: 'What tracks the offset', title: 'Residuals, nested models, the recursive split scan, Bland–Altman and influential filters', Page: OffsetDrivers },
  { key: 'two-problem', label: 'Two-problem decomposition', title: 'The HIPS offset as a baseline plus a compressed sensitivity, by site', Page: TwoProblem },
  { key: 'offset-shift', label: 'EC shift and the offset', title: 'Deming against the error ratio, the EC shift that zeros the intercept, and the implied missing EC by season', Page: OffsetShift },
  { key: 'moisture-mac', label: 'Moisture and MAC', title: 'Effective MAC against precipitable water and humidity; does moisture move the slope or the intercept?', Page: MoistureMac },
  { key: 'hips-envelope', label: 'HIPS operating envelope', title: 'Raw HIPS optics, mass per area, lot screen, blank domain and the Addis windows', Page: HipsEnvelope },
  { key: 'improve-envelope', label: 'Addis vs IMPROVE loading', title: 'The IMPROVE network in loading space: Addis windows, matched subsets, the fAbs sweep, smoke and R/T proxies', Page: ImproveEnvelope },
  { key: 'improve-origin', label: 'IMPROVE HIPS at EC = 0', title: 'Does IMPROVE HIPS read a nonzero Fabs at zero EC, and is its response curved? (ftir_26, ftir_37)', Page: ImproveOrigin },
]
