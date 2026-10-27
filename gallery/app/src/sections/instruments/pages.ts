/**
 * The Aethalometer tab's sub-pages, in menu order. Each is its own file under ./pages; the first is the
 * default. Adding one: write the component (it takes SectionPageProps) and list it here.
 */
import type { ComponentType } from 'react'
import type { SectionPage, SectionPageProps } from '../types'
import { Overview } from './pages/Overview'
import { FilterComparison } from './pages/FilterComparison'
import { Sources } from './pages/Sources'
import { Quality } from './pages/Quality'
import { Diurnal } from './pages/Diurnal'

export const AETH_PAGES: (SectionPage & { Page: ComponentType<SectionPageProps> })[] = [
  { key: 'overview', label: 'Overview', title: 'Daily black carbon at five wavelengths, by site and season', Page: Overview },
  { key: 'filters', label: 'Against the filters', title: 'The aethalometer against HIPS and FTIR EC: MAC, iron and method agreement', Page: FilterComparison },
  { key: 'sources', label: 'Sources (AAE)', title: 'Absorption Ångström exponent and the source classes it implies', Page: Sources },
  { key: 'quality', label: 'Data quality', title: 'Flow fix, smoothing, attenuation and coverage', Page: Quality },
  { key: 'diurnal', label: 'Diurnal', title: 'The day inside a 24 h filter', Page: Diurnal },
]
