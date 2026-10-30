/**
 * The Machine learning tab's sub-pages, in menu order. Each is its own file under ./pages and takes
 * SectionPageProps; its data is public/data/ml/<key>.json, written by gallery/data/ml_pages/<module>.py.
 */
import type { ComponentType } from 'react'
import type { SectionPage, SectionPageProps } from '../types'
import { Overview } from './pages/Overview'
import { Components } from './pages/Components'
import { Protocols } from './pages/Protocols'
import { JuneVariants } from './pages/JuneVariants'
import { Residual } from './pages/Residual'
import { Targets } from './pages/Targets'
import { ModelForm } from './pages/ModelForm'

export const ML_PAGES: (SectionPage & { Page: ComponentType<SectionPageProps> })[] = [
  { key: 'overview', label: 'Overview', title: 'Every model-fitting result in the notebooks, and where the gallery shows it', Page: Overview },
  { key: 'components', label: 'Components and CV', title: 'ftir_11, ftir_20, ftir_23, ftir_32: PLS factors under site-grouped and interleaved cross-validation (the June 2026 set: June 2026 variants)', Page: Components },
  { key: 'protocols', label: 'Protocols and transfer', title: 'ftir_13, ftir_15, ftir_17, ftir_18, ftir_21, ftir_22, ftir_31: app vs site-held-out end to end, transfer roundup, bootstrap intervals', Page: Protocols },
  { key: 'june', label: 'June 2026 variants', title: 'ftir_ec_calibration_2026_06_25 05–14, spartan_ec 05/07/08: CAL-0…CAL-7, chasing the origin, calibration stability', Page: JuneVariants },
  { key: 'residual', label: 'Residual learners', title: 'ftir_43, ftir_45, ftir_46: residual learners against null controls, the paired increment and mass', Page: Residual },
  { key: 'targets', label: 'What counts as EC', title: 'ftir_41, ftir_42, ftir_44, ftir_48: target definitions and the Adama TOR filters', Page: Targets },
  { key: 'model-form', label: 'Model form and VIP', title: 'ftir_54, ftir_55, ftir_51, ftir_07, ftir_08: other model forms, every combination tried, feature importance', Page: ModelForm },
]
