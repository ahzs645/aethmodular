import { SectionShell } from '../SectionShell'
import type { SectionPageProps } from '../types'
import { AETH_PAGES } from './pages'

/**
 * The Aethalometer tab: one sub-page per family of notebooks (./pages.ts), picked here or from the
 * header menu (URL `aep`). Every sub-page gets the same rows, measurement and site pick as the Weather
 * tab, with the aethalometer fields already on the rows (lib.ts).
 */
export function AethalometerPage(props: SectionPageProps) {
  return <SectionShell name="Aethalometer" pages={AETH_PAGES} props={props} />
}
