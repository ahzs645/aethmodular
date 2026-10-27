/**
 * The Instruments section (src/sections/sections.json: private, so a public build swaps this module for
 * ../omitted.ts and none of it is compiled). One tab so far, the aethalometer, with its sub-pages.
 */
import type { Section } from '../types'
import { AethalometerPage } from './AethalometerPage'
import { AETH_PAGES } from './pages'
import { useAethFields } from './lib'

const instruments: Section = {
  key: 'instruments',
  label: 'Instruments',
  tabs: [{
    key: 'aethalometer',
    label: 'Aethalometer',
    title: 'MA350 aethalometers at the focus sites: black carbon, AAE, MAC, flow fix, smoothing and diurnal cycles',
    pages: AETH_PAGES.map(({ key, label, title }) => ({ key, label, title })),
    Page: AethalometerPage,
  }],
  useFields: useAethFields,
}
export default instruments
