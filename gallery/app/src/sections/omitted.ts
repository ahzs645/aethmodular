/**
 * What a section's import resolves to in a public build that leaves it out (vite.config.ts,
 * src/sections/sections.json): no tabs, no fields, and none of the section's code.
 */
import type { Section } from './types'

const omitted: Section = { key: 'omitted', label: '', tabs: [], useFields: () => null }
export default omitted
