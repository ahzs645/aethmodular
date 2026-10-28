/**
 * The contract between App and a flagged section (src/sections/sections.json). A section module
 * default-exports a Section; a public build that leaves the section out resolves the same import
 * to src/sections/omitted.ts, whose Section has no tabs and no fields, so App needs no other check.
 */
import type { ComponentType } from 'react'
import type { AxesOpts } from '@/lib/axes'
import type { FilterRow, MetaFile } from '@/lib/types'

export interface SectionPage { key: string; label: string; title?: string
  /** left out of the header menu, the Investigations card and the page picker; opens only from a link (`aep=<key>`) */
  hidden?: boolean }

/** the page a section's tab opens on: its first page that is not hidden */
export const defaultPage = (pages: readonly SectionPage[]): string => (pages.find((p) => !p.hidden) ?? pages[0]).key

export interface SectionPageProps {
  rows: FilterRow[]
  meta: MetaFile
  /** the tab's measurement and crossplot axes (the Filters/Plot sidebar) */
  field: string; xField: string; yField: string
  axes: AxesOpts
  setField: (f: string) => void
  /** the Filters panel's SPARTAN pick, and a way to set it (absent outside a SPARTAN network) */
  picked: Set<string>
  onPick?: (codes: string[]) => void
  /** the open sub-page (URL `aep`) and a way to change it */
  page: string
  onPage: (key: string) => void
}

export interface SectionTab {
  key: string
  label: string
  title?: string
  /** sub-pages, listed in the header menu under the section's label; the first is the default */
  pages: SectionPage[]
  Page: ComponentType<SectionPageProps>
}

/** per-filter fields a section adds to every row (the same shape lib/weather and lib/aeronet use) */
export interface SectionFields {
  group: string
  fields: string[]
  field_units: Record<string, string>
  /** id -> values in `fields` order */
  rows: Record<string, (number | null)[]>
}

export interface Section {
  key: string
  label: string
  tabs: SectionTab[]
  /** a hook: the section's per-filter fields for the sites in view, or null */
  useFields: (codes: readonly string[]) => SectionFields | null
}
