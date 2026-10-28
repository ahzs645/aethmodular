import { scaleOrdinal, schemeDark2, schemeSet3, schemeTableau10 } from 'd3'

/** SPARTAN's October 2026 network composition figure, matched by eye. Shared
 * by the map and chemistry charts; these colours are fixed in both themes. */
export const SPARTAN_COMPOSITION_COLOURS: Readonly<Record<string, string>> = {
  bc: '#1a1a1a', teo: '#8c8c8c', dust: '#f9e81b', na: '#2f4ea0', cl: '#2aa198', k: '#b48ad8',
  no3: '#f47c20', nh4: '#ee5fa7', so4: '#ec2227', water: '#6dd0f5', oc: '#2e6b3c', rm: '#4cb847',
}

interface ColourSet {
  colours: readonly string[]
  /** Optional palette for the family bands of a hierarchical legend. */
  familyColours?: readonly string[]
  /** Named components retain their colour in any order or subset. */
  named?: Readonly<Record<string, string>>
}
const aliases: Record<string, readonly string[]> = {
  bc: ['BC', 'BC PM2.5', 'HIPS BC', 'BC (public HIPS)', 'BC (public SSR estimate)', 'Equivalent BC (SSR)', 'EC (FTIR)', 'EC (ChemSpec FTIR)', 'EC (TOR)'],
  teo: ['Trace Element Oxides', 'Trace elements'], dust: ['Fine Soil', 'Dust', 'Metals & crustal'],
  na: ['Sodium', 'Sodium Ion PM2.5'], cl: ['Chloride', 'Chlorine'], k: ['Potassium', 'Potassium Ion PM2.5'],
  no3: ['Nitrate', 'Nitrate Ion PM2.5'], nh4: ['Ammonium', 'Ammonium Ion PM2.5'], so4: ['Sulfate', 'Sulfate Ion PM2.5', 'Ions'],
  water: ['Water', 'Particle-bound water'], oc: ['OC', 'OC (FTIR)', 'OC (ChemSpec FTIR)', 'OC (TOR)', 'OM (FTIR)', 'Carbon'],
  rm: ['Residual matter', 'Residual Matter Only', 'Residual Matter with OC', 'Residual Matter with OM'],
}
const named = Object.fromEntries(Object.entries(SPARTAN_COMPOSITION_COLOURS).flatMap(([key, colour]) =>
  [key, ...(aliases[key] ?? [])].map(label => [label, colour])))

/** A categorical set has discrete swatches, never interpolated between them.
 * Gallery preserves the original composition chart's Tableau10 + Set3 range. */
export const CATEGORICAL_COLOUR_SETS: Readonly<Record<string, ColourSet>> = {
  'SPARTAN composition': { colours: Object.values(SPARTAN_COMPOSITION_COLOURS), named },
  Gallery: { colours: [...schemeTableau10, ...schemeSet3], familyColours: ['#5b6470', '#2171b5', '#b2182b', '#0f766e'] },
  'Tableau 10': { colours: schemeTableau10 },
  'Dark 2': { colours: schemeDark2 },
  'Set 3': { colours: schemeSet3 },
}
export const hasNamedColours = (name: string) => !!CATEGORICAL_COLOUR_SETS[name]?.named

export function categoricalStops(name: string, reverse = false): string[] {
  const set = CATEGORICAL_COLOUR_SETS[name] ?? CATEGORICAL_COLOUR_SETS.Gallery
  return reverse && !set.named ? [...set.colours].reverse() : [...set.colours]
}

/** Reuse for any category domain. Unmatched names in a semantic set use the
 * Gallery range, so an extra measured species is not labelled as a reconstruction. */
export function categoricalScale(name: string, domain: readonly string[], reverse = false, role: 'series' | 'family' = 'series') {
  const set = CATEGORICAL_COLOUR_SETS[name] ?? CATEGORICAL_COLOUR_SETS.Gallery
  const familyStops = role === 'family' ? set.familyColours : undefined
  const stops = familyStops ? (reverse ? [...familyStops].reverse() : familyStops) : categoricalStops(set.named ? 'Gallery' : name, reverse && !set.named)
  const unique = [...new Set(domain)]
  return scaleOrdinal<string, string>().domain(unique).range(unique.map((label, i) => set.named?.[label] ?? stops[i % stops.length]))
}
