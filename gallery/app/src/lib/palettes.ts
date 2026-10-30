import {
  color, interpolateCividis, interpolateGreens, interpolateGreys, interpolateInferno, interpolateLab, interpolateMagma,
  interpolateOranges, interpolatePlasma, interpolatePurples, interpolateReds, interpolateViridis, interpolateYlGnBu,
  interpolateYlOrRd, piecewise, quantize, scaleOrdinal, schemeDark2, schemeObservable10, schemeSet3, schemeTableau10,
} from 'd3'

/**
 * Every colour palette a chart can offer, one list per kind: sequential ramps for counts and
 * densities, diverging ramps for signed values, categorical sets for named groups. Each entry says
 * where its colours come from and whether they hold up for colour-vision deficiencies or in print.
 *
 * To add a palette, add an entry to its list. Every picker of that kind lists it
 * (components/ColourSchemePicker), its dark-theme steps and export colours are derived from it
 * (lib/colourSchemes), and the line under its name is built by paletteNote. 'Gallery' is each
 * kind's default when a chart names none, and stands in for a name a link no longer matches.
 *
 * Sources:
 *   ColorBrewer (Brewer, colorbrewer2.org), via d3-scale-chromatic
 *   matplotlib's perceptually uniform maps (van der Walt & Smith 2015; cividis: Nuñez et al. 2018)
 *   Crameri, Shephard & Heron (2020) Nat. Commun. 11, 5444; Scientific colour maps v8, MIT
 *   Okabe & Ito (2008) Colour Universal Design; Wong (2011) Nat. Methods 8, 441
 *   Tol (2021) Colour Schemes, SRON/EPS/TN/09-002 v3.2: qualitative sets are used as given, never
 *     interpolated, and never stretched past their length
 */
export type PaletteKind = 'sequential' | 'diverging' | 'categorical'

export interface Palette {
  /** shown in the picker and kept in the link as `<chart>.palette` */
  name: string
  /**
   * The light theme's colours. Sequential: 8 stops, low to high. Diverging: 7 stops, negative to
   * neutral to positive. Categorical: the category colours in order, never blended.
   */
  colours: readonly string[]
  /** where the colours come from */
  source: string
  /** stays apart for the common colour-vision deficiencies */
  safe?: boolean
  /** keeps its order printed in black and white */
  greyscale?: boolean
  /** anything else the reader should know, shown after the flags */
  note?: string
  /** categorical: labels that keep their colour in any order or subset; other labels take Gallery colours */
  named?: Readonly<Record<string, string>>
  /** categorical: colours for the family bands of a hierarchical legend */
  familyColours?: readonly string[]
  /**
   * categorical: the dark page's twin of each colour that would vanish on it (black on the dark
   * panel), by light colour. Charts still draw the light colour; lib/colourSchemes registers each
   * pair with lib/theme paintPair, so the page shows the twin and an export prints the original.
   */
  dark?: Readonly<Record<string, string>>
}

/** a d3 ramp sampled at 8 hex stops between `lo` and `hi`: the palest end of a light-first ramp barely shows on white */
const ramp = (f: (t: number) => string, lo = 0, hi = 1) => quantize((t) => color(f(lo + (hi - lo) * t))!.formatHex(), 8)

/** batlow, Scientific colour maps v8 (Crameri), at 32 even steps */
const BATLOW = [
  '#011959', '#08265b', '#0d315d', '#0f3b5f', '#114460', '#144c62', '#175362', '#1c5b62',
  '#246160', '#2d675d', '#386c58', '#457051', '#51744a', '#5e7843', '#6c7c3c', '#7a8035',
  '#89842f', '#99882c', '#a98c2c', '#b98f32', '#c9923c', '#d79448', '#e49756', '#ef9c67',
  '#f7a17a', '#fba78d', '#fdada0', '#fdb3b2', '#fdb9c3', '#fcbfd5', '#fbc6e7', '#faccfa',
]

/** SPARTAN's October 2026 network composition figure, matched by eye. Shared by the map and
 * chemistry charts; these colours are fixed in both themes. */
export const SPARTAN_COMPOSITION_COLOURS: Readonly<Record<string, string>> = {
  bc: '#1a1a1a', teo: '#8c8c8c', dust: '#f9e81b', na: '#2f4ea0', cl: '#2aa198', k: '#b48ad8',
  no3: '#f47c20', nh4: '#ee5fa7', so4: '#ec2227', water: '#6dd0f5', oc: '#2e6b3c', rm: '#4cb847',
}
const SPARTAN_ALIASES: Record<string, readonly string[]> = {
  bc: ['Black Carbon', 'BC', 'BC PM2.5', 'HIPS BC', 'BC (public HIPS)', 'BC (public SSR estimate)', 'Equivalent BC (SSR)', 'EC (FTIR)', 'EC (ChemSpec FTIR)', 'EC (TOR)'],
  teo: ['TEO', 'Trace Element Oxides', 'Trace elements', 'Trace metals'], dust: ['Fine Soil', 'Dust', 'Crustal elements'],
  na: ['Sodium', 'Sodium Ion PM2.5'], cl: ['Chloride', 'Chlorine'], k: ['Potassium', 'Potassium Ion PM2.5'],
  no3: ['Nitrate', 'Nitrate Ion PM2.5'], nh4: ['Ammonium', 'Ammonium Ion PM2.5'], so4: ['Sulfate', 'Sulfate Ion PM2.5', 'Ions'],
  water: ['Water', 'Particle-bound water'], oc: ['Org. Carbon', 'OC', 'OC (FTIR)', 'OC (ChemSpec FTIR)', 'OC (TOR)', 'OM (FTIR)', 'Carbon'],
  rm: ['Residual', 'Residual matter', 'Residual Matter Only', 'Residual Matter with OC', 'Residual Matter with OM'],
}
/**
 * The FTIR organic functional groups in the colours of the IMPROVE functional-group figure (Atmos. Environ. 2017,
 * Fig. 7, OM/OC by group): each the median pixel of that group's bars in the published image, listed bottom of
 * that figure's stack to top. The SPARTAN FTIR product has
 * no amine group: its colour is kept so a dataset that has one draws it as the figure does.
 */
export const FTIR_GROUP_COLOURS: Readonly<Record<string, string>> = {
  'Alkane CH': '#0035f4', 'Carboxylic COOH': '#77f94f', 'Non-acid CO': '#387e7f', 'Alcohol COH': '#ed72b2', 'Amine NH2': '#f1aa42',
}
/** the figure's own names for the groups */
const FTIR_GROUP_ALIASES: Record<string, readonly string[]> = {
  'Alkane CH': ['alkaneCH'], 'Carboxylic COOH': ['carboxylicCOH', 'Carboxylic COH'], 'Non-acid CO': ['carbonylCO', 'Carbonyl CO'],
  'Alcohol COH': ['alcoholCOH'], 'Amine NH2': ['amineNH2'],
}
const FTIR_GROUP_NAMED = Object.fromEntries(Object.entries(FTIR_GROUP_COLOURS).flatMap(([key, colour]) =>
  [key, ...(FTIR_GROUP_ALIASES[key] ?? [])].map((label) => [label, colour])))
/** the stack order of the functional groups, bottom to top, as the reference figure draws them */
export const FTIR_GROUP_ORDER = Object.keys(FTIR_GROUP_COLOURS)
/** the PMF paper's source colours (Anwar et al., Addis Ababa, Figure 7), keyed by the PMF export's source labels */
export const PMF_PAPER_COLOURS: Readonly<Record<string, string>> = {
  'Fossil Fuel Combustion': '#f4b6c2', 'Polluted Marine': '#e0191c', 'Sea Salt Mixed': '#0a14d0', 'Wood Burning': '#92e192', Charcoal: '#2e8b2e',
}
// the composition set colours the functional groups as well, so any chart on its default draws them as the figure does
const SPARTAN_NAMED = {
  ...FTIR_GROUP_NAMED,
  ...Object.fromEntries(Object.entries(SPARTAN_COMPOSITION_COLOURS).flatMap(([key, colour]) =>
    [key, ...(SPARTAN_ALIASES[key] ?? [])].map((label) => [label, colour]))),
}

export const PALETTES: Readonly<Record<PaletteKind, readonly Palette[]>> = {
  sequential: [
    { name: 'Gallery', colours: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'], source: 'ColorBrewer Blues, 8 classes' },
    { name: 'Viridis', colours: ramp(interpolateViridis), source: 'matplotlib viridis', safe: true },
    { name: 'Cividis', colours: ramp(interpolateCividis), source: 'matplotlib cividis, built for colour-vision deficiencies', safe: true },
    { name: 'Batlow', colours: ramp(piecewise(interpolateLab, BATLOW)), source: 'Crameri Scientific colour maps v8, batlow', safe: true },
    { name: 'Plasma', colours: ramp(interpolatePlasma), source: 'matplotlib plasma' },
    { name: 'Magma', colours: ramp(interpolateMagma, 0.1), source: 'matplotlib magma, without its black end' },
    { name: 'Inferno', colours: ramp(interpolateInferno, 0.1), source: 'matplotlib inferno, without its black end' },
    { name: 'Yellow–green–blue', colours: ramp(interpolateYlGnBu, 0.08), source: 'ColorBrewer YlGnBu' },
    { name: 'Yellow–orange–red', colours: ramp(interpolateYlOrRd, 0.08), source: 'ColorBrewer YlOrRd' },
    { name: 'Reds', colours: ramp(interpolateReds, 0.12), source: 'ColorBrewer Reds' },
    { name: 'Greens', colours: ramp(interpolateGreens, 0.08), source: 'ColorBrewer Greens' },
    { name: 'Oranges', colours: ramp(interpolateOranges, 0.08), source: 'ColorBrewer Oranges' },
    { name: 'Purples', colours: ramp(interpolatePurples, 0.08), source: 'ColorBrewer Purples' },
    { name: 'Greys', colours: ramp(interpolateGreys, 0.05, 0.95), source: 'ColorBrewer Greys', greyscale: true },
  ],
  diverging: [
    { name: 'Gallery', colours: ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'], source: 'ColorBrewer RdBu, 7 classes, blue negative' },
    { name: 'Red–yellow–blue', colours: ['#4575b4', '#91bfdb', '#e0f3f8', '#ffffbf', '#fee090', '#fc8d59', '#d73027'], source: 'ColorBrewer RdYlBu, 7 classes, blue negative' },
    { name: 'Purple–orange', colours: ['#542788', '#998ec3', '#d8daeb', '#f7f7f7', '#fee0b6', '#f1a340', '#b35806'], source: 'ColorBrewer PuOr, 7 classes', safe: true },
    { name: 'Brown–teal', colours: ['#01665e', '#5ab4ac', '#c7eae5', '#f5f5f5', '#f6e8c3', '#d8b365', '#8c510a'], source: 'ColorBrewer BrBG, 7 classes, teal negative', safe: true },
    { name: 'Purple–green', colours: ['#762a83', '#af8dc3', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#7fbf7b', '#1b7837'], source: 'ColorBrewer PRGn, 7 classes' },
    { name: 'Greys', colours: ['#252525', '#636363', '#bdbdbd', '#f7f7f7', '#bdbdbd', '#636363', '#252525'], source: 'ColorBrewer Greys, mirrored about zero', greyscale: true },
  ],
  categorical: [
    { name: 'SPARTAN composition', colours: Object.values(SPARTAN_COMPOSITION_COLOURS), named: SPARTAN_NAMED, source: "SPARTAN's October 2026 network composition figure", note: 'fixed component colours; additional species use Gallery colours' },
    // the original Stacked bars range
    { name: 'Gallery', colours: [...schemeTableau10, ...schemeSet3], familyColours: ['#5b6470', '#2171b5', '#b2182b', '#0f766e'], source: 'Tableau 10 followed by ColorBrewer Set3' },
    { name: 'FTIR functional groups', colours: Object.values(FTIR_GROUP_COLOURS), named: FTIR_GROUP_NAMED, source: 'IMPROVE functional-group figure, Atmos. Environ. 2017, Fig. 7 (sampled from the published image)', note: 'fixed group colours; other labels use Gallery colours' },
    { name: 'PMF paper', colours: Object.values(PMF_PAPER_COLOURS), named: PMF_PAPER_COLOURS, source: 'PMF paper (Anwar et al., Addis Ababa), Figure 7', note: 'fixed source colours; other labels use Gallery colours' },
    // Wong's colours with black moved from first to last, so the first seven categories are hues. On
    // the dark page black shows as a grey near R's Okabe-Ito grey (#999999), one step off it because
    // an export would print every #999999 on the page (Blanks uses it) back as black
    { name: 'Okabe–Ito', colours: ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7', '#000000'], dark: { '#000000': '#9c9c9c' }, source: 'Okabe & Ito, Colour Universal Design; Wong 2011', safe: true },
    { name: 'Tol bright', colours: ['#4477aa', '#ee6677', '#228833', '#ccbb44', '#66ccee', '#aa3377', '#bbbbbb'], source: 'Tol 2021, bright', safe: true },
    { name: 'Tol vibrant', colours: ['#ee7733', '#0077bb', '#33bbee', '#ee3377', '#cc3311', '#009988', '#bbbbbb'], source: 'Tol 2021, vibrant', safe: true },
    // Tol's grey for missing data (#dddddd) is not a category colour
    { name: 'Tol muted', colours: ['#cc6677', '#332288', '#ddcc77', '#117733', '#88ccee', '#882255', '#44aa99', '#999933', '#aa4499'], source: 'Tol 2021, muted', safe: true },
    { name: 'Tableau 10', colours: schemeTableau10, source: 'Tableau 10 (2016), via d3-scale-chromatic' },
    { name: 'Observable 10', colours: schemeObservable10, source: 'Observable 10 (2024), via d3-scale-chromatic' },
    { name: 'Dark 2', colours: schemeDark2, source: 'ColorBrewer Dark2' },
    { name: 'Set 3', colours: schemeSet3, source: 'ColorBrewer Set3', note: 'pale: for filled areas, not lines or points' },
  ],
}

export const paletteNames = (kind: PaletteKind) => PALETTES[kind].map((p) => p.name)
/** the named palette of a kind, or its Gallery palette for a name it does not have */
export const getPalette = (kind: PaletteKind, name: string): Palette =>
  PALETTES[kind].find((p) => p.name === name) ?? PALETTES[kind].find((p) => p.name === 'Gallery')!

/** the line under a palette's name in the picker */
export function paletteNote(kind: PaletteKind, name: string): string | undefined {
  const p = PALETTES[kind].find((q) => q.name === name)
  if (!p) return undefined
  const parts = [
    p.safe && 'colour-vision safe',
    p.greyscale && 'prints in black and white',
    // a categorical set repeats after its last colour (Tol: use no more than it has)
    kind === 'categorical' && !p.named && `${p.colours.length} colours`,
    p.note,
  ]
  return parts.filter(Boolean).join('; ') || undefined
}

/** a palette's colours in the light theme, in the chosen direction (a set with named colours keeps its order) */
export function paletteColours(kind: PaletteKind, name: string, reverse = false): string[] {
  const p = getPalette(kind, name)
  return reverse && !p.named ? [...p.colours].reverse() : [...p.colours]
}

export const hasNamedColours = (name: string) => !!PALETTES.categorical.find((p) => p.name === name)?.named

/**
 * An ordinal scale over any category domain. In a set with named colours, a matching label keeps
 * its colour and an unmatched one takes Gallery colours, so an extra measured species is not
 * labelled as a reconstruction component. `role: 'family'` uses the set's family colours if it has them.
 */
export function categoricalScale(name: string, domain: readonly string[], reverse = false, role: 'series' | 'family' = 'series') {
  const set = getPalette('categorical', name)
  const familyStops = role === 'family' ? set.familyColours : undefined
  const stops = familyStops ? (reverse ? [...familyStops].reverse() : familyStops) : paletteColours('categorical', set.named ? 'Gallery' : set.name, reverse && !set.named)
  const unique = [...new Set(domain)]
  return scaleOrdinal<string, string>().domain(unique).range(unique.map((label, i) => set.named?.[label] ?? stops[i % stops.length]))
}
