import { chemText } from './format.ts'
/**
 * Visual constants. Site colours are NOT invented here — they are exported
 * from research/ftir_hips_chem/scripts/config.py (SITES[site]['color']) and
 * arrive in meta.json, so the web gallery and the matplotlib figures agree.
 * Season colours likewise come from config.ETHIOPIA_SEASONS.
 */
export const MARGIN = { top: 28, right: 24, bottom: 52, left: 68 }

/** The light page: the default, and what every exported figure is drawn in. */
export const INK_LIGHT = {
  axis: '#5b6470',
  grid: '#e8ebef',
  text: '#1f2933',
  muted: '#7b8794',
  fit: '#111827',
  deming: '#c026d3',
  identity: '#9aa5b1',
  panel: '#ffffff',
  page: '#f6f7f9',
  border: '#dde2e8',
  /** the app accent (nav, selection, positive r) */
  accent: '#2171b5',
  positive: '#2171b5',
  negative: '#b2182b',
  /** "passes the test" green (held-out floor, built badge) */
  good: '#0f766e',
  /** a category that carries no signal: "no HIPS", "no carbon EC" */
  neutral: '#c7ccd1',
  /** an empty cell */
  empty: '#f3f5f8',
  /** alternate table / strip rows */
  zebra: '#f8f9fb',
  /** a shaded band behind marks (spectral regions left out, IQR bands) */
  shade: '#eef0f3',
  /** the warm shade: PTFE and other regions a spectrum keeps but analog selection ignores */
  shadeWarm: '#f1ece0',
  /** a picked row or card */
  tint: '#eef4fb',
  /** maps: sea and land */
  sea: '#f2f6fa',
  land: '#e4e9ef',
  /** a caution (orange text, outlines) */
  warn: '#b45309',
}
export type Ink = typeof INK_LIGHT

/**
 * The dark page. Its own steps, not an inversion: surfaces are blue-grey near #12161c, text and
 * axes are lifted to keep the light theme's contrast, and the few series colours that sit too
 * dark on the panel (accent, negative, good, the Deming magenta) are re-stepped lighter in the
 * same hue. Site and season colours are NOT here: they come from config.py and stay the same in
 * both themes, so a site keeps its colour in the decks and in the gallery.
 */
export const INK_DARK: Ink = {
  axis: '#a3adb8',
  grid: '#2a313a',
  text: '#e3e7ec',
  muted: '#8e99a6',
  fit: '#f1f4f8',
  deming: '#e070ec',
  identity: '#6b7684',
  panel: '#181d24',
  page: '#0f1318',
  border: '#2f3742',
  accent: '#5aa2e0',
  positive: '#5aa2e0',
  negative: '#ef6f74',
  good: '#2fb3a2',
  neutral: '#4b545f',
  empty: '#20262e',
  zebra: '#1d232b',
  shade: '#242b34',
  shadeWarm: '#2e2b23',
  tint: '#1c2f45',
  sea: '#141a21',
  land: '#2b323c',
  warn: '#f0a548',
}

/**
 * The chart ink, read at render time. lib/colorTheme swaps its values when the theme changes and
 * remounts the app, so a chart that reads INK.text while it renders draws in the right theme.
 * Never copy a value out of INK into a module-level constant: it would keep the first theme.
 */
export const INK: Ink = { ...INK_LIGHT }
export type ThemeName = 'light' | 'dark'
let current: ThemeName = 'light'
/** the stored choice (components/ThemeToggle): 'light' (the default), 'dark' or 'auto' (follow the system) */
export type ThemePref = 'light' | 'dark' | 'auto'
export const THEME_KEY = 'aeth-gallery:theme'
export function readThemePref(): ThemePref {
  try { const v = localStorage.getItem(THEME_KEY); return v === 'dark' || v === 'auto' ? v : 'light' } catch { return 'light' }
}
export function resolveTheme(pref: ThemePref): ThemeName {
  if (pref !== 'auto') return pref
  try { return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light' } catch { return 'light' }
}
/** read before any chart module copies a colour, so the first render is already in the stored theme */
function initialTheme(): ThemeName {
  const theme = resolveTheme(readThemePref())
  if (typeof document !== 'undefined') document.documentElement.dataset.theme = theme
  return theme
}
export const currentTheme = () => current
export const isDark = () => current === 'dark'
/** a colour that has its own step in the dark theme; in light it is `light` */
export const themed = (light: string, dark: string) => (current === 'dark' ? dark : light)

/**
 * One colour per baseline correction on every tab (docs/gallery-chart-standards.md §5):
 * spline (AIRSpec) blue, archived VIBES purple, completed full-range VIBES orange.
 * Keys match the method codes in the exports and lib/labels.ts PREPROCESSING.
 */
export const BASELINE_COLOR: Record<string, string> = {
  AIRSpec: '#2C6E9E',
  airspec: '#2C6E9E',
  /** archived truncated-grid VIBES (fit 4000–1425) */
  VIBES: '#7A4FA3',
  vibes: '#7A4FA3',
  /** archived full-range VIBES, preliminary rank cap 30 */
  'VIBES-full': '#9467bd',
  'vibes-full': '#9467bd',
  'VIBES-full-cut': '#b39ddb',
  'vibes-full-cut': '#b39ddb',
  /** completed full-range VIBES, rank 70: the current VIBES everywhere */
  'VIBES-full-r70': '#d95f0e',
  'vibes-full-r70': '#d95f0e',
  /** sensitivity: rank-70 VIBES used only over 4000–1425 */
  'VIBES-full-cut-r70': '#f2a65a',
  'vibes-full-cut-r70': '#f2a65a',
}
/** the baselines whose light colour sits too dark on the dark panel, re-stepped lighter in the same hue */
const BASELINE_DARK: Record<string, string> = { AIRSpec: '#4f95cc', airspec: '#4f95cc', VIBES: '#a07ad0', vibes: '#a07ad0' }

/**
 * Colour for a baseline as an export names it. Older exports call every VIBES run
 * "VIBES"; pass the run's background rank so a completed rank-70 run is drawn in
 * the current VIBES colour rather than the archived one.
 */
export function baselineColor(method: string, backgroundRank?: number | null): string {
  if ((method === 'VIBES' || method === 'vibes') && backgroundRank === 70) return BASELINE_COLOR['VIBES-full-r70']
  return BASELINE_COLOR[method] ?? INK.axis
}

/** IMPROVE filters drawn as context (analogs, held-out pool, reference traces). */
export const IMPROVE_GREY = '#8a8f98'
/** Addis pooled over every season (each season keeps its own colour). */
export const ADDIS_ALL = '#e0a800'

/** VIBES blank banks (Blanks tab and the Satoshi 2 Oct follow-up). */
export const BANK_COLOR: Record<string, string> = {
  AIRSpec: BASELINE_COLOR.AIRSpec,
  'VIBES-full-r70': BASELINE_COLOR['VIBES-full-r70'],
  'VIBES-lot251': '#a6761d',
  'VIBES-allblanks': '#7A4FA3',
  'VIBES-lot251-x1': '#1b9e77',
  'VIBES-allblanks-x1': '#e7298a',
}

/**
 * Spectral regions shaded behind every spectrum. `selection` marks what analog
 * selection ignores (PLS still sees it); the 1 Oct region is the agreed similarity
 * window (600–3600 cm⁻¹ without CO₂ and PTFE).
 */
export const SPECTRAL_REGION = {
  co2: { lo: 1800, hi: 2500, fill: INK_LIGHT.shade, label: 'CO₂ 1800–2500 cm⁻¹' },
  aboveCH: { lo: 3600, hi: 4000, fill: INK_LIGHT.shade, label: '>3600 cm⁻¹' },
  ptfe: { lo: 1050, hi: 1250, fill: INK_LIGHT.shadeWarm, label: 'PTFE C–F 1050–1250 cm⁻¹' },
  belowOct1: { lo: 500, hi: 600, fill: INK_LIGHT.shadeWarm, label: '<600 cm⁻¹' },
}
/** Hatching for regions dropped only by the 1 Oct similarity window. */
export const OCT1_HATCH = { fill: '#f6f1e7', stroke: '#d9c9a3' }

/** Trace styles shared by every spectrum plot: the pinned filter and the hovered one. */
export const TRACE = {
  pinned: { stroke: INK.text, strokeWidth: 2.2 },
  hovered: { stroke: INK.text, strokeOpacity: 0.55, strokeWidth: 1.6, strokeDasharray: '5 3' },
  bandOpacity: 0.22,
}

/**
 * Switch every colour constant above to a theme (lib/colorTheme calls it, then remounts the app).
 * The objects are changed in place so the modules that imported them see the new values.
 */
export function setInkTheme(theme: ThemeName) {
  current = theme
  const dark = theme === 'dark'
  Object.assign(INK, dark ? INK_DARK : INK_LIGHT)
  Object.assign(BASELINE_COLOR, BASELINE_LIGHT, dark ? BASELINE_DARK : {})
  BANK_COLOR.AIRSpec = BASELINE_COLOR.AIRSpec
  BANK_COLOR['VIBES-allblanks'] = BASELINE_COLOR.VIBES
  SPECTRAL_REGION.co2.fill = SPECTRAL_REGION.aboveCH.fill = INK.shade
  SPECTRAL_REGION.ptfe.fill = SPECTRAL_REGION.belowOct1.fill = INK.shadeWarm
  Object.assign(OCT1_HATCH, dark ? { fill: '#2a2720', stroke: '#6b5d3d' } : { fill: '#f6f1e7', stroke: '#d9c9a3' })
  TRACE.pinned.stroke = TRACE.hovered.stroke = INK.text
}
const BASELINE_LIGHT = { ...BASELINE_COLOR }
setInkTheme(initialTheme())

/**
 * Figures are exported in the light theme whatever the page shows (decks and papers are white):
 * lib/export maps each dark-theme colour on a figure back to its light step. A dark colour that
 * has no light twin is left as drawn.
 */
const PRINT = new Map<string, string>()
const norm = (c: string) => c.trim().toLowerCase()
export function registerPrintColours(dark: readonly string[], light: readonly string[]) {
  dark.forEach((d, i) => { if (light[i] && norm(d) !== norm(light[i])) PRINT.set(norm(d), light[i]) })
}
/** colours between two ramp steps (a heat map's interpolated cells) are found by lib/colourSchemes */
let resolver: (hex: string) => string | null = () => null
export const registerPrintResolver = (fn: (hex: string) => string | null) => { resolver = fn }
/** the light theme's colour for a hex the dark theme drew (null: not a dark-theme colour) */
export const printColour = (hex: string) => PRINT.get(norm(hex)) ?? resolver(norm(hex))
registerPrintColours(Object.values(INK_DARK), Object.values(INK_LIGHT))
registerPrintColours(Object.values(BASELINE_DARK), Object.keys(BASELINE_DARK).map((k) => BASELINE_LIGHT[k]))
registerPrintColours(['#2a2720', '#6b5d3d'], ['#f6f1e7', '#d9c9a3'])

/**
 * Text colour for a label drawn on a coloured cell: white on dark fills, the
 * normal ink on light ones, from the fill's relative luminance (WCAG), so every
 * heat map stays readable whatever its ramp (no per-chart value thresholds).
 */
export function inkOn(fill: string): string {
  const hex = fill.trim().replace('#', '')
  const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex
  const m = /^[0-9a-f]{6}$/i.test(full) ? full : null
  const rgb = m ? [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16))
    : (fill.match(/\d+(\.\d+)?/g) ?? []).slice(0, 3).map(Number)
  if (rgb.length < 3) return INK_LIGHT.text
  const luminance = (c: number[]) => {
    const lin = c.map((v) => { const x = v / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4 })
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
  }
  // whichever of white and the page ink contrasts more with the fill
  const L = luminance(rgb), ink = luminance([0x1f, 0x29, 0x33])
  return (1.05 / (L + 0.05)) > ((L + 0.05) / (ink + 0.05)) ? '#fff' : INK_LIGHT.text
}

/** Sequential + diverging ramps used by the heatmap and correlogram. */
export const RAMP_DIVERGING = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']

/**
 * Seven-stop diverging palettes (negative → neutral → positive), for charts that let the reader
 * choose (a correlogram's r; picked with lib/colourSchemes, which adds the sequential ones). 'Gallery' is RAMP_DIVERGING; the ColorBrewer ones are from
 * d3-scale-chromatic, sampled at the same stops; Purple–orange and Brown–teal stay apart for most
 * colour-vision deficiencies; Greys prints.
 */
export const DIVERGING_SCHEMES: Record<string, readonly string[]> = {
  Gallery: RAMP_DIVERGING,
  'Red–yellow–blue': ['#4575b4', '#91bfdb', '#e0f3f8', '#ffffbf', '#fee090', '#fc8d59', '#d73027'],
  'Purple–orange': ['#542788', '#998ec3', '#d8daeb', '#f7f7f7', '#fee0b6', '#f1a340', '#b35806'],
  'Brown–teal': ['#01665e', '#5ab4ac', '#c7eae5', '#f5f5f5', '#f6e8c3', '#d8b365', '#8c510a'],
  'Purple–green': ['#762a83', '#af8dc3', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#7fbf7b', '#1b7837'],
  Greys: ['#252525', '#636363', '#bdbdbd', '#f7f7f7', '#bdbdbd', '#636363', '#252525'],
}
export const RAMP_SEQUENTIAL = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']

export const FONT = {
  family: 'ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
  mono: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace',
}

/** "EC (FTIR)" + "ug/m3" -> "EC (FTIR) (µg/m³)". Units come from meta.json. */
/** A field's display label with its unit; chemistry subscripted (PM₂.₅, SO₄²⁻) wherever it is drawn. */
export function withUnit(field: string, units: Record<string, string> | undefined): string {
  const u = units?.[field]
  if (!u) return chemText(field)
  const pretty = u.replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹')
  return chemText(`${field} (${pretty})`)
}
