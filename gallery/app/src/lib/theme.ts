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
 * The chart ink. It always holds the light values: marks are drawn in light colours and lib/themeCss
 * maps each one to its dark twin in CSS (SVG attributes and inline HTML colours alike), so switching
 * the theme recolours the page without redrawing a chart, and an export is light as drawn.
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
/**
 * A colour that has its own step in the dark theme. It returns `light`, the colour the mark is drawn
 * in, and registers the pair so lib/themeCss shows `dark` on the dark page.
 */
export const themed = (light: string, dark: string) => paintPair(light, dark)
/** an INK colour as it shows in the current theme, for what CSS cannot recolour (a canvas raster) */
export const inkNow = (key: keyof Ink) => (current === 'dark' ? INK_DARK : INK_LIGHT)[key]

/**
 * Fill opacity for a filled scatter dot, by how crowded the chart is (docs/gallery-chart-standards.md §5).
 * On the dark panel a pale dot mixes toward the background and loses its hue, so site and season colours
 * blur together; the dark steps are lifted to keep them apart. `dimmed` (context and dots pushed back
 * by a focus) recedes the same in both themes. Dots are drawn at the light step and tagged data-dot;
 * lib/themeCss applies the dark step, so an export keeps the light one.
 */
export const DOT_OPACITY = {
  /** a few dozen dots, or dots that carry the point of the chart */
  sparse: { light: 0.85, dark: 0.92 },
  /** the default: a few hundred dots that overlap a little */
  standard: { light: 0.7, dark: 0.82 },
  /** crowded charts, where overlap should build up into darker areas */
  dense: { light: 0.45, dark: 0.6 },
  /** context behind other marks, and everything but the focused dot */
  dimmed: { light: 0.25, dark: 0.25 },
} as const
export type DotLevel = keyof typeof DOT_OPACITY
/** the opacity a dot is drawn at (the light step; the dark page lifts it in CSS) */
export const dotOpacity = (level: DotLevel) => DOT_OPACITY[level].light

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
 * Record the page theme (lib/colorTheme calls it). The colour constants above stay light: their dark
 * twins are paint pairs (below) that lib/themeCss applies in CSS. Only what CSS cannot reach (colour
 * ramps, inkOn labels, canvas rasters) reads the theme, and those charts re-render via useTheme().
 */
export function setInkTheme(theme: ThemeName) {
  current = theme
}
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

/**
 * Paint pairs: every light colour that has a dark twin. lib/themeCss turns each into CSS (an SVG mark
 * whose fill or stroke is the light hex, or an inline HTML colour of it, shows the twin on the dark
 * page or in a chart set to dark), and lib/export reads them to print a dark-page colour back to light.
 */
const PAIRS = new Map<string, string>()
const pairListeners = new Set<(light: string, dark: string) => void>()
export function paintPair(light: string, dark: string): string {
  const l = norm(light), d = norm(dark)
  if (l !== d && !PAIRS.has(l)) {
    PAIRS.set(l, d)
    registerPrintColours([d], [l])
    pairListeners.forEach((f) => f(l, d))
  }
  return light
}
export const paintPairs = (): ReadonlyMap<string, string> => PAIRS
export const onPaintPair = (f: (light: string, dark: string) => void) => { pairListeners.add(f); return () => { pairListeners.delete(f) } }
/** the dark twin of a light colour (null: it looks the same in both themes) */
export const darkTwin = (hex: string) => PAIRS.get(norm(hex)) ?? null
;(Object.keys(INK_LIGHT) as (keyof Ink)[]).forEach((k) => paintPair(INK_LIGHT[k], INK_DARK[k]))
Object.entries(BASELINE_DARK).forEach(([k, dark]) => paintPair(BASELINE_COLOR[k], dark))
paintPair(OCT1_HATCH.fill, '#2a2720')
paintPair(OCT1_HATCH.stroke, '#6b5d3d')

/**
 * Text colour for a label drawn on a coloured cell: white on dark fills, the
 * normal ink on light ones, from the fill's relative luminance (WCAG), so every
 * heat map stays readable whatever its ramp (no per-chart value thresholds).
 */
export function inkOn(fill: string): string {
  // a light ink cell shows its dark twin on the dark page: pick the label for what is seen
  return inkOnFill(current === 'dark' ? darkTwin(fill) ?? fill : fill)
}
/**
 * Dark label ink on a light cell: INK.text one step off, so lib/themeCss (which maps INK.text to the
 * light dark-theme text) leaves it dark on a light cell of a dark-page heat map.
 */
export const CELL_INK = '#1f2934'
/** inkOn for a fill as given, whatever the page theme (lib/export uses it for the light figure) */
export function inkOnFill(fill: string): string {
  const hex = fill.trim().replace('#', '')
  const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex
  const m = /^[0-9a-f]{6}$/i.test(full) ? full : null
  const rgb = m ? [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16))
    : (fill.match(/\d+(\.\d+)?/g) ?? []).slice(0, 3).map(Number)
  if (rgb.length < 3) return CELL_INK
  const luminance = (c: number[]) => {
    const lin = c.map((v) => { const x = v / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4 })
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
  }
  // whichever of white and the page ink contrasts more with the fill
  const L = luminance(rgb), ink = luminance([0x1f, 0x29, 0x33])
  return (1.05 / (L + 0.05)) > ((L + 0.05) / (ink + 0.05)) ? '#fff' : CELL_INK
}

/** Chart colour palettes (sequential, diverging and categorical) live in lib/palettes. */

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
