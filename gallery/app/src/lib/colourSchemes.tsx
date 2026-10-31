import * as d3 from 'd3'
import { isDark, paintPair, registerPrintColours, registerPrintResolver } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { useTheme } from '@/lib/colorTheme'
import { ColourSchemePicker, type SchemeChoice } from '@/components/ColourSchemePicker'
import { PALETTES, allPalettes, getPalette, hasNamedColours, paletteColours, type PaletteKind } from './palettes'
import { useCustomPalettes } from './customPalettes'

/**
 * One picker for a chart's colours (components/ColourSchemePicker), offering the palettes of one
 * kind from lib/palettes: diverging for signed values (a correlogram's r), sequential for counts
 * and densities (a hexbin), categorical for named groups. A chart asks for its stops and hands the
 * scheme to its colour ramp, which opens the list of palettes when clicked (components/ColorLegend):
 *
 *   const colours = useColourScheme('hexbin', 'sequential')
 *   const color = d3.scaleQuantize<string>().domain([0, max]).range(colours.stops)
 *   <ColorLegend scale={color} scheme={colours.scheme} …>
 *
 * A chart without a ramp puts `colours.picker` (a button with the same list) in its controls; a
 * categorical chart builds its scale with categoricalScale(colours.name, fields, colours.reverse).
 * The choice is kept in the URL as `<chart>.palette` / `<chart>.reverse`. New palettes go in
 * lib/palettes, never here; the reader's own (lib/customPalettes) are listed after them.
 */
export type SchemeKind = PaletteKind
const RAMPS = ['diverging', 'sequential'] as const
/**
 * The palettes a chart's picker lists: every general one, and a fixed set (one with named colours:
 * SPARTAN composition, FTIR functional groups, PMF paper) only on a chart that opens on it. On any
 * other chart's categories a fixed set matches no label and would only repeat Gallery. The
 * reader's own palettes come last.
 */
export const paletteChoices = (kind: SchemeKind, chartDefault: string) =>
  allPalettes(kind).filter((p) => !p.named || p.name === chartDefault).map((p) => p.name)
// a categorical colour with a dark twin (Okabe–Ito's black) shows the twin on the dark page, as
// every themed mark does; charts keep drawing the light colour, so exports print it
for (const p of PALETTES.categorical) for (const [l, d] of Object.entries(p.dark ?? {})) paintPair(l, d)
const lightStops = (kind: SchemeKind) => Object.fromEntries(PALETTES[kind].map((p) => [p.name, p.colours]))

/**
 * The dark theme's step of each scheme, not the light one flipped wholesale. A sequential ramp
 * keeps "more = further from the page": light-first ramps run from a mid-dark step that still
 * shows on the panel up to their palest (Viridis and the other dark-first ramps already do).
 * A diverging ramp keeps its two poles, lifted where they sit too dark, and its neutral middle
 * becomes a grey just above the panel, so "near zero" recedes as it does on white.
 */
const luminance = (c: string) => d3.lab(c).l
const DARK_MID = '#363c44'
function darkStops(kind: SchemeKind, stops: readonly string[]): string[] {
  if (kind === 'sequential') {
    if (luminance(stops[0]) <= luminance(stops[stops.length - 1])) return [...stops]
    const reversed = [...stops].reverse()
    const ramp = d3.interpolateRgbBasis(reversed)
    // start a quarter of the way in: the darkest step of a light-first ramp vanishes on the panel
    return d3.quantize((t) => d3.lab(ramp(0.12 + 0.88 * t)).formatHex(), stops.length)
  }
  const n = stops.length, mid = (n - 1) / 2
  const lift = (c: string) => { const l = d3.lab(c); if (l.l < 58) l.l = 58; return l.formatHex() }
  const lo = lift(stops[0]), hi = lift(stops[n - 1])
  return stops.map((_, i) => i === mid ? DARK_MID : d3.interpolateLab(DARK_MID, i < mid ? lo : hi)(Math.abs(i - mid) / mid))
}
// categorical sets keep their colours in both themes
const DARK_RAMPS = Object.fromEntries(RAMPS.map((kind) =>
  [kind, Object.fromEntries(PALETTES[kind].map((p) => [p.name, darkStops(kind, p.colours)]))])) as Record<typeof RAMPS[number], Record<string, string[]>>
// an exported heat map is drawn light (lib/export): each dark stop maps back to its light one, and a
// colour a chart interpolated between stops (d3 scaleLinear, interpolateRgbBasis) to the light colour
// at the same place on the light ramp
for (const kind of RAMPS) for (const [k, v] of Object.entries(DARK_RAMPS[kind])) registerPrintColours(v, lightStops(kind)[k])
const RAMP_SAMPLES: { rgb: [number, number, number]; light: string }[] = []
function addRampSamples(dark: readonly string[], lit: readonly string[]) {
  const add = (d: string, light: string) => {
    if (d === light) return
    const c = d3.rgb(d)
    RAMP_SAMPLES.push({ rgb: [c.r, c.g, c.b], light: d3.rgb(light).formatHex() })
  }
  const basisD = d3.interpolateRgbBasis([...dark]), basisL = d3.interpolateRgbBasis([...lit])
  for (let i = 0; i <= 400; i++) add(basisD(i / 400), basisL(i / 400))
  for (let s = 0; s < dark.length - 1; s++) for (let i = 0; i <= 60; i++) add(d3.interpolateRgb(dark[s], dark[s + 1])(i / 60), d3.interpolateRgb(lit[s], lit[s + 1])(i / 60))
}
for (const kind of RAMPS) for (const [k, dark] of Object.entries(DARK_RAMPS[kind])) addRampSamples(dark, lightStops(kind)[k])
const resolved = new Map<string, string | null>()
registerPrintResolver((hex) => {
  if (resolved.has(hex)) return resolved.get(hex)!
  const c = d3.rgb(hex)
  let best: string | null = null, bestD = 3 * 3.5 ** 2
  for (const p of RAMP_SAMPLES) {
    const d = (p.rgb[0] - c.r) ** 2 + (p.rgb[1] - c.g) ** 2 + (p.rgb[2] - c.b) ** 2
    if (d < bestD) { bestD = d; best = p.light }
  }
  resolved.set(hex, best)
  return best
})

/**
 * A reader's palette (lib/customPalettes) gets its dark steps the first time the dark page draws it,
 * by the same rule as the built-in ones, and its export colours with them. Keyed by its colours, so
 * an edited palette is worked out again.
 */
const customDark = new Map<string, string[]>()
function darkRamp(kind: typeof RAMPS[number], name: string): readonly string[] {
  const built = DARK_RAMPS[kind][name]
  if (built) return built
  const lit = getPalette(kind, name).colours
  const key = `${kind}|${lit.join(',')}`
  let dark = customDark.get(key)
  if (!dark) {
    dark = darkStops(kind, lit)
    customDark.set(key, dark)
    registerPrintColours(dark, [...lit])
    addRampSamples(dark, lit)
    resolved.clear()
  }
  return dark
}

/** a palette's colours in the current theme, in the chosen direction */
export function schemeStops(kind: SchemeKind, name: string, reverse = false): string[] {
  if (kind === 'categorical' || !isDark()) return paletteColours(kind, name, reverse)
  const stops = [...darkRamp(kind, name)]
  return reverse ? stops.reverse() : stops
}

export function useColourScheme(chartId: string, kind: SchemeKind, fallback = 'Gallery') {
  // ramps have their own dark steps, which CSS cannot map (a cell's colour is interpolated): redraw on a switch
  useTheme()
  // and on the reader saving, editing or deleting a palette of their own
  useCustomPalettes()
  const names = paletteChoices(kind, fallback)
  const [name, setName] = useSearchParam<string>(`${chartId}.palette`, fallback, names)
  const [rev, setRev] = useSearchParam<string>(`${chartId}.reverse`, '0', ['0', '1'])
  const reverse = rev === '1' && !(kind === 'categorical' && hasNamedColours(name))
  const stops = schemeStops(kind, name, reverse)
  const scheme: SchemeChoice = { kind, names, value: name, reverse, onChange: (n, r) => { setName(n); setRev(kind === 'categorical' && hasNamedColours(n) ? '0' : r ? '1' : '0') } }
  const picker = <ColourSchemePicker {...scheme} />
  return { name, reverse, stops, scheme, picker }
}
