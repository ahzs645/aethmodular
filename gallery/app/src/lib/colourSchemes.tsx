import * as d3 from 'd3'
import { DIVERGING_SCHEMES, RAMP_SEQUENTIAL, isDark, registerPrintColours, registerPrintResolver } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { useTheme } from '@/lib/colorTheme'
import { ColourSchemePicker, type SchemeChoice } from '@/components/ColourSchemePicker'

/**
 * The colour schemes a chart can offer, and one picker for them (components/ColourSchemePicker):
 * diverging schemes for signed values (a correlogram's r), sequential ones for counts and
 * densities (a hexbin). A chart asks for its stops and hands the scheme to its colour ramp, which
 * opens the list of palettes when clicked (components/ColorLegend):
 *
 *   const colours = useColourScheme('hexbin', 'sequential')
 *   const color = d3.scaleQuantize<string>().domain([0, max]).range(colours.stops)
 *   <ColorLegend scale={color} scheme={colours.scheme} …>
 *
 * A chart without a ramp puts `colours.picker` (a button with the same list) in its controls.
 * The choice is kept in the URL as `<chart>.palette` / `<chart>.reverse`.
 */
export type SchemeKind = 'diverging' | 'sequential'
const STOPS = 8
const sample = (f: (t: number) => string) => d3.quantize(f, STOPS)
export const SEQUENTIAL_SCHEMES: Record<string, readonly string[]> = {
  Gallery: RAMP_SEQUENTIAL,
  Viridis: sample(d3.interpolateViridis),
  Cividis: sample(d3.interpolateCividis),
  Plasma: sample(d3.interpolatePlasma),
  Magma: sample((t) => d3.interpolateMagma(1 - t * 0.9)).reverse(),
  Reds: sample((t) => d3.interpolateReds(0.12 + 0.88 * t)),
  Greens: sample((t) => d3.interpolateGreens(0.08 + 0.92 * t)),
  Oranges: sample((t) => d3.interpolateOranges(0.08 + 0.92 * t)),
  Purples: sample((t) => d3.interpolatePurples(0.08 + 0.92 * t)),
  Greys: sample((t) => d3.interpolateGreys(0.05 + 0.9 * t)),
}
export const SCHEMES: Record<SchemeKind, Record<string, readonly string[]>> = { diverging: DIVERGING_SCHEMES, sequential: SEQUENTIAL_SCHEMES }
/** a line under the picker: which schemes hold up for colour-vision deficiencies or in print */
export const SCHEME_NOTES: Record<string, string> = {
  'Purple–orange': 'colour-vision safe', 'Brown–teal': 'colour-vision safe', Cividis: 'colour-vision safe', Viridis: 'colour-vision safe',
  Greys: 'prints in black and white',
}

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
const DARK_SCHEMES: Record<SchemeKind, Record<string, string[]>> = {
  diverging: Object.fromEntries(Object.entries(DIVERGING_SCHEMES).map(([k, v]) => [k, darkStops('diverging', v)])),
  sequential: Object.fromEntries(Object.entries(SEQUENTIAL_SCHEMES).map(([k, v]) => [k, darkStops('sequential', v)])),
}
// an exported heat map is drawn light (lib/export): each dark stop maps back to its light one, and a
// colour a chart interpolated between stops (d3 scaleLinear, interpolateRgbBasis) to the light colour
// at the same place on the light ramp
for (const kind of ['diverging', 'sequential'] as const) for (const [k, v] of Object.entries(DARK_SCHEMES[kind])) registerPrintColours(v, SCHEMES[kind][k])
const RAMP_SAMPLES = (() => {
  const out: { rgb: [number, number, number]; light: string }[] = []
  const add = (dark: string, light: string) => {
    if (dark === light) return
    const c = d3.rgb(dark)
    out.push({ rgb: [c.r, c.g, c.b], light: d3.rgb(light).formatHex() })
  }
  for (const kind of ['diverging', 'sequential'] as const) for (const [k, dark] of Object.entries(DARK_SCHEMES[kind])) {
    const light = SCHEMES[kind][k]
    const basisD = d3.interpolateRgbBasis(dark), basisL = d3.interpolateRgbBasis([...light])
    for (let i = 0; i <= 400; i++) add(basisD(i / 400), basisL(i / 400))
    for (let s = 0; s < dark.length - 1; s++) for (let i = 0; i <= 60; i++) add(d3.interpolateRgb(dark[s], dark[s + 1])(i / 60), d3.interpolateRgb(light[s], light[s + 1])(i / 60))
  }
  return out
})()
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

export function schemeStops(kind: SchemeKind, name: string, reverse = false): string[] {
  const set = isDark() ? DARK_SCHEMES[kind] : SCHEMES[kind]
  const stops = [...(set[name] ?? Object.values(set)[0])]
  return reverse ? stops.reverse() : stops
}

export function useColourScheme(chartId: string, kind: SchemeKind, fallback = 'Gallery') {
  // ramps have their own dark steps, which CSS cannot map (a cell's colour is interpolated): redraw on a switch
  useTheme()
  const names = Object.keys(SCHEMES[kind])
  const [name, setName] = useSearchParam<string>(`${chartId}.palette`, fallback, names)
  const [rev, setRev] = useSearchParam<string>(`${chartId}.reverse`, '0', ['0', '1'])
  const reverse = rev === '1'
  const stops = schemeStops(kind, name, reverse)
  const scheme: SchemeChoice = { kind, value: name, reverse, onChange: (n, r) => { setName(n); setRev(r ? '1' : '0') } }
  const picker = <ColourSchemePicker {...scheme} />
  return { name, reverse, stops, scheme, picker }
}
