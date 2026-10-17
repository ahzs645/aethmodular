import * as d3 from 'd3'
import { DIVERGING_SCHEMES, RAMP_SEQUENTIAL } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { ColourSchemePicker } from '@/components/ColourSchemePicker'

/**
 * The colour schemes a chart can offer, and one picker for them (components/ColourSchemePicker):
 * diverging schemes for signed values (a correlogram's r), sequential ones for counts and
 * densities (a hexbin). A chart asks for its stops and gets the picker to put in its controls:
 *
 *   const colours = useColourScheme('hexbin', 'sequential')
 *   d3.scaleQuantize<string>().domain([0, max]).range(colours.stops) · controls: colours.picker
 *
 * The choice is kept in the URL as `<chart>.palette` / `<chart>.reverse`.
 */
export type SchemeKind = 'diverging' | 'sequential'
const STOPS = 8
const sample = (f: (t: number) => string) => d3.quantize(f, STOPS)
export const SEQUENTIAL_SCHEMES: Record<string, readonly string[]> = {
  Gallery: RAMP_SEQUENTIAL,
  Viridis: sample(d3.interpolateViridis),
  Cividis: sample(d3.interpolateCividis),
  Magma: sample((t) => d3.interpolateMagma(1 - t * 0.9)).reverse(),
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

export function schemeStops(kind: SchemeKind, name: string, reverse = false): string[] {
  const stops = [...(SCHEMES[kind][name] ?? Object.values(SCHEMES[kind])[0])]
  return reverse ? stops.reverse() : stops
}

export function useColourScheme(chartId: string, kind: SchemeKind, fallback = 'Gallery') {
  const names = Object.keys(SCHEMES[kind])
  const [name, setName] = useSearchParam<string>(`${chartId}.palette`, fallback, names)
  const [rev, setRev] = useSearchParam<string>(`${chartId}.reverse`, '0', ['0', '1'])
  const reverse = rev === '1'
  const stops = schemeStops(kind, name, reverse)
  const picker = <ColourSchemePicker kind={kind} value={name} reverse={reverse}
    onChange={(n, r) => { setName(n); setRev(r ? '1' : '0') }} />
  return { name, reverse, stops, picker }
}
