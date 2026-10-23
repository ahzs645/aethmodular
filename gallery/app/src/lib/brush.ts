import type { Selection } from 'd3'
import { INK } from '@/lib/theme'

/**
 * One brush look for every chart (docs/gallery-chart-standards.md §1): the
 * accent colour, a light fill and visible handles, instead of d3's grey default.
 * Call it right after `selection.call(brush)`; the svg's `animated` transition
 * is switched off on the brush so the rectangle does not trail the pointer.
 */
export const BRUSH_STYLE = { get fill() { return INK.accent }, fillOpacity: 0.14, handleOpacity: 0.6 } as const

export function styleBrush(sel: Selection<SVGGElement, unknown, null, undefined>) {
  sel.selectAll('rect').style('transition', 'none')
  sel.selectAll('.selection').attr('fill', BRUSH_STYLE.fill).attr('fill-opacity', BRUSH_STYLE.fillOpacity).attr('stroke', BRUSH_STYLE.fill)
  sel.selectAll('.handle').attr('fill', BRUSH_STYLE.fill).attr('fill-opacity', BRUSH_STYLE.handleOpacity)
}
