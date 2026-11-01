import type { ReactNode } from 'react'
import { Legend, type LegendShape } from '@/components/Legend'

/**
 * A static key: what each colour, mark and line means, for charts whose series
 * are fixed. It is `Legend` without `onToggle`/`onHover` (so it gets the tile
 * modes, export colour tags and the gallery's swatches), under the names the
 * research pages already use. New code can call `Legend` directly.
 */
export type KeyKind = 'dot' | 'square' | 'hollow' | 'diamond' | 'line' | 'dash' | 'bar' | 'faded'
export interface KeyItem { label: string; color: string; kind?: KeyKind }

/** a Key's diamond is drawn hollow (the marks it names are outlined); the rest are Legend's shapes or aliases */
const SHAPE: Record<KeyKind, LegendShape> = {
  dot: 'dot', square: 'square', hollow: 'hollow', diamond: 'hollow-diamond', line: 'line', dash: 'dash', bar: 'bar', faded: 'faded',
}

export function Key({ items, children, relabelFits = false }: {
  items: KeyItem[]
  /** extra legend content, e.g. a ColorLegend ramp */
  children?: ReactNode
  /** labels say "Deming" for a fit the browser refits: show the active estimator's name instead */
  relabelFits?: boolean
}) {
  return (
    <Legend items={items.map((it) => ({ label: it.label, color: it.color, shape: SHAPE[it.kind ?? 'dot'] }))}
      relabelFits={relabelFits} group={false}>{children}</Legend>
  )
}
