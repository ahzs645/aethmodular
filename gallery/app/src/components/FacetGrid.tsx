import { useRef, useState, type ReactNode } from 'react'
import { PANEL_TITLE_H, PanelCell } from '@/components/Panels'
import { SpectrumModal } from '@/components/FilterSpectrumPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { useFitHeight } from '@/lib/chartFit'
import { facetGrid, panelLetter } from '@/lib/facets'

/**
 * Lays out small multiples: as many columns as fit (lib/facets), each cell
 * handed its width. `render(item, cellWidth, index, cell)` draws one facet, usually a
 * SpectrumPlot or SweepPlot with `width={cellWidth}`.
 *
 * Optional chrome, the same as a composite's panels (components/Panels PanelCell): `title` and
 * `note` per item make the title row (`.facet-title`, kept in exports), `letters` puts a, b, …
 * before it, and `expandable` adds ⤢ to draw a facet larger in a dialog (`cell.expanded`, a wider
 * `cellWidth`). In a Dashboard tile `cell.height` is each row's share of the fitted height
 * (null elsewhere: draw at the facet's own height).
 */
export interface FacetCell { height: number | null; expanded: boolean }

export function FacetGrid<T>({ items, render, minCell, maxCols, maxCell, gap, title, note, label, letters, expandable, center, minHeight = 120 }: {
  items: T[]
  render: (item: T, cellWidth: number, index: number, cell: FacetCell) => ReactNode
  minCell?: number
  maxCols?: number
  /** the widest a cell grows (a fixed-shape panel, a square crossplot) */
  maxCell?: number
  gap?: number
  title?: (item: T, index: number) => ReactNode
  /** a readout at the right of the title row */
  note?: (item: T, index: number) => ReactNode
  /** plain-text name per facet, for the dialog and screen readers */
  label?: (item: T, index: number) => string
  letters?: boolean
  expandable?: boolean
  /** centre the columns when the cells do not fill the width */
  center?: boolean
  minHeight?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const fit = useFitHeight()
  const [open, setOpen] = useState<number | null>(null)
  const g = facetGrid(width, items.length, { minCell, maxCols, gap, maxCell })
  const chrome = !!(title || note || letters || expandable)
  const rows = Math.max(1, Math.ceil(items.length / g.cols))
  const height = fit === null ? null : Math.max(minHeight, (fit - g.gap * (rows - 1)) / rows - (chrome ? PANEL_TITLE_H : 0))
  const name = (it: T, i: number) => label?.(it, i) ?? (typeof title?.(it, i) === 'string' ? title!(it, i) as string : undefined)
  const big = typeof window === 'undefined' ? 900 : Math.max(320, Math.min(maxCell ? maxCell * 1.6 : 1000, window.innerWidth - 160))
  return (
    <div ref={ref} style={{ display: 'grid', gridTemplateColumns: `repeat(${g.cols}, ${g.cell}px)`, gap: g.gap, alignItems: 'start', justifyContent: center ? 'center' : undefined }}>
      {items.map((it, i) => chrome
        ? <PanelCell key={i} title={title?.(it, i)} note={note?.(it, i)} letter={letters ? panelLetter(i) : undefined} label={name(it, i)}
            onExpand={expandable ? () => setOpen(i) : undefined}>{render(it, g.cell, i, { height, expanded: false })}</PanelCell>
        : <div key={i}>{render(it, g.cell, i, { height, expanded: false })}</div>)}
      {open !== null && items[open] !== undefined && <SpectrumModal id={name(items[open], open) ?? ''} title={letters ? `Panel ${panelLetter(open)}` : 'Panel'} onClose={() => setOpen(null)}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>{render(items[open], big, open, { height: null, expanded: true })}</div>
      </SpectrumModal>}
    </div>
  )
}
