import { useRef, type ReactNode } from 'react'
import { useDimensions } from '@/hooks/useGalleryData'
import { facetGrid } from '@/lib/facets'

/**
 * Lays out small multiples: as many columns as fit (lib/facets), each cell
 * handed its width. `render(item, cellWidth)` draws one facet, usually a
 * SpectrumPlot or SweepPlot with `width={cellWidth}`.
 */
export function FacetGrid<T>({ items, render, minCell, maxCols, gap }: {
  items: T[]
  render: (item: T, cellWidth: number, index: number) => ReactNode
  minCell?: number
  maxCols?: number
  gap?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const g = facetGrid(width, items.length, { minCell, maxCols, gap })
  return (
    <div ref={ref} style={{ display: 'grid', gridTemplateColumns: `repeat(${g.cols}, ${g.cell}px)`, gap: g.gap, alignItems: 'start' }}>
      {items.map((it, i) => <div key={i}>{render(it, g.cell, i)}</div>)}
    </div>
  )
}
