import { useRef, useState, type ReactNode } from 'react'
import { ExpandedPanel, PANEL_TITLE_H, PanelCell, type PanelGroupProps } from '@/components/Panels'
import { PanelGroupControls, PanelGroupProvider, PanelSlot, usePanelGroupValue } from '@/components/PanelGroup'
import { useDimensions } from '@/hooks/useGalleryData'
import { useFitHeight } from '@/lib/chartFit'
import { facetGrid, panelLetter, pickedPanels } from '@/lib/facets'

/**
 * Lays out small multiples: as many columns as fit (lib/facets), each cell
 * handed its width. `render(item, cellWidth, index, cell)` draws one facet, usually a
 * SpectrumPlot or SweepPlot with `width={cellWidth}`.
 *
 * Optional chrome, the same as a composite's panels (components/Panels PanelCell): `title` and
 * `note` per item make the title row (`.facet-title`, kept in exports), `letters` puts a, b, …
 * before it, and ⤢ (on unless `expandable={false}`) draws a facet larger in a dialog (`cell.expanded`, a
 * wider `cellWidth`), on the title row or in the facet's corner. In a Dashboard tile `cell.height` is each
 * row's share of the fitted height (null elsewhere: draw at the facet's own height).
 *
 * The facets are one panel group (components/PanelGroup), as Panels': `share` gives the shared charts inside one
 * axis range under the gallery's Panel axes setting (`site` per item for Per site), crossplots draw one fit grouping
 * box for the grid, `legend` is drawn once under it, and `show`/`order` pick and order facets by `itemKey`.
 */
export interface FacetCell { height: number | null; expanded: boolean }

export function FacetGrid<T>({ items, render, minCell, maxCols, maxCell, gap, title, note, label, letters = false, expandable = true, center, minHeight = 120,
  itemKey, site, share, grouping, legend, controls, axesControl = true, dims, show, order, pick, mode }: {
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
  /** ⤢ on every facet (default true) */
  expandable?: boolean
  /** centre the columns when the cells do not fill the width */
  center?: boolean
  minHeight?: number
  /** a facet's key, for `show`/`order` and Own ranges (default: the item when it is a string, else its index) */
  itemKey?: (item: T, index: number) => string
  /** a facet's site, for Panel axes: Per site */
  site?: (item: T, index: number) => string | undefined
} & PanelGroupProps) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const fit = useFitHeight()
  const [open, setOpen] = useState<string | null>(null)
  const { value: group, joined } = usePanelGroupValue({ share, grouping, mode })
  const keyOf = (it: T, i: number) => itemKey?.(it, i) ?? (typeof it === 'string' ? it : String(i))
  const keys = items.map(keyOf)
  // the facets drawn, in order, each with its index in `items` (what `render` and `title` are handed)
  const drawn = pickedPanels(keys, show, order, pick).map((i) => ({ it: items[i], i, key: keys[i] }))
  const g = facetGrid(width, drawn.length, { minCell, maxCols, gap, maxCell })
  const head = !!(title || note || letters)
  const rows = Math.max(1, Math.ceil(drawn.length / g.cols))
  const height = fit === null ? null : Math.max(minHeight, (fit - g.gap * (rows - 1)) / rows - (head ? PANEL_TITLE_H : 0))
  const name = (it: T, i: number) => label?.(it, i) ?? (typeof title?.(it, i) === 'string' ? title!(it, i) as string : undefined)
  const big = typeof window === 'undefined' ? 900 : Math.max(320, Math.min(maxCell ? maxCell * 1.6 : 1000, window.innerWidth - 160))
  const at = open === null ? -1 : drawn.findIndex((d) => d.key === open)
  const shown = at >= 0 ? drawn[at] : null
  const sharing = !!share && !mode && drawn.length > 1
  return (
    <PanelGroupProvider value={group}>
      {!joined && <PanelGroupControls value={group} axesControl={sharing && axesControl} dims={dims ?? (site ? ['site'] : [])} extra={controls} />}
      <div ref={ref} style={{ display: 'grid', gridTemplateColumns: `repeat(${g.cols}, ${g.cell}px)`, gap: g.gap, alignItems: 'start', justifyContent: center ? 'center' : undefined }}>
        {drawn.map(({ it, i, key }, pos) => (
          <PanelCell key={key} title={title?.(it, i)} note={note?.(it, i)} letter={letters ? panelLetter(pos) : undefined} label={name(it, i)}
            onExpand={expandable ? () => setOpen(key) : undefined}>
            <PanelSlot panelKey={key} site={site?.(it, i)}>{render(it, g.cell, i, { height, expanded: false })}</PanelSlot>
          </PanelCell>
        ))}
        {shown && <ExpandedPanel name={name(shown.it, shown.i)} letter={letters ? panelLetter(at) : undefined} onClose={() => setOpen(null)}>
          <PanelSlot panelKey={shown.key} site={site?.(shown.it, shown.i)}>{render(shown.it, big, shown.i, { height: null, expanded: true })}</PanelSlot>
        </ExpandedPanel>}
      </div>
      {legend}
    </PanelGroupProvider>
  )
}
