import { useRef, useState, type ReactNode } from 'react'
import { ExpandIcon } from '@/components/Icons'
import { SpectrumModal } from '@/components/FilterSpectrumPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { ChartFit, useFitHeight } from '@/lib/chartFit'
import { panelLetter, panelRow, type PanelSize } from '@/lib/facets'

/**
 * Composite figures: different charts in one ChartFrame (a crossplot and its threshold sweep, a
 * chart and its table). One layout rule (lib/facets panelRow): side by side while every panel's
 * `min` fits, stacked below that; each panel is handed its width. Every panel gets the same
 * chrome as a facet (PanelCell): a title row with an optional letter (a, b, … for a caption to
 * point at), a readout on the right, and ⤢ to open it larger in a dialog. Titles are
 * `.facet-title`, so the whole-panel export keeps them; the ⤢ button is skipped.
 *
 * In a Dashboard tile or the one-chart view (lib/chartFit), each panel is handed its share of
 * the fitted height (all of it side by side, a split when stacked) as `height`, and charts inside
 * that read useFitHeight() see it too. Elsewhere `height` is null: draw at your own height.
 *
 *   <Panels panels={[
 *     { key: 'scatter', title: 'Dry against humid', min: 300, basis: 420, max: 460, grow: 0, render: ({ width }) => <Crossplot … maxSide={width - 82} /> },
 *     { key: 'sweep', title: 'Threshold sweep', min: 260, render: ({ width, height }) => <SweepPlot width={width} height={height ?? 240} … /> },
 *   ]} />
 *
 * Same-kind small multiples (one panel per site, season, …) use FacetGrid, which shares PanelCell.
 */
export interface PanelRender {
  width: number
  /** the panel's share of a fitted height (a tile, the one-chart view); null: the chart's own */
  height: number | null
  /** drawn in the ⤢ dialog */
  expanded: boolean
  index: number
}
export interface PanelDef extends PanelSize {
  key: string
  title?: ReactNode
  /** a readout at the right of the title row (n, ρ, bias …) */
  note?: ReactNode
  /** plain-text name for the dialog title and screen readers (default: the title when it is a string) */
  label?: string
  /** a ⤢ button that opens the panel larger (default true) */
  expandable?: boolean
  render: (p: PanelRender) => ReactNode
}

/** the title row's height, taken off a fitted height */
export const PANEL_TITLE_H = 22

export function Panels({ panels, gap = 24, letters = true, minHeight = 140, align = 'start', label }: {
  panels: PanelDef[]
  gap?: number
  /** a, b, … before each title (default on, with two panels or more: a composite is usually captioned panel by panel) */
  letters?: boolean
  /** the least height a panel is handed when a fitted height is split */
  minHeight?: number
  /** side by side: top-aligned, or centred on each other */
  align?: 'start' | 'center'
  /** names the group for screen readers */
  label?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const fit = useFitHeight()
  const [open, setOpen] = useState<number | null>(null)
  const { stacked, widths } = panelRow(width, panels, gap)
  const n = panels.length
  // one panel needs no letter
  if (n < 2) letters = false
  const titled = (p: PanelDef) => !!(p.title || p.note || letters || p.expandable !== false)
  const height = (p: PanelDef) => fit === null ? null
    : Math.max(minHeight, stacked ? (fit - gap * (n - 1)) / n - (titled(p) ? PANEL_TITLE_H : 0) : fit - (titled(p) ? PANEL_TITLE_H : 0))
  const shown = open !== null ? panels[open] : null
  const big = typeof window === 'undefined' ? 900 : Math.max(320, Math.min(1100, window.innerWidth - 160))

  return (
    <div ref={ref} className="panels" role="group" aria-label={label}
      style={{ display: 'flex', flexDirection: stacked ? 'column' : 'row', gap, alignItems: stacked ? 'stretch' : align === 'center' ? 'center' : 'flex-start' }}>
      {width > 0 && panels.map((p, i) => (
        <PanelCell key={p.key} width={widths[i]} title={p.title} note={p.note} letter={letters ? panelLetter(i) : undefined}
          label={panelLabel(p)} onExpand={p.expandable === false ? undefined : () => setOpen(i)}>
          <ChartFit height={height(p)}>{p.render({ width: widths[i], height: height(p), expanded: false, index: i })}</ChartFit>
        </PanelCell>
      ))}
      {shown && <SpectrumModal id={panelLabel(shown) ?? ''} title={letters ? `Panel ${panelLetter(open!)}` : 'Panel'} onClose={() => setOpen(null)}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <ChartFit height={null}>{shown.render({ width: Math.max(shown.min ?? 240, Math.min(big, shown.max ? shown.max * 1.5 : big)), height: null, expanded: true, index: open! })}</ChartFit>
        </div>
      </SpectrumModal>}
    </div>
  )
}

const panelLabel = (p: { label?: string; title?: ReactNode }) => p.label ?? (typeof p.title === 'string' ? p.title : undefined)

/**
 * One panel's chrome, shared by Panels and FacetGrid: the title row (letter, title, a readout on
 * the right, ⤢) over the chart. Without a title, note, letter or ⤢ it is a bare box of `width`.
 */
export function PanelCell({ width, title, note, letter, label, onExpand, children }: {
  width?: number
  title?: ReactNode
  note?: ReactNode
  letter?: string
  label?: string
  onExpand?: () => void
  children: ReactNode
}) {
  const head = title || note || letter || onExpand
  return (
    <div className="panel-cell" role={label ? 'group' : undefined} aria-label={label} style={{ width, minWidth: 0 }}>
      {head && <div className="facet-title panel-head">
        {letter && <span className="panel-letter">{letter}</span>}
        {title && <span className="panel-title">{title}</span>}
        {/* the readout and ⤢ sit together at the right; a row too narrow for both moves them under the title */}
        {(note || onExpand) && <span className="panel-tools">
          {note && <span className="panel-note">{note}</span>}
          {onExpand && <button type="button" className="btn quiet panel-expand-inline" onClick={onExpand}
            aria-label={`Open ${label ?? (letter ? `panel ${letter}` : 'this panel')} larger`} title="Open larger"><ExpandIcon /></button>}
        </span>}
      </div>}
      {children}
    </div>
  )
}
