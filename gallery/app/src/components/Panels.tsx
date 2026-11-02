import { Children, Fragment, isValidElement, useRef, useState, type ReactElement, type ReactNode } from 'react'
import { ExpandIcon } from '@/components/Icons'
import { SpectrumModal } from '@/components/FilterSpectrumPlot'
import { PanelGroupControls, PanelGroupProvider, PanelSlot, usePanelGroupValue, type ShareAxes } from '@/components/PanelGroup'
import { useDimensions } from '@/hooks/useGalleryData'
import { ChartFit, useFitHeight } from '@/lib/chartFit'
import { panelLetter, panelRow, panelWrap, pickedPanels, type PanelSize } from '@/lib/facets'
import type { PanelAxes, PanelDim } from '@/lib/axes'
import type { FitGrouping } from '@/lib/siteRegression'

/**
 * Composite figures: different charts in one ChartFrame (a crossplot and its threshold sweep, a
 * chart and its table). One layout rule (lib/facets panelRow): side by side while every panel's
 * `min` fits, stacked below that; each panel is handed its width. With `wrap` (lib/facets panelWrap)
 * the panels flow into equal columns instead of stacking all at once (the research and meeting `Grid`).
 * Every panel gets the same chrome as a facet (PanelCell): a title row with an optional letter (a, b,
 * … for a caption to point at), a readout on the right, and ⤢ to open it larger in a dialog (on the
 * title row, or in the panel's corner when it has none). Titles are `.facet-title`, so the
 * whole-panel export keeps them; the ⤢ button is skipped.
 *
 * In a Dashboard tile or the one-chart view (lib/chartFit), each panel is handed its share of
 * the fitted height (all of it side by side, a split when stacked) as `height`, and charts inside
 * that read useFitHeight() see it too. Elsewhere `height` is null: draw at your own height.
 *
 * The panels are one group (components/PanelGroup, the same as FacetGrid's): `share` gives the shared charts inside
 * one axis range under the gallery's Panel axes setting, crossplots draw one fit grouping box for all of them,
 * `legend` is drawn once under the panels (and exported), `controls` and the charts' hoisted controls sit in one row
 * above them, and `show`/`order` pick and order panels by key.
 *
 *   <Panels panels={[
 *     { key: 'scatter', title: 'Dry against humid', min: 300, basis: 420, max: 460, grow: 0, render: ({ width }) => <Crossplot … maxSide={width - 82} /> },
 *     { key: 'sweep', title: 'Threshold sweep', min: 260, render: ({ width, height }) => <SweepPlot width={width} height={height ?? 240} … /> },
 *   ]} legend={<Legend items={…} />} />
 *
 * Same-kind small multiples (one panel per site, season, …) use FacetGrid, which shares PanelCell and the group.
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
  /** a ⤢ button that opens the panel larger (default: the group's `expandable`) */
  expandable?: boolean
  /** the panel's site, for Panel axes: Per site (panels of one site share a range) */
  site?: string
  /** false: the panel draws at its own height (a table) under a fitted height, and the other panels share the fit
   * (default true); the host's fitting takes the panel's height off theirs */
  fit?: boolean
  render: (p: PanelRender) => ReactNode
}

/** What both layouts (Panels, FacetGrid) take for the group as a whole (components/PanelGroup). */
export interface PanelGroupProps {
  /** shared axis ranges for the shared charts inside, under the gallery's Panel axes setting (components/PanelGroup) */
  share?: ShareAxes
  /** one fit grouping for every crossplot inside, with no box; unset: the gallery's, one box for the group */
  grouping?: FitGrouping | 'none'
  /** drawn once under the panels, kept in exports (a Legend or Key) */
  legend?: ReactNode
  /** group-level controls, in one row above the panels with the charts' hoisted ones */
  controls?: ReactNode
  /** the Panel axes control in that row while ranges are shared across several panels (default true) */
  axesControl?: boolean
  /** what the panels split by, for the Panel axes control (default: site when the panels name one) */
  dims?: PanelDim[]
  /** the keys of the panels to draw (default all) */
  show?: readonly string[]
  /** keys to draw first, in this order; the rest follow in their own */
  order?: readonly string[]
  /** the reader's choice from lib/panelPick usePanelPick: its ids are both `show` and `order` (none chosen: no panel) */
  pick?: { readonly ids: readonly string[] }
  /** fix Panel axes for this figure (a meeting figure as presented) instead of following the gallery's setting */
  mode?: PanelAxes
}

/** the title row's height, taken off a fitted height */
export const PANEL_TITLE_H = 22

export function Panels({ panels, gap = 24, letters = true, minHeight = 140, align = 'start', label, wrap = false, maxCols, center = true, expandable = true,
  share, grouping, legend, controls, axesControl = true, dims, show, order, pick, mode }: {
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
  /** when the panels do not fit side by side, flow them into equal columns (lib/facets panelWrap) rather than one stack */
  wrap?: boolean
  /** with `wrap`, at most this many columns */
  maxCols?: number
  /** centre the panels when they do not fill the width (default true) */
  center?: boolean
  /** ⤢ on every panel (default true; a panel's own `expandable` wins) */
  expandable?: boolean
} & PanelGroupProps) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const fit = useFitHeight()
  const [open, setOpen] = useState<string | null>(null)
  const { value: group, joined } = usePanelGroupValue({ share, grouping, mode })
  const drawn = pickedPanels(panels.map((p) => p.key), show, order, pick).map((i) => panels[i])
  const n = drawn.length
  const lay = wrap ? panelWrap(width, drawn, gap, maxCols) : (() => { const r = panelRow(width, drawn, gap); return { ...r, cols: r.stacked ? 1 : n, rows: r.stacked ? n : 1 } })()
  // one panel needs no letter
  if (n < 2) letters = false
  const headed = (p: PanelDef) => !!(p.title || p.note || letters)
  // stacked, a panel at its own height (`fit: false`) takes no share of the rows
  const rows = lay.stacked ? drawn.filter((p) => p.fit !== false).length : lay.rows
  const height = (p: PanelDef) => fit === null || p.fit === false ? null
    : Math.max(minHeight, (fit - gap * (rows - 1)) / Math.max(1, rows) - (headed(p) ? PANEL_TITLE_H : 0))
  const at = open === null ? -1 : drawn.findIndex((p) => p.key === open)
  const shown = at >= 0 ? drawn[at] : null
  const big = typeof window === 'undefined' ? 900 : Math.max(320, Math.min(1100, window.innerWidth - 160))
  const grid = lay.cols > 1 && lay.rows > 1
  const sharing = !!share && !mode && n > 1

  return (
    <PanelGroupProvider value={group}>
      {!joined && <PanelGroupControls value={group} axesControl={sharing && axesControl} dims={dims ?? (drawn.some((p) => p.site) ? ['site'] : [])} extra={controls} />}
      <div ref={ref} className="panels" role="group" aria-label={label}
        style={grid
          ? { display: 'grid', gridTemplateColumns: `repeat(${lay.cols}, minmax(0, ${Math.max(...lay.widths)}px))`, gap, alignItems: align === 'center' ? 'center' : 'start', justifyContent: center ? 'safe center' : undefined }
          : { display: 'flex', flexDirection: lay.stacked ? 'column' : 'row', gap, alignItems: lay.stacked ? (center ? 'safe center' : 'stretch') : align === 'center' ? 'center' : 'flex-start', justifyContent: center && !lay.stacked ? 'safe center' : undefined }}>
        {width > 0 && drawn.map((p, i) => (
          <PanelCell key={p.key} width={lay.widths[i]} title={p.title} note={p.note} letter={letters ? panelLetter(i) : undefined}
            label={panelLabel(p)} onExpand={(p.expandable ?? expandable) ? () => setOpen(p.key) : undefined}>
            <PanelSlot panelKey={p.key} site={p.site}>
              <ChartFit height={height(p)}>{p.render({ width: lay.widths[i], height: height(p), expanded: false, index: i })}</ChartFit>
            </PanelSlot>
          </PanelCell>
        ))}
        {shown && <ExpandedPanel name={panelLabel(shown)} letter={letters ? panelLetter(at) : undefined} onClose={() => setOpen(null)}>
          <PanelSlot panelKey={shown.key} site={shown.site}>
            <ChartFit height={null}>{shown.render({ width: Math.max(shown.min ?? 240, Math.min(big, shown.max ? shown.max * 1.5 : big)), height: null, expanded: true, index: at })}</ChartFit>
          </PanelSlot>
        </ExpandedPanel>}
      </div>
      {legend}
    </PanelGroupProvider>
  )
}

const panelLabel = (p: { label?: string; title?: ReactNode }) => p.label ?? (typeof p.title === 'string' ? p.title : undefined)

/** the ⤢ dialog of a panel or facet */
export function ExpandedPanel({ name, letter, onClose, children }: { name?: string; letter?: string; onClose: () => void; children: ReactNode }) {
  return <SpectrumModal id={name ?? ''} title={letter ? `Panel ${letter}` : 'Panel'} onClose={onClose}>
    <div style={{ display: 'flex', justifyContent: 'center' }}>{children}</div>
  </SpectrumModal>
}

/**
 * One panel's chrome, shared by Panels and FacetGrid: the title row (letter, title, a readout on
 * the right, ⤢) over the chart. Without a title, note or letter there is no title row: ⤢ then sits in
 * the panel's top-right corner (shown on hover or focus), so the chart keeps its place.
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
  const head = title || note || letter
  const expand = onExpand && <button type="button" className={`btn quiet ${head ? 'panel-expand-inline' : 'panel-expand-float'}`} onClick={onExpand}
    aria-label={`Open ${label ?? (letter ? `panel ${letter}` : 'this panel')} larger`} title="Open larger"><ExpandIcon /></button>
  return (
    <div className="panel-cell" role={label ? 'group' : undefined} aria-label={label} style={{ width, minWidth: 0, position: head ? undefined : 'relative' }}>
      {head && <div className="facet-title panel-head">
        {letter && <span className="panel-letter">{letter}</span>}
        {title && <span className="panel-title">{title}</span>}
        {/* the readout and ⤢ sit together at the right; a row too narrow for both moves them under the title */}
        {(note || onExpand) && <span className="panel-tools">
          {note && <span className="panel-note">{note}</span>}
          {expand}
        </span>}
      </div>}
      {children}
      {!head && expand}
    </div>
  )
}

/** a child list as separate panels: fragments opened, nulls and booleans dropped */
export function panelChildren(children: ReactNode): ReactElement[] {
  const out: ReactElement[] = []
  const walk = (c: ReactNode) => Children.forEach(c, (k) => {
    if (isValidElement(k) && k.type === Fragment) walk((k.props as { children?: ReactNode }).children)
    else if (isValidElement(k)) out.push(k)
    else if (k != null && typeof k !== 'boolean' && k !== '') out.push(<span>{k}</span>)
  })
  walk(children)
  return out
}

/**
 * Panels from children: each child is one panel at least `min` wide, flowing into equal columns (two on a wide
 * screen, one on a phone). The research tabs' and meeting pages' `Grid` is this. It is the panel group of Panels
 * (PanelCell chrome, ⤢ in each panel's corner, `share`, one fit grouping box, `legend`, `show`/`order` by child key)
 * on the CSS track `repeat(auto-fit, minmax(min, 1fr))`, so a child that draws nothing (an ExplainOnly with Explain
 * off) leaves no empty column. Letters are off (the children carry their own titles). Children size themselves.
 */
export function PanelGrid({ children, min = 340, gap = 18, letters = false, label, expandable = true,
  share, grouping, legend, controls, axesControl = true, dims, show, order, pick, mode }: {
  children: ReactNode
  min?: number
  gap?: number
  letters?: boolean
  label?: string
  expandable?: boolean
} & PanelGroupProps) {
  const kids = panelChildren(children)
  const keys = kids.map((k, i) => (k.key != null ? String(k.key) : String(i)))
  const { value: group, joined } = usePanelGroupValue({ share, grouping, mode })
  const [open, setOpen] = useState<string | null>(null)
  const drawn = pickedPanels(keys, show, order, pick)
  const at = open === null ? -1 : drawn.findIndex((i) => keys[i] === open)
  const big = typeof window === 'undefined' ? 900 : Math.max(320, Math.min(1100, window.innerWidth - 160))
  return (
    <PanelGroupProvider value={group}>
      {!joined && <PanelGroupControls value={group} axesControl={!!share && !mode && drawn.length > 1 && axesControl} dims={dims} extra={controls} />}
      <div className="panel-grid" role="group" aria-label={label}
        style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(min(${min}px, 100%), 1fr))`, gap, alignItems: 'start' }}>
        {drawn.map((i, pos) => (
          <PanelCell key={keys[i]} letter={letters ? panelLetter(pos) : undefined} onExpand={expandable ? () => setOpen(keys[i]) : undefined}>
            <PanelSlot panelKey={keys[i]}>{kids[i]}</PanelSlot>
          </PanelCell>
        ))}
        {at >= 0 && <ExpandedPanel letter={letters ? panelLetter(at) : undefined} onClose={() => setOpen(null)}>
          {/* a child sizes itself to its box: in the dialog that box is the dialog's width */}
          <PanelSlot panelKey={keys[drawn[at]]}><ChartFit height={null}><div style={{ width: big }}>{kids[drawn[at]]}</div></ChartFit></PanelSlot>
        </ExpandedPanel>}
      </div>
      {legend}
    </PanelGroupProvider>
  )
}
