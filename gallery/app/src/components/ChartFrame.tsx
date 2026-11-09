import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { createPortal } from 'react-dom'
import { createContext, useCallback, useId, useContext, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { ExportButton } from '@/components/ExportDialog'
import { EXPLAIN_AVAILABLE, usePrefs } from '@/lib/prefs'
import { setSingleChartDisplay, useChartDisplayPrefs } from '@/lib/chartDisplayPrefs'
import { chartInkStyle } from '@/lib/chartDisplayStyles'
import { useSystemDark } from '@/lib/colorTheme'
import { useParamScope } from '@/lib/url'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'
import { AsideSlotCtx, TILE_CORNER_WIDTH, TILE_FIT_MODES, TileStatsCtx, type TileFitMode, ChartAside, SingleViewNav, SingleViewStats, StatsInPlotCtx, useSingleView } from '@/lib/singleView'
import type { ControlsSide, StatsPlace } from '@/lib/chartDisplay'
import { useExportRender } from '@/lib/exportRender'
import { FrameSlotProvider, useFrameSlot } from '@/lib/frameControls'
import { EmbedDrawnContext, useEmbedDraw } from '@/lib/embedChart'
import { copyText, currentView, viewUrl } from '@/lib/views'
import { HOVER_CARD_ATTR, HoverCardContext } from '@/lib/highlight'

// the option controls live in Controls.tsx; re-exported so the many imports from ChartFrame keep working
export { Select, Segmented, Toggle, sentence } from '@/components/Controls'
import { Segmented } from '@/components/Controls'
import { SectionAddButton, useSectionPick } from '@/components/SectionsMenu'

/**
 * What a frame's host adds to it. A Dashboard tile draws the frame compact: no header
 * line, the chart's name, actions and options folded into a ⋯ menu that shows on hover
 * (always on a touch screen), the way Grafana's panel menu and Power BI's visual header do.
 */
export interface FrameExtras {
  /** a tile: the ⋯ menu instead of the header line */
  compact?: boolean
  /** the tile's own actions in the menu (full window, duplicate, remove …); `close` shuts the menu */
  menu?: (close: () => void) => ReactNode
  /** ahead of the chart's own controls in the menu, e.g. the tile's field pickers */
  controls?: ReactNode
  /** after the name, e.g. the tile's fields */
  subtitle?: ReactNode
  /** arranging: a bar with the name to drag the tile by (pages/DashboardPage) */
  editBar?: ReactNode
  /** where the chart's fit statistics go (a choice in the ⋯ menu, shown when the chart has a fit); absent: behind the ⓘ */
  fitStats?: { mode: TileFitMode; set: (mode: TileFitMode) => void }
  /** more for the tile's ⓘ box, after the chart's tip, e.g. a field standing in for one the sites never measured */
  info?: ReactNode
}
export const FrameExtrasContext = createContext<FrameExtras | null>(null)

/**
 * The panel every chart sits in. Quiet by default: the title's first clause,
 * the controls, the chart. The explanation (the rest of the title, the
 * subtitle, the provenance line) is behind the "?" button, or shown on every
 * chart at once with the header's "explain" toggle.
 */
export function ChartFrame({
  title,
  subtitle,
  provenance,
  controls,
  children,
  exportable = true,
  id,
  tip,
  source,
  about,
  evidence,
  statsPlace,
  controlsSide,
  pageOnly = false,
  hoverCard,
}: {
  /** "Short name — longer clause": only the short name shows unless explaining */
  title: string
  subtitle?: string
  /** which notebooks this chart family stands in for */
  provenance?: ReactNode
  controls?: ReactNode
  children: ReactNode
  /** show the export button (off for text panels) */
  exportable?: boolean
  /** anchor id so a URL can jump straight to this chart */
  id?: string
  /** caveats and reading notes, behind an ⓘ beside the title instead of under the chart */
  tip?: ReactNode
  /** where the chart is built, e.g. "gallery/app/src/pages/MeetingFollowupPage.tsx › ThreePlots"; copied by the ref button */
  source?: string
  /** longer explanation (e.g. a meeting section's findings) shown with the subtitle, behind the "?" */
  about?: ReactNode
  /** Essential reference, population and evidence status; always visible and retained in exports. */
  evidence?: ReactNode
  /** where this chart's fit statistics go in the single-chart view unless the viewer set it for this chart; else the shared setting */
  statsPlace?: StatsPlace
  /** where this chart's options sit unless the viewer set it for this chart; else the shared setting. A tile and the single-chart view keep their own placement */
  controlsSide?: ControlsSide
  /** a summary of the page's settings rather than a chart: the single-chart view does not step to it, and hides it */
  pageOnly?: boolean
  /** this card's id for the linked hover, for a chart that reads the hover above this frame (lib/highlight useHighlight); default: its own */
  hoverCard?: string
}) {
  const ownCard = useId()
  // a hovered filter lights up the charts of this card only (lib/highlight)
  const card = hoverCard ?? ownCard
  const cardAttrs = { [HOVER_CARD_ATTR]: card }
  const method = useEstimator()
  const extras = useContext(FrameExtrasContext)
  // inside a Dashboard tile the anchor carries the tile's prefix: two tiles, two ids
  const scope = useParamScope()
  if (id && scope) id = `${scope}${id}`
  title = estimatorText(title, method)
  subtitle = subtitle ? estimatorText(subtitle, method) : subtitle
  const ref = useRef<HTMLElement>(null)
  const { explain, density } = usePrefs()
  const dash = title.indexOf(' — ')
  const short = dash > 0 ? title.slice(0, dash) : title
  const rest = dash > 0 ? title.slice(dash + 3) : ''
  const hasInfo = !!(rest || subtitle || provenance || about)
  const [refCopied, setRefCopied] = useState(false)
  const compact = !!extras?.compact
  // controls the chart's layout hoists into this options row (a panel group's Panel axes, lib/frameControls)
  const hoist = useFrameSlot()
  if (hoist.claimed || controls) controls = <>{controls}{hoist.slot}</>
  const allControls = (controls || extras?.controls) ? <>{extras?.controls}{controls}</> : null
  // a paste-able pointer for development: which chart, where its code lives, and the exact view
  const copyRef = () => {
    const text = [`Gallery chart: "${short}"${id ? ` (#${id})` : ''}`,
      `Source: ${source ?? 'search the title in gallery/app/src'}`,
      `View: ${window.location.href}`].join('\n')
    copyText(text).then(() => { setRefCopied(true); setTimeout(() => setRefCopied(false), 1400) }, () => {})
  }

  const anchor = id && method !== 'deming' ? `${id}-${method}` : id
  const chartKey = id ?? `${scope}${slug(short)}`
  const display = useChartDisplayPrefs()
  const own = display.charts[chartKey]
  // single-chart view (lib/singleView): the chart in view carries the arrows, the picker and its fit statistics,
  // and its options default to a column on the right, so the plot keeps the height under the title
  const single = useSingleView()
  const inView = !compact && single.on && single.active !== null && single.active === ref.current
  // its notes and readouts go to the stats box too, except while an export copies the chart (they belong in the file)
  const exportRender = useExportRender()
  const asideSlot = inView && !(exportRender && exportRender.panel === ref.current) ? single.slot : null
  const ownSide = compact ? undefined : controlsSide
  const side = own?.controlsSide ?? (inView ? single.side : ownSide ?? display.controlsSide)
  const stats = own?.statsPlace ?? statsPlace ?? display.statsPlace ?? 'options'
  const info = EXPLAIN_AVAILABLE && (own?.explain ?? explain)
  const readable = (own?.density ?? density) === 'readable'
  const systemDark = useSystemDark()
  const theme = own?.theme === 'auto' ? systemDark ? 'dark' : 'light' : own?.theme
  const appearance = {
    ...(theme ? chartInkStyle(theme) : {}),
    '--text-scale': readable ? 1.15 : 1,
    '--chart-text-scale': readable ? 1.12 : 1,
  } as CSSProperties
  const displayAttrs = { 'data-chart-key': chartKey, 'data-chart-title': short, 'data-controls-default': ownSide, 'data-stats-place': stats, 'data-chart-theme': theme, 'data-chart-explain': info ? '1' : '0', style: appearance }
  // focus mode: this chart alone, centred over the page with its options; `focus=<key>` in the URL, so a copied link
  // reopens it exactly as it is (every other setting already lives in the URL)
  const focusKey = anchor ?? slug(short)
  const [focused, setFocused] = useFocus(focusKey)
  // a section's embed draws the chart it is for, not the rest of that chart's page (lib/embedChart)
  const draw = useEmbedDraw(focusKey, inView)
  if (!draw) controls = null
  const [exportAsk, setExportAsk] = useState(0)
  // a tile's corner (TileCorner): its notes (ChartAside, Note) fold behind a ⚠; its fit statistics (RegressionOverlay)
  // go behind the ⓘ with its tip, in a column beside the plot, or above the plot (the tile's ⋯ menu: extras.fitStats).
  // While an export copies the tile, both go back in the chart, as on a page
  const [noteBox, setNoteBox] = useState<HTMLElement | null>(null)
  const [statsBox, setStatsBox] = useState<HTMLElement | null>(null)
  const [sideBox, setSideBox] = useState<HTMLElement | null>(null)
  const tileExporting = !!exportRender && exportRender.panel === ref.current
  const fitMode = extras?.fitStats?.mode ?? 'corner'
  const tileNotes = compact && !tileExporting ? noteBox : null
  const tileStats = !compact || tileExporting ? null : fitMode === 'corner' ? statsBox : fitMode === 'beside' ? sideBox : null
  if (compact) return (<HoverCardContext.Provider value={card}><FrameSlotProvider value={hoist.value}>
    <section className="panel compact" ref={ref} id={anchor} {...cardAttrs} {...displayAttrs}>
      {/* "About this chart" in the ⋯ menu follows Explain like every other explanation: off by default */}
      <TileMenu name={short} extras={extras!} about={info && hasInfo ? <>{rest && <p>{rest}</p>}{about && <div className="frame-about">{about}</div>}{subtitle && <p className="subtitle">{subtitle}</p>}{provenance && <p className="provenance">{provenance}</p>}</> : null}
        tip={tip} onNoteBox={setNoteBox} onStatsBox={setStatsBox} controls={side === 'top' ? allControls : null} export={exportable ? <ExportButton panelRef={ref} title={short} /> : null} onCopyRef={copyRef} copied={refCopied} />
      {info && hasInfo && <div className="compact-explanation"><strong>{short}</strong>{about}{subtitle && <p className="subtitle">{subtitle}</p>}{provenance && <p className="provenance">{provenance}</p>}</div>}
      {evidence && <p className="chart-evidence" data-export="text">{evidence}</p>}
      <div className={`frame-body controls-${side}`}>
        {side !== 'top' && allControls && <div className="frame-controls">{allControls}</div>}
        {/* the same elements in every fit mode, so switching does not remount the chart; the column is empty unless beside */}
        <div className={`tile-chart-row${fitMode === 'beside' ? ' fit-beside' : ''}`}>
          <div className="frame-chart"><AsideSlotCtx.Provider value={tileNotes}><TileStatsCtx.Provider value={tileStats}>{children}</TileStatsCtx.Provider></AsideSlotCtx.Provider></div>
          <div ref={setSideBox} className="tile-fit-side" data-export="skip" />
        </div>
      </div>
    </section>
  </FrameSlotProvider></HoverCardContext.Provider>)
  // single-chart view with its options in a column: the title bar heads that column, so the plot takes the panel's full height
  const headInColumn = inView && !!controls && side !== 'top' && !focused
  const head = (
    <header className="frame-head">
      <div>
        {/* in the single-chart view the title is the way through the page: ‹ n / N ›, then a picker named for this chart */}
        {inView
          ? <h2 className="single-head"><SingleViewNav>{tip && <InfoTip>{tip}</InfoTip>}</SingleViewNav></h2>
          : <h2>
            {short}
            {info && rest && <span className="title-rest"> — {rest}</span>}
            {tip && <InfoTip>{tip}</InfoTip>}
          </h2>}
        {inView && info && rest && <p className="title-rest">{rest}</p>}
        {info && about && <div className="frame-about">{about}</div>}
        {info && subtitle && <p className="subtitle">{subtitle}</p>}
        {info && provenance && <p className="provenance">{provenance}</p>}
      </div>
      <div className="frame-actions">
        {/* on a phone these three fold into the ⋯ menu (FrameMoreMenu); the export dialog still opens from here */}
        <span className="frame-wide">
          {hasInfo && EXPLAIN_AVAILABLE && (
            <button type="button" className={`btn quiet ${info ? 'on' : ''}`} title={info ? 'Hide the explanation' : 'What this chart shows and where it comes from'} aria-pressed={info} onClick={() => setSingleChartDisplay(chartKey, { explain: !info })}>
              ?
            </button>
          )}
          {!pageOnly && <SectionAddButton chart={focusKey} title={short} />}
          <RefMenu onCopyRef={copyRef} refCopied={refCopied} link={() => focusLink(focusKey)} />
          {/* the whole figure (every facet and the legend), not just the first <svg> */}
          {exportable && <ExportButton panelRef={ref} title={short} ask={exportAsk} />}
        </span>
        <FrameMoreMenu link={() => focusLink(focusKey)} onCopyRef={copyRef} chart={pageOnly ? null : focusKey} title={short}
          explain={hasInfo && EXPLAIN_AVAILABLE ? { on: info, toggle: () => setSingleChartDisplay(chartKey, { explain: !info }) } : null}
          onExport={exportable ? () => setExportAsk((n) => n + 1) : null} />
        {focused
          ? <button type="button" className="btn quiet" title="Back to the page (Esc)" onClick={() => setFocused(false)}>Close ✕</button>
          : <button type="button" className="btn quiet" title="Focus: this chart alone, centred, with its options" aria-label="Focus on this chart" onClick={() => setFocused(true)}><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg></button>}
      </div>
    </header>
  )
  return (<HoverCardContext.Provider value={card}><FrameSlotProvider value={hoist.value}>
    {focused && <div className="focus-scrim" onClick={() => setFocused(false)} />}
    <section className={`panel${focused ? ' focused' : ''}`} ref={ref} id={anchor} {...(pageOnly ? {} : { 'data-frame': '', 'data-frame-title': short, 'data-frame-key': focusKey })} {...cardAttrs} {...displayAttrs}>{/* data-frame: a chart the single-chart view steps through (lib/singleView) */}
      {!headInColumn && head}
      {evidence && <p className="chart-evidence" data-export="text">{evidence}</p>}
      {/* the fit statistics in single-chart view: at the head of an options column, else above the options row */}
      {inView && (!controls || side === 'top') && <SingleViewStats />}
      {/* the same elements whichever side the options are on: a different tree would remount the chart's
          own wrapper under a chart that is not remounted, and its size observer would keep the old width */}
      <div className={`frame-body controls-${controls ? side : 'top'}`}>
        {controls && <div className="frame-controls">{headInColumn && head}{inView && side !== 'top' && <SingleViewStats />}{controls}</div>}
        <div className="frame-chart"><AsideSlotCtx.Provider value={asideSlot}><StatsInPlotCtx.Provider value={stats === 'plot'}><EmbedDrawnContext.Provider value={draw}>{draw && children}</EmbedDrawnContext.Provider></StatsInPlotCtx.Provider></AsideSlotCtx.Provider></div>
      </div>
    </section>
  </FrameSlotProvider></HoverCardContext.Provider>)
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48)
const FOCUS_EVENT = 'gallery-focus'

/** A link that opens the gallery on one chart in focus mode, with every setting as it is now: one json-url token
 * (lib/views), so a long list of filters stays short, or the plain address if it is too long even for that. */
export function focusLink(key: string): Promise<string> {
  const view = currentView()
  view.q.focus = key
  return viewUrl(view).catch(() => {
    const url = new URL(window.location.href)
    url.searchParams.set('focus', key)
    return url.toString()
  })
}

/** whether this chart is the focused one (`focus` in the query); one at a time, kept in sync across frames */
function useFocus(key: string): [boolean, (on: boolean) => void] {
  const read = () => new URL(window.location.href).searchParams.get('focus') === key
  const [on, setOn] = useState(read)
  useEffect(() => {
    const sync = () => setOn(read())
    sync()
    window.addEventListener('popstate', sync)
    window.addEventListener(FOCUS_EVENT, sync)
    return () => { window.removeEventListener('popstate', sync); window.removeEventListener(FOCUS_EVENT, sync) }
  }, [key])  // eslint-disable-line react-hooks/exhaustive-deps
  const set = (v: boolean) => {
    const url = new URL(window.location.href)
    if (v) url.searchParams.set('focus', key)
    else url.searchParams.delete('focus')
    window.history.replaceState(null, '', url)
    window.dispatchEvent(new Event(FOCUS_EVENT))
  }
  // while focused: Esc leaves (unless a menu inside handled it first) and the page behind does not scroll
  useEffect(() => {
    if (!on) return
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape' && !e.defaultPrevented && !document.querySelector('[aria-expanded="true"]')) set(false) }
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', esc)
    return () => { document.body.style.overflow = prev; window.removeEventListener('keydown', esc) }
  }, [on])  // eslint-disable-line react-hooks/exhaustive-deps
  return [on, set]
}

/**
 * A phone's ⋯ beside Focus: the explanation, the chart's link and reference, and export, which the header lays out
 * as separate buttons on a wider screen (styles.css shows one or the other). The same menu style as Ref's.
 */
function FrameMoreMenu({ link, onCopyRef, explain, onExport, chart, title }: {
  link: () => Promise<string>; onCopyRef: () => void; explain: { on: boolean; toggle: () => void } | null; onExport: (() => void) | null
  /** this chart's key and name for the section being built (null: a page summary, which is not offered) */
  chart: string | null; title: string
}) {
  const pick = useSectionPick(chart ?? '', title)
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState<'link' | 'ref' | null>(null)
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const menuId = useId()
  const place = useFloatingMenu(open, trigger, menu)
  const positioned = place !== null
  useEffect(() => {
    if (open && positioned) menu.current?.querySelector<HTMLButtonElement>('[role=menuitem]')?.focus({ preventScroll: true })
  }, [open, positioned])
  useEffect(() => {
    if (!open) return
    const off = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node) && !menu.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); setOpen(false); trigger.current?.focus({ preventScroll: true }) } }
    window.addEventListener('pointerdown', off)
    window.addEventListener('keydown', esc)
    return () => { window.removeEventListener('pointerdown', off); window.removeEventListener('keydown', esc) }
  }, [open])
  const done = (what: 'link' | 'ref') => { setCopied(what); setTimeout(() => { setCopied(null); setOpen(false) }, 1000) }
  return (
    <div className="frame-menu-root frame-more" ref={root}>
      <button ref={trigger} type="button" className="btn quiet frame-more-btn" aria-haspopup="menu" aria-controls={menuId} aria-expanded={open}
        aria-label="More for this chart" title="Explanation, link, reference and export" onClick={() => setOpen((v) => !v)}>⋯</button>
      {open && createPortal(<div ref={menu} id={menuId} className="frame-menu floating" role="menu" aria-label="More for this chart"
        style={place ?? { position: 'fixed', visibility: 'hidden' }}
        onKeyDown={e => {
          const items = [...menu.current!.querySelectorAll<HTMLButtonElement>('[role=menuitem]')]
          const at = items.indexOf(document.activeElement as HTMLButtonElement)
          if (['ArrowDown', 'ArrowUp'].includes(e.key)) { e.preventDefault(); items[(at + (e.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length]?.focus({ preventScroll: true }) }
          else if (e.key === 'Tab') setOpen(false)
        }}>
        {explain && <button type="button" role="menuitem" onClick={() => { explain.toggle(); setOpen(false) }}>
          <strong>{explain.on ? 'Hide the explanation' : 'Explain this chart'}</strong><span>What it shows and where it comes from</span>
        </button>}
        {chart && pick.offered && <button type="button" role="menuitem" onClick={() => { pick.toggle(); setOpen(false) }}>
          <strong>{pick.on ? `Take out of ${pick.name}` : 'Add to section'}</strong><span>{pick.on ? 'It is in this section with these settings' : `Keeps its sites, filters and options, in ${pick.name}`}</span>
        </button>}
        <button type="button" role="menuitem" onClick={() => { copyText(link()).then(() => done('link'), () => {}) }}>
          <strong>{copied === 'link' ? 'Link copied' : 'Copy link to this chart'}</strong><span>Opens the gallery on this chart, with every setting as it is now</span>
        </button>
        {onExport && <button type="button" role="menuitem" onClick={() => { setOpen(false); onExport() }}>
          <strong>Export…</strong><span>PNG, JPEG, SVG or PDF, with a preview</span>
        </button>}
        <button type="button" role="menuitem" onClick={() => { onCopyRef(); done('ref') }}>
          <strong>{copied === 'ref' ? 'Reference copied' : 'Copy reference'}</strong><span>Title, source file and view URL, for development</span>
        </button>
      </div>, document.body)}
    </div>
  )
}

/** "Ref ▾": share this chart as it is now, or copy the development reference (focus is its own button beside) */
function RefMenu({ onCopyRef, refCopied, link }: { onCopyRef: () => void; refCopied: boolean; link: () => Promise<string> }) {
  const [open, setOpen] = useState(false)
  const [linkCopied, setLinkCopied] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const menuId = useId()
  const place = useFloatingMenu(open, trigger, menu)
  const positioned = place !== null
  const close = () => { setOpen(false); trigger.current?.focus({ preventScroll: true }) }
  useEffect(() => {
    if (open && positioned) menu.current?.querySelector<HTMLButtonElement>('[role=menuitem]')?.focus({ preventScroll: true })
  }, [open, positioned])
  useEffect(() => {
    if (!open) return
    const off = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node) && !menu.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); setOpen(false); trigger.current?.focus({ preventScroll: true }) } }
    window.addEventListener('pointerdown', off)
    window.addEventListener('keydown', esc)
    return () => { window.removeEventListener('pointerdown', off); window.removeEventListener('keydown', esc) }
  }, [open])
  const copyLink = () => { copyText(link()).then(() => {
    setLinkCopied(true)
    setTimeout(() => { setLinkCopied(false); setOpen(false) }, 1200)
  }, () => setOpen(false)) }
  return (
    <div className="frame-menu-root" ref={root}>
      <button ref={trigger} type="button" className="btn quiet" aria-haspopup="menu" aria-controls={menuId} aria-expanded={open} onClick={() => setOpen((v) => !v)}
        title="Share this chart with its settings, focus on it, or copy a development reference">
        {refCopied ? 'Copied' : 'Ref'}<svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
      </button>
      {open && createPortal(<div ref={menu} id={menuId} className="frame-menu floating" role="menu" aria-label="Chart reference actions"
        style={place ?? { position: 'fixed', visibility: 'hidden' }}
        onKeyDown={e => {
          const items = [...menu.current!.querySelectorAll<HTMLButtonElement>('[role=menuitem]')]
          const at = items.indexOf(document.activeElement as HTMLButtonElement)
          if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
            e.preventDefault()
            const next = e.key === 'Home' ? 0 : e.key === 'End' ? items.length - 1 : (at + (e.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length
            items[next]?.focus({ preventScroll: true })
          } else if (e.key === 'Tab') close()
        }}>
        <button type="button" role="menuitem" onClick={copyLink}>
          <strong>{linkCopied ? 'Link copied' : 'Copy link to this chart'}</strong>
          <span>Opens the gallery on this chart alone, with every setting as it is now</span>
        </button>
        <button type="button" role="menuitem" onClick={() => { onCopyRef(); setOpen(false) }}>
          <strong>Copy reference</strong><span>Title, source file and view URL, for development</span>
        </button>
      </div>, document.body)}
    </div>
  )
}

/**
 * A tile's ⋯ button and its menu: the chart's name and fields, the tile's actions, export,
 * "About this chart", then the chart's options. A click in another tile or Esc closes it;
 * clicks in a picker's or the export dialog's own popup (drawn outside the tile) do not.
 */
function TileMenu({ name, extras, about, tip, onNoteBox, onStatsBox, controls, export: exportNode, onCopyRef, copied }: {
  name: string; extras: FrameExtras; about: ReactNode; tip?: ReactNode
  onNoteBox: (el: HTMLElement | null) => void; onStatsBox: (el: HTMLElement | null) => void
  controls: ReactNode; export: ReactNode; onCopyRef: () => void; copied: boolean
}) {
  const [open, setOpen] = useState(false)
  // the menu is fixed to the window, placed from the ⋯ button: the tile's panel scrolls (overflow),
  // so a menu inside it would be clipped at the tile's edge. It opens upward from a tile low in the
  // window, and is kept inside the window on the left and right
  const [place, setPlace] = useState<React.CSSProperties>({})
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const off = (e: PointerEvent) => {
      const t = e.target as Element
      if (t.closest?.('.dash-main') && !root.current?.parentElement?.contains(t)) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    // a fixed menu would stay put while its tile scrolls away under it (a scroll inside the menu,
    // or in a picker's list opened from it, moves nothing)
    const scroll = (e: Event) => {
      const t = e.target
      if (t === document || (t instanceof Node && root.current && t.contains(root.current))) setOpen(false)
    }
    const resize = () => setOpen(false)
    window.addEventListener('pointerdown', off)
    window.addEventListener('keydown', esc)
    window.addEventListener('scroll', scroll, { passive: true, capture: true })
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('pointerdown', off); window.removeEventListener('keydown', esc)
      window.removeEventListener('scroll', scroll, { capture: true }); window.removeEventListener('resize', resize)
    }
  }, [open])
  const close = () => setOpen(false)
  // the fit statistics choice only for a chart that draws a fit (in the plot, or in the ⓘ box or column)
  const [hasFit, setHasFit] = useState(false)
  const toggle = (btn: HTMLElement) => {
    if (open) { setOpen(false); return }
    setHasFit(!!root.current?.closest('section')?.querySelector('.regression-summary, .fit-stack'))
    const r = btn.getBoundingClientRect(), W = window.innerWidth, H = window.innerHeight
    const width = Math.min(380, W - 16)
    const left = Math.max(8, Math.min(r.right - width, W - width - 8))
    const up = r.top > H * 0.55
    setPlace(up
      ? { left, width, bottom: H - r.top + 4, maxHeight: Math.min(520, r.top - 12) }
      : { left, width, top: r.bottom + 4, maxHeight: Math.min(520, H - r.bottom - 12) })
    setOpen(true)
  }
  return (
    <div ref={root} className={`tile-head${open ? ' open' : ''}`}>
      {extras.editBar}
      <TileCorner kind="notes" onBox={onNoteBox} />
      <TileCorner kind="info" onBox={onStatsBox}>{tip || extras.info ? <>{tip}{extras.info}</> : null}</TileCorner>
      <button type="button" className="tile-menu-btn" aria-expanded={open} aria-label={`${name}: menu`} title={name}
        onClick={(e) => toggle(e.currentTarget)}>⋯</button>
      {open && <div className="tile-menu" role="dialog" aria-label={name} style={place}>
        <div className="tile-menu-title"><strong>{name}</strong>{extras.subtitle && <span className="frame-sub">{extras.subtitle}</span>}</div>
        <div className="tile-menu-actions">
          {extras.menu?.(close)}
          {exportNode}
          <button type="button" className="btn quiet" onClick={onCopyRef} title="Copy a reference to this chart (title, source file, view URL)">{copied ? 'Copied' : 'Ref'}</button>
        </div>
        {extras.fitStats && hasFit && <span className="tile-legend-mode">
          <Segmented label="Fit statistics" value={extras.fitStats.mode} options={[...TILE_FIT_MODES]} onChange={extras.fitStats.set}
            optionLabel={(m) => ({ corner: 'ⓘ', beside: 'Beside', plot: 'In plot' }[m as TileFitMode])}
            title="ⓘ: behind the button in the corner · Beside: a column next to the plot · In plot: the box above the axes" />
        </span>}
        {about && <details className="tile-about"><summary>About this chart</summary>{about}</details>}
        {controls && <div className="frame-controls">{controls}</div>}
      </div>}
    </div>
  )
}

/**
 * A note on the chart: under it, or beside its options when it is the chart in single-chart view (ChartAside).
 * A data caveat (values not drawn, rows pinned to an edge) is always shown; `explain` marks a note for the reader
 * (method, provenance, how to read it), shown only with Explain on (the header toggle or the chart's own).
 */
export function Note({ explain, children }: { explain?: boolean; children: ReactNode }) {
  return <ChartAside><p className={explain ? 'chart-note legend-note' : 'chart-note'}>{children}</p></ChartAside>
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="frame-empty">{children}</p>
}

/**
 * Swaps the two axis fields.
 *
 * Not cosmetic: OLS of y-on-x is not symmetric, so swapping changes the
 * reported slope (the ETAD Fabs/MAC-vs-EC pair reads 1.898 one way and 0.402
 * the other — the same fit, two different questions). Deming is symmetric at
 * lambda=1, so watching which number moves tells you which one you meant.
 */
export function SwapButton({ onClick, title }: { onClick: () => void; title?: string }) {
  return (
    <button type="button" className="btn" onClick={onClick} title={title ?? 'Swap the x and y axes'} aria-label="Swap x and y axes">
      ⇅ swap
    </button>
  )
}

/**
 * An ⓘ that reveals a note on hover or keyboard focus. The body stays open
 * while the pointer is over it, so links inside remain clickable. Off unless the
 * viewer turns on Chart notes (Display and data → Interface).
 */
export function InfoTip({ children, label = 'Notes on this chart' }: { children: ReactNode; label?: string }) {
  const tip = useFloatingTip('start')
  // hidden unless the viewer turns on "Chart notes" (lib/prefs `notes`, off by default)
  const { notes } = usePrefs()
  if (!notes) return null
  return (
    <span className="info-tip" tabIndex={0} aria-label={label} aria-describedby={tip.place ? tip.id : undefined}
      onMouseEnter={e => tip.show(e.currentTarget)} onMouseLeave={tip.closeSoon}
      onFocus={e => tip.show(e.currentTarget)} onBlur={tip.closeSoon}
      onClick={e => tip.show(e.currentTarget)} onKeyDown={e => { if (e.key === 'Escape') { e.stopPropagation(); tip.close() } }}>
      <span className="info-tip-icon" aria-hidden="true">i</span>
      {tip.body(children)}
    </span>
  )
}

/**
 * A note's body floats on the page (portalled to <body>, position: fixed) and is kept inside the window:
 * a panel, a scrolling tile, a drawer or the window edge can never cut it off, the way a body positioned
 * inside the panel was cut off at a phone's edge. `align`: the body starts at the icon, or ends at it.
 */
function useFloatingTip(align: 'start' | 'end') {
  const id = useId()
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const trigger = useRef<HTMLElement | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const width = align === 'start' ? 420 : 360
  // placed by the shared floating layer (hooks/useFloatingMenu: flips above when there is no room below,
  // follows the visual viewport); no height cap but the window's
  const place = useFloatingMenu(anchor !== null, trigger, box, Infinity, { width, align })
  const closeTimer = useRef<ReturnType<typeof setTimeout>>()
  const cancelClose = () => clearTimeout(closeTimer.current)
  const close = () => { cancelClose(); setAnchor(null) }
  const closeSoon = () => { cancelClose(); closeTimer.current = setTimeout(() => setAnchor(null), 120) }
  useEffect(() => {
    // a note is a passing hint: scrolling or resizing the page closes it rather than carrying it along
    if (!anchor) return
    const shut = (e: Event) => { if (!box.current?.contains(e.target as Node)) setAnchor(null) }
    window.addEventListener('resize', shut)
    window.addEventListener('scroll', shut, true)
    return () => { window.removeEventListener('resize', shut); window.removeEventListener('scroll', shut, true) }
  }, [anchor])
  useEffect(() => cancelClose, [])
  const show = (el: HTMLElement) => { cancelClose(); trigger.current = el; setAnchor(el) }
  const body = (children: ReactNode) => anchor && createPortal(<div ref={box} id={id} role="tooltip" className="info-tip-body floating"
    style={place ?? { position: 'fixed', visibility: 'hidden', width }}
    onMouseEnter={cancelClose} onMouseLeave={closeSoon}>{children}</div>, document.body)
  return { id, place, show, close, closeSoon, body }
}

/**
 * A button in a tile's corner whose box floats beside it on hover, focus or click, placed like InfoTip's body (inside
 * the window, flipping above when there is no room below), so a small tile keeps its height for the chart:
 *   notes  ⚠  the chart's notes (ChartAside: values not drawn, filters left out, readouts), instead of under the plot
 *   info   ⓘ  its fit statistics (RegressionOverlay), instead of above the plot, then the chart's `tip`
 * The box is always mounted, since the chart portals into its slot (`onBox`), and hidden while closed. The button
 * shows only while there is something to read (an explain-only note hidden by the tile's explain setting does not count).
 */
function TileCorner({ kind, onBox, children }: { kind: 'notes' | 'info'; onBox: (el: HTMLElement | null) => void; children?: ReactNode }) {
  const id = useId()
  const [open, setOpen] = useState(false)
  const [count, setCount] = useState(0)
  const trigger = useRef<HTMLButtonElement>(null)
  const box = useRef<HTMLDivElement | null>(null)
  const slot = useRef<HTMLDivElement | null>(null)
  const setSlot = useCallback((el: HTMLDivElement | null) => { slot.current = el; onBox(el) }, [onBox])
  const place = useFloatingMenu(open, trigger, box, Infinity, { width: TILE_CORNER_WIDTH, align: 'end' })
  const closeTimer = useRef<ReturnType<typeof setTimeout>>()
  const cancelClose = () => clearTimeout(closeTimer.current)
  const show = () => { cancelClose(); setOpen(true) }
  const close = () => { cancelClose(); setOpen(false) }
  const closeSoon = () => { cancelClose(); closeTimer.current = setTimeout(() => setOpen(false), 120) }
  const fixed = !!children
  useEffect(() => {
    const el = slot.current
    if (!el) return
    const check = () => setCount((fixed ? 1 : 0) + [...el.children].filter((c) => getComputedStyle(c).display !== 'none').length)
    check()
    const mo = new MutationObserver(check)
    mo.observe(el, { childList: true, subtree: true, characterData: true })
    const panel = el.closest('section')
    if (panel) mo.observe(panel, { attributes: true, attributeFilter: ['data-chart-explain'] })
    return () => mo.disconnect()
  }, [fixed])
  useEffect(() => { if (!count) setOpen(false) }, [count])
  useEffect(() => {
    if (!open) return
    const shut = (e: Event) => { if (!box.current?.contains(e.target as Node)) setOpen(false) }
    window.addEventListener('resize', shut)
    window.addEventListener('scroll', shut, true)
    return () => { window.removeEventListener('resize', shut); window.removeEventListener('scroll', shut, true) }
  }, [open])
  useEffect(() => cancelClose, [])
  const label = kind === 'notes' ? `${count} note${count === 1 ? '' : 's'} on this chart` : 'Fit statistics and notes on this chart'
  return <>
    {count > 0 && <button ref={trigger} type="button" className={`tile-corner-btn ${kind}`} data-export="skip" aria-label={label} aria-expanded={open}
      aria-describedby={open ? id : undefined}
      onMouseEnter={show} onMouseLeave={closeSoon} onFocus={show} onBlur={closeSoon}
      onClick={() => (open ? close() : show())} onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); close() } }}>
      {kind === 'notes'
        ? <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
          <path d="M8 1.8 15 14.2H1Z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
          <path d="M8 6.2v3.6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="11.9" r="0.85" fill="currentColor" />
        </svg>
        : <span aria-hidden="true">i</span>}
    </button>}
    <div ref={box} id={id} role="tooltip" data-export="skip" className={`info-tip-body floating tile-corner ${kind}${open ? ' open' : ''}`}
      style={open && place ? place : { position: 'fixed', left: 0, top: 0, width: TILE_CORNER_WIDTH, visibility: 'hidden', pointerEvents: 'none' }}
      onMouseEnter={cancelClose} onMouseLeave={closeSoon}>
      <div ref={setSlot} className="tile-corner-slot" />
      {children && <div className="tile-corner-tip">{children}</div>}
    </div>
  </>
}

/** Background prose the reader can do without: rendered only when the header's "explain" toggle is on. */
export function ExplainOnly({ children }: { children: ReactNode }) {
  const { explain } = usePrefs()
  return explain ? <>{children}</> : null
}
