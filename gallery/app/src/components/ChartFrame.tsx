import { Combobox } from '@/components/Combobox'
import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { createPortal } from 'react-dom'
import { createContext, useId, useContext, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { ExportButton } from '@/components/ExportDialog'
import { EXPLAIN_AVAILABLE, usePrefs } from '@/lib/prefs'
import { setSingleChartDisplay, useChartDisplayPrefs } from '@/lib/chartDisplayPrefs'
import { chartInkStyle } from '@/lib/chartDisplayStyles'
import { useSystemDark } from '@/lib/colorTheme'
import { useParamScope } from '@/lib/url'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'
import { AsideSlotCtx, ChartAside, SingleViewNav, SingleViewStats, useSingleView } from '@/lib/singleView'
import { useExportRender } from '@/lib/exportRender'

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
}) {
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
  const allControls = (controls || extras?.controls) ? <>{extras?.controls}{controls}</> : null
  // a paste-able pointer for development: which chart, where its code lives, and the exact view
  const copyRef = () => {
    const text = [`Gallery chart: "${short}"${id ? ` (#${id})` : ''}`,
      `Source: ${source ?? 'search the title in gallery/app/src'}`,
      `View: ${window.location.href}`].join('\n')
    navigator.clipboard?.writeText(text).then(() => { setRefCopied(true); setTimeout(() => setRefCopied(false), 1400) })
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
  const side = own?.controlsSide ?? (inView ? single.side : display.controlsSide)
  const info = EXPLAIN_AVAILABLE && (own?.explain ?? explain)
  const readable = (own?.density ?? density) === 'readable'
  const systemDark = useSystemDark()
  const theme = own?.theme === 'auto' ? systemDark ? 'dark' : 'light' : own?.theme
  const appearance = {
    ...(theme ? chartInkStyle(theme) : {}),
    '--text-scale': readable ? 1.15 : 1,
    '--chart-text-scale': readable ? 1.12 : 1,
  } as CSSProperties
  const displayAttrs = { 'data-chart-key': chartKey, 'data-chart-title': short, 'data-chart-theme': theme, 'data-chart-explain': info ? '1' : '0', style: appearance }
  // focus mode: this chart alone, centred over the page with its options; `focus=<key>` in the URL, so a copied link
  // reopens it exactly as it is (every other setting already lives in the URL)
  const focusKey = anchor ?? slug(short)
  const [focused, setFocused] = useFocus(focusKey)
  const [exportAsk, setExportAsk] = useState(0)
  if (compact) return (
    <section className="panel compact" ref={ref} id={anchor} {...displayAttrs}>
      <TileMenu name={short} extras={extras!} about={hasInfo ? <>{rest && <p>{rest}</p>}{about && <div className="frame-about">{about}</div>}{subtitle && <p className="subtitle">{subtitle}</p>}{provenance && <p className="provenance">{provenance}</p>}</> : null}
        tip={tip} controls={side === 'top' ? allControls : null} export={exportable ? <ExportButton panelRef={ref} title={short} /> : null} onCopyRef={copyRef} copied={refCopied} />
      {info && hasInfo && <div className="compact-explanation"><strong>{short}</strong>{about}{subtitle && <p className="subtitle">{subtitle}</p>}{provenance && <p className="provenance">{provenance}</p>}</div>}
      <div className={`frame-body controls-${side}`}>
        {side !== 'top' && allControls && <div className="frame-controls">{allControls}</div>}
        <div className="frame-chart">{children}</div>
      </div>
    </section>
  )
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
          <RefMenu onCopyRef={copyRef} refCopied={refCopied} link={() => focusLink(focusKey)} />
          {/* the whole figure (every facet and the legend), not just the first <svg> */}
          {exportable && <ExportButton panelRef={ref} title={short} ask={exportAsk} />}
        </span>
        <FrameMoreMenu link={() => focusLink(focusKey)} onCopyRef={copyRef}
          explain={hasInfo && EXPLAIN_AVAILABLE ? { on: info, toggle: () => setSingleChartDisplay(chartKey, { explain: !info }) } : null}
          onExport={exportable ? () => setExportAsk((n) => n + 1) : null} />
        {focused
          ? <button type="button" className="btn quiet" title="Back to the page (Esc)" onClick={() => setFocused(false)}>Close ✕</button>
          : <button type="button" className="btn quiet" title="Focus: this chart alone, centred, with its options" aria-label="Focus on this chart" onClick={() => setFocused(true)}><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg></button>}
      </div>
    </header>
  )
  return (<>
    {focused && <div className="focus-scrim" onClick={() => setFocused(false)} />}
    <section className={`panel${focused ? ' focused' : ''}`} ref={ref} id={anchor} data-frame data-frame-title={short} data-frame-key={focusKey} {...displayAttrs}>{/* data-frame: a chart the single-chart view steps through (lib/singleView) */}
      {!headInColumn && head}
      {/* the fit statistics in single-chart view: at the head of an options column, else above the options row */}
      {inView && (!controls || side === 'top') && <SingleViewStats />}
      {/* the same elements whichever side the options are on: a different tree would remount the chart's
          own wrapper under a chart that is not remounted, and its size observer would keep the old width */}
      <div className={`frame-body controls-${controls ? side : 'top'}`}>
        {controls && <div className="frame-controls">{headInColumn && head}{inView && side !== 'top' && <SingleViewStats />}{controls}</div>}
        <div className="frame-chart"><AsideSlotCtx.Provider value={asideSlot}>{children}</AsideSlotCtx.Provider></div>
      </div>
    </section>
  </>)
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48)
const FOCUS_EVENT = 'gallery-focus'

/** a link that opens the gallery on one chart in focus mode, with every setting as it is now */
export function focusLink(key: string) {
  const url = new URL(window.location.href)
  url.searchParams.set('focus', key)
  return url.toString()
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
function FrameMoreMenu({ link, onCopyRef, explain, onExport }: {
  link: () => string; onCopyRef: () => void; explain: { on: boolean; toggle: () => void } | null; onExport: (() => void) | null
}) {
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
        <button type="button" role="menuitem" onClick={() => navigator.clipboard?.writeText(link()).then(() => done('link'))}>
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
function RefMenu({ onCopyRef, refCopied, link }: { onCopyRef: () => void; refCopied: boolean; link: () => string }) {
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
  const copyLink = () => navigator.clipboard?.writeText(link()).then(() => {
    setLinkCopied(true)
    setTimeout(() => { setLinkCopied(false); setOpen(false) }, 1200)
  })
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
function TileMenu({ name, extras, about, tip, controls, export: exportNode, onCopyRef, copied }: {
  name: string; extras: FrameExtras; about: ReactNode; tip?: ReactNode; controls: ReactNode; export: ReactNode; onCopyRef: () => void; copied: boolean
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
  const toggle = (btn: HTMLElement) => {
    if (open) { setOpen(false); return }
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
      {tip && <FloatingInfoTip>{tip}</FloatingInfoTip>}
      <button type="button" className="tile-menu-btn" aria-expanded={open} aria-label={`${name}: menu`} title={name}
        onClick={(e) => toggle(e.currentTarget)}>⋯</button>
      {open && <div className="tile-menu" role="dialog" aria-label={name} style={place}>
        <div className="tile-menu-title"><strong>{name}</strong>{extras.subtitle && <span className="frame-sub">{extras.subtitle}</span>}</div>
        <div className="tile-menu-actions">
          {extras.menu?.(close)}
          {exportNode}
          <button type="button" className="btn quiet" onClick={onCopyRef} title="Copy a reference to this chart (title, source file, view URL)">{copied ? 'Copied' : 'Ref'}</button>
        </div>
        {about && <details className="tile-about"><summary>About this chart</summary>{about}</details>}
        {controls && <div className="frame-controls">{controls}</div>}
      </div>}
    </div>
  )
}

export function Select({
  label,
  value,
  options,
  onChange,
  title,
  optionLabel,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
  title?: string
  /** display text per option when the value is a code (e.g. a site code shown by name) */
  optionLabel?: (o: string) => string
}) {
  const method = useEstimator()
  return <Combobox label={label} value={value} title={title} onChange={onChange}
    options={options.map((o) => ({ key: o, label: estimatorText(optionLabel ? optionLabel(o) : o, method) }))} />
}

/**
 * Sentence case for option text: capitalise the first word unless it is a single
 * letter (x, y, n) or already starts with a capital, digit or symbol. Display
 * only; the option values (often URL params) stay as written.
 */
export const sentence = (s: string) => /^[a-z]{2}/.test(s) ? s[0].toUpperCase() + s.slice(1) : s

/**
 * Segmented control: for 2–5 mutually exclusive options that deserve to be visible at once (more
 * than five, or long labels: Select). Labels follow the page's estimator ('Deming' reads 'OLS'
 * under OLS), as Select's do.
 */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  title,
  optionLabel,
}: {
  label?: string
  value: T
  options: readonly T[]
  onChange: (v: T) => void
  title?: string
  /** display text per option; defaults to the option in sentence case */
  optionLabel?: (o: T) => string
}) {
  const method = useEstimator()
  const text = (o: T) => estimatorText(optionLabel ? optionLabel(o) : sentence(o), method)
  // too long for one row in a column beside the chart: stacked there (styles.css .seg.long); four or more
  // options wrap into a broken grid well before 28 characters
  const chars = options.reduce((n, o) => n + text(o).length, 0)
  const long = chars > 28 || (options.length >= 4 && chars > 20)
  return (
    <span className="control" title={title}>
      {label}
      <span className={`seg${long ? ' long' : ''}`} role="radiogroup">
        {options.map((o) => (
          <button key={o} type="button" role="radio" aria-checked={o === value} className={o === value ? 'active' : ''} onClick={() => onChange(o)}>
            {text(o)}
          </button>
        ))}
      </span>
    </span>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
  title,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  title?: string
}) {
  return (
    <label className="toggle" title={title}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

/** A data caveat under a chart (values not drawn, rows pinned to an edge). Always shown: it is about the data, not the reader. */
/** a note on the chart: under it, or beside its options when it is the chart in single-chart view (ChartAside) */
export function Note({ children }: { children: ReactNode }) {
  return <ChartAside><p className="chart-note">{children}</p></ChartAside>
}

/** A hint for the reader; shown only when explanations are on. */
export function Hint({ children }: { children: ReactNode }) {
  const { explain } = usePrefs()
  return explain ? <span className="control hint">{children}</span> : null
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
 * while the pointer is over it, so links inside remain clickable.
 */
export function InfoTip({ children, label = 'Notes on this chart' }: { children: ReactNode; label?: string }) {
  const tip = useFloatingTip('start')
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
  const [place, setPlace] = useState<React.CSSProperties | null>(null)
  const closeTimer = useRef<ReturnType<typeof setTimeout>>()
  const cancelClose = () => clearTimeout(closeTimer.current)
  const close = () => { cancelClose(); setPlace(null) }
  const closeSoon = () => { cancelClose(); closeTimer.current = setTimeout(() => setPlace(null), 120) }
  useEffect(() => {
    const shut = () => setPlace(null)
    window.addEventListener('resize', shut)
    window.addEventListener('scroll', shut, true)
    return () => { cancelClose(); window.removeEventListener('resize', shut); window.removeEventListener('scroll', shut, true) }
  }, [])
  const show = (el: HTMLElement) => {
    cancelClose()
    const r = el.getBoundingClientRect()
    const width = Math.min(align === 'start' ? 420 : 360, window.innerWidth - 24)
    const want = align === 'start' ? r.left - 8 : r.right - width
    const below = window.innerHeight - r.bottom > 160
    setPlace({ position: 'fixed', width, left: Math.max(12, Math.min(want, window.innerWidth - width - 12)),
      ...(below ? { top: r.bottom + 6, bottom: 'auto', maxHeight: window.innerHeight - r.bottom - 18 }
        : { top: 'auto', bottom: window.innerHeight - r.top + 6, maxHeight: r.top - 18 }) })
  }
  const body = (children: ReactNode) => place && createPortal(<div id={id} role="tooltip" className="info-tip-body floating" style={place}
    onMouseEnter={cancelClose} onMouseLeave={closeSoon}>{children}</div>, document.body)
  return { id, place, show, close, closeSoon, body }
}

/** A tile's corner note: the same floating body as InfoTip, from a button in the tile's corner. */
function FloatingInfoTip({ children }: { children: ReactNode }) {
  const tip = useFloatingTip('end')
  return <>
    <button type="button" className="tile-info-btn info-tip-icon" aria-label="Notes on this chart" aria-describedby={tip.place ? tip.id : undefined}
      onMouseEnter={e => tip.show(e.currentTarget)} onMouseLeave={tip.closeSoon}
      onFocus={e => tip.show(e.currentTarget)} onBlur={tip.closeSoon}
      onClick={e => tip.show(e.currentTarget)} onKeyDown={e => { if (e.key === 'Escape') { e.stopPropagation(); tip.close() } }}>i</button>
    {tip.body(children)}
  </>
}

/** Background prose the reader can do without: rendered only when the header's "explain" toggle is on. */
export function ExplainOnly({ children }: { children: ReactNode }) {
  const { explain } = usePrefs()
  return explain ? <>{children}</> : null
}
