import { Combobox } from '@/components/Combobox'
import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { createPortal } from 'react-dom'
import { createContext, useId, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { ExportButton } from '@/components/ExportDialog'
import { usePrefs } from '@/lib/prefs'
import { useParamScope } from '@/lib/url'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'

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
  const { explain } = usePrefs()
  const [info, setInfo] = useState(explain)
  useEffect(() => setInfo(explain), [explain])
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
  const [side, cycleSide] = useControlsSide(id ?? short)
  // focus mode: this chart alone, centred over the page with its options; `focus=<key>` in the URL, so a copied link
  // reopens it exactly as it is (every other setting already lives in the URL)
  const focusKey = anchor ?? slug(short)
  const [focused, setFocused] = useFocus(focusKey)
  if (compact) return (
    <section className="panel compact" ref={ref} id={anchor}>
      <TileMenu name={short} extras={extras!} about={hasInfo ? <>{rest && <p>{rest}</p>}{about && <div className="frame-about">{about}</div>}{subtitle && <p className="subtitle">{subtitle}</p>}{provenance && <p className="provenance">{provenance}</p>}</> : null}
        tip={tip} controls={allControls} export={exportable ? <ExportButton panelRef={ref} title={short} /> : null} onCopyRef={copyRef} copied={refCopied} />
      {children}
    </section>
  )
  return (<>
    {focused && <div className="focus-scrim" onClick={() => setFocused(false)} />}
    <section className={`panel${focused ? ' focused' : ''}`} ref={ref} id={anchor}>
      <header className="frame-head">
        <div>
          <h2>
            {short}
            {info && rest && <span className="title-rest"> — {rest}</span>}
            {tip && <InfoTip>{tip}</InfoTip>}
          </h2>
          {info && about && <div className="frame-about">{about}</div>}
          {info && subtitle && <p className="subtitle">{subtitle}</p>}
          {info && provenance && <p className="provenance">{provenance}</p>}
        </div>
        <div className="frame-actions">
          {controls && <ControlsSideButton side={side} onClick={cycleSide} />}
          {hasInfo && (
            <button type="button" className={`btn quiet ${info ? 'on' : ''}`} title={info ? 'Hide the explanation' : 'What this chart shows and where it comes from'} aria-pressed={info} onClick={() => setInfo((v) => !v)}>
              ?
            </button>
          )}
          <RefMenu onCopyRef={copyRef} refCopied={refCopied} link={() => focusLink(focusKey)} focused={focused} onFocus={() => setFocused(true)} />
          {/* the whole figure (every facet and the legend), not just the first <svg> */}
          {exportable && <ExportButton panelRef={ref} title={short} />}
          {focused
            ? <button type="button" className="btn quiet" title="Back to the page (Esc)" onClick={() => setFocused(false)}>Close ✕</button>
            : <button type="button" className="btn quiet" title="Focus: this chart alone, centred, with its options" aria-label="Focus on this chart" onClick={() => setFocused(true)}><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg></button>}
        </div>
      </header>
      {/* the same elements whichever side the options are on: a different tree would remount the chart's
          own wrapper under a chart that is not remounted, and its size observer would keep the old width */}
      <div className={`frame-body controls-${controls ? side : 'top'}`}>
        {controls && <div className="frame-controls">{controls}</div>}
        <div className="frame-chart">{children}</div>
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

/** "Ref ▾": share this chart as it is now, open it in focus mode, or copy the development reference */
function RefMenu({ onCopyRef, refCopied, link, focused, onFocus }: {
  onCopyRef: () => void; refCopied: boolean; link: () => string; focused: boolean; onFocus: () => void
}) {
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
        {!focused && <button type="button" role="menuitem" onClick={() => { setOpen(false); onFocus() }}>
          <strong>Focus on this chart</strong><span>Alone and centred, with its options (Esc to leave)</span>
        </button>}
        <button type="button" role="menuitem" onClick={() => { onCopyRef(); setOpen(false) }}>
          <strong>Copy reference</strong><span>Title, source file and view URL, for development</span>
        </button>
      </div>, document.body)}
    </div>
  )
}

/**
 * Where a chart's options sit: above it (the default), or in a column to its right or left, for a
 * tall chart on a wide screen. Per chart and per viewer (localStorage), a convenience, not view state:
 * it stays out of the URL. Under 820px wide the column falls back to the row above (styles.css).
 */
type ControlsSide = 'top' | 'right' | 'left'
const SIDES: ControlsSide[] = ['top', 'right', 'left']
const SIDE_KEY = 'aeth-gallery:controls-side'
const readSides = (): Record<string, ControlsSide> => { try { return JSON.parse(localStorage.getItem(SIDE_KEY) ?? '{}') } catch { return {} } }

function useControlsSide(key: string): [ControlsSide, () => void] {
  const [side, setSide] = useState<ControlsSide>(() => readSides()[key] ?? 'top')
  const cycle = () => {
    const next = SIDES[(SIDES.indexOf(side) + 1) % SIDES.length]
    setSide(next)
    try {
      const all = readSides()
      if (next === 'top') delete all[key]; else all[key] = next
      localStorage.setItem(SIDE_KEY, JSON.stringify(all))
    } catch { /* blocked storage: the choice still holds until reload */ }
  }
  return [side, cycle]
}

const SIDE_NEXT: Record<ControlsSide, string> = { top: 'to the right of the chart', right: 'to the left of the chart', left: 'back above the chart' }

/** A card outline with the options' place filled in; a click moves them on (top → right → left). */
function ControlsSideButton({ side, onClick }: { side: ControlsSide; onClick: () => void }) {
  const bar = side === 'top' ? { x: 2, y: 2, width: 12, height: 3 } : { x: side === 'right' ? 10 : 2, y: 2, width: 4, height: 12 }
  return (
    <button type="button" className={`btn quiet ${side !== 'top' ? 'on' : ''}`} onClick={onClick}
      title={`Options ${side === 'top' ? 'above' : `to the ${side} of`} the chart; click to put them ${SIDE_NEXT[side]}`}
      aria-label={`Move the options ${SIDE_NEXT[side]}`}>
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" style={{ display: 'block' }}>
        <rect x="1" y="1" width="14" height="14" rx="2" fill="none" stroke="currentColor" strokeWidth="1.3" />
        <rect {...bar} rx="0.8" fill="currentColor" />
      </svg>
    </button>
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
export function Note({ children }: { children: ReactNode }) {
  return <p className="chart-note">{children}</p>
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
  return (
    <span className="info-tip" tabIndex={0} aria-label={label}>
      <span className="info-tip-icon" aria-hidden="true">i</span>
      <span className="info-tip-body" role="tooltip">{children}</span>
    </span>
  )
}

/** A corner note is portalled outside the scrolling tile, so it cannot be clipped. */
function FloatingInfoTip({ children }: { children: ReactNode }) {
  const id = useId()
  const [place, setPlace] = useState<React.CSSProperties | null>(null)
  const closeTimer = useRef<ReturnType<typeof setTimeout>>()
  const cancelClose = () => clearTimeout(closeTimer.current)
  const closeSoon = () => { cancelClose(); closeTimer.current = setTimeout(() => setPlace(null), 120) }
  useEffect(() => {
    const close = () => setPlace(null)
    window.addEventListener('resize', close)
    window.addEventListener('scroll', close, true)
    return () => { cancelClose(); window.removeEventListener('resize', close); window.removeEventListener('scroll', close, true) }
  }, [])
  const show = (el: HTMLElement) => {
    cancelClose()
    const r = el.getBoundingClientRect()
    const width = Math.min(360, window.innerWidth - 24)
    const below = window.innerHeight - r.bottom > 160
    setPlace({ position: 'fixed', width, left: Math.max(12, Math.min(r.right - width, window.innerWidth - width - 12)),
      ...(below ? { top: r.bottom + 6, maxHeight: window.innerHeight - r.bottom - 18 } : { bottom: window.innerHeight - r.top + 6, maxHeight: r.top - 18 }) })
  }
  return <>
    <button type="button" className="tile-info-btn info-tip-icon" aria-label="Notes on this chart" aria-describedby={place ? id : undefined}
      onMouseEnter={e => show(e.currentTarget)} onMouseLeave={closeSoon}
      onFocus={e => show(e.currentTarget)} onBlur={closeSoon}
      onClick={e => show(e.currentTarget)} onKeyDown={e => { if (e.key === 'Escape') { e.stopPropagation(); cancelClose(); setPlace(null) } }}>i</button>
    {place && createPortal(<div id={id} role="tooltip" className="info-tip-body floating" style={place}
      onMouseEnter={cancelClose} onMouseLeave={closeSoon}>{children}</div>, document.body)}
  </>
}

/** Background prose the reader can do without: rendered only when the header's "explain" toggle is on. */
export function ExplainOnly({ children }: { children: ReactNode }) {
  const { explain } = usePrefs()
  return explain ? <>{children}</> : null
}
