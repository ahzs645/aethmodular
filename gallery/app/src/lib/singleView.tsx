import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Combobox } from '@/components/Combobox'
import { StepButton } from '@/components/StepButton'
import { FrameFit } from '@/lib/chartFit'
import { useExportRender } from '@/lib/exportRender'
import { useSearchParam } from '@/lib/url'
import type { ControlsSide } from '@/lib/chartDisplay'

/**
 * Single-chart view: the page shows one chart at a time, filling the window, and ‹ › (in the chart's
 * title line, or ← → on the keyboard) step through the page's charts in order. The title itself becomes
 * a searchable picker of every chart on the page, and the regression summary is drawn just above the
 * chart's options instead of above its axes. The sidebar stays as it was: when
 * it is open, the page sits beside it.
 *
 * The charts are found in the page itself: every ChartFrame marks its panel `data-frame`, and
 * the provider hides everything else under the page root (Shell's `data-single-root`) with
 * `data-single-hidden`, so no page has to list its charts. Remembered in this browser, like
 * the other display toggles; the Dashboard has its own presenting mode and leaves this off.
 */
interface SingleView {
  on: boolean
  setOn: (v: boolean) => void
  /** whether this tab offers the view (not the Dashboard) */
  available: boolean
  /** the chart in view, and where it is in the page's order */
  active: HTMLElement | null
  index: number
  count: number
  title: string
  /** every chart on the page, for the picker: its name and the section it sits under */
  charts: { title: string; group?: string }[]
  go: (by: number) => void
  jump: (i: number) => void
  /** the stats box above the chart's options, where the active chart's regression summary goes */
  slot: HTMLElement | null
  setSlot: (n: HTMLElement | null) => void
  /** where the chart in view has its options (a chart's own placement still wins) */
  side: ControlsSide
  setSide: (v: ControlsSide) => void
}

const Ctx = createContext<SingleView>({ on: false, setOn: () => {}, available: false, active: null, index: 0, count: 0, title: '', charts: [], go: () => {}, jump: () => {}, slot: null, setSlot: () => {}, side: 'right', setSide: () => {} })

/**
 * The on/off switch, kept in App so the provider can turn it off on the Dashboard. In the link (`view=single`),
 * with the chart in view (`chart=`, its frame key) and where its options sit (`side=`), so a shared link opens
 * on the same chart laid out the same way.
 */
export function useSingleViewPref(): [boolean, (v: boolean) => void] {
  const [view, setView] = useSearchParam<'page' | 'single'>('view', 'page', ['page', 'single'])
  return [view === 'single', useCallback((v: boolean) => setView(v ? 'single' : 'page'), [setView])]
}

/** a query parameter written beside the ones useSearchParam keeps (null removes it) */
function setQuery(name: string, value: string | null) {
  const url = new URL(window.location.href)
  if ((url.searchParams.get(name) ?? null) === value) return
  if (value === null) url.searchParams.delete(name)
  else url.searchParams.set(name, value)
  window.history.replaceState(null, '', url)
  window.dispatchEvent(new Event('gallery:querychange'))
}

/** left in view beside the chart: the sidebar, its scrim, drawers and the brush bar */
const KEEP = '.side-dock, .sheet-scrim, .drawer, .brush-bar, .focus-scrim'
const MARKS = ['data-single-hidden', 'data-single-path', 'data-single-active']

/** the page's charts in reading order: frames outside the sidebar and drawers, not inside another frame */
const framesIn = (root: Element) => [...root.querySelectorAll<HTMLElement>('section[data-frame]')]
  .filter((el) => !el.closest('.side-dock, .drawer') && !el.parentElement?.closest('[data-frame]'))

/** each chart's section: the last page heading before it (outside any chart), on pages that have them */
function sectionsOf(root: Element, frames: HTMLElement[]) {
  const heads = [...root.querySelectorAll<HTMLElement>('h1, h2, h3')].filter((h) => !h.closest('[data-frame], .side-dock, .drawer'))
  return frames.map((f) => {
    let last: string | undefined
    for (const h of heads) if (h.compareDocumentPosition(f) & Node.DOCUMENT_POSITION_FOLLOWING) last = h.textContent?.trim() || last
    return last
  })
}

function clearMarks() {
  for (const el of document.querySelectorAll(MARKS.map((m) => `[${m}]`).join(','))) for (const m of MARKS) el.removeAttribute(m)
}

/** hide every branch beside the one that holds `el`, from the page root down */
function mark(root: Element, el: HTMLElement) {
  el.setAttribute('data-single-active', '')
  for (let n: HTMLElement | null = el; n && n !== root; n = n.parentElement) {
    if (n !== el) n.setAttribute('data-single-path', '')
    if (n instanceof HTMLDetailsElement) n.open = true
    for (const sib of n.parentElement?.children ?? []) if (sib !== n && !sib.matches(KEEP)) sib.setAttribute('data-single-hidden', '')
  }
}

export function SingleViewProvider({ on: wanted, setOn, available, page, children }: { on: boolean; setOn: (v: boolean) => void; available: boolean; page: string; children: ReactNode }) {
  const on = wanted && available
  const [frames, setFrames] = useState<HTMLElement[]>([])
  const [index, setIndex] = useState(0)
  const [slot, setSlot] = useState<HTMLElement | null>(null)
  const want = useRef<HTMLElement | null>(null)
  const at = useRef(0)
  const [side, setSide] = useSearchParam<ControlsSide>('side', 'right', ['right', 'left', 'top'])
  // the link's chart (`chart=`, or a `focus=` link's): opened once its frame has drawn
  const asked = useRef<string | null>(null)
  const shown = useRef<string | null>(null)
  // read once, as the link arrived: the chart in view is written back to `chart=` as soon as one shows
  const [linked] = useState(() => { const q = new URL(window.location.href).searchParams; return q.get('chart') ?? q.get('focus') })
  useEffect(() => {
    // the link's chart on the page the link opened (a re-run for the same page keeps it); a new tab or meeting
    // starts at its first chart
    if (shown.current === null || shown.current === page) asked.current = linked
    else asked.current = null
    shown.current = page
    want.current = null; at.current = 0; setIndex(0)
  }, [page])

  useEffect(() => {
    document.documentElement.dataset.single = on ? '1' : '0'
    if (!on) { clearMarks(); setFrames([]); setQuery('chart', null); return }
    let raf = 0
    const apply = () => {
      raf = 0
      // looked up each time: the page root is replaced when the data finishes loading
      const root = document.querySelector('[data-single-root]')
      if (!root) return
      const list = framesIn(root)
      if (asked.current) {
        const found = list.find((f) => f.dataset.frameKey === asked.current)
        if (found) { want.current = found; asked.current = null }
      }
      let i = want.current ? list.indexOf(want.current) : -1
      if (i < 0) i = Math.max(0, Math.min(at.current, list.length - 1))
      const el = list[i] ?? null
      // the page re-rendered without moving: nothing to redo (attributes are not observed, so no loop)
      if (!(el && el.hasAttribute('data-single-active') && el === want.current)) {
        clearMarks()
        if (el) mark(root, el)
      }
      want.current = el
      at.current = i
      setIndex(i)
      setFrames((prev) => prev.length === list.length && prev.every((f, k) => f === list[k]) ? prev : list)
    }
    apply()
    // charts mount, load and re-key as the page fills in
    const mo = new MutationObserver(() => { if (!raf) raf = requestAnimationFrame(apply) })
    mo.observe(document.body, { childList: true, subtree: true })
    return () => { mo.disconnect(); if (raf) cancelAnimationFrame(raf) }
  }, [on, page])
  useEffect(() => () => { clearMarks(); delete document.documentElement.dataset.single }, [])

  const jump = useCallback((to: number) => {
    const root = document.querySelector('[data-single-root]')
    if (!root) return
    const list = framesIn(root)
    if (!list.length) return
    const i = (to + list.length) % list.length
    // a focus link's overlay belongs to the chart it opened on, not the next one
    if (new URL(window.location.href).searchParams.has('focus')) { setQuery('focus', null); window.dispatchEvent(new Event('gallery-focus')) }
    clearMarks()
    mark(root, list[i])
    want.current = list[i]
    at.current = i
    setIndex(i)
    setFrames(list)
    window.scrollTo({ top: 0 })
  }, [])
  const go = useCallback((by: number) => jump(at.current + by), [jump])

  // ← → and Page Up / Down step, unless a control has the keys
  useEffect(() => {
    if (!on) return
    const key = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return
      if ((e.target as Element)?.closest?.('input, textarea, select, [contenteditable], [role="listbox"], [role="combobox"], [role="slider"], [role="menu"], [role="dialog"], .drawer')) return
      if (['ArrowRight', 'PageDown'].includes(e.key)) { e.preventDefault(); go(1) }
      else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(-1) }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [on, go])

  const active = on ? frames[index] ?? null : null
  // the chart in view, in the link
  useEffect(() => { if (active?.dataset.frameKey) setQuery('chart', active.dataset.frameKey) }, [active])
  const charts = useMemo(() => {
    const root = document.querySelector('[data-single-root]')
    const groups = root ? sectionsOf(root, frames) : []
    return frames.map((f, i) => ({ title: f.dataset.frameTitle ?? `Chart ${i + 1}`, group: groups[i] }))
  }, [frames])
  const value = useMemo<SingleView>(() => ({
    on, setOn, available, active, index, count: on ? frames.length : 0,
    title: active?.dataset.frameTitle ?? '', charts, go, jump, slot, setSlot, side, setSide,
  }), [on, setOn, available, active, index, frames.length, charts, go, jump, slot, side, setSide])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useSingleView = () => useContext(Ctx)

/**
 * Around the page (App's Shell): in single-chart view, the height every chart on the page may draw
 * at, so the chart in view ends at the bottom of the window. Charts read their height above their
 * own ChartFrame, so this sits over the whole page rather than in the frame; only one chart shows.
 * Measured on the chart in view and corrected after each layout until it fits (at most six tries
 * per chart and window size, the way a Dashboard tile fits its chart); a chart that ignores the
 * number keeps its own height.
 */
export function SingleFit({ children }: { children: ReactNode }) {
  const { active } = useSingleView()
  // while the chart in view is exported (lib/exportRender) it is drawn at page size, not fitted to the window
  const render = useExportRender()
  const exporting = !!render && render.panel === active
  const [fit, setFit] = useState<number | null>(null)
  const tries = useRef(0)
  // the fitting has found its height (or given up): the zoom below waits for it, or it would scale a chart still growing
  const settled = useRef(false)
  const [size, setSize] = useState(() => window.innerHeight)
  useEffect(() => {
    if (!active) return
    const resize = () => setSize(window.innerHeight)
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [active])
  useEffect(() => { tries.current = 0; if (!active) setFit(null) }, [active, size])
  // what sits above the chart (the title and its explanation, the statistics once they arrive, the
  // options) settles after the chart: each change starts the fitting over
  const [, refit] = useState(0)
  useEffect(() => {
    if (!active) return
    const above = active.querySelectorAll(':scope > .frame-head, :scope > .single-stats, :scope > .frame-body > .frame-controls')
    const ro = new ResizeObserver(() => { tries.current = 0; refit((n) => n + 1) })
    for (const el of above) ro.observe(el)
    return () => ro.disconnect()
  }, [active])
  // a legend under a plot that leaves room beside it goes beside, stacked (styles.css [data-legend-beside]);
  // otherwise it stays under. Moving it frees height, so the plot grows into the room it left: it only goes
  // back under once there is clearly no room, and only comes beside with room to spare, so it never flips
  useEffect(() => {
    const chart = active?.querySelector('.frame-chart')
    if (!chart || exporting) return
    let raf = 0
    const place = () => {
      raf = 0
      let moved = false
      for (const legend of chart.querySelectorAll<HTMLElement>('.legend')) {
        const wrap = legend.parentElement
        if (!wrap || legend.parentElement?.closest('.legend')) continue
        const plots = [...wrap.querySelectorAll('svg')].filter((s) => !legend.contains(s) && !s.parentElement?.closest('svg'))
        if (!plots.length) continue
        const rects = plots.map((p) => p.getBoundingClientRect())
        const span = Math.max(...rects.map((r) => r.right)) - Math.min(...rects.map((r) => r.left))
        // stacked, the legend is as wide as its widest item (a Legend can hold its items in an inner .legend); a colour
        // ramp's item stacks too, its label over the ramp, which stands up beside the plot (components/ColorLegend .ramp-v)
        const width = (c: Element) => {
          const upright = c.querySelector('.ramp-v')
          if (!upright) return c.getBoundingClientRect().width
          return Math.max(Number(upright.getAttribute('width')) || 0, ...[...c.children].filter((k) => !k.querySelector('.ramp-v') && !k.classList.contains('ramp-v')).map((k) => k.getBoundingClientRect().width))
        }
        const stacked = Math.max(0, ...[...legend.querySelectorAll('.legend-item, .legend-group-name, .legend-note')].map(width))
        const beside = wrap.hasAttribute('data-legend-beside')
        const want = wrap.clientWidth - span >= stacked + 32 + (beside ? 0 : 56)
        if (want !== beside) { wrap.toggleAttribute('data-legend-beside', want); moved = true }
        // beside the whole column (a caption above the plot, notes under it), not only its first row
        if (want) wrap.style.setProperty('--legend-rows', String(Math.max(1, wrap.children.length - 1)))
      }
      if (moved) { tries.current = 0; refit((n) => n + 1) }
    }
    const later = () => { if (!raf) raf = requestAnimationFrame(place) }
    place()
    const ro = new ResizeObserver(later)
    ro.observe(chart)
    const mo = new MutationObserver(later)
    mo.observe(chart, { childList: true, subtree: true, attributes: true, attributeFilter: ['width', 'height'] })
    return () => {
      ro.disconnect(); mo.disconnect(); if (raf) cancelAnimationFrame(raf)
      for (const el of chart.querySelectorAll<HTMLElement>('[data-legend-beside]')) { el.removeAttribute('data-legend-beside'); el.style.removeProperty('--legend-rows') }
    }
  }, [active, exporting])
  useLayoutEffect(() => {
    // while an export copies the chart, its size holds (the summary back above the axes makes it taller)
    if (!active || exporting) return
    if (tries.current >= 6) { settled.current = true; return }
    const chart = active.querySelector('.frame-chart')
    if (!chart) return
    const BOTTOM = 18
    const pad = parseFloat(getComputedStyle(active).paddingBottom) || 0
    // the panel's room in the window: an options column taller than that scrolls inside it (styles.css --single-room)
    active.style.setProperty('--single-room', `${Math.max(240, Math.round(size - BOTTOM - pad - (active.getBoundingClientRect().top + window.scrollY)))}px`)
    const guess = fit === null
    // first a guess from where the chart starts, then the chart's own overflow or room left (the chart, not the
    // panel: an options column longer than the window must not shrink the plot)
    // measured unscaled: the zoom below scales a chart that cannot fit, and must not feed back into its height
    const rect = chart.getBoundingClientRect()
    const bottom = rect.top + rect.height / (Number((chart as HTMLElement).style.zoom) || 1)
    const next = Math.max(240, Math.round(guess
      ? size - rect.top - window.scrollY - BOTTOM - 40
      : fit + size - BOTTOM - pad - (bottom + window.scrollY)))
    if (!guess && Math.abs(next - fit) <= 2) { settled.current = true; return }
    settled.current = false
    tries.current += 1
    setFit(next)
  })
  // A chart that draws at its own size (a fixed height, a capped side) is scaled up to the room once the fitting
  // has settled, up to ZOOM_MAX, when it leaves both height and width spare. A chart that follows the fitted height
  // already fills it and stays at 1. Never down: a chart that lays itself out from its room (rows that fit, a width
  // it fills) would redraw against the scale. CSS zoom reflows the layout and keeps pointer positions true; it is
  // off while the chart is exported, so the file is drawn at its size.
  useEffect(() => {
    const chart = active?.querySelector<HTMLElement>('.frame-chart')
    if (!chart || exporting) return
    let timer = 0, steps = 0
    // the chart's slot, measured at zoom 1 where the chart fills it (its own width is set once it is scaled)
    let slot = 0
    const scale = () => {
      timer = 0
      if (!settled.current) { later(); return }
      const k0 = Number(chart.style.zoom) || 1
      // what the chart draws (plots and legends), as laid out now: its natural size is that at zoom 1
      const drawn = [...chart.querySelectorAll<Element>('svg, canvas, img, .legend')]
        .filter((m) => !m.parentElement?.closest('svg, .legend') && m.getClientRects().length)
      if (!drawn.some((m) => m.tagName !== 'DIV') || chart.querySelector('table')) return
      const rects = drawn.map((m) => m.getBoundingClientRect())
      const box = chart.getBoundingClientRect()
      const natW = (Math.max(...rects.map((r) => r.right)) - Math.min(...rects.map((r) => r.left))) / k0
      const natH = box.height / k0
      const roomH = size - 18 - (parseFloat(getComputedStyle(active!).paddingBottom) || 0) - (box.top + window.scrollY)
      if (k0 === 1 && !chart.style.width) slot = box.width
      const roomW = slot || box.width
      const want = Math.max(1, Math.min(ZOOM_MAX, roomH / natH, roomW / natW))
      const k = Math.abs(want - 1) < 0.04 ? 1 : +want.toFixed(3)
      if (Math.abs(k - k0) <= 0.01 || steps++ >= 4) return
      // scaled up, the chart's own width shrinks by the scale, so it still spans the room
      if (k === 1) { chart.style.zoom = ''; chart.style.width = ''; return }
      chart.style.width = `${Math.round(roomW / k)}px`
      chart.style.zoom = String(k)
    }
    const later = () => { if (timer) clearTimeout(timer); timer = window.setTimeout(scale, 450) }
    later()
    const ro = new ResizeObserver(later)
    ro.observe(chart)
    return () => { ro.disconnect(); if (timer) clearTimeout(timer); chart.style.zoom = ''; chart.style.width = '' }
  }, [active, exporting, size])
  return <FrameFit height={active && !exporting ? fit : null}>{children}</FrameFit>
}

/** how far the single-chart view scales up a chart that draws at its own size (SingleFit) */
const ZOOM_MAX = 1.6

/** The header button: on, the page shows one chart at a time. */
export function SingleViewToggle({ icon }: { icon: ReactNode }) {
  const { on, setOn, available } = useSingleView()
  if (!available) return null
  return <button type="button" className={`btn quiet icon-btn${on ? ' on' : ''}`} aria-pressed={on} aria-label="One chart at a time"
    onClick={() => setOn(!on)}
    title={on ? 'One chart at a time: ‹ › or ← → step through the page (click to show the whole page)' : 'One chart at a time: each chart fills the page, with its fit statistics beside the arrows; ‹ › or ← → for the next'}>
    {icon}
  </button>
}

function Steps() {
  const { index, count, go } = useSingleView()
  const many = count > 1
  return <span className="single-steps">
    <StepButton dir="prev" label="Previous chart" title="Previous chart (←)" disabled={!many} onClick={() => go(-1)} />
    <span className="single-count" aria-live="polite">{count ? `${index + 1} / ${count}` : 'no charts'}</span>
    <StepButton dir="next" label="Next chart" title="Next chart (→)" disabled={!many} onClick={() => go(1)} />
  </span>
}

/** Every chart on the page by name, under its section's heading, with a search box (the Combobox the field pickers use). */
function ChartPicker() {
  const { index, charts, jump } = useSingleView()
  if (!charts.length) return null
  const options = charts.map((c, i) => ({ key: String(i), label: c.title, group: c.group, hint: String(i + 1) }))
  return <Combobox label="" title="Jump to a chart on this page" value={String(index)} options={options} search onChange={(k) => jump(Number(k))} />
}

/**
 * The title line of the chart in view (components/ChartFrame): the arrows and the chart picker stand in
 * for the heading, then `children` (the chart's notes ⓘ). The header's button leads back to the whole page.
 */
export function SingleViewNav({ children }: { children?: ReactNode }) {
  return <>
    <Steps />
    <span className="single-title"><ChartPicker /></span>
    {children}
  </>
}

/**
 * Directly above the chart's options: its fit statistics, drawn here instead of above the axes, and its
 * notes and readouts (ChartAside). Left out of exports (data-export="skip"); an open export draws them
 * back in the chart.
 */
export function SingleViewStats() {
  const { setSlot } = useSingleView()
  return <div ref={setSlot} className="single-stats" data-export="skip" />
}

/** ChartFrame's stats box while its chart is the one in view (and not being exported), else null */
export const AsideSlotCtx = createContext<HTMLElement | null>(null)

/** true when the chart keeps its fit statistics above the plot in this view too (lib/chartDisplay StatsPlace) */
export const StatsInPlotCtx = createContext(false)

/**
 * A chart's notes and readouts (what the reader is told about the plot, under or beside it): in place,
 * or in the stats box beside the chart's options when the chart is the one in single-chart view, so the
 * plot has the room. components/ChartFrame Note uses it; a chart with its own note markup wraps it.
 */
export function ChartAside({ children }: { children: ReactNode }) {
  const slot = useContext(AsideSlotCtx)
  return slot ? createPortal(children, slot) : <>{children}</>
}
