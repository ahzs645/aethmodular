import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Combobox } from '@/components/Combobox'
import { StepButton } from '@/components/StepButton'
import { FrameFit } from '@/lib/chartFit'
import { useExportRender } from '@/lib/exportRender'

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
}

const Ctx = createContext<SingleView>({ on: false, setOn: () => {}, available: false, active: null, index: 0, count: 0, title: '', charts: [], go: () => {}, jump: () => {}, slot: null, setSlot: () => {} })

const KEY = 'aeth-gallery:single'
/** the on/off switch, kept in App so the provider can turn it off on the Dashboard */
export function useSingleViewPref(): [boolean, (v: boolean) => void] {
  const [on, setOnState] = useState(() => { try { return localStorage.getItem(KEY) === '1' } catch { return false } })
  const setOn = useCallback((v: boolean) => {
    setOnState(v)
    try { localStorage.setItem(KEY, v ? '1' : '0') } catch { /* the choice still holds for the session */ }
  }, [])
  return [on, setOn]
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
  // a new tab or meeting starts at its first chart
  useEffect(() => { want.current = null; at.current = 0; setIndex(0) }, [page])

  useEffect(() => {
    document.documentElement.dataset.single = on ? '1' : '0'
    if (!on) { clearMarks(); setFrames([]); return }
    let raf = 0
    const apply = () => {
      raf = 0
      // looked up each time: the page root is replaced when the data finishes loading
      const root = document.querySelector('[data-single-root]')
      if (!root) return
      const list = framesIn(root)
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
  const charts = useMemo(() => {
    const root = document.querySelector('[data-single-root]')
    const groups = root ? sectionsOf(root, frames) : []
    return frames.map((f, i) => ({ title: f.dataset.frameTitle ?? `Chart ${i + 1}`, group: groups[i] }))
  }, [frames])
  const value = useMemo<SingleView>(() => ({
    on, setOn, available, active, index, count: on ? frames.length : 0,
    title: active?.dataset.frameTitle ?? '', charts, go, jump, slot, setSlot,
  }), [on, setOn, available, active, index, frames.length, charts, go, jump, slot])
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
        // stacked, the legend is as wide as its widest item (a Legend can hold its items in an inner .legend)
        const stacked = Math.max(0, ...[...legend.querySelectorAll('.legend-item, .legend-group-name, .legend-note')].map((c) => c.getBoundingClientRect().width))
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
    if (!active || exporting || tries.current >= 6) return
    const chart = active.querySelector('.frame-chart')
    if (!chart) return
    const BOTTOM = 18
    const pad = parseFloat(getComputedStyle(active).paddingBottom) || 0
    // the panel's room in the window: an options column taller than that scrolls inside it (styles.css --single-room)
    active.style.setProperty('--single-room', `${Math.max(240, Math.round(size - BOTTOM - pad - (active.getBoundingClientRect().top + window.scrollY)))}px`)
    const guess = fit === null
    // first a guess from where the chart starts, then the chart's own overflow or room left (the chart, not the
    // panel: an options column longer than the window must not shrink the plot)
    const next = Math.max(240, Math.round(guess
      ? size - chart.getBoundingClientRect().top - window.scrollY - BOTTOM - 40
      : fit + size - BOTTOM - pad - (chart.getBoundingClientRect().bottom + window.scrollY)))
    if (!guess && Math.abs(next - fit) <= 2) return
    tries.current += 1
    setFit(next)
  })
  return <FrameFit height={active && !exporting ? fit : null}>{children}</FrameFit>
}

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
 * Directly above the chart's options: its fit statistics, drawn here instead of above the axes. Left
 * out of exports (data-export="skip"); an open export draws them back in the chart.
 */
export function SingleViewStats() {
  const { setSlot } = useSingleView()
  return <div ref={setSlot} className="single-stats" data-export="skip" />
}
