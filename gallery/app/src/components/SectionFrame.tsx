import { useEffect, useRef, useState } from 'react'
import { EMBED_CHART_MESSAGE, type EmbedChartMessage } from '@/lib/embedChart'
import { itemHref, itemView, sameItem, type SectionItem } from '@/lib/sections'
import type { ViewState } from '@/lib/views'

/** a card grows to a tall chart up to here, and past that the chart scrolls inside it */
const MAX_HEIGHT = 1100
/** a page that has not shown the chart by then does not have it */
const FIND_MS = 45000
/** Every chart is a running page, so they load a few at a time, in the order they scroll near. */
const AT_ONCE = 2
let loading = 0
const waiting: (() => void)[] = []
const next = () => { while (loading < AT_ONCE && waiting.length) { loading++; waiting.shift()!() } }
function takeTurn(go: () => void) { waiting.push(go); next(); return () => { const i = waiting.indexOf(go); if (i >= 0) waiting.splice(i, 1) } }
function endTurn() { loading--; next() }

/**
 * Only the charts near the screen stay alive, so a section's memory does not grow with its length: past this many
 * the longest-kept one out of view is let go, and reopens from its saved settings when it scrolls back.
 */
const maxLive = () => (window.innerWidth <= 900 ? 3 : 8)
interface Live { near: boolean; release: () => void }
const live: Live[] = []
function keepLive(slot: Live) {
  live.push(slot)
  while (live.length > maxLive()) {
    const out = live.find((s) => !s.near && s !== slot)
    if (!out) return
    live.splice(live.indexOf(out), 1)
    out.release()
  }
}
const dropLive = (slot: Live) => { const i = live.indexOf(slot); if (i >= 0) live.splice(i, 1) }

/**
 * One chart of a section, in its own embed of the view it was picked in. `src` stays as it opened while the embed is
 * alive, since the section saving a changed setting must not reload the chart it came from.
 */
export function SectionFrame({ item, height, fixed, onView, onChart }: {
  item: SectionItem
  height: number
  /** a height the reader set: the card does not grow to the chart */
  fixed: boolean
  /** a setting changed in the chart (absent: a shared section, which is read-only) */
  onView?: (view: ViewState) => void
  /** the chart's key changed with its settings (a title that names its fields) */
  onChart?: (chart: string) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const frame = useRef<HTMLIFrameElement>(null)
  const [src, setSrc] = useState<string | null>(null)
  const [state, setState] = useState<'waiting' | 'shown' | 'missing'>('waiting')
  const [content, setContent] = useState(0)
  const [near, setNear] = useState(false)
  const latest = useRef({ item, onView, onChart })
  latest.current = { item, onView, onChart }
  const slot = useRef<Live>({ near: false, release: () => {} })

  useEffect(() => {
    const s = slot.current
    const io = new IntersectionObserver(([e]) => { s.near = e.isIntersecting; setNear(e.isIntersecting) }, { rootMargin: '500px 0px' })
    io.observe(box.current!)
    return () => io.disconnect()
  }, [])

  // a turn to load, while near and not alive
  useEffect(() => {
    if (!near || src) return
    return takeTurn(() => setSrc(itemHref(latest.current.item, true)))
  }, [near, src])

  useEffect(() => {
    if (!src) return
    const s = slot.current
    let turn = true
    const finish = () => { if (turn) { turn = false; endTurn() } }
    s.release = () => { setSrc(null); setState('waiting') }
    keepLive(s)
    let seen = false
    const giveUp = window.setTimeout(() => { if (!seen) { setState('missing'); finish() } }, FIND_MS)
    const onMessage = (e: MessageEvent<EmbedChartMessage>) => {
      if (e.origin !== window.location.origin || e.data?.type !== EMBED_CHART_MESSAGE || e.source !== frame.current?.contentWindow) return
      const { item: now, onView: view, onChart: chart } = latest.current
      if (!seen) {
        if (e.data.active !== now.chart) return
        seen = true
        setState('shown')
        window.clearTimeout(giveUp)
        finish()
      }
      // grows only: a chart fits the height it is given, so a card that shrank to it would close up
      setContent((c) => Math.max(c, e.data.height))
      if (e.data.active && e.data.active !== now.chart) chart?.(e.data.active)
      const next = itemView(e.data.view)
      if (!sameItem(now, { chart: now.chart, view: next })) view?.(next)
    }
    window.addEventListener('message', onMessage)
    return () => { window.clearTimeout(giveUp); window.removeEventListener('message', onMessage); dropLive(s); finish() }
  }, [src])

  return (
    <div ref={box} className="sec-frame" style={{ height: fixed ? height : Math.max(height, Math.min(content, MAX_HEIGHT)) }}>
      {state === 'missing'
        ? <p className="sec-frame-note">This chart was not found on its page, so it is not drawn here. The page may have changed since it was picked. <a href={itemHref(item, false)} target="_blank" rel="noreferrer">Open the page ↗</a></p>
        : state === 'waiting' && <p className="sec-frame-note" role="status">Loading chart…</p>}
      {src && <iframe ref={frame} src={src} title={item.title} className={state === 'shown' ? 'shown' : ''} />}
    </div>
  )
}
