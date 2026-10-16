import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useHeaderNav, type ChartTarget } from '@/lib/headerNav'
import index from '@/chartIndex.json'
import { SearchIcon } from '@/components/Icons'

/**
 * ⌘K / Ctrl K: search every tab, meeting page and chart (by title, or by the component that
 * draws it: "hexbin", "ridgeline", "PCA"). Charts come from src/chartIndex.json, which
 * scripts/build-chart-index.mjs reads out of the source, so new charts are found without
 * registering them.
 */
interface Item extends ChartTarget { kind: 'Tab' | 'Meeting' | 'Chart'; label: string; where: string; words: string }

const CHARTS = (index as { charts: { title: string; id?: string; component?: string; tab: string; meeting?: string; file: string }[] }).charts
// "DensityHexbin" → "density hexbin", so a search for the kind of chart finds it
const spaced = (s?: string) => (s ?? '').replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase()

export function CommandPaletteButton() {
  const nav = useHeaderNav()
  const [open, setOpen] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setOpen((o) => !o) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  if (!nav) return null
  const mac = /Mac|iPhone|iPad/.test(navigator.platform)
  return <>
    <button type="button" className="btn quiet icon-btn" onClick={() => setOpen(true)} aria-label="Search" title={`Search tabs, meetings and charts (${mac ? '⌘' : 'Ctrl'} K)`}>
      <SearchIcon />
    </button>
    {open && <Palette onClose={() => setOpen(false)} />}
  </>
}

function Palette({ onClose }: { onClose: () => void }) {
  const nav = useHeaderNav()!
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const tabLabel = (t: string) => nav.tabs.find((x) => x.key === t)?.label ?? t
  const meetingLabel = (m?: string) => nav.meetings.find((x) => x.key === m)?.label ?? m ?? ''
  const items = useMemo<Item[]>(() => [
    ...nav.tabs.map((t): Item => ({ kind: 'Tab', label: t.label, where: t.title ?? '', tab: t.key, words: `${t.label} ${t.title ?? ''} tab`.toLowerCase() })),
    ...nav.meetings.map((m): Item => ({ kind: 'Meeting', label: m.label, where: 'Meeting', tab: 'meeting', meeting: m.key, words: `${m.label} meeting`.toLowerCase() })),
    // the tabs this build offers (a share build drops some)
    ...CHARTS.filter((c) => nav.tabs.some((t) => t.key === c.tab)).map((c): Item => {
      const where = c.tab === 'meeting' ? `Meeting · ${meetingLabel(c.meeting)}` : tabLabel(c.tab)
      return { kind: 'Chart', label: c.title.replace(/^…\s*/, ''), where, tab: c.tab, meeting: c.meeting, id: c.id, title: c.title,
        words: `${c.title} ${where} ${spaced(c.component)} ${c.id ?? ''}`.toLowerCase() }
    }),
  ], [nav]) // eslint-disable-line react-hooks/exhaustive-deps
  const shown = useMemo(() => {
    const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean)
    if (!terms.length) return items.filter((i) => i.kind !== 'Chart')
    // every term must match; titles that start with the query come first
    return items.filter((i) => terms.every((t) => i.words.includes(t)))
      .sort((a, b) => Number(b.label.toLowerCase().startsWith(terms[0])) - Number(a.label.toLowerCase().startsWith(terms[0])))
      .slice(0, 60)
  }, [items, q])
  useEffect(() => { setActive(0) }, [q])
  useEffect(() => {
    input.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])
  useEffect(() => { list.current?.querySelector('.palette-item.active')?.scrollIntoView({ block: 'nearest' }) }, [active])
  const choose = (i?: Item) => { if (!i) return; onClose(); nav.goTo(i) }
  return createPortal(
    <div className="export-backdrop palette-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Search the gallery">
        <input ref={input} className="palette-input" type="search" placeholder="Search tabs, meetings and charts (e.g. hexbin, PCA, humidity, starting point)…" value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(shown.length - 1, a + 1)) }
            else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(0, a - 1)) }
            else if (e.key === 'Enter') { e.preventDefault(); choose(shown[active]) }
          }} />
        <div ref={list} className="palette-list" role="listbox">
          {shown.map((i, k) => (
            <button key={`${i.kind}-${i.tab}-${i.meeting ?? ''}-${i.id ?? ''}-${i.label}`} type="button" role="option" aria-selected={k === active}
              className={`palette-item${k === active ? ' active' : ''}`} onMouseEnter={() => setActive(k)} onClick={() => choose(i)}>
              <span className="palette-kind">{i.kind}</span>
              <span className="palette-label">{i.label}</span>
              <span className="palette-where">{i.where}</span>
            </button>
          ))}
          {!shown.length && <p className="site-empty">Nothing matches “{q}”.</p>}
        </div>
        <p className="palette-foot">↑ ↓ to move · Enter to open · Esc to close</p>
      </div>
    </div>,
    document.body,
  )
}
