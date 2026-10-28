import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useHeaderNav, type ChartTarget } from '@/lib/headerNav'
import index from '@/chartIndex.json'
import { SearchIcon } from '@/components/Icons'

/**
 * ⌘K / Ctrl K: search every tab, meeting page and chart (by title, or by the component that
 * draws it: "hexbin", "ridgeline", "PCA"). Charts come from src/chartIndex.json, which
 * scripts/build-chart-index.mjs reads out of the source, so new charts are found without
 * registering them.
 *
 * Results come in sections: with nothing typed, the pages opened from here lately, then every tab
 * under the header menu's groups (components/NavMenu) and the meeting pages; once something is
 * typed, the matching tabs, meetings and charts, the typed words marked. A tab is described by
 * its title, or, lacking one, by the charts on it.
 */
type Kind = 'tab' | 'meeting' | 'chart'
interface Item extends ChartTarget {
  kind: Kind
  /** unique across the palette, and what "Recent" remembers */
  key: string
  label: string
  /** one line beside or under the label: a tab's or meeting's title, the rest of a chart's title */
  desc: string
  /** where a chart sits (its tab or meeting page) */
  where?: string
  words: string
}
interface Section { name: string; items: Item[] }

const CHARTS = (index as { charts: { title: string; id?: string; component?: string; tab: string; meeting?: string; file: string }[] }).charts
// "DensityHexbin" → "density hexbin", so a search for the kind of chart finds it
const spaced = (s?: string) => (s ?? '').replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase()
/** a chart title's head (what ChartFrame shows as its heading) and the rest after ' — ' */
const head = (t: string) => t.replace(/^…\s*/, '').split(' — ')[0].trim()
const rest = (t: string) => t.split(' — ').slice(1).join(' — ').trim()
/** the header menu's groups, in its order (App NAV_GROUPS); a tab without one goes under Investigations */
const GROUP_ORDER = ['Charts', 'Sites', 'Calibration', 'Investigations']
const MAX_CHARTS = 50
const RECENT_KEY = 'gallery:palette-recent'
const RECENT_MAX = 5

function readRecent(): string[] {
  try { const v = JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]'); return Array.isArray(v) ? v.filter((x) => typeof x === 'string') : [] } catch { return [] }
}
function remember(key: string) {
  try { localStorage.setItem(RECENT_KEY, JSON.stringify([key, ...readRecent().filter((k) => k !== key)].slice(0, RECENT_MAX))) } catch { /* private window: no history */ }
}

/** the charts on a page by their heads, for a tab with no title of its own: "Boxplot · Violin · Beeswarm · +2 more" */
function chartSummary(heads: string[]) {
  const u = [...new Set(heads)]
  if (!u.length) return ''
  return u.length > 4 ? `${u.slice(0, 3).join(' · ')} · +${u.length - 3} more` : u.join(' · ')
}

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

/** `text` with every searched word marked */
function Hit({ text, re }: { text: string; re: RegExp | null }): ReactNode {
  if (!re || !text) return text
  return text.split(re).map((part, k) => k % 2 ? <mark key={k} className="palette-hit">{part}</mark> : <Fragment key={k}>{part}</Fragment>)
}

function Palette({ onClose }: { onClose: () => void }) {
  const nav = useHeaderNav()!
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const [recent] = useState(readRecent)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLDivElement>(null)

  const items = useMemo<Item[]>(() => {
    const offered = new Set(nav.tabs.map((t) => t.key))
    // the tabs this build offers (a share build drops some)
    const charts = CHARTS.filter((c) => offered.has(c.tab))
    const tabLabel = (t: string) => nav.tabs.find((x) => x.key === t)?.label ?? t
    const meetingLabel = (m?: string) => nav.meetings.find((x) => x.key === m)?.label ?? m ?? ''
    const headsOn = (tab: string, meeting?: string) => charts.filter((c) => c.tab === tab && (!meeting || c.meeting === meeting)).map((c) => head(c.title))
    const sameComponent = new Map(charts.map((c) => [`chart:${c.tab}:${c.meeting ?? ''}:${c.id ?? ''}:${c.title}`, c.component ?? c.file]))
    // the Meeting tab is its pages, listed under Meetings
    const tabs = nav.tabs.filter((t) => !(t.key === 'meeting' && nav.meetings.length))
    return [
      ...tabs.map((t): Item => {
        const desc = t.title ?? chartSummary(headsOn(t.key))
        return { kind: 'tab', key: `tab:${t.key}`, label: t.label, desc, tab: t.key, words: `${t.label} ${desc} ${t.group ?? ''} tab`.toLowerCase() }
      }),
      ...(offered.has('meeting') ? nav.meetings : []).map((m): Item => {
        const n = headsOn('meeting', m.key).length
        const desc = m.title ?? (n ? `${n} chart${n === 1 ? '' : 's'}` : '')
        return { kind: 'meeting', key: `meeting:${m.key}`, label: m.label, desc, tab: 'meeting', meeting: m.key, words: `${m.label} ${desc} meeting`.toLowerCase() }
      }),
      ...charts.map((c): Item => {
        const where = c.tab === 'meeting' ? meetingLabel(c.meeting) : tabLabel(c.tab)
        return { kind: 'chart', key: `chart:${c.tab}:${c.meeting ?? ''}:${c.id ?? ''}:${c.title}`, label: head(c.title), desc: rest(c.title), where,
          tab: c.tab, meeting: c.meeting, id: c.id, title: c.title,
          words: `${c.title} ${where} ${spaced(c.component)} ${c.id ?? ''}`.toLowerCase() }
      }),
    ].reduce<Item[]>((out, i) => {
      // one row per chart: a chart drawn under two titles (a variant's longer one) or twice on a page reads as one;
      // the row keeps the fuller title and is found by either
      const same = out.findIndex((j) => j.key === i.key || (i.kind === 'chart' && j.kind === 'chart' && j.tab === i.tab && j.meeting === i.meeting && j.label === i.label && sameComponent.get(j.key) === sameComponent.get(i.key)))
      if (same < 0) out.push(i)
      else if (i.desc.length > out[same].desc.length) out[same] = { ...i, words: `${i.words} ${out[same].words}` }
      else out[same] = { ...out[same], words: `${out[same].words} ${i.words}` }
      return out
    }, [])
  }, [nav])
  const groupOf = useMemo(() => new Map(nav.tabs.map((t) => [`tab:${t.key}`, t.group ?? 'Investigations'])), [nav.tabs])

  const terms = useMemo(() => [...new Set(q.trim().toLowerCase().split(/\s+/).filter(Boolean))], [q])
  const re = useMemo(() => terms.length
    ? new RegExp(`(${[...terms].sort((a, b) => b.length - a.length).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
    : null, [terms])

  const { sections, total, chartsFound } = useMemo(() => {
    if (!terms.length) {
      const byKey = new Map(items.map((i) => [i.key, i]))
      const groups = new Map<string, Item[]>()
      for (const i of items.filter((x) => x.kind === 'tab')) {
        const g = groupOf.get(i.key) ?? 'Investigations'
        groups.set(g, [...(groups.get(g) ?? []), i])
      }
      const order = [...groups.keys()].sort((a, b) => (GROUP_ORDER.indexOf(a) + 1 || 99) - (GROUP_ORDER.indexOf(b) + 1 || 99))
      const out: Section[] = [
        { name: 'Recent', items: recent.flatMap((k) => byKey.get(k) ?? []) },
        // the Investigations cards lead their group
        ...order.map((g) => ({ name: g, items: groups.get(g)!.sort((a, b) => Number(b.tab === 'investigations') - Number(a.tab === 'investigations')) })),
        { name: 'Meetings', items: items.filter((i) => i.kind === 'meeting') },
      ].filter((s) => s.items.length)
      return { sections: out, total: 0, chartsFound: 0 }
    }
    const phrase = terms.join(' ')
    // every word must match; then labels that are the query, start with it, or hold every word come first
    const score = (i: Item) => {
      const l = i.label.toLowerCase()
      return l === phrase ? 4 : l.startsWith(phrase) ? 3 : terms.every((t) => l.includes(t)) ? (terms.some((t) => new RegExp(`(^|[^a-z0-9])${t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`).test(l)) ? 2 : 1.5) : 0
    }
    const found = items.filter((i) => terms.every((t) => i.words.includes(t)))
      .map((i, k) => ({ i, k, s: score(i) })).sort((a, b) => b.s - a.s || a.k - b.k)
    const of = (kind: Kind) => found.filter((x) => x.i.kind === kind)
    const charts = of('chart')
    return {
      // the section with the best match first ("ridgeline" is a chart before it is a word in the Distribution tab's
      // description), else tabs, meetings, charts; Enter opens the first row
      sections: [
        { name: 'Tabs', found: of('tab') },
        { name: 'Meetings', found: of('meeting') },
        { name: 'Charts', found: charts.slice(0, MAX_CHARTS) },
      ].filter((x) => x.found.length).sort((a, b) => b.found[0].s - a.found[0].s)
        .map((x) => ({ name: x.name, items: x.found.map((f) => f.i) })),
      total: found.length,
      chartsFound: charts.length,
    }
  }, [items, terms, recent, groupOf])
  const flat = useMemo(() => sections.flatMap((s) => s.items), [sections])

  useEffect(() => { setActive(0); list.current?.scrollTo({ top: 0 }) }, [q])
  useEffect(() => {
    input.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])
  useEffect(() => { list.current?.querySelector('.palette-item.active')?.scrollIntoView({ block: 'nearest' }) }, [active])
  const choose = (i?: Item) => {
    if (!i) return
    remember(i.key)
    onClose()
    nav.goTo(i)
  }
  const move = (by: number) => setActive((a) => flat.length ? (a + by + flat.length) % flat.length : 0)

  const shownCharts = Math.min(chartsFound, MAX_CHARTS)
  const status = !terms.length
    ? `${CHARTS.filter((c) => nav.tabs.some((t) => t.key === c.tab)).length} charts to search`
    : !total ? 'No results'
      : chartsFound > shownCharts ? `${total} results · first ${shownCharts} charts shown`
        : `${total} result${total === 1 ? '' : 's'}`
  let n = -1
  return createPortal(
    <div className="export-backdrop palette-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Search the gallery">
        <div className="palette-bar">
          <span className="palette-icon" aria-hidden><SearchIcon /></span>
          <input ref={input} className="palette-input" type="text" placeholder="Search tabs, meetings and charts" value={q}
            role="combobox" aria-expanded="true" aria-controls="palette-list" aria-autocomplete="list"
            aria-activedescendant={flat.length ? `palette-opt-${active}` : undefined}
            autoComplete="off" autoCorrect="off" autoCapitalize="off" spellCheck={false} enterKeyHint="go"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); move(1) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1) }
              else if (e.key === 'Enter') { e.preventDefault(); choose(flat[active]) }
            }} />
          <button type="button" className="palette-close" onClick={onClose} aria-label="Close search" title="Close (Esc)">Esc</button>
        </div>
        <div ref={list} id="palette-list" className="palette-list" role="listbox" aria-label="Results">
          {sections.map((s) => (
            <div key={s.name} role="group" aria-label={s.name} className="palette-section">
              <div className="palette-head" aria-hidden>{s.name}{s.name === 'Charts' && chartsFound > s.items.length && <span className="palette-head-note">first {s.items.length} of {chartsFound}</span>}</div>
              {s.items.map((i) => {
                const k = ++n
                return (
                  <button key={`${s.name}:${i.key}`} id={`palette-opt-${k}`} type="button" role="option" aria-selected={k === active} tabIndex={-1}
                    className={`palette-item${k === active ? ' active' : ''}`} onMouseMove={() => { if (k !== active) setActive(k) }} onClick={() => choose(i)}>
                    <span className="palette-label"><Hit text={i.label} re={re} /></span>
                    {i.desc && <span className="palette-desc"><Hit text={i.desc} re={re} /></span>}
                    {i.where && <span className="palette-where"><Hit text={i.where} re={re} /></span>}
                  </button>
                )
              })}
            </div>
          ))}
          {!flat.length && <p className="palette-empty">Nothing matches “{q.trim()}”. Try a chart type (hexbin, ridgeline), a measurement (humidity, EC) or a page name.</p>}
        </div>
        <div className="palette-foot">
          <span className="palette-count" aria-live="polite">{status}</span>
          <span className="palette-keys" aria-hidden><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>Enter</kbd> open</span><span><kbd>Esc</kbd> close</span></span>
        </div>
      </div>
    </div>,
    document.body,
  )
}
