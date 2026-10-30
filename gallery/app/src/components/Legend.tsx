import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { useEffect, useRef, useState, type MouseEvent } from 'react'
import { usePrefs } from '@/lib/prefs'
import { useChartLabel, useFitHeight, useTileLegend } from '@/lib/chartFit'

/**
 * The event that holds a legend's hovered view (detail: an item's label, or null to let go), dispatched on the chart's
 * panel (section.panel). The export dialog's Focus sends it so the file is drawn with one series picked out, as a hover
 * shows it; shift-click on an item does the same on the page.
 */
export const LEGEND_FOCUS = 'legend-focus'

/** in a Dashboard tile, a legend longer than this shows its first items and "+N more" */
const TILE_LIMIT = 8

export interface LegendItem {
  label: string
  parent?: string
  color: string
  shape?: 'dot' | 'ring' | 'square' | 'band' | 'line' | 'dashed' | 'diamond'
  /** shown after the label in muted mono, e.g. "n=190" */
  detail?: string
}

/**
 * A legend that is also a filter. When `onToggle` is given, clicking an item
 * hides that series (react-graph-gallery's "interactive inline legend"
 * pattern); the chart reads `hidden` and drops those marks. When `onHover` is
 * given, pointing at (or tabbing to) an item reports its label so the chart
 * can dim the other series; `highlighted` echoes it back to dim the other
 * legend items too. Without either it is a plain key.
 */
export function Legend({
  items,
  hidden,
  onToggle,
  onHover,
  highlighted,
  note,
}: {
  items: LegendItem[]
  hidden?: Set<string>
  onToggle?: (label: string) => void
  onHover?: (label: string | null) => void
  highlighted?: string | null
  note?: string
}) {
  const label = useChartLabel()
  const displayItems = items.filter(it => label(it.label))
  const method = useEstimator()
  const { explain } = usePrefs()
  const live = !!(onToggle || onHover)
  // in a fitted tile a long legend would take the chart's height: the first few, then "+N more"
  const inTile = useFitHeight() !== null
  const tile = useTileLegend()
  const [all, setAll] = useState(false)
  // a held hover (LEGEND_FOCUS, or shift-click): leaving an item goes back to it rather than to none
  const [pinned, setPinned] = useState<string | null>(null)
  const [root, setRoot] = useState<HTMLDivElement | null>(null)
  const live$ = useRef({ onHover, items, pinned })
  live$.current = { onHover, items, pinned }
  const pin = (l: string | null) => { setPinned(l); live$.current.onHover?.(l) }
  useEffect(() => {
    const panel = root?.closest('section.panel')
    if (!panel) return
    const on = (e: Event) => {
      const { onHover: hover, items: its } = live$.current
      if (!hover) return
      const l = (e as CustomEvent<string | null>).detail
      const mine = l && its.some((it) => it.label === l) ? l : null
      setPinned(mine); hover(mine)
    }
    panel.addEventListener(LEGEND_FOCUS, on)
    return () => panel.removeEventListener(LEGEND_FOCUS, on)
  }, [root])
  const leave = onHover ? () => onHover(live$.current.pinned) : undefined
  // a tile's legend setting: off draws nothing; bottom and right draw every item
  const long = inTile && (tile?.mode ?? 'auto') === 'auto' && displayItems.length > TILE_LIMIT + 1
  const shown = long && !all ? displayItems.slice(0, TILE_LIMIT) : displayItems
  if (tile?.mode === 'off' || (!displayItems.length && !(note && explain))) return null
  // "Site · Season" labels: grouped one line per site, the site named once; with a single site
  // its name is dropped altogether (the chart says which site). Other labels stay as they are.
  const PREFIX = ' · '
  const parts = shown.map((it) => {
    const text = label(it.label)
    const k = text.indexOf(PREFIX)
    return { it, group: it.parent ?? (k > 0 ? text.slice(0, k) : null), text: k > 0 ? text.slice(k + PREFIX.length) : text }
  })
  const groups = [...new Set(parts.map((p) => p.group).filter((g): g is string => !!g))]
  const grouped = groups.length > 0
  const item = ({ it, group, text }: (typeof parts)[number]) => {
    const off = !!(hidden?.has(it.label) || (group && hidden?.has(group)))
    const dim = !!highlighted && highlighted !== it.label && !off
    const held = pinned === it.label
    const cls = ['legend-item', onToggle ? 'clickable' : '', onHover ? 'hoverable' : '', off ? 'off' : '', dim ? 'dim' : '', held ? 'pinned' : ''].filter(Boolean).join(' ')
    // shift-click holds this item's hovered view (again lets go); a plain click toggles it as before
    const click = onToggle || onHover ? (e: MouseEvent) => {
      if (e.shiftKey && onHover) { e.preventDefault(); pin(held ? null : it.label) } else onToggle?.(it.label)
    } : undefined
    return (
      <span
        key={it.label}
        data-export-series={it.label} data-export-color={it.color}
        className={cls}
        onClick={click}
        onKeyDown={onToggle ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle(it.label) } } : undefined}
        onMouseEnter={onHover ? () => onHover(it.label) : undefined}
        onMouseLeave={leave}
        onFocus={onHover ? () => onHover(it.label) : undefined}
        onBlur={leave}
        title={[group ? `${group} · ${text}` : '', onToggle ? (off ? 'click to show' : 'click to hide') : '',
          onHover ? (held ? 'shift-click to let go' : 'shift-click to hold this view') : ''].filter(Boolean).join(' — ') || undefined}
        role={onToggle ? 'button' : undefined}
        aria-pressed={onToggle ? !off : undefined}
        aria-label={group ? `${group} · ${text}` : undefined}
        tabIndex={live ? 0 : undefined}
      >
        <span
          className={`swatch ${it.shape ?? 'dot'}`}
          style={it.shape === 'dashed' || it.shape === 'ring' ? { color: it.color } : { background: it.color }}
        />
        {estimatorText(text, method)}
        {it.detail && <span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{it.detail}</span>}
      </span>
    )
  }
  // a group's name with no item of its own toggles its whole group: hides every shown member,
  // or shows them all again once all are hidden (one onToggle per member, so each chart's own
  // toggle rule still decides what may hide)
  const groupName = (g: string) => {
    const kids = parts.filter((p) => p.group === g)
    if (!onToggle || !kids.length) return g
    const anyOn = kids.some((p) => !hidden?.has(p.it.label))
    const flip = () => kids.filter((p) => !anyOn || !hidden?.has(p.it.label)).forEach((p) => onToggle(p.it.label))
    return (
      <span className={`legend-group-toggle${anyOn ? '' : ' off'}`} role="button" tabIndex={0} aria-pressed={anyOn}
        title={`${g} — click to ${anyOn ? 'hide' : 'show'} all ${kids.length}`}
        onClick={flip} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip() } }}>{g}</span>
    )
  }
  return (
    <div ref={setRoot} className={`legend${grouped ? ' grouped' : ''}`}>
      {parts.filter((p) => !grouped || (!p.group && !groups.includes(p.it.label))).map(item)}
      {grouped && groups.map((g) => <div key={g} className="legend-group">
        <span className="legend-group-name">{parts.some(p => p.it.label === g) ? item(parts.find(p => p.it.label === g)!) : groupName(g)}</span>
        {parts.filter((p) => p.group === g).map(item)}
      </div>)}
      {long && <button type="button" className="legend-more" onClick={() => setAll(!all)} aria-expanded={all}
        title={all ? 'Show the first few only' : 'Show every item (the chart shrinks to make room, or the tile scrolls)'}>
        {all ? 'fewer' : `+${displayItems.length - TILE_LIMIT} more`}</button>}
      {note && explain && <span className="legend-note">{estimatorText(note, method)}</span>}
    </div>
  )
}

/**
 * The state behind a reactive legend: which series are hidden and which one
 * the pointer is on. `show(label)` is the filter, `dim(label)` the opacity
 * factor for a series (1, or `low` while another series is hovered), and
 * `props` spreads straight onto `<Legend>`. A hovered series that is hidden
 * dims nothing, so the chart never fades out entirely.
 */
export function useLegend(initialHidden?: Iterable<string>) {
  const [hidden, setHidden] = useState<Set<string>>(() => new Set(initialHidden))
  const [hover, setHover] = useState<string | null>(null)
  const active = hover !== null && !hidden.has(hover) ? hover : null
  return {
    hidden,
    setHidden,
    hover: active,
    setHover,
    show: (label: string) => !hidden.has(label),
    dim: (label: string, low = 0.15) => (active === null || active === label ? 1 : low),
    toggle: (label: string) => setHidden((h) => toggleIn(h, label)),
    props: { hidden, highlighted: active, onToggle: (l: string) => setHidden((h) => toggleIn(h, l)), onHover: setHover },
  }
}

/** Small helper so charts can keep a hidden-set in one line. */
export function toggleIn(set: Set<string>, label: string): Set<string> {
  const next = new Set(set)
  if (next.has(label)) next.delete(label)
  else next.add(label)
  return next
}
