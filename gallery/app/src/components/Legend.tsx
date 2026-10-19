import { estimatorText, useEstimator } from '@/lib/regressionPrefs'
import { useState } from 'react'
import { usePrefs } from '@/lib/prefs'
import { useChartLabel, useFitHeight, useTileLegend } from '@/lib/chartFit'

/** in a Dashboard tile, a legend longer than this shows its first items and "+N more" */
const TILE_LIMIT = 8

export interface LegendItem {
  label: string
  color: string
  shape?: 'dot' | 'ring' | 'square' | 'band' | 'line' | 'dashed'
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
  // a tile's legend setting: off draws nothing; bottom and right draw every item
  const long = inTile && (tile?.mode ?? 'auto') === 'auto' && displayItems.length > TILE_LIMIT + 1
  const shown = long && !all ? displayItems.slice(0, TILE_LIMIT) : displayItems
  if (tile?.mode === 'off' || (!displayItems.length && !(note && explain))) return null
  return (
    <div className="legend">
      {shown.map((it) => {
        const off = hidden?.has(it.label) ?? false
        const dim = !!highlighted && highlighted !== it.label && !off
        const cls = ['legend-item', onToggle ? 'clickable' : '', onHover ? 'hoverable' : '', off ? 'off' : '', dim ? 'dim' : ''].filter(Boolean).join(' ')
        return (
          <span
            key={it.label}
            className={cls}
            onClick={onToggle ? () => onToggle(it.label) : undefined}
            onKeyDown={onToggle ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle(it.label) } } : undefined}
            onMouseEnter={onHover ? () => onHover(it.label) : undefined}
            onMouseLeave={onHover ? () => onHover(null) : undefined}
            onFocus={onHover ? () => onHover(it.label) : undefined}
            onBlur={onHover ? () => onHover(null) : undefined}
            title={onToggle ? (off ? 'click to show' : 'click to hide') : undefined}
            role={onToggle ? 'button' : undefined}
            aria-pressed={onToggle ? !off : undefined}
            tabIndex={live ? 0 : undefined}
          >
            <span
              className={`swatch ${it.shape ?? 'dot'}`}
              style={it.shape === 'dashed' || it.shape === 'ring' ? { color: it.color } : { background: it.color }}
            />
            {estimatorText(label(it.label), method)}
            {it.detail && <span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{it.detail}</span>}
          </span>
        )
      })}
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
