import { useState } from 'react'
import { usePrefs } from '@/lib/prefs'

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
  const { explain } = usePrefs()
  const live = !!(onToggle || onHover)
  return (
    <div className="legend">
      {items.map((it) => {
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
            {it.label}
            {it.detail && <span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{it.detail}</span>}
          </span>
        )
      })}
      {note && explain && <span className="legend-note">{note}</span>}
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
