import { usePrefs } from '@/lib/prefs'

export interface LegendItem {
  label: string
  color: string
  shape?: 'dot' | 'square' | 'band' | 'line' | 'dashed'
  /** shown after the label in muted mono, e.g. "n=190" */
  detail?: string
}

/**
 * A legend that is also a filter. When `onToggle` is given, clicking an item
 * hides that series (react-graph-gallery's "interactive inline legend"
 * pattern); the chart reads `hidden` and drops those marks. Without
 * `onToggle` it is a plain key.
 */
export function Legend({
  items,
  hidden,
  onToggle,
  note,
}: {
  items: LegendItem[]
  hidden?: Set<string>
  onToggle?: (label: string) => void
  note?: string
}) {
  const { explain } = usePrefs()
  return (
    <div className="legend">
      {items.map((it) => {
        const off = hidden?.has(it.label) ?? false
        const cls = ['legend-item', onToggle ? 'clickable' : '', off ? 'off' : ''].filter(Boolean).join(' ')
        return (
          <span
            key={it.label}
            className={cls}
            onClick={onToggle ? () => onToggle(it.label) : undefined}
            title={onToggle ? (off ? 'click to show' : 'click to hide') : undefined}
            role={onToggle ? 'button' : undefined}
          >
            <span
              className={`swatch ${it.shape ?? 'dot'}`}
              style={it.shape === 'dashed' ? { color: it.color } : { background: it.color }}
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

/** Small helper so charts can keep a hidden-set in one line. */
export function toggleIn(set: Set<string>, label: string): Set<string> {
  const next = new Set(set)
  if (next.has(label)) next.delete(label)
  else next.add(label)
  return next
}
