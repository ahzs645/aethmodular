import type { ReactNode } from 'react'
import { FONT } from '@/lib/theme'

/**
 * The pieces the "Linearising HIPS at Addis" page's charts share (pages/HipsLinearisePage.tsx, pages/HipsDecimals.tsx):
 * a row of headline numbers under a chart, and a plain table.
 */

/** a row of headline numbers under a chart */
export function Readouts({ items }: { items: { label: string; value: ReactNode; color?: string; sub?: string }[] }) {
  return <div className="readouts">
    {items.map((it) => <div key={it.label} className="stat-tile">
      <div className="stat-label">{it.label}</div>
      <div className="stat-value"><strong style={it.color ? { color: it.color } : undefined}>{it.value}</strong></div>
      {it.sub && <div className="stat-sub">{it.sub}</div>}
    </div>)}
  </div>
}

/** a plain table in the gallery's table style (styles.css .census-table); numeric columns right-aligned in mono */
export interface Column<T> { key: string; label: ReactNode; value: (row: T) => ReactNode; num?: boolean }
export function DataTable<T>({ rows, columns, rowKey, highlight }: { rows: T[]; columns: Column<T>[]; rowKey: (r: T, i: number) => string; highlight?: (r: T) => boolean }) {
  return (
    <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
      <table className="census-table" style={{ minWidth: Math.min(880, columns.length * 88) }}>
        <thead><tr>{columns.map((c) => <th key={c.key} style={{ textTransform: 'none', letterSpacing: 0, textAlign: c.num ? 'right' : 'left' }}>{c.label}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={rowKey(r, i)} style={highlight?.(r) ? { background: 'var(--accent-soft)' } : undefined}>
          {columns.map((c) => <td key={c.key} style={c.num ? { fontFamily: FONT.mono, textAlign: 'right', whiteSpace: 'nowrap' } : undefined}>{c.value(r)}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
  )
}
