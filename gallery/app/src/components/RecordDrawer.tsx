import { useEffect, useState } from 'react'
import { fmt } from '@/lib/stats'
import type { FilterRecord } from '@/lib/highlight'
import { FilterSpectrum } from '@/components/FilterSpectrum'

/** A value as a drawer cell: numbers to 3 decimals, missing as an em dash. */
const cell = (v: FilterRecord['fields'][number][1]) =>
  v === null || v === undefined || v === '' ? '—' : typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : fmt(v, 3)) : v

/** The "what this chart knows" table, shared by both drawers. */
export function RecordSection({ rec }: { rec: FilterRecord }) {
  return (
    <section className="drawer-section">
      <h3>From {rec.source}</h3>
      <table className="placement record">
        <tbody>
          {rec.fields.map(([k, v]) => (
            <tr key={k}>
              <td className="f">{k}</td>
              <td className="v">{cell(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rec.note && <p className="chart-note" style={{ display: 'block' }}>{rec.note}</p>}
    </section>
  )
}

/**
 * The drawer for a filter the gallery's SPARTAN export does not carry: IMPROVE
 * calibration test filters, Bishoftu, the Adama quartz campaign. It shows what
 * the clicked chart knows, in the same side panel as the full sample drawer,
 * so every single-filter mark in the app opens the same way.
 */
export function RecordDrawer({ rec, onClose }: { rec: FilterRecord; onClose: () => void }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const copy = () => navigator.clipboard?.writeText(rec.id).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })

  return (
    <aside className="drawer" role="dialog" aria-label={`Filter ${rec.id}`}>
      <header className="drawer-head">
        <div>
          <div className="drawer-id">{rec.id}</div>
          <div className="drawer-sub">{[rec.site, rec.date].filter(Boolean).join(' · ')}</div>
        </div>
        <button type="button" className="btn quiet" onClick={onClose} aria-label="Close" title="Close (Esc)">✕</button>
      </header>
      <div className="drawer-actions">
        <span className="u">Not in the SPARTAN filter export, so only this chart's values are shown.</span>
        <span style={{ flex: 1 }} />
        <button type="button" className="btn" onClick={copy}>{copied ? 'copied' : 'copy id'}</button>
      </div>
      <RecordSection rec={rec} />
      <FilterSpectrum id={rec.id} />
    </aside>
  )
}
