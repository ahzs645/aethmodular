import { useEffect, useState } from 'react'
import { fmt } from '@/lib/stats'
import type { FilterRecord } from '@/lib/highlight'
import { FilterSpectrum } from '@/components/FilterSpectrum'
import { DrawerShell } from '@/components/DrawerShell'
import { XIcon } from '@/components/Icons'

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
 * The drawer for a filter the gallery's exports do not carry: Bishoftu, the Adama
 * quartz campaign, and the few IMPROVE spectral samples with no IMPROVE network row
 * (the rest open the full sample drawer once App finds their row, lib/improveRecord). It shows what
 * the clicked chart knows, in the same side panel as the full sample drawer,
 * so every single-filter mark in the app opens the same way.
 */
export function RecordDrawer({ rec, onClose, lookup }: {
  rec: FilterRecord
  onClose: () => void
  /** an IMPROVE filter being looked up in the IMPROVE network export (App openRecord), or not found there */
  lookup?: { status: 'loading' } | { status: 'missing'; reason: string } | null
}) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const copy = () => navigator.clipboard?.writeText(rec.id).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })

  return (
    <DrawerShell label={`Filter ${rec.id}`} onClose={onClose}>
      <header className="drawer-head">
        <div>
          <div className="drawer-id">{rec.id}</div>
          <div className="drawer-sub">{[rec.site, rec.date].filter(Boolean).join(' · ')}</div>
        </div>
        <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
      </header>
      <div className="drawer-actions">
        <span className="u">{lookup?.status === 'loading' ? 'Looking for this filter in the IMPROVE network export…'
          : lookup?.status === 'missing' ? `Not in the IMPROVE network export (${lookup.reason}), so only this chart's values are shown.`
          : rec.id.startsWith('improve:') ? 'Not in the IMPROVE network export, so only this chart\'s values are shown.'
          : 'Not in the SPARTAN filter export, so only this chart\'s values are shown.'}</span>
        <span style={{ flex: 1 }} />
        <button type="button" className="btn" onClick={copy}>{copied ? 'Copied' : 'Copy ID'}</button>
      </div>
      <RecordSection rec={rec} />
      <FilterSpectrum id={rec.id} />
    </DrawerShell>
  )
}
