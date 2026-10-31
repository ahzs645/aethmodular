import { Component, createContext, useContext, useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Combobox } from '@/components/Combobox'
import { useHighlightActions } from '@/lib/highlight'
import { useSearchParam } from '@/lib/url'
import { parseCohort, rowsCsv, selectCohort, type CohortFilter } from '@/lib/dashboardCohort'
import { metricDefinition, measuredRows } from '@/lib/metricDefinition'
import { downloadBlob, slug } from '@/lib/export'
import type { DashboardChart, DashboardData } from './registry'
import type { PublicSiteStatus } from '@/lib/spartanPublic'
import type { DetailRequest } from '@/lib/dashboardDetails'

/** A shared, portalled dialog with keyboard containment and focus return. */
export function DashboardDialog({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  const root = useRef<HTMLDivElement>(null)
  const heading = useId()
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null
    const el = root.current!
    el.querySelector<HTMLElement>('button, input, [tabindex="0"]')?.focus()
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); close.current(); return }
      if (e.key !== 'Tab') return
      const items = [...el.querySelectorAll<HTMLElement>('button:not(:disabled), input, textarea, select, [tabindex="0"], a[href]')].filter(x => x.getClientRects().length)
      const first = items[0], last = items[items.length - 1]
      if (e.shiftKey && (document.activeElement === first || !el.contains(document.activeElement))) { e.preventDefault(); last?.focus() }
      else if (!e.shiftKey && (document.activeElement === last || !el.contains(document.activeElement))) { e.preventDefault(); first?.focus() }
    }
    window.addEventListener('keydown', key, true)
    return () => { window.removeEventListener('keydown', key, true); before?.focus({ preventScroll: true }) }
  }, [])
  return createPortal(<div className="map-dialog-backdrop" onPointerDown={e => { if (e.target === e.currentTarget) onClose() }}>
    <div ref={root} className="panel dashboard-dialog" role="dialog" aria-modal="true" aria-labelledby={heading}>
      <header className="frame-head"><h2 id={heading}>{title}</h2><button type="button" className="btn" onClick={onClose} aria-label="Close dialog">Close ×</button></header>
      {children}
    </div>
  </div>, document.body)
}

/** Render failures are isolated to their tile. Data failures have their own retry state. */
export class TileErrorBoundary extends Component<{ children: ReactNode; name: string; resetKey: string }, { error: string }> {
  state = { error: '' }
  static getDerivedStateFromError(error: Error) { return { error: error.message || 'The chart could not be drawn.' } }
  componentDidUpdate(prev: Readonly<{ children: ReactNode; name: string; resetKey: string }>) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: '' })
  }
  render() {
    if (this.state.error) return <section className="panel compact tile-failure" role="alert"><h3>{this.props.name} could not be drawn</h3>
      <p>{this.state.error}</p><button type="button" className="btn" onClick={() => this.setState({ error: '' })}>Retry chart</button></section>
    return this.props.children
  }
}

export const TileData = createContext<DashboardData | null>(null)
export function useTileCohort(data: DashboardData) {
  const [text, setText] = useSearchParam<string>('cohort', '')
  const filter = useMemo(() => data.snapshot ? {} : parseCohort(text), [text, data.snapshot])
  const rows = useMemo(() => selectCohort(data.rows, data.referenceRows ?? data.rows, filter), [data.rows, data.referenceRows, filter])
  return { filter, setFilter: (v: CohortFilter) => setText(Object.keys(v).length ? JSON.stringify(v) : ''), rows }
}

export function CohortControls({ data, filter, setFilter, rows }: { data: DashboardData; filter: CohortFilter; setFilter: (v: CohortFilter) => void; rows: DashboardData['rows'] }) {
  const sites = [...new Set(data.rows.map(r => r.site))].sort()
  const seasons = [...new Set(data.rows.filter(r => !filter.site || r.site === filter.site).map(r => r.season))].sort()
  return <fieldset className="tile-cohort-controls"><legend>Filters for this tile</legend>
    <p>Page filters affect every connected tile. Narrow this tile below, or retain its current cohort as a reference. Units, methods and source refreshes stay shared.</p>
    <Combobox label="Site" value={filter.site ?? ''} onChange={site => setFilter({ ...filter, site, season: undefined })}
      options={[{ key: '', label: 'All sites in view' }, ...sites.map(s => ({ key: s, label: s }))]} />
    <Combobox label="Season" value={filter.season ?? ''} onChange={season => setFilter({ ...filter, season })}
      options={[{ key: '', label: 'All seasons in view' }, ...seasons.map(s => ({ key: s, label: s }))]} />
    <label>From <input className="select" type="date" value={filter.from ?? ''} max={filter.to} onChange={e => setFilter({ ...filter, from: e.target.value })} /></label>
    <label>To <input className="select" type="date" value={filter.to ?? ''} min={filter.from} onChange={e => setFilter({ ...filter, to: e.target.value })} /></label>
    <div className="dash-sheet-actions"><button className="btn" type="button" disabled={!rows.length || !!filter.ids} onClick={() => setFilter({ ids: rows.map(r => r.id), label: `${rows.length} filters retained` })}>Retain reference cohort</button>
      <button className="btn quiet" type="button" onClick={() => setFilter({})}>Use page filters</button></div>
    {filter.ids && <p>Reference membership is retained; data values can refresh. Missing source records are reported in the tile information.</p>}
  </fieldset>
}

/** the tile's measured rows, for its ⋯ menu's "View data" */
export function tileDataRequest(rows: DashboardData['rows'], chart: DashboardChart, fields: string[]): DetailRequest {
  return { title: `${chart.label}: measured filters in this tile`, rows: measuredRows(rows, fields, chart.group === 'Evolution' || chart.kind === 'stat'), fields,
    note: chart.kind === 'scatter' || chart.kind === 'hexbin' ? 'Finite paired measurements before chart-specific axis clipping, legend hiding or grouping. Use a chart-specific contributing-row action where offered for an individual aggregate.' : 'Measured rows in this tile. Individual aggregate actions use their own exact contributing rows.' }
}

/** a tile's effective filters, measurement methods and source, opened from its ⋯ menu ("Data and filters") */
export function TileInformation({ data, chart, fields, filter, onClose, onFilters }: { data: DashboardData; chart: DashboardChart; fields: string[]; filter: CohortFilter; onClose: () => void; onFilters: () => void }) {
  const finite = measuredRows(data.rows, fields, chart.group === 'Evolution' || chart.kind === 'stat')
  const dates = data.rows.map(r => r.date).filter(Boolean).sort()
  const n = new Set(data.rows.map(r => r.id)).size
  const label = filter.ids ? 'Reference cohort' : filter.site || filter.season || filter.from || filter.to ? 'Tile + page filters' : 'Page filters'
  return <>
    <DashboardDialog title={`${chart.label} · data and filters`} onClose={onClose}>
      <p><strong>{label}</strong> · {n} physical filters · {fields.length ? `${finite.length} with finite selected measurements` : 'Chart-specific population'}</p>
      <p>{[...new Set(data.rows.map(r => r.site))].join(', ') || 'No sites'} · {dates[0] ?? 'No dates'} to {dates[dates.length - 1] ?? '—'}</p>
      <p>Seasons: {[...new Set(data.rows.map(r => r.season))].join('; ') || 'None'}. Calendar: {data.meta.season_convention}.</p>
      {filter.ids && <p>{filter.ids.length - n} retained IDs are unavailable under the current source selection. Re-select their source sites to make them available. This retains membership, not a frozen dataset.</p>}
      {fields.map(f => { const m = metricDefinition(f, data.meta); return <div key={f} className="metric-definition"><strong>{f} {m.unit && `(${m.unit})`}</strong><p>{m.method}. {m.missing}</p>{m.caveat && <p>{m.caveat}</p>}</div> })}
      <p>Source export: {data.provenance?.generated as string ?? data.meta.generated}. {data.provenance?.source as string ?? ''}</p>
      <p>Source mode: {String(data.provenance?.sourceMode ?? 'bundled')}; baseline: {String(data.provenance?.baseline ?? 'exported')}; MAC: {String(data.provenance?.mac ?? data.meta.mac_value)}; basis: {data.meta.basis ?? 'concentration'}.</p>
      {Array.isArray(data.provenance?.publicSources) && (data.provenance.publicSources as PublicSiteStatus[]).filter(s => data.rows.some(r => r.code === s.code)).map(s => <p key={s.code}>{s.code}: {s.state}; source updated {s.updated ?? 'not supplied'}; retrieved {s.checked}. {s.message ?? ''}</p>)}
      <button type="button" className="btn" onClick={() => { onClose(); onFilters() }}>Edit this tile’s filters</button>
    </DashboardDialog>
  </>
}

export function DetailsDialog({ request, onClose }: { request: DetailRequest; onClose: () => void }) {
  const actions = useHighlightActions()
  const data = useContext(TileData)
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(0)
  const rows = request.rows.filter(r => `${r.id} ${r.site} ${r.date} ${r.season}`.toLowerCase().includes(query.toLowerCase()))
  const fields = request.fields.length ? request.fields : data?.meta.fields.slice(0, 3) ?? []
  const count = 50
  return <DashboardDialog title={request.title} onClose={onClose}>
    <p>{request.rows.length} contributing physical filters. {request.note}</p>
    <div className="dashboard-data-actions"><label>Find a filter <input className="select" type="search" value={query} onChange={e => { setQuery(e.target.value); setPage(0) }} /></label>
      <button type="button" className="btn" onClick={() => downloadBlob(new Blob([rowsCsv(rows, fields)], { type: 'text/csv;charset=utf-8' }), `${slug(request.title)}.csv`)}>Download CSV ({rows.length})</button></div>
    <div className="dashboard-data-table" tabIndex={0} role="region" aria-label="Contributing filters table"><table className="census-table"><thead><tr><th>Filter</th><th>Site</th><th>Date</th><th>Season</th>{fields.map(f => <th key={f}>{f}{data?.meta.field_units[f] ? ` (${data.meta.field_units[f]})` : ''}</th>)}<th>Exclusion reason</th></tr></thead>
      <tbody>{rows.slice(page * count, (page + 1) * count).map(r => <tr key={r.id}><td><button type="button" className="btn quiet" onClick={() => { onClose(); actions.openRecord({ id: r.id, site: r.site, date: r.date, source: request.title, fields: fields.map(f => [f, r[f] == null ? null : String(r[f])]) }) }}>{r.id}</button></td>
        <td>{r.site}</td><td>{r.date}</td><td>{r.season}</td>{fields.map(f => <td key={f}>{typeof r[f] === 'number' ? (r[f] as number).toPrecision(5) : '—'}</td>)}<td>{r.exclusion_reason ?? (r.excluded ? 'Export exclusion' : '—')}</td></tr>)}</tbody></table></div>
    {!rows.length && <p>No filters match this search.</p>}
    <div className="dashboard-data-actions"><button type="button" className="btn" disabled={!page} onClick={() => setPage(p => p - 1)}>Previous</button><span>{rows.length ? page * count + 1 : 0}–{Math.min((page + 1) * count, rows.length)} of {rows.length}</span><button type="button" className="btn" disabled={(page + 1) * count >= rows.length} onClick={() => setPage(p => p + 1)}>Next</button></div>
  </DashboardDialog>
}
