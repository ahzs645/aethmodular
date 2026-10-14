import { useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { useFilterEdits, type FilterGroup } from '@/lib/filterEdits'
import { useHighlight } from '@/lib/highlight'
import { fmt } from '@/lib/stats'
import { chemText } from '@/lib/format'
import { prettyUnit } from '@/components/FieldSelect'
import type { FilterRow, MetaFile } from '@/lib/types'

/**
 * The side panel for a bin of filters (a hex, a histogram bar): every filter in it,
 * with its site, date and the chart's values, and the same exclude / keep edits as the
 * right-click menu, for the whole bin or for the ticked rows. Excluded rows stay listed
 * (struck through) so they can be put back from here.
 *
 * Clicking a row opens that filter's own drawer; closing it comes back to this list.
 */
export function GroupDrawer({ group, byId, meta, fields, onClose }: {
  group: FilterGroup
  /** every filter the tab could show, before hand edits, so excluded rows can still be listed */
  byId: Map<string, FilterRow>
  meta: MetaFile
  /** the chart's fields, when the group does not name its own */
  fields: string[]
  onClose: () => void
}) {
  const ed = useFilterEdits()
  const hl = useHighlight()
  const [ticked, setTicked] = useState<Set<string>>(new Set())
  useEffect(() => setTicked(new Set()), [group])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const cols = (group.fields ?? fields).filter((f, i, a) => a.indexOf(f) === i)
  const rows = useMemo(() => group.ids.map((id) => byId.get(id)).filter((r): r is FilterRow => !!r)
    .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : a.id < b.id ? -1 : 1)), [group, byId])
  const live = rows.filter((r) => !ed.dropped.has(r.id))
  const out = rows.filter((r) => ed.dropped.has(r.id))
  const sites = d3.rollups(rows, (v) => v.length, (r) => r.site).sort((a, b) => b[1] - a[1])
  const colorOf = (site: string) => meta.sites.find((s) => s.name === site)?.color ?? '#9aa5b1'
  const unit = (f: string) => (meta.field_units?.[f] ? prettyUnit(meta.field_units[f]) : '')
  const tick = (id: string) => setTicked((cur) => { const n = new Set(cur); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const tickedLive = live.filter((r) => ticked.has(r.id)).map((r) => r.id)
  const all = live.length > 0 && tickedLive.length === live.length

  return (
    <aside className="drawer" role="dialog" aria-label={`${rows.length} filters`}>
      <header className="drawer-head">
        <div>
          <div className="drawer-id">{rows.length} filter{rows.length === 1 ? '' : 's'}</div>
          <div className="drawer-sub">{chemText(group.title)}</div>
        </div>
        <button type="button" className="btn quiet" onClick={onClose} aria-label="Close" title="Close (Esc)">✕</button>
      </header>

      <div className="drawer-actions">
        {ed.enabled ? <>
          <button type="button" className="btn" disabled={!live.length} onClick={() => ed.drop(live.map((r) => r.id))} title="Remove these filters from every chart; the Filters panel lists them and can put them back">Exclude all {live.length}</button>
          <button type="button" className="btn" disabled={!live.length} onClick={() => ed.keep(live.map((r) => r.id))} title="Show only these filters in every chart">Keep only these</button>
          {out.length > 0 && <button type="button" className="btn quiet" onClick={() => ed.restore(out.map((r) => r.id))}>Put {out.length} back</button>}
        </> : <span className="u">Hand exclusions apply on the network tabs only.</span>}
      </div>

      <div className="drawer-section">
        <h3>Sites</h3>
        <div className="group-sites">
          {sites.map(([s, n]) => <span key={s} className="group-site"><span className="swatch" style={{ background: colorOf(s) }} />{s} <span className="u">{n}</span></span>)}
        </div>
        {cols.length > 0 && <table className="placement group-summary">
          <thead><tr><th>Field</th><th>Median</th><th>Range</th></tr></thead>
          <tbody>
            {cols.map((f) => {
              const vs = rows.map((r) => r[f]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
              return <tr key={f}><td>{chemText(f)}{unit(f) ? <span className="u"> {unit(f)}</span> : null}</td>
                <td className="mono">{vs.length ? fmt(d3.median(vs)!, 3) : '–'}</td>
                <td className="mono">{vs.length ? `${fmt(d3.min(vs)!, 3)} – ${fmt(d3.max(vs)!, 3)}` : '–'}</td></tr>
            })}
          </tbody>
        </table>}
      </div>

      <div className="drawer-section">
        {ed.enabled && <div className="group-tick-bar">
          <label className="toggle"><input type="checkbox" checked={all} disabled={!live.length}
            onChange={() => setTicked(all ? new Set() : new Set(live.map((r) => r.id)))} />{tickedLive.length ? `${tickedLive.length} ticked` : 'Tick rows'}</label>
          {tickedLive.length > 0 && <>
            <button type="button" className="btn quiet" onClick={() => { ed.drop(tickedLive); setTicked(new Set()) }}>Exclude ticked</button>
            <button type="button" className="btn quiet" onClick={() => { ed.keep(tickedLive); setTicked(new Set()) }}>Keep only ticked</button>
          </>}
        </div>}
        <table className="placement group-rows">
          <thead><tr>{ed.enabled && <th />}<th>Filter</th><th>Date</th>{cols.map((f) => <th key={f} className="num">{chemText(f)}</th>)}{ed.enabled && <th />}</tr></thead>
          <tbody>
            {rows.map((r) => {
              const gone = ed.dropped.has(r.id)
              return (
                <tr key={r.id} className={gone ? 'gone' : ''} onMouseEnter={() => !gone && hl.setHover(r.id)} onMouseLeave={() => hl.setHover(null)}>
                  {ed.enabled && <td><input type="checkbox" aria-label={`Tick ${r.id}`} disabled={gone} checked={ticked.has(r.id) && !gone} onChange={() => tick(r.id)} /></td>}
                  <td><button type="button" className="link-btn" disabled={gone} title={gone ? 'Excluded by hand' : `${r.site}: open this filter`} onClick={() => hl.openSample(r.id)}>
                    <span className="swatch" style={{ background: colorOf(r.site) }} />{r.id}</button></td>
                  <td className="mono">{r.date}</td>
                  {cols.map((f) => <td key={f} className="mono num">{typeof r[f] === 'number' ? fmt(r[f] as number, 3) : '–'}</td>)}
                  {ed.enabled && <td>{gone
                    ? <button type="button" className="btn quiet" title="Put this filter back" onClick={() => ed.restore([r.id])}>↺</button>
                    : <button type="button" className="btn quiet" title="Exclude this filter" aria-label={`Exclude ${r.id}`} onClick={() => ed.drop([r.id])}>×</button>}</td>}
                </tr>
              )
            })}
          </tbody>
        </table>
        {rows.length < group.ids.length && <p className="u">{group.ids.length - rows.length} filters in the bin are not in the current data.</p>}
      </div>
    </aside>
  )
}
