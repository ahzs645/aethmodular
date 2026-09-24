import { useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Segmented, SwapButton } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { regression, fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import type { FilterRecord } from '@/lib/highlight'
import { RecordSection } from '@/components/RecordDrawer'
import { FilterSpectrum } from '@/components/FilterSpectrum'

const SCOPES = ['Same site', 'All sites'] as const

/** English ordinal suffix: 1st, 2nd, 3rd, 11th, 51st. */
const ordinal = (n: number) => (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th')

/** Ratios worth reading off a single filter. Each is a pair of field names. */
const RATIOS: { label: string; num: string; den: string; why: string }[] = [
  { label: 'HIPS BC / EC (FTIR)', num: 'HIPS BC', den: 'EC (FTIR)', why: 'optical vs calibrated FTIR; the crossplot slope, per filter' },
  { label: 'OC / EC (ChemSpec FTIR)', num: 'OC (ChemSpec FTIR)', den: 'EC (ChemSpec FTIR)', why: 'high = secondary or biomass, low = fresh combustion' },
  { label: 'OM / OC (FTIR)', num: 'OM (FTIR)', den: 'OC (FTIR)', why: 'oxygenation of the organics; 1.4–2.2 is typical' },
  { label: 'EC (ChemSpec FTIR) / PM2.5', num: 'EC (ChemSpec FTIR)', den: 'PM2.5 mass', why: 'FTIR-derived EC share of mass' },
]

/**
 * The drawer that opens when a sample is clicked. Charts only ever show two
 * or three numbers per filter; this shows all of them, and *where each sits*
 * within its site — a value means nothing without its distribution.
 *
 * Non-modal on purpose: the charts stay live behind it so the pinned point
 * is still visible in context.
 */
export function SampleDrawer({
  row,
  allRows,
  meta,
  xField,
  yField,
  focusField,
  pinned,
  onPin,
  onOpen,
  onClose,
  context,
}: {
  /** what the clicked chart knows about this filter (e.g. an AIRSpec prediction); shown first */
  context?: FilterRecord | null
  row: FilterRow
  /** every exported filter, unfiltered — neighbours and percentiles need the whole site */
  allRows: FilterRow[]
  meta: MetaFile
  xField: string
  yField: string
  focusField: string
  pinned: boolean
  onPin: (id: string | null) => void
  onOpen: (id: string) => void
  onClose: () => void
}) {
  const [cx, setCx] = useState(xField)
  const [cy, setCy] = useState(yField)
  const [scope, setScope] = useState<(typeof SCOPES)[number]>('Same site')
  const [copied, setCopied] = useState(false)

  // follow the bar's pair when it changes, but let the drawer diverge
  useEffect(() => { setCx(xField) }, [xField])
  useEffect(() => { setCy(yField) }, [yField])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowLeft' && prev) onOpen(prev.id)
      if (e.key === 'ArrowRight' && next) onOpen(next.id)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const site = meta.sites.find((s) => s.name === row.site)
  const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

  // the site's clean filters in date order — the frame everything is judged against
  const siteRows = useMemo(
    () => allRows.filter((r) => r.site === row.site && !r.excluded).sort((a, b) => (a.date < b.date ? -1 : 1)),
    [allRows, row.site]
  )
  const idx = siteRows.findIndex((r) => r.id === row.id)
  const prev = idx > 0 ? siteRows[idx - 1] : null
  const next = idx >= 0 && idx < siteRows.length - 1 ? siteRows[idx + 1] : null
  const neighbours = useMemo(() => {
    if (idx < 0) return []
    return siteRows.slice(Math.max(0, idx - 3), idx + 4)
  }, [siteRows, idx])

  /** per field: the sample's value, its site percentile, and the site's quartiles */
  const placement = useMemo(() => {
    const out: Record<string, { v: number; pct: number; min: number; q1: number; med: number; q3: number; max: number; n: number }> = {}
    for (const f of meta.fields) {
      const v = row[f]
      if (!isNum(v)) continue
      const vals = siteRows.map((r) => r[f]).filter(isNum).sort(d3.ascending)
      if (vals.length < 4) continue
      const below = vals.filter((x) => x < v).length
      out[f] = {
        v,
        pct: (below / vals.length) * 100,
        min: vals[0],
        q1: d3.quantile(vals, 0.25) ?? vals[0],
        med: d3.quantile(vals, 0.5) ?? vals[0],
        q3: d3.quantile(vals, 0.75) ?? vals[0],
        max: vals[vals.length - 1],
        n: vals.length,
      }
    }
    return out
  }, [row, siteRows, meta.fields])

  const ratios = RATIOS.map((r) => {
    const a = row[r.num]
    const b = row[r.den]
    const ok = isNum(a) && isNum(b) && b !== 0
    // site median of the same ratio, so the number has a reference
    const siteVals = siteRows
      .map((s) => (isNum(s[r.num]) && isNum(s[r.den]) && (s[r.den] as number) !== 0 ? (s[r.num] as number) / (s[r.den] as number) : null))
      .filter((v): v is number => v !== null)
    return { ...r, value: ok ? (a as number) / (b as number) : null, siteMedian: siteVals.length ? d3.median(siteVals) ?? null : null }
  }).filter((r) => r.value !== null)

  // ---- mini crossplot
  const cross = useMemo(() => {
    const pool = (scope === 'Same site' ? siteRows : allRows.filter((r) => !r.excluded))
      .map((r) => ({ r, x: r[cx], y: r[cy] }))
      .filter((p): p is { r: FilterRow; x: number; y: number } => isNum(p.x) && isNum(p.y))
    const stats = regression(pool.map((p) => p.x), pool.map((p) => p.y))
    return { pool, stats }
  }, [scope, siteRows, allRows, cx, cy])
  const W = 400
  const m = { top: 12, right: 14, bottom: 46, left: 62 }
  const iw = W - m.left - m.right
  // square plot area, like every crossplot in the gallery
  const ih = iw
  const H = ih + m.top + m.bottom
  const xs = d3.scaleLinear().domain([Math.min(0, d3.min(cross.pool, (p) => p.x) ?? 0), (d3.max(cross.pool, (p) => p.x) ?? 1) * 1.05]).range([0, iw]).nice()
  const ys = d3.scaleLinear().domain([Math.min(0, d3.min(cross.pool, (p) => p.y) ?? 0), (d3.max(cross.pool, (p) => p.y) ?? 1) * 1.05]).range([ih, 0]).nice()
  const here = isNum(row[cx]) && isNum(row[cy]) ? { x: row[cx] as number, y: row[cy] as number } : null
  const resid = here && cross.stats ? here.y - (cross.stats.slope * here.x + cross.stats.intercept) : null

  const link = () => {
    const url = new URL(window.location.href)
    const p = new URLSearchParams(url.hash.replace(/^#/, ''))
    p.set('pin', row.id)
    url.hash = p.toString()
    navigator.clipboard?.writeText(url.toString()).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <aside className="drawer" role="dialog" aria-label={`Sample ${row.id}`}>
      <header className="drawer-head">
        <div>
          <div className="drawer-id">
            <span className="swatch" style={{ background: site?.color ?? INK.muted }} />
            {row.id}
            {row.excluded && <span className="badge danger">excluded</span>}
            {pinned && <span className="badge">pinned</span>}
          </div>
          <div className="drawer-sub">
            {row.site} ({row.code}) · {row.date} · {row.season}
            {idx >= 0 && <> · filter {idx + 1} of {siteRows.length} at this site</>}
          </div>
          {row.excluded && <div className="drawer-reason">{row.exclusion_reason}</div>}
        </div>
        <button type="button" className="btn quiet" onClick={onClose} aria-label="Close" title="Close (Esc)">✕</button>
      </header>

      <div className="drawer-actions">
        <button type="button" className="btn" disabled={!prev} onClick={() => prev && onOpen(prev.id)} title="Previous filter at this site (←)">← {prev?.date ?? '—'}</button>
        <button type="button" className="btn" disabled={!next} onClick={() => next && onOpen(next.id)} title="Next filter at this site (→)">{next?.date ?? '—'} →</button>
        <span style={{ flex: 1 }} />
        <button type="button" className="btn" onClick={() => onPin(pinned ? null : row.id)}>{pinned ? 'unpin' : 'pin'}</button>
        <button type="button" className="btn" onClick={link}>{copied ? 'copied' : 'copy link'}</button>
      </div>

      {context && <RecordSection rec={context} />}

      {/* Addis filters resolve to their similarity trace via etad_ids.json; nothing renders without one */}
      <FilterSpectrum id={row.id} />

      <section className="drawer-section">
        <h3>Where this filter sits at {row.site}</h3>
        <p className="chart-note">Each bar is the site's clean range for that measurement; the darker band is the interquartile range, the tick is the site median, the dot is this filter. Percentile is within the site.</p>
        {meta.field_groups.map((g) => {
          const fields = g.fields.filter((f) => f in placement)
          if (!fields.length) return null
          return (
            <table key={g.label} className="placement">
              <thead>
                <tr><th colSpan={4}>{g.label}</th></tr>
              </thead>
              <tbody>
                {fields.map((f) => {
                  const p = placement[f]
                  const s = d3.scaleLinear().domain([p.min, p.max]).range([0, 100])
                  const hot = f === xField || f === yField || f === focusField
                  return (
                    <tr key={f} className={hot ? 'hot' : ''}>
                      <td className="f">{f}</td>
                      <td className="v">{fmt(p.v, 3)} <span className="u">{prettyUnit(meta.field_units[f])}</span></td>
                      <td className="strip" title={`min ${fmt(p.min, 2)} · q1 ${fmt(p.q1, 2)} · median ${fmt(p.med, 2)} · q3 ${fmt(p.q3, 2)} · max ${fmt(p.max, 2)} · n=${p.n}`}>
                        <span className="track" />
                        <span className="iqr" style={{ left: `${s(p.q1)}%`, width: `${Math.max(1, s(p.q3) - s(p.q1))}%` }} />
                        <span className="med" style={{ left: `${s(p.med)}%` }} />
                        <span className="dot" style={{ left: `${s(p.v)}%`, background: site?.color ?? INK.text }} />
                      </td>
                      <td className="pct">{p.pct.toFixed(0)}<span className="u">{ordinal(Math.round(p.pct))}</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )
        })}
      </section>

      {ratios.length > 0 && (
        <section className="drawer-section">
          <h3>Ratios</h3>
          <table className="placement">
            <tbody>
              {ratios.map((r) => (
                <tr key={r.label}>
                  <td className="f" title={r.why}>{r.label}</td>
                  <td className="v">{fmt(r.value, 3)}</td>
                  <td className="u" colSpan={2}>site median {fmt(r.siteMedian, 3)} · {r.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="drawer-section">
        <h3>This filter on any crossplot</h3>
        <div className="frame-controls" style={{ marginBottom: 6 }}>
          <FieldSelect label="x" value={cx} meta={meta} onChange={setCx} />
          <SwapButton onClick={() => { setCx(cy); setCy(cx) }} />
          <FieldSelect label="y" value={cy} meta={meta} onChange={setCy} />
          <Segmented value={scope} options={SCOPES} onChange={setScope} />
        </div>
        {cross.pool.length < 3 ? (
          <p className="frame-empty">Fewer than 3 filters carry both fields.</p>
        ) : (
          <svg width={W} height={H} style={{ display: 'block', maxWidth: '100%' }}>
            <g transform={`translate(${m.left},${m.top})`}>
              <YAxis scale={ys} x={0} label={cy} gridWidth={iw} tickCount={5} labelX={-48} />
              <XAxis scale={xs} y={ih} label={cx} tickCount={5} />
              {cross.pool.map((p) => (
                <circle key={p.r.id} cx={xs(p.x)} cy={ys(p.y)} r={2.4} fill={scope === 'All sites' ? (p.r.color as string) : site?.color ?? INK.muted} fillOpacity={0.35} style={{ cursor: 'pointer' }} onClick={() => onOpen(p.r.id)}>
                  <title>{p.r.id} · {p.r.date}</title>
                </circle>
              ))}
              {cross.stats && (
                <line
                  x1={xs(xs.domain()[0])} y1={ys(cross.stats.slope * xs.domain()[0] + cross.stats.intercept)}
                  x2={xs(xs.domain()[1])} y2={ys(cross.stats.slope * xs.domain()[1] + cross.stats.intercept)}
                  stroke={INK.fit} strokeWidth={1.5} pointerEvents="none"
                />
              )}
              {here && (
                <>
                  <line x1={xs(here.x)} x2={xs(here.x)} y1={0} y2={ih} stroke={INK.text} strokeDasharray="3 3" strokeOpacity={0.4} />
                  <line x1={0} x2={iw} y1={ys(here.y)} y2={ys(here.y)} stroke={INK.text} strokeDasharray="3 3" strokeOpacity={0.4} />
                  <circle cx={xs(here.x)} cy={ys(here.y)} r={7} fill={site?.color ?? INK.text} stroke="#111827" strokeWidth={2} />
                </>
              )}
            </g>
          </svg>
        )}
        {here ? (
          <p className="chart-note" style={{ fontFamily: FONT.mono }}>
            {cross.stats && <>n = {cross.stats.n} · R² = {fmt(cross.stats.r2, 3)} · OLS slope = {fmt(cross.stats.slope, 3)}</>}
            {resid !== null && <> · this filter is {fmt(Math.abs(resid), 2)} {resid >= 0 ? 'above' : 'below'} the fit</>}
          </p>
        ) : (
          <p className="chart-note">This filter does not carry both {cx} and {cy}.</p>
        )}
      </section>

      {neighbours.length > 1 && (
        <section className="drawer-section">
          <h3>Neighbours in time at {row.site}</h3>
          <table className="placement">
            <thead>
              <tr><th>filter</th><th>date</th><th>{focusField}</th><th>{focusField === xField ? yField : xField}</th></tr>
            </thead>
            <tbody>
              {neighbours.map((n) => (
                <tr key={n.id} className={n.id === row.id ? 'hot' : 'link'} onClick={() => n.id !== row.id && onOpen(n.id)}>
                  <td className="f">{n.id}</td>
                  <td className="u">{n.date}</td>
                  <td className="v">{fmt(n[focusField] as number | null, 2)}</td>
                  <td className="v">{fmt(n[focusField === xField ? yField : xField] as number | null, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </aside>
  )
}
