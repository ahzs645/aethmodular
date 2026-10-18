import { fitUncertainty } from '@/lib/fitUncertainty'
import { chemText } from '@/lib/format'
import { loadJson } from '@/lib/data'
import { RegressionLines, RegressionReadout, RegressionGroupLines, RegressionGroupReadouts, regressionGroupReadoutLayout } from '@/components/RegressionOverlay'
import { useRegressions, fitsFor, equation, FitGroupingControl } from '@/lib/regressionPrefs'
import { groupedRegressions } from '@/lib/siteRegression'
import { useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Select, SwapButton } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { regression, fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import type { FilterRecord } from '@/lib/highlight'
import { RecordSection } from '@/components/RecordDrawer'
import { FilterSpectrum } from '@/components/FilterSpectrum'
import { useSearchParam } from '@/lib/url'
import { NO_PMF, PMF_SOURCE, pmfShareField } from '@/lib/pmfGroups'
import { cleanReference, placements, stripPosition, ratioReference, isFiniteNumber, type ReferenceRow, type SpartanReference } from '@/lib/referencePopulation'
import { DrawerShell } from '@/components/DrawerShell'


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
 * within a selectable site or network reference population.
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
  /** every exported filter, unfiltered — reference distributions use the full available population */
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
  const { methods, grouping, showCI } = useRegressions()
  const [cx, setCx] = useState(xField)
  const [cy, setCy] = useState(yField)
  const referenceOptions = ['same', ...meta.sites.map(s => s.name), 'study', 'network']
  const [reference, setReference] = useSearchParam<string>('sampleRef', 'same', referenceOptions)
  const [network, setNetwork] = useState<SpartanReference | null>(null)
  const [networkError, setNetworkError] = useState('')
  const [copied, setCopied] = useState(false)

  // follow the bar's pair when it changes, but let the drawer diverge
  useEffect(() => { setCx(xField) }, [xField])
  useEffect(() => { setCy(yField) }, [yField])

  useEffect(() => {
    if (reference !== 'network' || network) return
    let live = true
    setNetworkError('')
    loadJson<SpartanReference>('spartan_reference.json')
      .then((data) => {
        if (data.schema_version !== 1 || !Array.isArray(data.rows)) throw new Error('Unsupported reference export')
        if (live) setNetwork(data)
      })
      .catch(e => { if (live) setNetworkError(String(e)) })
    return () => { live = false }
  }, [reference, network])

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
  const isNum = isFiniteNumber
  const referenceSite = reference === 'same' ? row.site : reference === 'network' || reference === 'study' ? undefined : reference
  const referenceLabel = reference === 'network' ? 'SPARTAN network' : reference === 'study' ? 'Four gallery sites' : referenceSite!
  const referenceRows = useMemo(() => cleanReference(reference === 'network' ? network?.rows ?? [] : allRows, referenceSite), [reference, network, allRows, referenceSite])
  const knownIds = useMemo(() => new Set(allRows.map(r => r.id)), [allRows])
  const referenceDates = referenceRows.map(r => r.date).filter(Boolean).sort()

  // Navigation remains within the selected filter's actual site.
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

  const placement = useMemo(() => placements(row, meta.fields, referenceRows), [row, referenceRows, meta.fields])
  const referenceCoverage = useMemo(() => Object.fromEntries(meta.fields.map(f => [f, referenceRows.filter(r => isNum(r[f])).length])), [meta.fields, referenceRows])
  const unavailableFields = meta.fields.filter(f => isNum(row[f]) && !(f in placement))

  const ratios = RATIOS.map((r) => {
    const a = row[r.num]
    const b = row[r.den]
    const ok = isNum(a) && isNum(b) && b !== 0
    return { ...r, value: ok ? (a as number) / (b as number) : null, reference: ratioReference(referenceRows, r.num, r.den) }
  }).filter((r) => r.value !== null)

  // ---- mini crossplot
  const cross = useMemo(() => {
    const pool = referenceRows
      .map((r) => ({ r, x: r[cx], y: r[cy] }))
      .filter((p): p is { r: ReferenceRow; x: number; y: number } => isNum(p.x) && isNum(p.y))
    const stats = regression(pool.map((p) => p.x), pool.map((p) => p.y), fitUncertainty(pool.map(p => p.r), cx, cy, meta))
    const fitGroups = groupedRegressions(pool, grouping, { x: p => p.x, y: p => p.y, site: p => p.r.site, season: p => String(p.r.season ?? ""), seasonColor: p => meta.seasons.find(se => se.name === p.r.season)?.color ?? INK.muted,
      color: p => meta.sites.find(s => s.code === p.r.code)?.color ?? INK.muted,
      options: members => fitUncertainty(members.map(p => p.r), cx, cy, meta) })
    return { pool, stats, fitGroups }
  }, [referenceRows, cx, cy, meta, grouping])
  const W = 400
  const m = { top: 12, right: 14, bottom: 46, left: 62 }
  const iw = W - m.left - m.right
  // square plot area, like every crossplot in the gallery
  const ih = iw
  const siteFits = grouping !== 'pooled' && methods.length > 0
  const H = ih + m.top + m.bottom + (siteFits ? regressionGroupReadoutLayout(cross.fitGroups.length, W, methods, showCI).height : 0)
  const here = isNum(row[cx]) && isNum(row[cy]) ? { x: row[cx] as number, y: row[cy] as number } : null
  const axisPoints = here ? [...cross.pool, here] : cross.pool
  const xs = d3.scaleLinear().domain([Math.min(0, d3.min(axisPoints, p => p.x) ?? 0), Math.max(0, d3.max(axisPoints, p => p.x) ?? 1) * 1.05 || 1]).range([0, iw]).nice()
  const ys = d3.scaleLinear().domain([Math.min(0, d3.min(axisPoints, p => p.y) ?? 0), Math.max(0, d3.max(axisPoints, p => p.y) ?? 1) * 1.05 || 1]).range([ih, 0]).nice()

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
    <DrawerShell label={`Sample ${row.id}`} onClose={onClose}>
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
        <button type="button" className="btn" onClick={() => onPin(pinned ? null : row.id)}>{pinned ? 'Unpin' : 'Pin'}</button>
        <button type="button" className="btn" onClick={link}>{copied ? 'Copied' : 'Copy link'}</button>
      </div>

      {context && <RecordSection rec={context} />}

      {/* Addis filters resolve to their similarity trace via etad_ids.json; nothing renders without one */}
      <FilterSpectrum id={row.id} />

      {/* the day's PMF solution, at a site that has one (lib/pmfGroups) */}
      <PmfDay row={row} meta={meta} />

      <section className="drawer-section">
        <h3>Where this filter sits {referenceLabel === row.site ? `at ${row.site}` : `against ${referenceLabel}`}</h3>
        <div className="frame-controls" style={{ margin: '8px 0' }}>
          <Select label="Reference population" value={reference} options={referenceOptions}
            optionLabel={v => v === 'same' ? `Filter's site (${row.site})` : v === 'study' ? 'Four gallery sites' : v === 'network' ? 'SPARTAN network — public filters' : v}
            onChange={setReference} />
        </div>
        <p className="reference-summary">{referenceLabel} · {reference === 'network' && !network ? '…' : referenceRows.length.toLocaleString()} filters after known exclusions{referenceDates.length > 0 && <> · {referenceDates[0]} to {referenceDates[referenceDates.length - 1]}</>}. Coverage varies by measurement; hover a bar for n. Uses all dates, independently of the main graph filters.</p>
        {reference === 'network' && <p className="reference-summary">Public SPARTAN chemistry at {network?.n_sites ?? '…'} sites, weighted by physical filter count. Carbon uses the public FTIR-derived reports. HIPS comes from batches 1–51 (27 sites); public BC is not substituted for HIPS BC. FTIR functional groups are unavailable.</p>}
        {reference === 'network' && !network && <p className="frame-empty">{networkError ? `Could not load the SPARTAN reference: ${networkError}` : 'Loading public SPARTAN filter reference…'}</p>}
        <p className="chart-note reference-guide">Bars show the reference range; the darker band is its interquartile range and the tick is its median. The dot retains this filter's value. Percentile is the percentage of measured reference values strictly below it. Values outside the range are marked at the nearest edge.</p>
        {meta.field_groups.map((g) => {
          const fields = g.fields.filter(f => isNum(row[f]))
          if (!fields.length) return null
          return (
            <table key={g.label} className="placement">
              <thead>
                <tr><th colSpan={4}>{g.label}</th></tr>
              </thead>
              <tbody>
                {fields.map((f) => {
                  const p = placement[f]
                  if (!p) return <tr key={f}>
                    <td className="f">{chemText(f)}</td>
                    <td className="v">{fmt(row[f] as number, 3)} <span className="u">{prettyUnit(meta.field_units[f])}</span></td>
                    <td className="u" colSpan={2}>Reference unavailable · n={referenceCoverage[f]}</td>
                  </tr>
                  const s = (value: number) => stripPosition(value, p)
                  const outside = p.v < p.min ? 'below' : p.v > p.max ? 'above' : null
                  const hot = f === xField || f === yField || f === focusField
                  return (
                    <tr key={f} className={hot ? 'hot' : ''}>
                      <td className="f">{chemText(f)}</td>
                      <td className="v">{fmt(p.v, 3)} <span className="u">{prettyUnit(meta.field_units[f])}</span></td>
                      <td className="strip" title={`${referenceLabel}: min ${fmt(p.min, 2)} · q1 ${fmt(p.q1, 2)} · median ${fmt(p.med, 2)} · q3 ${fmt(p.q3, 2)} · max ${fmt(p.max, 2)} · n=${p.n}${outside ? ` · filter is ${outside} the reference range; dot shown at the edge` : ''}`}>
                        <span className="track" />
                        <span className="iqr" style={{ left: `${s(p.q1)}%`, width: `${Math.max(1, s(p.q3) - s(p.q1))}%` }} />
                        <span className="med" style={{ left: `${s(p.med)}%` }} />
                        <span className={`dot${outside ? ' outside' : ''}`} data-outside-range={outside ?? undefined} style={{ left: `${s(p.v)}%`, background: site?.color ?? INK.text }} />
                      </td>
                      <td className="pct">{p.pct.toFixed(0)}<span className="u">{ordinal(Math.round(p.pct))}</span>{outside && <span className="u" style={{ display: 'block' }}>{outside} range</span>}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )
        })}
        {unavailableFields.length > 0 && <details className="reference-unavailable"><summary>{unavailableFields.length} measurements without a reference percentile</summary><p>{unavailableFields.join(', ')}. A percentile needs at least four finite reference values for that measurement.</p></details>}
      </section>

      {ratios.length > 0 && (
        <section className="drawer-section">
          <h3>Ratios</h3>
          <table className="placement">
            <tbody>
              {ratios.map((r) => (
                <tr key={r.label}>
                  <td className="f" title={r.why}>{chemText(r.label)}</td>
                  <td className="v">{fmt(r.value, 3)}</td>
                  <td className="u" colSpan={2}>{referenceLabel} median {fmt(r.reference.median, 3)} · n={r.reference.n}{r.reference.n === 0 && ' (unavailable)'} · {r.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="drawer-section">
        <h3>This filter on any crossplot</h3>
        <p className="reference-summary">Reference: {referenceLabel} · {cross.pool.length.toLocaleString()} paired filters. The highlighted filter remains {row.id} from {row.site}.</p>
        <div className="frame-controls" style={{ marginBottom: 6 }}>
          <FieldSelect label="x" value={cx} meta={meta} onChange={setCx} ratios={false} />
          <SwapButton onClick={() => { setCx(cy); setCy(cx) }} />
          <FieldSelect label="y" value={cy} meta={meta} onChange={setCy} ratios={false} />
          <FitGroupingControl />
        </div>
        {cross.pool.length < 3 ? (
          <p className="frame-empty">Fewer than 3 filters in {referenceLabel} carry both {cx} and {cy}. Choose another pair or reference population.</p>
        ) : (
          <svg width={W} height={H} style={{ display: 'block', maxWidth: '100%' }}>
            <g transform={`translate(${m.left},${m.top})`}>
              <YAxis scale={ys} x={0} label={cy} gridWidth={iw} tickCount={5} labelX={-48} />
              <XAxis scale={xs} y={ih} label={cx} tickCount={5} />
              {cross.pool.map((p) => (
                <circle key={p.r.id} cx={xs(p.x)} cy={ys(p.y)} r={2.4} fill={meta.sites.find(s => s.code === p.r.code)?.color ?? INK.muted} fillOpacity={0.35} style={{ cursor: knownIds.has(p.r.id) ? 'pointer' : 'default' }} onClick={() => { if (knownIds.has(p.r.id)) onOpen(p.r.id) }}>
                  <title>{p.r.id} · {p.r.site} · {p.r.date} · {cx}: {fmt(p.x, 3)} · {cy}: {fmt(p.y, 3)}</title>
                </circle>
              ))}
              {grouping !== 'pooled' ? <RegressionGroupLines groups={cross.fitGroups} x={xs} y={ys} width={iw} height={ih} />
                : <><RegressionLines stats={cross.stats} x={xs} y={ys} domain={xs.domain()} width={iw} height={ih} /><RegressionReadout stats={cross.stats} width={260} /></>}
              {here && (
                <>
                  <line x1={xs(here.x)} x2={xs(here.x)} y1={0} y2={ih} stroke={INK.text} strokeDasharray="3 3" strokeOpacity={0.4} />
                  <line x1={0} x2={iw} y1={ys(here.y)} y2={ys(here.y)} stroke={INK.text} strokeDasharray="3 3" strokeOpacity={0.4} />
                  <circle cx={xs(here.x)} cy={ys(here.y)} r={7} fill={site?.color ?? INK.text} stroke="#111827" strokeWidth={2} />
                </>
              )}
            </g>
            {siteFits && <RegressionGroupReadouts groups={cross.fitGroups} width={W} y={ih + m.top + m.bottom} />}
          </svg>
        )}
        {here ? (
          <p className="chart-note" style={{ fontFamily: FONT.mono }}>
            {grouping === 'pooled' && cross.stats && <>n = {cross.stats.n} · R² = {fmt(cross.stats.r2, 3)}{fitsFor(cross.stats, methods).map(fit => <span key={fit.method}> · {equation(fit)}</span>)}</>}

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
    </DrawerShell>
  )
}

/** The filter's PMF day: its dominant source and each factor's share, where its site has a PMF solution. */
function PmfDay({ row, meta }: { row: FilterRow; meta: MetaFile }) {
  const sources = meta.pmf_sources ?? []
  const label = row[PMF_SOURCE]
  if (!sources.length || typeof label !== 'string') return null
  const shares = sources.map((s) => ({ ...s, v: row[pmfShareField(s.name)] })).filter((s): s is typeof s & { v: number } => typeof s.v === 'number')
    .sort((a, b) => b.v - a.v)
  const dominant = meta.pmf_groups?.find((g) => g.name === label)
  return (
    <section className="drawer-section">
      <h3>PMF sources on this day</h3>
      {label === NO_PMF || !shares.length
        ? <p className="chart-note">No PMF day for this filter: the PMF solution at {row.site} does not include its sampling date.</p>
        : <>
          <p className="pmf-dominant">
            Dominant: <span className="pmf-dot" style={{ background: dominant?.color ?? INK.neutral }} /> <strong>{label}</strong> · {fmt(shares[0].v * 100, 0)} % of the PMF factors' total
          </p>
          <div className="pmf-shares">
            {shares.map((s) => <div key={s.name} className="pmf-share" title={`${s.name}: ${fmt(s.v * 100, 1)} %`}>
              <span className="pmf-share-name">{s.name}</span>
              <span className="pmf-share-bar"><span style={{ width: `${Math.max(1, s.v * 100)}%`, background: s.color }} /></span>
              <span className="pmf-share-v">{fmt(s.v * 100, 0)} %</span>
            </div>)}
          </div>
        </>}
    </section>
  )
}
