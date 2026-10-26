import { fitUncertainty } from '@/lib/fitUncertainty'
import { PAPER_COLORS, PAPER_ORDER, paperSeasonOf } from '@/lib/pmfPaper'
import { fieldHelp } from '@/lib/glossary'
import { seasonOnly } from '@/lib/seasons'
import { chemText } from '@/lib/format'
import { useFilterReference } from '@/hooks/useFilterReference'
import { SitePicker } from '@/components/SitePicker'
import { RegressionLines, RegressionGroupLines, useRegressionSummary } from '@/components/RegressionOverlay'
import { useRegressions, FitGroupingControl } from '@/lib/regressionPrefs'
import { groupedRegressions } from '@/lib/siteRegression'
import { Fragment, useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Select } from '@/components/ChartFrame'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { RATIO_GROUP } from '@/lib/viewFields'
import { XAxis, YAxis } from '@/components/Axes'
import { regression, fmt } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import type { FilterRecord } from '@/lib/highlight'
import { RecordSection } from '@/components/RecordDrawer'
import { FilterSpectrum } from '@/components/FilterSpectrum'
import { HipsReadings } from '@/components/NativeFilterMeasurements'
import { NO_PMF, PMF_SOURCE, pmfShareField } from '@/lib/pmfGroups'
import { placements, stripPosition, ratioPlacement, ratioReference, isFiniteNumber, type Placement, type ReferenceRow } from '@/lib/referencePopulation'
import { DrawerShell, DrawerScroll } from '@/components/DrawerShell'


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
  studyRows,
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
  studyRows: FilterRow[]
  meta: MetaFile
  xField: string
  yField: string
  focusField: string
  pinned: boolean
  onPin: (id: string | null) => void
  onOpen: (id: string) => void
  onClose: () => void
}) {
  const { methods, grouping } = useRegressions()
  const [cx, setCx] = useState(xField)
  const [cy, setCy] = useState(yField)
  const crossMeta = useMemo(() => ({ ...meta, field_groups: meta.field_groups.filter(g => g.label !== RATIO_GROUP) }), [meta])
  // Selecting the other axis's measurement swaps the pair, as in the main gallery.
  const chooseCrossField = (role: AxisRole, field: string) => {
    if (role === 'y') { if (field === cx) setCx(cy); setCy(field) }
    else { if (field === cy) setCy(cx); setCx(field) }
  }
  const swapCrossFields = () => { setCx(cy); setCy(cx) }
  const ref = useFilterReference(row, meta, allRows, studyRows)
  const { mode: reference, rows: referenceRows, label: referenceLabel } = ref
  const [copied, setCopied] = useState(false)

  // follow the bar's pair when it changes, but let the drawer diverge
  useEffect(() => { setCx(xField) }, [xField])
  useEffect(() => { setCy(yField) }, [yField])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as Element)?.closest('input, textarea, select, [role="combobox"], .axis-popover')) return
      if (document.querySelector('.axis-popover, .map-dialog-backdrop')) return
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowLeft' && prev) onOpen(prev.id)
      if (e.key === 'ArrowRight' && next) onOpen(next.id)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const site = meta.sites.find((s) => s.name === row.site)
  const isNum = isFiniteNumber
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
  // the group the raw HIPS readouts sit under (Optical (HIPS)); without one they close the section
  const hipsGroup = meta.field_groups.find((g) => /HIPS/i.test(g.label))?.label ?? null

  const ratios = RATIOS.map((r) => {
    const a = row[r.num]
    const b = row[r.den]
    const ok = isNum(a) && isNum(b) && b !== 0
    const value = ok ? (a as number) / (b as number) : null
    return { ...r, value, reference: ratioReference(referenceRows, r.num, r.den), place: value === null ? null : ratioPlacement(value, referenceRows, r.num, r.den) }
  }).filter((r) => r.value !== null)

  // one measurement (or ratio) against the reference: value, range strip with IQR and median, percentile
  const placeRow = (key: string, label: string, p: Placement, unit: string, hot = false, title?: string) => {
    const s = (value: number) => stripPosition(value, p)
    const outside = p.v < p.min ? 'below' : p.v > p.max ? 'above' : null
    return (
      <tr key={key} className={hot ? 'hot' : ''}>
        <td className="f" title={title}>{label}</td>
        <td className="v">{fmt(p.v, 3)} <span className="u">{unit}</span></td>
        <td className="strip" title={`${referenceLabel}: min ${fmt(p.min, 2)} · q1 ${fmt(p.q1, 2)} · median ${fmt(p.med, 2)} · q3 ${fmt(p.q3, 2)} · max ${fmt(p.max, 2)} · n=${p.n}${outside ? ` · filter is ${outside} the reference range; dot shown at the edge` : ''}`}>
          <span className="track" />
          <span className="iqr" style={{ left: `${s(p.q1)}%`, width: `${Math.max(1, s(p.q3) - s(p.q1))}%` }} />
          <span className="med" style={{ left: `${s(p.med)}%` }} />
          <span className={`dot${outside ? ' outside' : ''}`} data-outside-range={outside ?? undefined} style={{ left: `${s(p.v)}%`, background: site?.color ?? INK.text }} />
        </td>
        <td className="pct">{p.pct.toFixed(0)}<span className="u">{ordinal(Math.round(p.pct))}</span>{outside && <span className="u" style={{ display: 'block' }}>{outside} range</span>}</td>
      </tr>
    )
  }

  // ---- mini crossplot
  const cross = useMemo(() => {
    const pool = referenceRows
      .map((r) => ({ r, x: r[cx], y: r[cy] }))
      .filter((p): p is { r: ReferenceRow; x: number; y: number } => isNum(p.x) && isNum(p.y))
    const stats = regression(pool.map((p) => p.x), pool.map((p) => p.y), fitUncertainty(pool.map(p => p.r), cx, cy, meta))
    const fitGroups = groupedRegressions(pool, grouping, { x: p => p.x, y: p => p.y, site: p => p.r.site, season: p => String(p.r.season ?? ""), seasonColor: p => meta.seasons.find(se => se.name === p.r.season)?.color ?? INK.muted,
      color: p => ref.sites.find(s => s.code === p.r.code)?.color ?? INK.muted,
      options: members => fitUncertainty(members.map(p => p.r), cx, cy, meta) })
    return { pool, stats, fitGroups }
  }, [referenceRows, cx, cy, meta, grouping, ref.sites])
  const W = 400
  const m = { top: 12, right: 14, bottom: 46, left: 62 }
  const iw = W - m.left - m.right
  // square plot area, like every crossplot in the gallery
  const ih = iw
  const siteFits = grouping !== 'pooled' && methods.length > 0
  const summary = useRegressionSummary({ stats: cross.stats, groups: siteFits ? cross.fitGroups : undefined, width: W - 24 })
  const top = m.top + summary.height + 12
  const H = ih + top + m.bottom
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
            {/* the site is already named: its season without the "Site · " prefix */}
            {row.site} ({row.code}) · {row.date}{row.season ? <> · {seasonOnly(String(row.season))}</> : null}
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
        <div className="frame-controls" style={{ margin: '8px 0' }} title={`${referenceLabel} · ${ref.loading ? '…' : referenceRows.length.toLocaleString()} filters after known exclusions${referenceDates.length > 0 ? ` · ${referenceDates[0]} to ${referenceDates[referenceDates.length - 1]}` : ''}. Coverage varies by measurement; hover a bar for n. Uses all dates, independently of the main graph filters.`}>
          <Select label="Reference population" value={reference} options={ref.options}
            optionLabel={v => v === 'same' ? `Filter's site (${row.site})` : v === 'study' ? 'Four gallery sites' : v === 'network' ? 'All SPARTAN sites' : v === 'sites' ? 'Choose multiple sites…' : v}
            onChange={ref.setMode} />
        </div>
        {reference === 'sites' && <details className="reference-sites" open>
          <summary>Reference sites · {ref.codes.length} selected</summary>
          <SitePicker items={ref.sites.map(s => ({ key: s.code, label: s.name, sub: s.code, lat: s.lat, lon: s.lon, color: s.color, n: s.n_filters, selected: ref.codes.includes(s.code) }))}
            onToggle={ref.toggleSite} onAll={() => ref.setMode('network')} allLabel="Use all SPARTAN sites" onClear={ref.clearSites} noun="reference sites" />
        </details>}
        {reference === 'network' && <p className="reference-summary">{ref.loading ? '…' : ref.networkSize} SPARTAN sites with filter data, weighted by physical filter count. Each measurement uses its own available coverage; missing measurements are not filled with zero.</p>}
        {ref.loading && <p className="frame-empty">Loading reference filters…</p>}
        {ref.error && <p className="frame-empty" role="alert">Could not load the reference: {ref.error}</p>}
        <p className="chart-note reference-guide">Bars show the reference range; the darker band is its interquartile range and the tick is its median. The dot retains this filter's value. Percentile is the percentage of measured reference values strictly below it. Values outside the range are marked at the nearest edge.</p>
        {!ref.loading && !ref.error && meta.field_groups.map((g) => {
          const fields = g.fields.filter(f => isNum(row[f]))
          // the raw HIPS readouts follow the delivered HIPS values
          const hipsHere = g.label === hipsGroup
          if (!fields.length) return hipsHere ? <HipsReadings key={g.label} id={row.id} /> : null
          return (
            <Fragment key={g.label}>
            <DrawerScroll label={`${g.label} measurements`}><table className="placement">
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
                  return placeRow(f, chemText(f), p, prettyUnit(meta.field_units[f]), f === xField || f === yField || f === focusField, chemText(fieldHelp(f)))
                })}
              </tbody>
            </table></DrawerScroll>
            {hipsHere && <HipsReadings id={row.id} />}
            </Fragment>
          )
        })}
        {!hipsGroup && <HipsReadings id={row.id} />}
        {!ref.loading && !ref.error && ratios.length > 0 && <DrawerScroll label="Measurement ratios"><table className="placement">
          <thead><tr><th colSpan={4}>Ratios</th></tr></thead>
          <tbody>{ratios.map((r) => r.place ? placeRow(r.label, chemText(r.label), r.place, '', false, r.why)
            : <tr key={r.label}><td className="f" title={r.why}>{chemText(r.label)}</td><td className="v">{fmt(r.value, 3)}</td><td className="u" colSpan={2}>Reference unavailable · n={r.reference.n}</td></tr>)}</tbody>
        </table></DrawerScroll>}
        {unavailableFields.length > 0 && <details className="reference-unavailable"><summary>{unavailableFields.length} measurements without a reference percentile</summary><p>{unavailableFields.join(', ')}. A percentile needs at least four finite reference values for that measurement.</p></details>}
      </section>

      <AxisFieldsProvider meta={crossMeta} value={{ x: cx, y: cy, field: cx }} set={chooseCrossField} ratios={false}>
        <section className="drawer-section drawer-crossplot">
          <div className="drawer-crossplot-head">
            <h3 title={`Reference: ${referenceLabel} · ${ref.loading ? '…' : cross.pool.length.toLocaleString()} paired filters; the highlighted filter is ${row.id} from ${row.site}.`}>This filter on any crossplot</h3>
            <button type="button" className="btn quiet drawer-swap" onClick={swapCrossFields} title="Swap x and y" aria-label="Swap x and y">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 3v6h-6" /><path d="M21 9l-2.7-2.7A9 9 0 1 0 21 12" />
              </svg>
            </button>
          </div>
          <div className="frame-controls" style={{ marginBottom: 6 }}><FitGroupingControl /></div>
          {ref.loading ? <p className="frame-empty">Loading reference crossplot…</p> : cross.pool.length < 3 ? (
            <>
              <div className="frame-controls">
                <FieldSelect label="x" value={cx} meta={crossMeta} onChange={f => chooseCrossField('x', f)} ratios={false} />
                <FieldSelect label="y" value={cy} meta={crossMeta} onChange={f => chooseCrossField('y', f)} ratios={false} />
              </div>
              <p className="frame-empty">Fewer than 3 filters in {referenceLabel} carry both {cx} and {cy}. Choose another pair or reference population.</p>
            </>
          ) : (
            <DrawerScroll label="Filter reference crossplot"><svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', height: 'auto' }}>
              <g transform="translate(12,6)">{summary.node}</g>
              <g className="regression-plot" transform={`translate(${m.left},${top})`}>
                <YAxis scale={ys} x={0} label={withUnit(cy, meta.field_units)} gridWidth={iw} tickCount={5} labelX={-48} pick="y" />
                <XAxis scale={xs} y={ih} label={withUnit(cx, meta.field_units)} tickCount={5} pick="x" />
                {cross.pool.map((p) => (
                  <circle key={p.r.id} cx={xs(p.x)} cy={ys(p.y)} r={2.4} fill={ref.sites.find(s => s.code === p.r.code)?.color ?? INK.muted} fillOpacity={0.35} style={{ cursor: knownIds.has(p.r.id) ? 'pointer' : 'default' }} onClick={() => { if (knownIds.has(p.r.id)) onOpen(p.r.id) }}>
                    <title>{p.r.id} · {p.r.site} · {p.r.date} · {cx}: {fmt(p.x, 3)} · {cy}: {fmt(p.y, 3)}</title>
                  </circle>
                ))}
                {grouping !== 'pooled' ? <RegressionGroupLines groups={cross.fitGroups} x={xs} y={ys} width={iw} height={ih} />
                  : <RegressionLines stats={cross.stats} x={xs} y={ys} domain={xs.domain()} width={iw} height={ih} />}
                {here && (
                  <>
                    <line x1={xs(here.x)} x2={xs(here.x)} y1={0} y2={ih} stroke={INK.text} strokeDasharray="3 3" strokeOpacity={0.4} />
                    <line x1={0} x2={iw} y1={ys(here.y)} y2={ys(here.y)} stroke={INK.text} strokeDasharray="3 3" strokeOpacity={0.4} />
                    <circle cx={xs(here.x)} cy={ys(here.y)} r={7} fill={site?.color ?? INK.text} stroke={INK.fit} strokeWidth={2} />
                  </>
                )}
              </g>
            </svg></DrawerScroll>
          )}
          {!here && <p className="chart-note">This filter does not carry both {cx} and {cy}.</p>}
        </section>
      </AxisFieldsProvider>

      {neighbours.length > 1 && (
        <section className="drawer-section">
          <h3>Neighbours in time at {row.site}</h3>
          <DrawerScroll label="Neighbouring filters"><table className="placement">
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
          </table></DrawerScroll>
        </section>
      )}
    </DrawerShell>
  )
}

/**
 * The filter's PMF day: its dominant source, the day's bar from the PMF paper's Figure 7 (the
 * sources stacked in the paper's order and colours) and each factor's share, with the paper's
 * season for the day. Shown where the site has a PMF solution.
 */
function PmfDay({ row, meta }: { row: FilterRow; meta: MetaFile }) {
  const sources = meta.pmf_sources ?? []
  const label = row[PMF_SOURCE]
  if (!sources.length || typeof label !== 'string' || label === NO_PMF) return null
  const colour = (name: string, fallback: string) => PAPER_COLORS[name] ?? fallback
  const shares = sources.map((s) => ({ ...s, color: colour(s.name, s.color), v: row[pmfShareField(s.name)] })).filter((s): s is typeof s & { v: number } => typeof s.v === 'number' && Number.isFinite(s.v))
    .sort((a, b) => b.v - a.v)
  if (!shares.length) return null
  const stack = [...shares].sort((a, b) => PAPER_ORDER.indexOf(a.name) - PAPER_ORDER.indexOf(b.name))
  const total = shares.reduce((t, s) => t + s.v, 0) || 1
  const season = paperSeasonOf(row.date)
  return (
    <section className="drawer-section">
      <h3>PMF sources on this day{season && <span className="pmf-season" title="The PMF paper's season for this date"><span className="pmf-dot" style={{ background: season.color }} />{season.name}</span>}</h3>
      <p className="pmf-dominant">
        Dominant: <span className="pmf-dot" style={{ background: colour(label, INK.neutral) }} /> <strong>{label}</strong> · {fmt(shares[0].v * 100, 0)} % of the PMF factors' total
      </p>
      <div className="pmf-daybar" role="img" aria-label={`The day's sources: ${stack.map((s) => `${s.name} ${fmt(s.v * 100, 0)} %`).join(', ')}`}>
        {stack.map((s) => <span key={s.name} style={{ width: `${(s.v / total) * 100}%`, background: s.color }} title={`${s.name}: ${fmt(s.v * 100, 1)} %`} />)}
      </div>
      <div className="pmf-shares">
        {shares.map((s) => <div key={s.name} className="pmf-share" title={`${s.name}: ${fmt(s.v * 100, 1)} %`}>
          <span className="pmf-share-name"><span className="pmf-dot" style={{ background: s.color }} />{s.name}</span>
          <span className="pmf-share-bar"><span style={{ width: `${Math.max(1, s.v * 100)}%`, background: s.color }} /></span>
          <span className="pmf-share-v">{fmt(s.v * 100, 0)} %</span>
        </div>)}
      </div>
    </section>
  )
}
