import { fitUncertainty } from '@/lib/fitUncertainty'
import { PAPER_COLORS, PAPER_ORDER, paperSeasonOf } from '@/lib/pmfPaper'
import { fieldHelp } from '@/lib/glossary'
import { seasonOnly } from '@/lib/seasons'
import { chemText } from '@/lib/format'
import { useFilterReference, type SeasonOf } from '@/hooks/useFilterReference'
import { useChartGrouping } from '@/components/GroupingDialog'
import { Legend } from '@/components/Legend'
import { useSearchParam } from '@/lib/url'
import { MultiCombobox } from '@/components/MultiCombobox'
import { RegressionLines, RegressionGroupLines, useRegressionSummary } from '@/components/RegressionOverlay'
import { useRegressions, FitGroupingControl } from '@/lib/regressionPrefs'
import { bandRegressions, groupedRegressions } from '@/lib/siteRegression'
import { Fragment, useEffect, useMemo, useState } from 'react'
import * as d3 from 'd3'
import { FieldSelect, prettyUnit } from '@/components/FieldSelect'
import { Select, Toggle } from '@/components/ChartFrame'
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
import { placements, quantilePlacements, stripPosition, ratioPlacement, ratioReference, isFiniteNumber, type Placement, type ReferenceRow } from '@/lib/referencePopulation'
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

/** the drawer's include/exclude button, by where the filter stands now */
const EXCLUSION_ACTION = { 'export-out': 'Include in charts', 'export-in': 'Exclude again', hand: 'Put back', in: 'Exclude' } as const
const EXCLUSION_TITLE = {
  'export-out': 'The data export excludes this filter: put it back in every chart (kept in the link)',
  'export-in': 'Put back from the data export\'s exclusions: leave it out of the charts again',
  hand: 'Excluded by hand: put it back in every chart',
  in: 'Leave this filter out of every chart (excluded by hand, kept in the link; the reason can be added in Filters › excluded)',
} as const

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
  exclusion,
  seasonOf,
}: {
  /** the gallery's season of a filter (App seasonOf), so the reference's filters are grouped as the charts' are */
  seasonOf?: SeasonOf
  /** whether the charts draw this filter: left out by the data export (or put back), excluded by hand, or in */
  exclusion?: { state: 'export-out' | 'export-in' | 'hand' | 'in'; toggle: () => void }
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
  const ref = useFilterReference(row, meta, allRows, studyRows, seasonOf)
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
    // an excluded filter steps through its site's others from its own place
    () => allRows.filter((r) => r.site === row.site && (!r.excluded || r.id === row.id)).sort((a, b) => (a.date < b.date ? -1 : 1)),
    [allRows, row.site, row.id]
  )
  const idx = siteRows.findIndex((r) => r.id === row.id)
  const prev = idx > 0 ? siteRows[idx - 1] : null
  const next = idx >= 0 && idx < siteRows.length - 1 ? siteRows[idx + 1] : null
  const neighbours = useMemo(() => {
    if (idx < 0) return []
    return siteRows.slice(Math.max(0, idx - 3), idx + 4)
  }, [siteRows, idx])

  // the whole IMPROVE network comes as quantiles (its rows are only a sample, for the crossplot)
  const network = ref.summary
  const placement = useMemo(() => network ? quantilePlacements(row, meta.fields, network) : placements(row, meta.fields, referenceRows), [row, referenceRows, meta.fields, network])
  const referenceCoverage = useMemo(() => Object.fromEntries(meta.fields.map(f => [f, network ? network.quantiles[f]?.n ?? 0 : referenceRows.filter(r => isNum(r[f])).length])), [meta.fields, referenceRows, network])
  const referenceCount = network?.n ?? referenceRows.length
  const unavailableFields = meta.fields.filter(f => isNum(row[f]) && !(f in placement) && !ref.partial.has(f))
  // the group the raw HIPS readouts sit under (Optical (HIPS)); without one they close the section
  const hipsGroup = meta.field_groups.find((g) => /HIPS/i.test(g.label))?.label ?? null

  const ratios = RATIOS.map((r) => {
    const a = row[r.num]
    const b = row[r.den]
    const ok = isNum(a) && isNum(b) && b !== 0
    const value = ok ? (a as number) / (b as number) : null
    return { ...r, value, reference: ratioReference(referenceRows, r.num, r.den), place: value === null ? null : ratioPlacement(value, referenceRows, r.num, r.den) }
  // no ratio has network quantiles: against the IMPROVE summary they are left out rather than read from its sample
  }).filter((r) => r.value !== null && !network && !ref.partial.has(r.num) && !ref.partial.has(r.den))
  // several reference sites: a measurement that some of them never measured is left out (useFilterReference)
  const partialHere = [...ref.partial].filter(([f]) => isNum(row[f]))

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
  // the crossplot's grouping, as on the gallery's crossplots (components/GroupingDialog): bands of a measurement,
  // its own fits ("Fit per group") and colours; the first pick turns the per-group fits on
  const paired = useMemo(() => referenceRows.filter((r) => isNum(r[cx]) && isNum(r[cy])) as unknown as FilterRow[], [referenceRows, cx, cy])
  const [fitPerGroup, setFitPerGroup] = useSearchParam<string>('sampleFitGroup', '0')
  const { grp, picker: groupPicker, missingNote } = useChartGrouping('sample', paired, () => setFitPerGroup('1'))
  const perBand = !!grp && fitPerGroup === '1'
  const seasonColor = (season: unknown) => meta.seasons.find(se => se.name === season)?.color ?? INK.muted
  const siteColor = (code: unknown) => ref.sites.find(s => s.code === code)?.color ?? INK.muted
  const cross = useMemo(() => {
    const pool = referenceRows
      .map((r) => ({ r, x: r[cx], y: r[cy] }))
      .filter((p): p is { r: ReferenceRow; x: number; y: number } => isNum(p.x) && isNum(p.y))
    const stats = regression(pool.map((p) => p.x), pool.map((p) => p.y), fitUncertainty(pool.map(p => p.r), cx, cy, meta))
    const options = (members: typeof pool) => fitUncertainty(members.map(p => p.r), cx, cy, meta)
    const fitGroups = perBand && grp
      ? bandRegressions(pool, { labels: grp.labels, labelOf: p => grp.labelOf(p.r as unknown as FilterRow), colorOf: grp.colorOf }, { x: p => p.x, y: p => p.y, options })
      : groupedRegressions(pool, grouping, { x: p => p.x, y: p => p.y, site: p => p.r.site, season: p => String(p.r.season ?? ''), seasonColor: p => seasonColor(p.r.season),
        color: p => siteColor(p.r.code), options })
    return { pool, stats, fitGroups }
  }, [referenceRows, cx, cy, meta, grouping, ref.sites, perBand, grp]) // eslint-disable-line react-hooks/exhaustive-deps
  // the points take the colours of what splits them: a grouping's bands, else the seasons when fitting per season, else the sites
  const colourBy: 'band' | 'season' | 'site' = grp ? 'band' : grouping === 'season' ? 'season' : 'site'
  const pointColor = (r: ReferenceRow) => colourBy === 'band' ? grp!.colorOf(grp!.labelOf(r as unknown as FilterRow) ?? '') : colourBy === 'season' ? seasonColor(r.season) : siteColor(r.code)
  const crossLegend = colourBy === 'band' && grp
    ? grp.labels.map((l, i) => ({ label: l, color: grp.colors[i] }))
    : colourBy === 'season' ? ref.season.options.filter(se => cross.pool.some(p => p.r.season === se)).map(se => ({ label: se, color: seasonColor(se) })) : []
  const W = 400
  const m = { top: 12, right: 14, bottom: 46, left: 62 }
  const iw = W - m.left - m.right
  // square plot area, like every crossplot in the gallery
  const ih = iw
  const siteFits = (perBand || grouping !== 'pooled') && methods.length > 0
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
            {exclusion?.state === 'export-in' ? <span className="badge" title="Excluded in the data export, put back in the charts">excluded · put back</span>
              : exclusion?.state === 'hand' ? <span className="badge danger" title="Excluded by hand">excluded by hand</span>
              : row.excluded && <span className="badge danger">excluded</span>}
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
        {exclusion && <button type="button" className="btn" onClick={exclusion.toggle} title={EXCLUSION_TITLE[exclusion.state]}>{EXCLUSION_ACTION[exclusion.state]}</button>}
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
        <div className="frame-controls" style={{ margin: '8px 0' }} title={`${referenceLabel} · ${ref.loading ? '…' : referenceCount.toLocaleString()} filters after known exclusions${network ? ` · ${network.date_min} to ${network.date_max}` : referenceDates.length > 0 ? ` · ${referenceDates[0]} to ${referenceDates[referenceDates.length - 1]}` : ''}. Coverage varies by measurement; hover a bar for n. Uses all dates, independently of the main graph filters.`}>
          <Select label="Reference population" value={reference} options={ref.options}
            optionLabel={v => v === 'same' ? `Filter's site (${row.site})` : v === 'study' ? 'Four gallery sites' : v === 'network' ? 'All SPARTAN sites' : v === 'improve' ? 'All IMPROVE sites' : v === 'sites' ? 'Choose multiple sites…' : v}
            onChange={ref.setMode} />
          {ref.season.available && ref.season.options.length > 1 && <Select label="Season" value={ref.season.value}
            options={['all', ...(ref.season.own ? ['same'] : []), ...ref.season.options]}
            optionLabel={v => v === 'all' ? 'All seasons' : v === 'same' ? `This filter's season (${seasonOnly(ref.season.own ?? '')})` : v}
            onChange={ref.season.set} title="Compare against the reference in every season, in this filter's own season, or in one season (each filter's season under the gallery's calendar for its site)" />}
          {/* the gallery's several-at-once dropdown, the networks as its headings */}
          {reference === 'sites' && <MultiCombobox label="Sites" value={ref.codes} onChange={ref.setSites} placeholder="Choose sites…"
            title={`Reference sites, SPARTAN and IMPROVE${ref.improveNote ? ` (${ref.improveNote})` : ''}. With several sites, only what every chosen site measures is placed.`}
            options={ref.sites.map(s => ({ key: s.code, label: s.name, group: s.network, hint: s.n_filters.toLocaleString(), title: `${s.code} · ${s.n_filters.toLocaleString()} filters` }))} />}
        </div>
        {network && <p className="reference-summary">{network.n_sites} IMPROVE sites, {network.n.toLocaleString()} filters ({network.date_min.slice(0, 4)}–{network.date_max.slice(0, 4)}). Percentiles and ranges use every filter; the crossplot below shows an even sample of {network.sample_per_site} per site.</p>}
        {reference === 'network' && <p className="reference-summary">{ref.loading ? '…' : ref.networkSize} SPARTAN sites with filter data, weighted by physical filter count. Each measurement uses its own available coverage; missing measurements are not filled with zero.</p>}
        {ref.loading && <p className="frame-empty">Loading reference filters…</p>}
        {ref.error && <p className="frame-empty" role="alert">Could not load the reference: {ref.error}</p>}
        <p className="chart-note reference-guide">Bars show the reference range; the thicker band is its interquartile range and the tick is its median. The dot retains this filter's value. Percentile is the percentage of measured reference values strictly below it. Values outside the range are marked at the nearest edge.</p>
        {!ref.loading && !ref.error && meta.field_groups.map((g) => {
          const fields = g.fields.filter(f => isNum(row[f]) && !ref.partial.has(f))
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
        {!ref.loading && partialHere.length > 0 && (network
          ? <details className="reference-unavailable"><summary>{partialHere.length} measurement{partialHere.length === 1 ? '' : 's'} IMPROVE does not make</summary>
            <p>{partialHere.map(([f]) => chemText(f)).join(', ')}. IMPROVE measures EC and OC by TOR (not FTIR), and no ions or most elements, so these have no IMPROVE reference.</p></details>
          : <details className="reference-unavailable"><summary>{partialHere.length} measurement{partialHere.length === 1 ? '' : 's'} not measured at every reference site</summary>
            <ul>{partialHere.map(([f, lacking]) => <li key={f}>{chemText(f)}: not at {lacking.length <= 3 ? lacking.join(', ') : `${lacking.slice(0, 3).join(', ')} and ${lacking.length - 3} more`}</li>)}</ul>
            <p>A percentile against sites that never measured it would really be against the others only, so these are left out. Choose sites that all measure them to see them.</p></details>)}
        {unavailableFields.length > 0 && <details className="reference-unavailable"><summary>{unavailableFields.length} measurements without a reference percentile</summary><p>{unavailableFields.join(', ')}. A percentile needs at least four finite reference values for that measurement.</p></details>}
      </section>

      <AxisFieldsProvider meta={crossMeta} value={{ x: cx, y: cy, field: cx }} set={chooseCrossField} ratios={false}>
        <section className="drawer-section drawer-crossplot">
          <div className="drawer-crossplot-head">
            <h3 title={`Reference: ${referenceLabel} · ${ref.loading ? '…' : cross.pool.length.toLocaleString()} paired filters${network ? ` (an even sample of ${network.sample_per_site} per site from ${network.n.toLocaleString()})` : ''}; the highlighted filter is ${row.id} from ${row.site}.`}>This filter on any crossplot</h3>
            <button type="button" className="btn quiet drawer-swap" onClick={swapCrossFields} title="Swap x and y" aria-label="Swap x and y">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 3v6h-6" /><path d="M21 9l-2.7-2.7A9 9 0 1 0 21 12" />
              </svg>
            </button>
          </div>
          <div className="frame-controls" style={{ marginBottom: 6 }}>
            {!perBand && <FitGroupingControl />}
            {groupPicker}
            {grp && <Toggle label="Fit per group" checked={perBand} onChange={v => setFitPerGroup(v ? '1' : '0')} title="One fit per band of the grouping instead of the fit grouping" />}
          </div>
          {missingNote('so they are not coloured or fitted by band')}
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
                  <circle key={p.r.id} cx={xs(p.x)} cy={ys(p.y)} r={2.4} fill={pointColor(p.r)} fillOpacity={0.35} style={{ cursor: knownIds.has(p.r.id) ? 'pointer' : 'default' }} onClick={() => { if (knownIds.has(p.r.id)) onOpen(p.r.id) }}>
                    <title>{p.r.id} · {p.r.site} · {p.r.date} · {cx}: {fmt(p.x, 3)} · {cy}: {fmt(p.y, 3)}</title>
                  </circle>
                ))}
                {perBand || grouping !== 'pooled' ? <RegressionGroupLines groups={cross.fitGroups} x={xs} y={ys} width={iw} height={ih} />
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
          {!ref.loading && cross.pool.length >= 3 && crossLegend.length > 0 && <Legend items={crossLegend} />}
          {!here && <p className="chart-note">This filter does not carry both {cx} and {cy}.</p>}
          {here && (exclusion?.state === 'export-out' || exclusion?.state === 'hand') && <p className="chart-note">Drawn here for reference only: the gallery's charts leave this filter out until you put it back.</p>}
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
