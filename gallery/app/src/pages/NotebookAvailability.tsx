import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Select } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { dataUrl } from '@/lib/data'
import { useSearchParam } from '@/lib/url'
import { FONT, INK } from '@/lib/theme'
import { calendarMonths, studyOverlap, type AvailabilitySite, type CompletenessRule, type NotebookAvailability } from '@/lib/notebookAvailability'

const fmt = (n: number | null | undefined) => n == null ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: 1 })
const siteLabel = (s: AvailabilitySite) => s.name === s.code ? s.code : `${s.code} · ${s.name}`

export function NetworkAvailability({ data }: { data: NotebookAvailability }) {
  const { network } = data
  const [productKey, setProduct] = useSearchParam<string>('availability.product', 'HIPS', network.products.map(p => p.key))
  const [query, setQuery] = useSearchParam<string>('availability.productSearch', '')
  const [presence, setPresence] = useSearchParam<string>('availability.presence', 'All sites', ['All sites', 'Public + HIPS', 'Public only', 'HIPS only', 'Lookup only'])
  const [year, setYear] = useSearchParam<string>('availability.productYear', 'All years')
  const sites = useMemo(() => network.sites.filter(site => {
    const hasPublic = Object.entries(site.products).some(([key, p]) => key !== 'HIPS' && (p.n_samples ?? 0) > 0)
    const hasHips = (site.products.HIPS?.n_samples ?? 0) > 0
    const kind = hasPublic ? hasHips ? 'Public + HIPS' : 'Public only' : hasHips ? 'HIPS only' : 'Lookup only'
    return (presence === 'All sites' || kind === presence) && `${site.code} ${site.name}`.toLowerCase().includes(query.toLowerCase())
  }), [network, query, presence])
  const product = network.products.find(p => p.key === productKey) ?? network.products[0]
  const monthly = useMemo(() => network.monthly.filter(r => r.product === product.key), [network, product.key])
  const years = [...new Set(monthly.map(r => r.month.slice(0, 4)))].sort()
  const currentYear = years.includes(year) ? year : 'All years'
  const newer = network.sites.reduce((sum, s) => sum + Object.values(s.products).filter(p => p.newer_date_max).length, 0)
  return <>
    <ChartFrame id="availability-products" title="Network product coverage" subtitle="The saved SPARTAN coverage analysis plus the all-site HIPS notebook. Each product has its own unit and sampling population. Filter counts are unique IDs within a product; timestamp counts are not filter counts. Products overlap, so column totals must not be added." controls={<>
      <label className="control">Search sites <input className="select" aria-label="Search product coverage sites" type="search" value={query} onChange={e => setQuery(e.target.value)} /></label>
      <Select label="Presence" value={presence} options={['All sites', 'Public + HIPS', 'Public only', 'HIPS only', 'Lookup only']} onChange={setPresence} />
    </>}>
      <div className="availability-summary">
        <span><strong>{network.sites.length}</strong> sites in the combined inventory</span>
        <span><strong>{network.sites.filter(s => Object.entries(s.products).some(([k, p]) => k !== 'HIPS' && (p.n_samples ?? 0) > 0)).length}</strong> with dated public products</span>
        <span><strong>{network.sites.filter(s => (s.products.HIPS?.n_samples ?? 0) > 0).length}</strong> with HIPS sample filters</span>
        <span><strong>{fmt(network.sites.reduce((sum, s) => sum + (s.products.HIPS?.n_samples ?? 0), 0))}</strong> finite HIPS bridge filters · blanks kept separate</span>
      </div>
      <p className="availability-context">Saved analysis snapshot. The public inventory refreshed on {network.public_release} has later dates in {newer} site/product series. The cadence counts below retain the saved analysis; the Filter explorer view uses the gallery’s filter exports.</p>
      <p className="chart-note">HIPS monthly coverage uses the saved filter bridge. Its separate inventory reports {fmt(network.hips_inventory_total)} sample filters and {fmt(network.sites.reduce((sum, s) => sum + s.blanks, 0))} blanks. These snapshots differ. Availability also includes the {network.hips_nonpositive} finite zero/negative Fabs values omitted by the notebook’s positive-only heatmap.</p>
      <p className="chart-note">Select a product heading to inspect its monthly sampling and gaps. Counts use filters for chemistry/HIPS and distinct timestamps for daily/hourly products. A dash means no dated records in this analysis.</p>
      <div className="chart-scroll availability-matrix" tabIndex={0} role="region" aria-label="Site by product coverage, scroll for all products">
        <table className="census-table availability-product-table"><thead><tr><th>Site</th>{network.products.map(p => <th key={p.key} className={p.key === product.key ? 'active' : ''}><button className="pill-link" onClick={() => setProduct(p.key)} aria-pressed={p.key === product.key}>{p.label}</button><small>{p.unit}</small></th>)}<th>HIPS blanks<small>separate inventory</small></th></tr></thead><tbody>
          {sites.map(s => <tr key={s.code}><th scope="row"><strong>{s.code}</strong>{s.name !== s.code && <small>{s.name}</small>}</th>{network.products.map(p => <td key={p.key} className={(s.products[p.key]?.n_samples ?? 0) > 0 ? 'available' : ''} title={`${s.code} · ${p.label}: ${fmt(s.products[p.key]?.n_samples)} ${p.unit}; ${s.products[p.key]?.date_min ?? 'undated'} to ${s.products[p.key]?.date_max ?? 'undated'}`}>{fmt(s.products[p.key]?.n_samples)}</td>)}<td>{fmt(s.blanks)}</td></tr>)}
        </tbody></table>
      </div>
      {!sites.length && <Empty>No sites match this search and presence filter.</Empty>}
      <Note>Public BC and HIPS absorption can be related through an assumed MAC; their presence does not establish independent validation. This matrix includes PM10 separately from PM2.5. {network.future_inventory_endpoints.length} advertised inventory endpoints occur after the release date and are omitted from the newer-date comparison.</Note>
    </ChartFrame>
    <ChartFrame id="availability-product-months" title="Monthly product coverage" subtitle="Months with no observations stay visible. Pale cells mark zero samples inside the reported span; white cells lie outside that span. Sample dates are not analysis dates. Blank HIPS filters are excluded." controls={<>
      <Select label="Product" value={product.key} options={network.products.map(p => p.key)} optionLabel={key => network.products.find(p => p.key === key)?.label ?? key} onChange={setProduct} />
      <Select label="Year" value={currentYear} options={['All years', ...years]} onChange={setYear} />
    </>}>
      <p className="facet-title">{product.label} · {product.unit} per month · {sites.length} sites shown</p>
      <MonthMatrix sites={sites} product={product.key} rows={monthly} year={currentYear} />
      <Note>Counts are shown in each cell’s hover text. Darker blue means more observations within this product. This is sampling coverage, not a concentration heatmap.</Note>
    </ChartFrame>
    <ChartFrame id="availability-cadence" title="Sampling cadence and gaps" subtitle="Saved cadence statistics for the selected product. The nominal ratio compares observed samples with a 9-day filter, daily or hourly schedule across the first–last span. It is not instrument uptime and can exceed 100% when sampling was more frequent." >
      <p className="facet-title">{product.label} · {product.unit}</p>
      <div className="chart-scroll" tabIndex={0} role="region" aria-label="Sampling cadence table"><table className="census-table availability-table"><thead><tr><th>Site</th><th>Observations</th><th>First sample</th><th>Last sample</th><th>Median interval · days</th><th>95th percentile · days</th><th>Longest gap · days</th><th>Nominal ratio</th><th>Separate HIPS inventory · filters</th><th>Inventory ends</th></tr></thead><tbody>{sites.map(s => {
        const p = s.products[product.key]
        return <tr key={s.code}><th scope="row">{s.code}</th><td>{fmt(p?.n_samples)}</td><td>{p?.date_min ?? '—'}</td><td>{p?.date_max ?? '—'}</td><td>{fmt(p?.median_interval_h == null ? null : p.median_interval_h / 24)}</td><td>{fmt(p?.p95_interval_h == null ? null : p.p95_interval_h / 24)}</td><td>{fmt(p?.max_gap_days)}</td><td>{p?.completeness_pct == null ? '—' : `${fmt(p.completeness_pct)}%`}</td><td>{fmt(p?.inventory_n_samples)}</td><td>{p?.inventory_date_max ?? p?.newer_date_max ?? '—'}</td></tr>
      })}</tbody></table></div>
      <Note>{product.key === 'HIPS' ? 'HIPS cadence statistics were not exported by the source notebook. Missing statistics are left as dashes.' : 'All intervals include gaps. The underlying counts and ratios are preserved, including values above 100%.'}</Note>
    </ChartFrame>
  </>
}

function MonthMatrix({ sites, product, rows, year }: { sites: AvailabilitySite[]; product: string; rows: NotebookAvailability['network']['monthly']; year: string }) {
  const months = useMemo(() => {
    const ordered = rows.map(r => r.month).sort()
    return ordered.length ? calendarMonths(ordered[0], ordered[ordered.length - 1]).filter(m => year === 'All years' || m.startsWith(year)) : []
  }, [rows, year])
  const counts = useMemo(() => new Map(rows.map(r => [`${r.code}|${r.month}`, r.count])), [rows])
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref)
  const labelW = 170, cell = Math.max(14, Math.min(42, (width - labelW - 10) / Math.max(1, months.length)))
  const w = Math.max(width, labelW + months.length * cell + 10), h = 48 + sites.length * 24
  const color = d3.scaleSequentialLog(d3.interpolateBlues).domain([1, Math.max(2, ...rows.map(r => r.count))])
  if (!months.length) return <Empty>No monthly records for this product and year.</Empty>
  return <div className="chart-scroll availability-month-matrix" ref={ref} tabIndex={0} role="region" aria-label="Monthly sampling heatmap, scroll for all months and sites"><svg width={w} height={h} fontFamily={FONT.family} role="img" aria-label={`${sites.length} sites, ${months.length} months, ${product} sampling counts`}>
    {months.map((month, j) => <g key={month}>{(j === 0 || month.endsWith('-01')) && <text x={labelW + j * cell} y={13} fontSize={11} fill={INK.text}>{month.slice(0, 4)}</text>}{(cell > 25 || [1, 4, 7, 10].includes(+month.slice(5))) && <text x={labelW + j * cell + cell / 2} y={32} textAnchor="middle" fontSize={9} fill={INK.muted}>{month.slice(5)}</text>}</g>)}
    {sites.map((s, i) => <g key={s.code}><text x={labelW - 8} y={48 + i * 24 + 14} textAnchor="end" fontSize={11} fill={INK.text}><title>{siteLabel(s)}</title>{s.name === s.code ? s.code : `${s.code} · ${s.name.slice(0, 17)}${s.name.length > 17 ? '…' : ''}`}</text>{months.map((month, j) => {
      const n = counts.get(`${s.code}|${month}`) ?? 0, p = s.products[product]
      const inside = p?.date_min && p.date_max && month >= p.date_min.slice(0, 7) && month <= p.date_max.slice(0, 7)
      return <rect key={month} x={labelW + j * cell} y={48 + i * 24} width={cell - 1} height={20} fill={n ? color(n) : inside ? INK.empty : INK.panel} stroke={n || inside ? 'none' : INK.grid}><title>{s.code} · {month}: {n.toLocaleString()} observations{!inside && !n ? ' (outside reported span)' : ''}</title></rect>
    })}</g>)}
  </svg></div>
}

export function StudyAvailability({ data }: { data: NotebookAvailability }) {
  const [rule, setRule] = useSearchParam<CompletenessRule>('availability.completeness', 'Any finite IR', ['Any finite IR', 'At least 50%', 'At least 80%'])
  const [selection, setSelection] = useSearchParam<string>('availability.studySite', 'All four sites', ['All four sites', ...data.study.map(s => s.code)])
  const sites = data.study.filter(s => selection === 'All four sites' || s.code === selection)
  const summaries = sites.map(s => ({ site: s, ...studyOverlap(s, rule) }))
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref)
  const allDays = data.study.flatMap(s => [...s.ftir.days, ...s.hips.days, ...(s.aeth?.record_days ?? [])]).sort()
  const w = Math.max(760, width), labelW = 165, h = 36 + sites.length * 118
  const x = d3.scaleUtc().domain([new Date(allDays[0]), new Date(allDays[allDays.length - 1])]).range([labelW, w - 22])
  const jpl = data.study.find(s => s.code === 'USPA')
  const jplDays = jpl ? [...jpl.ftir.days, ...jpl.hips.days].sort() : []
  const jplLast = jplDays[jplDays.length - 1]
  return <>
    <ChartFrame id="availability-study" title="Study sample-day overlap" subtitle="Recomputed from the dedicated availability notebook using the current unified filter input and all four canonical 9am aethalometer files. Each mark is an observed local sample-day. Empty periods are not bridged." controls={<>
      <Select label="Site" value={selection} options={['All four sites', ...data.study.map(s => s.code)]} optionLabel={c => data.study.find(s => s.code === c)?.name ?? c} onChange={setSelection} />
      <Select label="Aethalometer completeness" value={rule} options={['Any finite IR', 'At least 50%', 'At least 80%']} onChange={v => setRule(v as CompletenessRule)} />
    </>}>
      <p className="availability-context">FTIR-EC and HIPS require finite concentrations. Aethalometer coverage requires finite IR BCc on the processed local <code>day_9am</code>. Availability includes flagged filters and values below fitting thresholds.</p>
      <div className="chart-scroll" ref={ref} tabIndex={0} role="region" aria-label="Sample-day strips, scroll horizontally for dates"><svg width={w} height={h} fontFamily={FONT.family} role="img" aria-label="FTIR, HIPS and aethalometer sample-day availability">
        {x.ticks(d3.utcMonth.every(6)!).map(t => <g key={+t}><line x1={x(t)} x2={x(t)} y1={24} y2={h} stroke={INK.grid} /><text x={x(t)} y={14} fontSize={10} textAnchor="middle" fill={INK.muted}>{d3.utcFormat('%b %Y')(t)}</text></g>)}
        {summaries.map(({ site: s, aeth }, i) => <g key={s.code} transform={`translate(0,${32 + i * 118})`}>
          <text x={8} y={12} fontSize={12} fill={s.color} fontWeight={600}>{s.name} · {s.code}</text>
          {([{ label: 'FTIR-EC', days: s.ftir.days, color: '#7755a2' }, { label: 'HIPS Fabs', days: s.hips.days, color: '#d28032' }, { label: 'Aethalometer', days: aeth, color: '#277bb5' }]).map((stream, k) => <g key={stream.label}>
            <text x={labelW - 9} y={34 + k * 24} textAnchor="end" fontSize={11} fill={INK.muted}>{stream.label}</text>
            {stream.days === null ? <text x={labelW + 5} y={34 + k * 24} fontSize={11} fill={INK.muted}>Completeness unavailable in this source</text> : stream.days.map(day => <line key={day} x1={x(new Date(day))} x2={x(new Date(day))} y1={23 + k * 24} y2={37 + k * 24} stroke={stream.color} strokeWidth={1.5}><title>{s.name} · {stream.label} · {day}</title></line>)}
          </g>)}
        </g>)}
      </svg></div>
      <div className="chart-scroll" tabIndex={0} role="region" aria-label="Day and physical filter overlap counts"><table className="census-table availability-table"><thead><tr><th>Site</th><th>FTIR days</th><th>HIPS days</th><th>Aeth days</th><th>FTIR + HIPS days</th><th>All three days</th><th>Same filter · FTIR + HIPS</th><th>Same filter + aeth day</th></tr></thead><tbody>{summaries.map(r => <tr key={r.site.code}><th scope="row">{r.site.name}</th><td>{r.site.ftir.days.length}</td><td>{r.site.hips.days.length}</td><td>{fmt(r.aeth?.length)}</td><td>{r.both.length}</td><td>{fmt(r.triple?.length)}</td><td>{r.matched}</td><td>{fmt(r.matchedWithAeth)}</td></tr>)}</tbody></table></div>
      <Note>Same-day intersections compare calendar dates exactly, with no ±1-day matching. “Same filter” separately matches canonical base filter IDs across FTIR and HIPS; replicates count once. A shared date alone does not prove a shared physical filter. Dashes mean unavailable, while 0 means a measured zero overlap.</Note>
    </ChartFrame>
    <ChartFrame id="availability-study-audit" title="Coverage checks and notebook differences" exportable={false}>
      <p className="availability-context">JPL sample-date check: the last finite FTIR/HIPS sample is {jplLast ?? 'unavailable'}{jplLast ? jplLast <= '2023-12-31' ? '; none occur after December 2023 in this study input.' : '; this input now contains samples after December 2023.' : '.'} Later analysis dates do not extend sampling coverage.</p>
      <div className="chart-scroll" tabIndex={0} role="region" aria-label="Reported versus finite measurement days"><table className="census-table availability-table"><thead><tr><th>Site</th><th>FTIR reported / finite days</th><th>HIPS reported / finite days</th><th>Last FTIR analysis</th><th>Last HIPS analysis</th><th>Aeth finite / stored days</th><th>Mean minute completeness</th></tr></thead><tbody>{sites.map(s => <tr key={s.code}><th scope="row">{s.name}</th><td>{s.ftir.reported_days} / {s.ftir.days.length}</td><td>{s.hips.reported_days} / {s.hips.days.length}</td><td>{s.ftir.analysis_last ?? '—'}</td><td>{s.hips.analysis_last ?? '—'}</td><td>{s.aeth ? `${s.aeth.days.length} / ${s.aeth.record_days.length}` : '—'}</td><td>{s.aeth?.mean_completeness == null ? 'Unavailable' : `${fmt(s.aeth.mean_completeness)}%`}</td></tr>)}</tbody></table></div>
      <ul className="availability-reading">
        <li>The original strip-chart notebook counted parameter-row presence. Finite-value coverage is lower where reported concentrations are missing.</li>
        <li>Its aethalometer paths were placeholders for three sites; its Addis file used a different processing population. This view uses the four processed files configured in the shared analysis helpers.</li>
        <li>Addis has finite IR BCc but no minute-completeness field. A 50% or 80% rule therefore leaves its aethalometer coverage unavailable.</li>
        <li>{data.study.reduce((sum, s) => sum + (s.aeth?.over100_days ?? 0), 0)} stored aethalometer day(s) report completeness above 100%; source values are retained. Availability does not certify flow correction or measurement quality.</li>
      </ul>
    </ChartFrame>
  </>
}

export function PriorCoverage({ data }: { data: NotebookAvailability }) {
  return <ChartFrame id="availability-prior-comparison" title="Prior SPARTAN–IMPROVE coverage comparison" subtitle="Saved orientation notebook table. These are the populations used in that analysis, rather than the current full network exports." exportable={false}>
    <p className="availability-context">Historical analysis snapshot: SPARTAN covers the four study sites; IMPROVE covers its cleaned analysis population through July 2025. Use Filter explorer for the current exported network inventory. EC is FTIR-derived at SPARTAN and thermal-optical at IMPROVE.</p>
    <div className="chart-scroll" tabIndex={0} role="region" aria-label="Historical cross-network coverage"><table className="census-table availability-table"><thead><tr><th>Population</th><th>Measurement</th><th>Finite observations</th><th>Sites</th><th>First sample</th><th>Last sample</th></tr></thead><tbody>{data.orientation.map(r => <tr key={`${r.network}-${r.quantity}`}><td>{r.network === 'SPARTAN' ? 'SPARTAN · four study sites' : 'IMPROVE · cleaned snapshot'}</td><td>{r.quantity}</td><td>{fmt(r.n)}</td><td>{r.n_sites}</td><td>{r.date_min}</td><td>{r.date_max}</td></tr>)}</tbody></table></div>
    <Note>The networks are compared as separate populations, without a same-site or same-date join. Parameter-specific counts must not be summed into physical filter totals.</Note>
  </ChartFrame>
}

export function AvailabilitySources({ data }: { data: NotebookAvailability }) {
  return <div className="availability-sources"><details><summary>Notebook sources, data snapshots and counting rules</summary>
    <p>Packaged {data.generated.slice(0, 10)}. Repackaging does not refresh the underlying observations. The JSON includes each source’s SHA-256 hash.</p>
    <p><a href={dataUrl('availability.json')} download="availability.json">Download the coverage data and provenance</a></p>
    <ul>{data.sources.map(s => <li key={s.path}><span>{s.role}</span><code>{s.path}</code></li>)}</ul>
  </details></div>
}
