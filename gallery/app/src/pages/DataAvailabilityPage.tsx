import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Select } from '@/components/ChartFrame'
import { FieldSelect } from '@/components/FieldSelect'
import { useDimensions } from '@/hooks/useGalleryData'
import { useExport } from '@/lib/data'
import { availabilityRecords, monthlyAvailability, recentCutoff, samplingStatus, type AvailabilityFile, type DatedRecord } from '@/lib/availability'
import type { MetaFile } from '@/lib/types'
import type { SpartanIndex } from '@/lib/spartan'
import { useSearchParam } from '@/lib/url'
import { INK, FONT } from '@/lib/theme'
import { HIDDEN_DUPLICATES } from '@/fieldLineage'
import type { NotebookAvailability } from '@/lib/notebookAvailability'
import { AeronetAvailability, AvailabilitySources, NetworkAvailability, PriorCoverage, StudyAvailability } from './NotebookAvailability'

interface InventorySite { code: string; name: string; location?: string; n_filters: number; date_min: string; date_max: string; coverage?: Record<string, number>; filters?: string }
interface Inventory { generated?: string; note?: string; fields?: string[]; field_units: Record<string, string>; field_groups?: MetaFile['field_groups']; sites: InventorySite[] }
const VIEWS = ['Network coverage', 'Study overlap', 'Filter explorer', 'AERONET photometer', 'Prior comparison']
const today = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Vancouver', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date())

export function DataAvailabilityPage() {
  const [view, setView] = useSearchParam<string>('availability.view', 'Network coverage', VIEWS)
  const { data, error } = useExport<NotebookAvailability>('availability.json', d => d.schema_version === 1 ? null : 'Unsupported availability data version')
  return <>
    <ChartFrame id="data-availability" title="Data availability" exportable={false} controls={<Select label="View" value={view} options={VIEWS} onChange={setView} />}>
      <p className="chart-note">Explore the existing network analyses, four-site sample-day overlaps, exported filter records, and the AERONET sun photometer against the filters. Each view identifies its own source population and counting unit.</p>
    </ChartFrame>
    {view === 'Filter explorer' ? <FilterAvailabilityExplorer /> : error ? <Empty>{error}</Empty> : !data ? <Empty>Loading notebook coverage analyses…</Empty> : view === 'Study overlap' ? <StudyAvailability data={data} /> : view === 'Prior comparison' ? <PriorCoverage data={data} /> : view === 'AERONET photometer' ? <AeronetAvailability data={data.aeronet} /> : <NetworkAvailability data={data} />}
    {data && <AvailabilitySources data={data} />}
  </>
}

function FilterAvailabilityExplorer() {
  const [network, setNetwork] = useSearchParam<string>('availability.network', 'SPARTAN', ['SPARTAN', 'IMPROVE'])
  const [yearsText, setYears] = useSearchParam<string>('availability.years', '2')
  const [asOf, setAsOf] = useSearchParam<string>('availability.asof', today())
  const [status, setStatus] = useSearchParam<string>('availability.status', 'All', ['All', 'Recent samples', 'No recent samples', 'No samples'])
  const [query, setQuery] = useSearchParam<string>('availability.search', '')
  const [siteCode, setSite] = useSearchParam<string>('availability.site', '')
  const [fieldChoice, setField] = useSearchParam<string>('availability.field', '')
  const spartan = useExport<SpartanIndex>(network === 'SPARTAN' ? 'spartan_network/index.json' : null)
  const fallback = useExport<Inventory>(network === 'SPARTAN' && spartan.error ? 'spartan_map.json' : null)
  const improve = useExport<Inventory>(network === 'IMPROVE' ? 'improve_samples/index.json' : null)
  const inventory: Inventory | null = network === 'IMPROVE' ? improve.data : spartan.data ?? fallback.data
  const reference = useExport<{ fields: string[]; rows: (DatedRecord & { code: string })[] }>(network === 'SPARTAN' && spartan.error ? 'spartan_reference.json' : null)
  const sites = inventory?.sites ?? []
  const years = Math.max(0, Math.min(100, Math.round(Number(yearsText) || 0)))
  const cutoff = recentCutoff(asOf, years)
  const shown = useMemo(() => sites.filter(s => (!query || `${s.name} ${s.code} ${s.location ?? ''}`.toLowerCase().includes(query.toLowerCase())) && (status === 'All' || samplingStatus(s.date_max, cutoff, asOf) === status)).sort((a, b) => a.name.localeCompare(b.name)), [sites, query, status, cutoff, asOf])
  const selected = sites.find(s => s.code === siteCode) ?? shown[0]
  const detail = useExport<AvailabilityFile>(selected ? network === 'IMPROVE' ? `improve_samples/${selected.code}.json` : selected.filters ?? null : null)
  const records = useMemo(() => (detail.data ? availabilityRecords(detail.data) : reference.data?.rows.filter(r => r.code === selected?.code) ?? []).filter(r => r.date <= asOf), [detail.data, reference.data, selected?.code, asOf])
  const fields = (inventory?.fields ?? Object.keys(inventory?.field_units ?? {})).filter(f => !HIDDEN_DUPLICATES.includes(f))
  const field = fields.includes(fieldChoice) ? fieldChoice : fields.includes('EC (FTIR)') ? 'EC (FTIR)' : fields.includes('EC (TOR)') ? 'EC (TOR)' : fields[0] ?? ''
  const meta = useMemo(() => ({ fields, field_units: inventory?.field_units ?? {}, field_groups: [{ label: 'Available measurements', fields }] }) as MetaFile, [fields.join('|'), inventory])
  const sampling = useMemo(() => monthlyAvailability(records, field), [records, field])
  const coverage = useMemo(() => fields.map(f => ({ field: f, n: records.filter(r => typeof r[f] === 'number' && Number.isFinite(r[f])).length })), [records, fields.join('|')])
  const error = network === 'IMPROVE' ? improve.error : spartan.error && fallback.error ? fallback.error : ''
  return <>
    <ChartFrame id="availability-filter-inventory" title="Filter inventory — sites, measurements and sampling history" subtitle="Coverage of the locally exported records. A site with no recent samples may have stopped sampling or its export may be incomplete. EC/OC at SPARTAN are FTIR-derived; IMPROVE carbon is thermal-optical. Inventory spans are sample dates, not continuous measurement coverage. Counts include exported exclusions so availability is independent of chart filters." controls={<>
      <Select label="Network" value={network} options={['SPARTAN', 'IMPROVE']} onChange={v => { setNetwork(v); setSite(''); setField('') }} />
      <label className="control">No samples in past <input className="range-value" aria-label="Inactivity threshold in years" type="number" min={0} max={100} step={1} value={years} onChange={e => { if (e.target.value !== '') setYears(String(Math.max(0, Math.min(100, +e.target.value)))) }} /> years</label>
      <label className="control">As of <input className="select" aria-label="Availability reference date" type="date" value={asOf} onChange={e => { if (e.target.value) setAsOf(e.target.value) }} /></label>
      <Select label="Sampling" value={status} options={['All', 'Recent samples', 'No recent samples', 'No samples']} onChange={setStatus} />
      <label className="control">Search sites <input className="select" type="search" value={query} onChange={e => setQuery(e.target.value)} /></label>
    </>}>
      {error ? <Empty>{error}</Empty> : !inventory ? <Empty>Loading site inventory…</Empty> : <>
        <p className="availability-context">{network === 'IMPROVE' ? 'Validated IMPROVE cleaned population: retained samples have positive, available EC and absorption. This is a paired analysis population, not every historical sample. Sites without map coordinates are included.' : 'SPARTAN gallery filter inventory. The fallback inventory describes public PM2.5 chemistry; its filter histories can additionally include HIPS-only records. Network product coverage is shown in the separate coverage view.'}</p>
        <div className="availability-summary"><span><strong>{sites.length}</strong> sites in inventory</span><span><strong>{sites.filter(s => samplingStatus(s.date_max, cutoff, asOf) === 'No recent samples').length}</strong> without a sample since {cutoff}</span><span><strong>{sites.reduce((sum, s) => sum + s.n_filters, 0).toLocaleString()}</strong> exported records</span></div>
        <SamplingSpans sites={shown} selected={selected?.code ?? ''} onSelect={setSite} cutoff={cutoff} />
        <div className="chart-scroll availability-inventory"><table className="census-table availability-table"><thead><tr><th>Site · select to inspect</th><th>Exported filters</th><th>First sample</th><th>Last sample</th><th>Sampling</th></tr></thead><tbody>{shown.map(s => <tr key={s.code} className={s.code === selected?.code ? 'active' : ''}><td><button type="button" className="pill-link" aria-pressed={s.code === selected?.code} onClick={() => setSite(s.code)}>{s.name} · {s.code}</button></td><td>{s.n_filters.toLocaleString()}</td><td>{s.date_min || '—'}</td><td>{s.date_max || '—'}</td><td>{samplingStatus(s.date_max, cutoff, asOf)}</td></tr>)}</tbody></table></div>
        {!shown.length && <Empty>No sites match these choices. Change the search or sampling filter.</Empty>}
        <Note>{inventory.generated ? `Exported ${inventory.generated.slice(0, 10)}. ` : ''}“No recent samples” describes the export, not confirmed site closure. The reference date and threshold are editable.</Note>
      </>}
    </ChartFrame>
    {selected && <ChartFrame id="availability-history" title={`Sampling history — ${selected.name} · ${selected.code}`} subtitle="One bar per consecutive calendar month, including months without samples. Counts use unique filter IDs (SPARTAN) or site/date/POC sample keys (IMPROVE). A jump means at least a twofold change and three filters between consecutive nonempty months. Gaps and jumps describe sampling, not changes in aerosol concentration." controls={<><Select label="Inspect site" value={selected.code} options={sites.map(s => s.code)} optionLabel={code => `${sites.find(s => s.code === code)?.name} · ${code}`} onChange={setSite} /><FieldSelect label="Measurement" value={field} meta={meta} onChange={setField} ratios={false} /></>}>
      {detail.loading || reference.loading ? <Empty>Loading filter dates and measurements…</Empty> : records.length ? <>
        <p className="facet-title">{sampling.measured.toLocaleString()} of {sampling.records.toLocaleString()} dated filters carry {field} · through {asOf}</p>
        <MonthlyCounts months={sampling.months} field={field} />
        <div className="availability-diagnostics"><div><strong>Sampling gaps</strong>{sampling.gaps.length ? <ul>{sampling.gaps.map(g => <li key={g.start}>{g.start}–{g.end}: {g.months} empty month{g.months === 1 ? '' : 's'}</li>)}</ul> : <p>No empty months inside the observed sampling span.</p>}</div><div><strong>Sampling jumps</strong>{sampling.jumps.length ? <ul>{sampling.jumps.map(j => <li key={j.period}>{j.period}: {j.before} → {j.after} filters</li>)}</ul> : <p>No jumps meet the stated rule.</p>}</div></div>
      </> : <Empty>{detail.error || reference.error ? 'Filter-level export is unavailable for this site; the inventory above still shows its reported span.' : 'No dated filters on or before the reference date.'}</Empty>}
    </ChartFrame>}
    {selected && records.length > 0 && <ChartFrame id="availability-measurements" title={`Measurement coverage — ${selected.name}`} subtitle="Finite values per measurement in the selected site's exported filters through the reference date. Missing measurements remain missing; duplicate ChemSpec EC/OC products are omitted." >
      <div className="availability-coverage">{coverage.map(c => <div key={c.field}><span>{c.field}</span><meter min={0} max={records.length} value={c.n} aria-label={`${c.field}: ${c.n} of ${records.length}`} /><span>{c.n.toLocaleString()} / {records.length.toLocaleString()}</span></div>)}</div>
    </ChartFrame>}
  </>
}

function SamplingSpans({ sites, selected, onSelect, cutoff }: { sites: InventorySite[]; selected: string; onSelect: (code: string) => void; cutoff: string }) {
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref)
  const dated = sites.filter(s => Number.isFinite(Date.parse(s.date_min)) && Number.isFinite(Date.parse(s.date_max)))
  const labelW = 180, w = Math.max(600, width), height = dated.length * 23 + 38
  const extent = d3.extent(dated.flatMap(s => [new Date(s.date_min), new Date(s.date_max)])) as [Date, Date]
  const x = d3.scaleTime().domain(dated.length ? extent : [new Date('2000-01-01'), new Date()]).range([labelW, w - 24])
  return <div ref={ref} className="chart-scroll availability-spans" tabIndex={0} role="region" aria-label="Sampling spans by site"><svg width={w} height={height} fontFamily={FONT.family} role="img" aria-label="First to last exported sample by site">
    {x.ticks(6).map(t => <g key={+t}><line x1={x(t)} x2={x(t)} y2={height - 24} stroke={INK.grid} /><text x={x(t)} y={height - 5} textAnchor="middle" fontSize={11} fill={INK.muted}>{d3.timeFormat('%Y')(t)}</text></g>)}
    {dated.map((s, i) => <g key={s.code} role="button" tabIndex={0} aria-label={`Inspect ${s.name}`} onClick={() => onSelect(s.code)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(s.code) } }} style={{ cursor: 'pointer' }}><title>{s.name}: {s.date_min}–{s.date_max} · {s.n_filters} exported filters; span does not imply continuous sampling</title><text x={labelW - 10} y={i * 23 + 17} textAnchor="end" fill={INK.text} fontSize={11}>{s.name.length > 23 ? `${s.name.slice(0, 22)}…` : s.name}</text><rect x={x(new Date(s.date_min))} y={i * 23 + 7} width={Math.max(3, x(new Date(s.date_max)) - x(new Date(s.date_min)))} height={12} rx={3} fill={s.code === selected ? INK.accent : s.date_max < cutoff ? '#aa6b78' : '#7b9ebc'} /></g>)}
  </svg></div>
}
function MonthlyCounts({ months, field }: { months: { period: string; samples: number; measured: number }[]; field: string }) {
  const ref = useRef<HTMLDivElement>(null), { width } = useDimensions(ref)
  const w = Math.max(width, months.length * 9 + 55), h = 230
  const x = d3.scaleBand().domain(months.map(m => m.period)).range([44, w - 10]).padding(.12)
  const y = d3.scaleLinear().domain([0, Math.max(1, d3.max(months, m => m.samples) ?? 1)]).range([190, 15]).nice()
  return <div ref={ref} className="chart-scroll" tabIndex={0} role="region" aria-label="Monthly sample counts, scroll for all months"><svg width={w} height={h} fontFamily={FONT.family}>
    {y.ticks(4).map(t => <g key={t}><line x1={44} x2={w - 10} y1={y(t)} y2={y(t)} stroke={INK.grid} /><text x={36} y={y(t)} dy=".35em" textAnchor="end" fontSize={11} fill={INK.muted}>{t}</text></g>)}
    {months.map((m, i) => <g key={m.period}><title>{m.period}: {m.samples} physical filters, {m.measured} with {field}</title><rect x={x(m.period)} y={y(m.samples)} width={x.bandwidth()} height={190 - y(m.samples)} fill="#d6e0e9" /><rect x={x(m.period)} y={y(m.measured)} width={x.bandwidth()} height={190 - y(m.measured)} fill={INK.accent} />{(i === 0 || m.period.endsWith('-01')) && <text x={x(m.period)} y={209} fontSize={10} fill={INK.muted}>{m.period.slice(0, 4)}</text>}</g>)}

  </svg><p className="chart-note">Blue: with {field} · pale: all filters</p></div>
}
