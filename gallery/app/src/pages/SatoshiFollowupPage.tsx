import { useEffect, useMemo, useRef, useState, useId } from 'react'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import * as d3 from 'd3'
import { ChartFrame, ExplainOnly, Note, Segmented, Select } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { Combobox } from '@/components/Combobox'
import { SatoshiSpectra } from '@/components/SatoshiSpectra'
import { BankPicker, BaselinePicker, SeasonPicker } from '@/components/Pickers'
import { useHeaderHeight, useSidePanelOpen } from '@/hooks/useSidePanel'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { useHighlight, type FilterRecord } from '@/lib/highlight'
import { BANK_COLOR, BASELINE_COLOR, FONT, INK, OCT1_HATCH, SPECTRAL_REGION } from '@/lib/theme'
import { f2, f3, pct, signed, coef, coefSigned } from '@/lib/format'
import { ALL_ADDIS, ANALYSIS_CALENDAR_NOTE, shortSeason } from '@/lib/seasons'
import { baselineName, ecAxis } from '@/lib/labels'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Key } from '@/components/Key'
import { IntervalStrip } from '@/components/IntervalStrip'
import { BandCell, FitCell } from '@/components/TableCells'
import { ANALOG_SELECTION_EXCLUDED, OCT1_ALSO_EXCLUDED, SpectralRegions } from '@/components/SpectralRegions'

/**
 * Satoshi's questions after the 1 Oct 2026 meeting (his email of 2 Oct). Every
 * number comes from research/ftir_hips_chem/output/tables/satoshi_followup_20261002
 * via gallery/data/export_satoshi_followup.py; nothing is refitted in the browser.
 */

const SATOSHI_PATH = 'satoshi/followup_20261002.json'
const schemaV1 = (d: { schema_version: number }) => d.schema_version === 1 ? null : 'unsupported data version'
type Row = Record<string, any>
type Cols = Record<string, (number | string | null)[]>
interface SatoshiData {
  schema_version: number
  summary: Row
  season_colors: Record<string, string>
  groups: string[]
  seasons: string[]
  overlap: Row[]
  overlap_props: Row[]
  members: Cols
  swap_arms: Row[]
  swap_effects: Row[]
  swap_reproduction: Row[]
  ladder: Row[]
  ladder_points: Cols
  vip: { wn: Record<string, number[]>; models: Record<string, { baseline: string; family: string; mask: string; group: string; k: number; vip: number[] }> }
  vip_bands: Row[]
  vip_similarity: Row[]
  envelope: Row[]
  envelope_points: Cols
  years: Row[]
  headlines: Row
}

const SPLINE = 'AIRSpec'
const VIBES = 'VIBES-full-r70'
const BASELINES = [SPLINE, VIBES] as const
type Baseline = (typeof BASELINES)[number]
/** Agreed colours (docs/gallery-chart-standards.md §5): spline blue, completed full-range VIBES orange. */
const BASE_COLOR = BASELINE_COLOR
const SHORT: Record<string, string> = { [SPLINE]: baselineName(SPLINE, { short: true }), [VIBES]: baselineName(VIBES, { short: true }) }
const LONG: Record<string, string> = { [SPLINE]: `${baselineName(SPLINE)} (4000–1425)`, [VIBES]: baselineName(VIBES) }
const MASKS = ['oct1', 'presented'] as const
type Mask = (typeof MASKS)[number]
const MASK_SHORT: Record<Mask, string> = {
  oct1: '600–3600 cm⁻¹ · exclude CO₂ 1800–2500 + PTFE 1050–1250',
  presented: '500–3600 cm⁻¹ · exclude CO₂ 1800–2500',
}
const MASK_DESCRIPTION: Record<Mask, string> = {
  oct1: 'Use 600–3600 cm⁻¹, excluding CO₂ (1800–2500 cm⁻¹) and PTFE (1050–1250 cm⁻¹).',
  presented: 'Use 500–3600 cm⁻¹, excluding CO₂ (1800–2500 cm⁻¹). The PTFE band remains included.',
}
const SHARED_COLOR = '#4b5563'
const STANDARD_COLOR = '#6b7280'

/** Log-axis labels at 1, 2 and 5 of each decade only. */
const logTick = (v: number) => { const d = +(v / 10 ** Math.floor(Math.log10(v))).toFixed(6); return [1, 2, 5].includes(d) ? String(+(+v).toPrecision(2)) : '' }
const minus = (t: string) => t.replace(/^-/, '−')
const range = (a: any[]) => `${minus(f2(a?.[0]))} to ${minus(f2(a?.[1]))}`

export function SatoshiFollowupPage() {
  const { data, error } = useExport<SatoshiData>(SATOSHI_PATH, schemaV1)
  if (!data) return <ExportState title="Satoshi's questions" what="the 2 Oct follow-up" error={error}
    command="uv run --no-sync python research/ftir_hips_chem/workflows/run_satoshi_followup_20261002.py && uv run --no-sync python gallery/data/export_satoshi_followup.py" />
  return <SatoshiView data={data} />
}

function SatoshiView({ data }: { data: SatoshiData }) {
  // 'sregion' is the similarity spectra's shading key; the mask gets its own so the two never overwrite each other
  const [mask, setMask] = useSearchParam<Mask>('satmask', 'oct1', MASKS)
  const oct1 = useOct1()
  const hl = useHighlight()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') hl.setPinned(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [hl])
  return (
    <>
    <SimilarityRegionSidebar mask={mask} onMask={setMask} />
    <PageToc>
      <p className="chart-note">{ANALYSIS_CALENDAR_NOTE}</p>
      <ExplainOnly><Conclusion s={data} o={oct1.data} /></ExplainOnly>
      <ExplainOnly><Answers data={data} /></ExplainOnly>
      <Overlap data={data} mask={mask} />
      <SatoshiSpectra membership={data.members} groups={data.groups} region={mask} />
      <Swap data={data} mask={mask} />
      <Vip data={data} mask={mask} />
      <Ladder data={data} mask={mask} />
      <Envelope data={data} mask={mask} />
      <Oct1Sections o={oct1} />
      <ExplainOnly><Methods data={data} /></ExplainOnly>
    </PageToc>
    </>
  )
}

function SimilarityRegionSidebar({ mask, onMask }: { mask: Mask; onMask: (mask: Mask) => void }) {
  const [open, setOpen] = useSidePanelOpen()
  const top = useHeaderHeight()
  return <>
    <button type="button" className={`subset-tab ${open ? 'open' : ''}`} style={{ top: top + 24 }}
      aria-expanded={open} onClick={() => setOpen(!open)} title={open ? 'Hide the filters' : 'Show the filters'}>
      <span className="subset-tab-label">Filters</span>
    </button>
    {open && <aside className="panel subset subset-panel" style={{ top, height: `calc(100vh - ${top}px)` }} aria-label="Filters">
      <header className="subset-head"><strong>Filters</strong>
        <span style={{ marginLeft: 'auto' }}><button type="button" className="btn quiet" onClick={() => setOpen(false)} aria-label="Close filters" title="Close">✕</button></span>
      </header>
      <Combobox label="Similarity region" value={mask} options={MASKS.map(key => ({ key, label: MASK_SHORT[key] }))} onChange={onMask} />
      <p className="chart-note">{MASK_DESCRIPTION[mask]}</p>
      <ExplainOnly><p className="chart-note">This region chooses the analogs on the charts. The spline baseline only covers 4000–1425 cm⁻¹, so its analogs are identical under both regions; only the VIBES analogs change. Every PLS model is still fitted on each baseline's full predictor grid.</p></ExplainOnly>
    </aside>}
  </>
}

// ------------------------------------------------------------------ answers
function Answers({ data }: { data: SatoshiData }) {
  const h = data.headlines
  const ov = (g: string) => h.overlap[g]
  const eff = h.intercept_effects
  const training = data.seasons.filter((g) => eff[g].training > eff[g].baseline)
  const close = data.seasons.filter((g) => !training.includes(g) && eff[g].training > 0.5 * eff[g].baseline)
  // composition of the two "only" sets, straight from the table below
  const prop = (g: string, part: string) => data.overlap_props.find((p) => p.mask === 'presented' && p.group === g && p.part === part)
  const ecRatio = data.groups.map((g) => prop(g, 'VIBES only')!.tor_ec_median / prop(g, 'spline only')!.tor_ec_median)
  const higherOcEc = data.groups.filter((g) => prop(g, 'VIBES only')!.oc_ec_median > prop(g, 'spline only')!.oc_ec_median)
  return (
    <ChartFrame source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Answers" title="Satoshi's questions, 2 Oct · original 1 Oct region — short answers (similarity region as presented 1 Oct; completed rank-70 VIBES)" exportable={false}>
      <table className="census-table">
        <thead><tr><th style={{ width: '24%' }}>Question</th><th>Short answer</th></tr></thead>
        <tbody>
          <tr><td>How different are the analogs the two baselines select?</td>
            <td><strong>Mostly different.</strong> Of 500 analogs per set, spline and VIBES share {data.seasons.map((g, i) => <span key={g}>{i ? ', ' : ''}{ov(g).shared} ({shortSeason(g)})</span>)} and {ov(ALL_ADDIS).shared} for all Addis. Filters only VIBES picks carry {f2(d3.min(ecRatio))}–{f2(d3.max(ecRatio))}× the median TOR EC of filters only the spline picks, and lower OC/EC{higherOcEc.length ? ` except in ${higherOcEc.map(shortSeason).join(', ')}` : ''}.</td></tr>
          <tr><td>Is it the training set or the baseline correction?</td>
            <td>{training.length === 0 ? <><strong>Mostly the baseline correction.</strong> Calibrating the same cohort on VIBES instead of spline moves the Addis intercept by {data.seasons.map((g, i) => <span key={g}>{i ? ', ' : ''}{f2(eff[g].baseline)} ({shortSeason(g)})</span>)} µg/m³; swapping which baseline chose the analogs moves it by {data.seasons.map((g, i) => <span key={g}>{i ? ', ' : ''}{f2(eff[g].training)}</span>)} (size of the median shift, averaged over the two arms). Spline calibrations stay within about 1.5 µg/m³ of a zero intercept whichever analogs they get; VIBES calibrations go strongly negative whichever analogs they get.{close.map((g) => { const e = data.swap_effects.find((x) => x.mask === 'presented' && x.group === g && x.metric === 'addis_intercept' && x.effect === 'training set' && x.held_fixed === 'calibrated on VIBES'); return e ? ` In ${shortSeason(g)} the training set matters too: calibrated on VIBES, switching to VIBES-selected analogs shifts the intercept by ${minus(f2(e.median))} µg/m³.` : '' }).join('')}</>
              : <>Swapping the <em>training set</em> moves the Addis intercept more than swapping the <em>baseline</em> in {training.map(shortSeason).join(', ')}; the baseline matters more in {data.seasons.filter((g) => !training.includes(g)).map(shortSeason).join(', ') || 'no season'}.</>} Swap test: each cohort calibrated on both baselines over 100 site-disjoint splits.</td></tr>
          <tr><td>What does PLS "see" as EC? (VIP)</td>
            <td>Spline analog models use the <strong>same bands</strong> as the standard model (Spearman ρ between VIP profiles {range(h.vip_spearman[SPLINE])}). VIBES analog models <strong>do not</strong> (ρ {range(h.vip_spearman[VIBES])}): {pct(h.vibes_analog_vip2_1050_1425[0])}–{pct(h.vibes_analog_vip2_1050_1425[1])} of their VIP² lies in 1050–1425 cm⁻¹ (PTFE band and below), where the spline baseline has no channels.</td></tr>
          <tr><td>Fitting or extrapolation?</td>
            <td><strong>Not in the fitting.</strong> Analog models reproduce TOR EC in cross-validation (R² spline {range(h.analog_cv_r2[SPLINE])}, VIBES {range(h.analog_cv_r2[VIBES])}; mean bias at most {f3(h.analog_cv_bias_max_abs)} µg/m³), with the usual regression-to-the-mean pattern (slope below 1, small positive intercept). The bias appears only at the Addis stage. It is not simply a range problem: Addis filters carry a median {f2(h.standard_addis?.[SPLINE]?.addis_loading_median_ug)} µg of HIPS-implied EC against {f2(h.standard_addis?.[SPLINE]?.train_loading_median_ug)} µg in the standard calibration set ({pct(h.standard_loading_beyond_p95?.[SPLINE])} above its 95th percentile). But the VIBES analogs cover that loading range (Belg {pct(h.loading_beyond_p95?.[VIBES]?.['Belg (Mar-May)'])}, Kiremt {pct(h.loading_beyond_p95?.[VIBES]?.['Kiremt (Jun-Sep)'])} of Addis above their 95th percentile) and still give the most negative intercepts ({range(h.analog_addis_intercept[VIBES])} µg/m³). Most Addis spectra also sit inside the models' score space.</td></tr>
          <tr><td>Are the analogs on the lower envelope of the standard model?</td>
            <td><strong>No.</strong> Out of sample, the standard model places the analogs at a median residual percentile of {range(h.envelope_pct_median[SPLINE]).replace(/\.\d+/g, '')} (spline) and {range(h.envelope_pct_median[VIBES]).replace(/\.\d+/g, '')} (VIBES) among filters with similar TOR EC; 50 is typical. Debus et al. (2022) calibrated on 2015 filters; this pool runs {data.summary.improve_dates?.[0]} to {data.summary.improve_dates?.[1]}, so whether such samples existed in 2015 cannot be checked from these spectra.</td></tr>
        </tbody>
      </table>
      <Note>IMPROVE comparisons are FTIR EC vs TOR EC (µg/m³). Addis is a test set compared with HIPS Fabs / MAC 10, an optical proxy, not thermal EC. Deming throughout, with the meeting's error ratios.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ shared bits

function GroupSelect({ data, value, onChange, label = 'Analogs of' }: { data: SatoshiData; value: string; onChange: (v: string) => void; label?: string }) {
  return <SeasonPicker as="select" label={label} value={value} groups={data.groups} onChange={onChange} />
}
function BaselineSeg({ value, onChange }: { value: string; onChange: (v: Baseline) => void }) {
  return <BaselinePicker value={value} options={BASELINES} short onChange={(v) => onChange(v as Baseline)} />
}
/** The analog set a baseline uses under a region: the spline set is the same under both. */
const maskFor = (baseline: string, mask: Mask): Mask => (baseline === SPLINE ? 'presented' : mask)

function improveRecord(id: number | string, site: string, source: string, fields: FilterRecord['fields']): FilterRecord {
  return { id: `improve:${id}`, source, site, fields }
}

// ------------------------------------------------------------------ 1. overlap
const PARTS = ['spline only', 'shared', 'VIBES only'] as const
const PART_COLOR: Record<string, string> = { 'spline only': BASE_COLOR[SPLINE], shared: SHARED_COLOR, 'VIBES only': BASE_COLOR[VIBES] }

function Overlap({ data, mask }: { data: SatoshiData; mask: Mask }) {
  const [group, setGroup] = useSearchParam<string>('sovgroup', 'Kiremt (Jun-Sep)', data.groups)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const rows = data.groups.map((g) => data.overlap.find((o) => o.mask === mask && o.group === g)!)
  const changed = data.overlap.filter((o) => o.mask === 'presented_vs_oct1')
  const m = { top: 8, right: 16, bottom: 30, left: 130 }
  const rowH = 30
  const iw = Math.max(240, width - m.left - m.right)
  const x = d3.scaleLinear().domain([0, 1000]).range([0, iw])
  const props = PARTS.map((p) => data.overlap_props.find((o) => o.mask === mask && o.group === group && o.part === p)).filter(Boolean) as Row[]
  return (
    <ChartFrame id="satoshi-overlap" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Overlap"
      title="Analog overlap — how many of the 500 analogs per set both baselines select"
      subtitle="Each bar is the union of the two 500-filter analog sets for one Addis group: left, filters only the spline baseline selects; middle, filters both select; right, filters only VIBES selects. A bar with a wide middle would mean the baselines agree on the training set."
      controls={<GroupSelect data={data} value={group} onChange={setGroup} label="Properties for" />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={m.top + rows.length * rowH + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            {rows.map((o, i) => {
              const segs = [{ part: 'spline only', n: o.spline_only }, { part: 'shared', n: o.shared }, { part: 'VIBES only', n: o.vibes_only }]
              let acc = 0
              return (
                <g key={o.group} transform={`translate(0,${i * rowH})`}>
                  <text x={-8} y={rowH / 2 - 4} dy="0.32em" textAnchor="end" fontSize={12} fontWeight={600} fill={data.season_colors[o.group] ?? INK.text}>{o.group}</text>
                  {segs.map((s) => {
                    const x0 = acc; acc += s.n
                    return (
                      <g key={s.part} onMouseEnter={(e) => tip.show(e, [`${o.group} · ${s.part}`, `${s.n} filters`, `shared ${o.shared} of 500 (Jaccard ${f2(o.jaccard)})`, `sites in common between the two sets: ${o.site_overlap}`])} onMouseLeave={tip.hide}>
                        <rect x={x(x0)} y={2} width={Math.max(0, x(acc) - x(x0))} height={rowH - 12} fill={PART_COLOR[s.part]} fillOpacity={s.part === 'shared' ? 0.95 : 0.8} />
                        {x(acc) - x(x0) > 26 && <text x={(x(x0) + x(acc)) / 2} y={rowH / 2 - 4} dy="0.32em" textAnchor="middle" fontSize={11} fill="white" fontWeight={600}>{s.n}</text>}
                      </g>
                    )
                  })}
                </g>
              )
            })}
            <XAxis scale={x} y={rows.length * rowH - 6} tickCount={5} label="" />
          </g>
        </svg>
        {tip.node}
      </div>
      <Key items={[{ label: 'Spline baseline only', color: PART_COLOR['spline only'], kind: 'bar' }, { label: 'Both baselines', color: SHARED_COLOR, kind: 'bar' }, { label: 'VIBES baseline only', color: PART_COLOR['VIBES only'], kind: 'bar' }]} />
      <table className="census-table" style={{ marginTop: 10 }}>
        <thead><tr><th>{group}</th><th>Filters</th><th>Sites</th><th>TOR EC median (90th) µg/m³</th><th>EC loading median, µg</th><th>TOR OC/EC median</th><th>Fabs / TOR EC median, m²/g</th><th>Lot 251</th><th>Most common sites</th></tr></thead>
        <tbody>{props.map((p) => <tr key={p.part}><td style={{ color: PART_COLOR[p.part], fontWeight: 600 }}>{p.part}</td><td>{p.n}</td><td>{p.n_sites}</td><td>{f2(p.tor_ec_median)} ({f2(p.tor_ec_p90)})</td><td>{f2(p.loading_median_ug)}</td><td>{f2(p.oc_ec_median)}</td><td>{f2(p.mac_ratio_median)}</td><td>{pct(p.lot251_fraction)}</td><td>{p.top_sites}</td></tr>)}</tbody>
      </table>
      <OverlapScatter data={data} mask={mask} group={group} />
      <Note>The agreed 1 Oct region changes the VIBES analogs themselves: it keeps {changed.map((c, i) => <span key={c.group}>{i ? ', ' : ''}{c.shared} ({shortSeason(c.group)})</span>)} of the 500 VIBES analogs selected as presented. Ranks of the shared filters barely agree between baselines (Spearman ρ {f2(d3.min(rows, (r) => r.shared_rank_spearman))} to {f2(d3.max(rows, (r) => r.shared_rank_spearman))}).</Note>
    </ChartFrame>
  )
}

/** TOR EC vs OC/EC of the three parts: where the two analog sets sit in composition. */
function OverlapScatter({ data, mask, group }: { data: SatoshiData; mask: Mask; group: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const hl = useHighlight()
  const clip = useId()
  const mb = data.members
  const idx = useMemo(() => mb.mask.map((_, i) => i).filter((i) => mb.mask[i] === mask && mb.group[i] === group && (mb.tor_ec_ugm3[i] as number) > 0 && (mb.tor_oc_ec[i] as number) > 0), [mb, mask, group])
  const m = { top: 12, right: 16, bottom: 46, left: 60 }
  const iw = Math.max(240, Math.min(width, 760) - m.left - m.right), ih = 300
  const x = d3.scaleLog().domain([0.01, 5]).range([0, iw]).clamp(true)
  const y = d3.scaleLog().domain([0.3, 40]).range([ih, 0]).clamp(true)
  const order = (p: string) => (p === 'shared' ? 2 : 1)
  const sorted = [...idx].sort((a, b) => order(String(mb.part[a])) - order(String(mb.part[b])))
  return (
    <div ref={ref} style={{ position: 'relative', marginTop: 12 }}>
      <svg width={iw + m.left + m.right} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
        <g transform={`translate(${m.left},${m.top})`}>
          <defs><clipPath id={clip}><rect width={iw} height={ih} /></clipPath></defs>
          <YAxis scale={y as any} x={0} label="TOR OC/EC" tickCount={5} gridWidth={iw} labelX={-44} format={logTick} />
          <XAxis scale={x as any} y={ih} label="TOR EC (µg/m³, log)" tickCount={5} format={logTick} labelDy={38} />
          <g clipPath={`url(#${clip})`}>
            <rect width={iw} height={ih} fill="transparent" onClick={() => hl.setPinned(null)} />
            {sorted.map((i) => {
              const part = String(mb.part[i]), id = `improve:${mb.filter_id[i]}`
              const focus = id === hl.pinnedId || id === hl.hoverId
              return <circle key={i} cx={x(mb.tor_ec_ugm3[i] as number)} cy={y(mb.tor_oc_ec[i] as number)} r={focus ? 6 : 3} fill={PART_COLOR[part]} fillOpacity={0.6}
                stroke={focus ? INK.text : 'none'} strokeWidth={2} style={{ cursor: 'pointer' }}
                onMouseEnter={(e) => { tip.show(e, [`IMPROVE ${mb.Site[i]} · filter ${mb.filter_id[i]}`, part, `TOR EC ${f2(mb.tor_ec_ugm3[i])} µg/m³ · OC/EC ${f2(mb.tor_oc_ec[i])}`, `rank spline ${mb.rank_spline[i] ?? '—'} · rank VIBES ${mb.rank_vibes[i] ?? '—'}`, 'click to pin and open its record']); hl.setHover(id) }}
                onMouseLeave={() => { tip.hide(); hl.setHover(null) }}
                onClick={() => { hl.setPinned(id); hl.openRecord(improveRecord(mb.filter_id[i] as number, String(mb.Site[i]), `Satoshi 2 Oct · analogs of ${group} · ${part}`, [
                  ['in analog set', part], ['rank, spline selection', mb.rank_spline[i]], ['rank, VIBES selection', mb.rank_vibes[i]],
                  ['Pearson r, spline', mb.r_spline[i]], ['Pearson r, VIBES', mb.r_vibes[i]], ['TOR EC (µg/m³)', mb.tor_ec_ugm3[i]],
                  ['TOR OC/EC', mb.tor_oc_ec[i]], ['Fabs / TOR EC (m²/g)', mb.mac_ratio[i]], ['TOR EC loading (µg/filter)', mb.loading_ug[i]], ['IMPROVE lot', mb.lot[i]]])) }} />
            })}
          </g>
        </g>
      </svg>
      {tip.node}
    </div>
  )
}

// ------------------------------------------------------------------ 2. swap
const ARMS = [
  { sel: SPLINE, cal: SPLINE, label: 'spline-selected → spline' },
  { sel: VIBES, cal: SPLINE, label: 'VIBES-selected → spline' },
  { sel: SPLINE, cal: VIBES, label: 'spline-selected → VIBES' },
  { sel: VIBES, cal: VIBES, label: 'VIBES-selected → VIBES' },
] as const
const SWAP_METRICS = ['intercept', 'slope'] as const

function Swap({ data, mask }: { data: SatoshiData; mask: Mask }) {
  const [metric, setMetric] = useSearchParam<(typeof SWAP_METRICS)[number]>('sswap', 'intercept', SWAP_METRICS)
  const col = metric === 'intercept' ? 'addis_intercept' : 'addis_slope'
  const arm = (g: string, sel: string, cal: string) => data.swap_arms.find((a) => a.group === g && a.selected_on === sel && a.mask === (sel === SPLINE ? 'presented' : mask) && a.calibrated_on === cal)
  const ref0 = metric === 'intercept' ? 0 : 1
  const eff = data.swap_effects.filter((e) => e.mask === mask && e.metric === col)
  const effCell = (g: string, effect: string, fixed: string) => {
    const e = eff.find((x) => x.group === g && x.effect === effect && x.held_fixed === fixed)
    return e ? <BandCell signed median={e.median} lo={e.p10} hi={e.p90} /> : '—'
  }
  return (
    <ChartFrame id="satoshi-swap" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Swap"
      title={`Training set or baseline — Addis Deming ${metric} when each analog set is calibrated on each baseline`}
      subtitle="Each analog set (selected on the spline or VIBES baseline) is calibrated on both baselines with the same site-grouped cross-validation. Dots are medians over 100 site-disjoint calibration splits; whiskers span the 10th–90th percentile; open diamonds are the protocol split shown in the Follow up meeting view. Colour is the baseline the PLS model was calibrated on; shape is the baseline that chose the analogs."
      controls={<Segmented label="Readout" value={metric} options={SWAP_METRICS} onChange={setMetric} />}>
      <IntervalStrip labelWidth={190} panels={[{ label: `Test set (Addis) Deming ${metric}, FTIR EC vs Fabs / MAC 10`, reference: ref0 }]}
        blocks={data.groups.map((g) => ({ key: g, title: shortSeason(g), color: data.season_colors[g], rows: ARMS.map((arm_) => {
          const a = arm(g, arm_.sel, arm_.cal)
          if (!a) return { key: arm_.label, label: arm_.label, marks: [[]] }
          const lo = a[`${col}_p10`], hi = a[`${col}_p90`], med = a[`${col}_median`], prot = a[`${col}_protocol`]
          return { key: arm_.label, label: arm_.label,
            tip: [`${g} · ${arm_.label}`, `median ${coef(med)} [10–90 %: ${coef(lo)} to ${coef(hi)}]`, `protocol split ${coef(prot)} (k ${a.k_protocol})`, `PLS factors median ${f2(a.k_median)} · IMPROVE test R² median ${coef(a.improve_test_R2_median)}`, `Addis R² median ${coef(a.addis_R2_median)}`],
            marks: [[{ lo, hi, mid: med, color: BASE_COLOR[arm_.cal], shape: arm_.sel === SPLINE ? 'dot' as const : 'square' as const, r: 5 }, { mid: prot, color: INK.text, shape: 'diamond' as const, r: 5 }]] }
        }) }))} />
      <Key items={[{ label: 'Calibrated on spline', color: BASE_COLOR[SPLINE], kind: 'bar' }, { label: 'Calibrated on VIBES', color: BASE_COLOR[VIBES], kind: 'bar' }, { label: 'Analogs selected on spline', color: INK.axis, kind: 'dot' }, { label: 'Analogs selected on VIBES', color: INK.axis, kind: 'square' }, { label: 'Protocol split', color: INK.text, kind: 'diamond' }]} />
      <table className="census-table" style={{ marginTop: 10 }}>
        <thead>
          <tr><th rowSpan={2}>Analogs of</th><th colSpan={2}>Training-set effect: VIBES-selected minus spline-selected</th><th colSpan={2}>Baseline effect: calibrated on VIBES minus spline (same cohort, same split)</th></tr>
          <tr><th>calibrated on spline</th><th>calibrated on VIBES</th><th>spline-selected cohort</th><th>VIBES-selected cohort</th></tr>
        </thead>
        <tbody>{data.groups.map((g) => <tr key={g}><td style={{ color: data.season_colors[g], fontWeight: 600 }}>{g}</td>
          <td>{effCell(g, 'training set', 'calibrated on spline')}</td><td>{effCell(g, 'training set', 'calibrated on VIBES')}</td>
          <td>{effCell(g, 'baseline', 'selected on spline')}</td><td>{effCell(g, 'baseline', 'selected on VIBES')}</td></tr>)}</tbody>
      </table>
      <Note>Change in the Addis Deming {metric}, median [10th, 90th percentile]. The baseline effect is paired: same cohort and same split, two baselines. The training-set effect compares different cohorts, so its interval is over all pairs of splits and is wider by construction. The diagonal arms reproduce the Follow up meeting view's seasonal calibrations exactly (k, slope and intercept checked to 1e-6).</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. VIP

function Vip({ data, mask }: { data: SatoshiData; mask: Mask }) {
  const [baseline, setBaseline] = useSearchParam<Baseline>('svipbase', VIBES, BASELINES)
  const [group, setGroup] = useSearchParam<string>('svipgroup', 'Kiremt (Jun-Sep)', data.groups)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const [hover, setHover] = useState<number | null>(null)
  const wn = data.vip.wn[baseline]
  const mm = maskFor(baseline, mask)
  const std = data.vip.models[`standard|${baseline}`]
  const ana = data.vip.models[`analog|${baseline}|${mm}|${group}`]
  const m = { top: 14, right: 18, bottom: 46, left: 56 }
  const iw = Math.max(300, width - m.left - m.right), ih = 280
  const x = d3.scaleLinear().domain([4000, baseline === SPLINE ? 1400 : 500]).range([0, iw])
  const yMax = Math.max(d3.max(std.vip) ?? 1, d3.max(ana?.vip ?? [1]) ?? 1)
  const y = d3.scaleLinear().domain([0, yMax * 1.05]).nice().range([ih, 0])
  const line = (v: number[]) => d3.line<number>().x((_, i) => x(wn[i])).y((d) => y(d)).defined((d) => Number.isFinite(d))(v) ?? ''
  const sim = data.vip_similarity.find((s) => s.model_id === `analog|${baseline}|${mm}|${group}`)
  const bands = (mid: string) => data.vip_bands.filter((b) => b.model_id === mid)
  const bs = bands(`standard|${baseline}`), ba = bands(`analog|${baseline}|${mm}|${group}`)
  const shades = mm === 'oct1' ? [...ANALOG_SELECTION_EXCLUDED, ...OCT1_ALSO_EXCLUDED] : ANALOG_SELECTION_EXCLUDED
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const px = e.nativeEvent.offsetX - m.left
    const w = x.invert(px)
    const i = d3.minIndex(wn, (v) => Math.abs(v - w))
    setHover(i)
    tip.show(e, [`${wn[i].toFixed(0)} cm⁻¹`, `standard model VIP ${f2(std.vip[i])}`, `${group} analog model VIP ${f2(ana?.vip[i])}`])
  }
  return (
    <ChartFrame id="satoshi-vip" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Vip"
      title="What PLS sees as EC — VIP of the standard all-IMPROVE model vs the analog-trained model"
      subtitle="Variable importance in projection (Wold VIP, Takahama et al. 2019 Eq. 14) for each calibration, on the baseline's full predictor grid. VIP ≥ 1 marks channels that carry more than their share of the explained TOR EC variance. The standard model is the all-IMPROVE lot-251 calibration (frozen site-disjoint split); the analog model is that season's 500-filter calibration from the meeting tab."
      controls={<><BaselineSeg value={baseline} onChange={setBaseline} /><GroupSelect data={data} value={group} onChange={setGroup} /></>}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            <SpectralRegions x={x} height={ih} regions={shades} />
            <YAxis scale={y} x={0} label="VIP" tickCount={5} gridWidth={iw} labelX={-38} />
            <XAxis scale={x} y={ih} label="Wavenumber (cm⁻¹)" tickCount={8} labelDy={38} />
            <line x1={0} x2={iw} y1={y(1)} y2={y(1)} stroke={INK.axis} strokeDasharray="3 3" />
            <path d={line(std.vip)} fill="none" stroke={STANDARD_COLOR} strokeWidth={1.4} />
            {ana && <path d={line(ana.vip)} fill="none" stroke={data.season_colors[group]} strokeWidth={1.6} strokeOpacity={0.9} />}
            {hover != null && <line x1={x(wn[hover])} x2={x(wn[hover])} y1={0} y2={ih} stroke={INK.text} strokeOpacity={0.4} />}
            <rect width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => { setHover(null); tip.hide() }} />
          </g>
        </svg>
        {tip.node}
      </div>
      <Key items={[{ label: `Standard model (all IMPROVE lot 251, k ${std.k})`, color: STANDARD_COLOR, kind: 'line' }, { label: `${group} analog model (k ${ana?.k ?? '—'})`, color: data.season_colors[group], kind: 'line' }, { label: 'Excluded from similarity, both regions', color: SPECTRAL_REGION.co2.fill, kind: 'bar' }, { label: 'Excluded only in the agreed region', color: OCT1_HATCH.stroke, kind: 'bar' }]} />
      <table className="census-table" style={{ marginTop: 10 }}>
        <thead><tr><th>Band (cm⁻¹)</th><th>Channels</th><th>Mean VIP, standard</th><th>Mean VIP, analog</th><th>Share of VIP², standard</th><th>Share of VIP², analog</th></tr></thead>
        <tbody>{bs.map((b) => { const a = ba.find((x) => x.band === b.band); return <tr key={b.band}><td>{b.band}</td><td>{b.n_channels}</td><td>{f2(b.vip_mean)}</td><td>{f2(a?.vip_mean)}</td><td>{pct(b.vip2_share)}</td><td style={{ fontWeight: a && a.vip2_share > b.vip2_share * 1.5 ? 700 : 400 }}>{pct(a?.vip2_share)}</td></tr> })}</tbody>
      </table>
      {sim && <Note>Agreement between the two profiles: Spearman ρ {f2(sim.spearman_vs_standard)}; {pct(sim.top200_overlap)} of the 200 highest-VIP channels are shared; {sim.vip_ge1_both} channels have VIP ≥ 1 in both ({sim.vip_ge1_standard} standard, {sim.vip_ge1_analog} analog). VIP depends on the number of PLS factors, so compare shapes, not heights. Shares of VIP² also scale with the number of channels in a band; use the mean-VIP columns to compare bands of different width.</Note>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. fit vs extrapolation
const STAGES = ['calibration fit', 'cross-validation', 'IMPROVE test sites', 'Addis test set'] as const
const LADDER_METRICS = ['slope', 'intercept ÷ median reference', 'R²', 'mean bias'] as const
type LadderMetric = (typeof LADDER_METRICS)[number]
const metricOf: Record<LadderMetric, (r: Row) => number | null> = {
  slope: (r) => r.deming_slope,
  'intercept ÷ median reference': (r) => r.intercept_rel ?? (r.deming_intercept != null && r.x_median ? r.deming_intercept / r.x_median : null),
  'R²': (r) => r.R2,
  'mean bias': (r) => r.bias,
}

function Ladder({ data, mask }: { data: SatoshiData; mask: Mask }) {
  const [baseline, setBaseline] = useSearchParam<Baseline>('sladbase', SPLINE, BASELINES)
  const [group, setGroup] = useSearchParam<string>('sladgroup', 'Kiremt (Jun-Sep)', data.groups)
  const [metric, setMetric] = useSearchParam<LadderMetric>('sladmetric', 'intercept ÷ median reference', LADDER_METRICS)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const mm = maskFor(baseline, mask)
  const L = data.ladder
  const analogId = `analog|${baseline}|${mm}|${group}`
  const stdId = `standard|${baseline}`
  const find = (q: Row) => L.find((r) => Object.entries(q).every(([k, v]) => r[k] === v))
  const series = [
    { key: 'analog', label: `${shortSeason(group)} analog model, on its own analogs`, color: data.season_colors[group], dash: undefined as string | undefined,
      rows: STAGES.map((s) => s === 'Addis test set' ? find({ model_id: analogId, stage: s, eval_group: group }) : find({ model_id: analogId, stage: s, population: 'own' })) },
    { key: 'std-analogs', label: 'Standard model, on the same analogs', color: STANDARD_COLOR, dash: '5 3',
      rows: STAGES.map((s) => s === 'Addis test set' ? find({ model_id: stdId, stage: s, eval_group: group }) : find({ model_id: stdId, stage: s, population: 'analogs', mask: mm, group })) },
    { key: 'std-own', label: 'Standard model, on all its lot-251 filters', color: '#b8bfc7', dash: '2 3',
      rows: STAGES.map((s) => s === 'Addis test set' ? undefined : find({ model_id: stdId, stage: s, population: 'own' })) },
  ]
  const get = metricOf[metric]
  const vals = series.flatMap((s) => s.rows.map((r) => (r ? get(r) : null))).filter((v): v is number => v != null && Number.isFinite(v))
  const ref0 = metric === 'slope' || metric === 'R²' ? 1 : 0
  const m = { top: 14, right: 20, bottom: 40, left: 64 }
  const iw = Math.max(300, Math.min(width, 860) - m.left - m.right), ih = 260
  const xb = d3.scalePoint<string>().domain(STAGES as unknown as string[]).range([0, iw]).padding(0.5)
  const y = d3.scaleLinear().domain(d3.extent([...vals, ref0]) as [number, number]).nice().range([ih, 0])
  const ex = find({ model_id: analogId, stage: 'Addis test set', eval_group: group })
  const exStd = find({ model_id: stdId, stage: 'Addis test set', eval_group: group })
  const unit = metric === 'mean bias' ? ' (µg/m³)' : ''
  return (
    <ChartFrame id="satoshi-ladder" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Ladder"
      title="Fitting or extrapolation — the same readout at every stage, from calibration fit to Addis"
      subtitle="Calibration fit: in-sample predictions of the calibration set. Cross-validation: site-grouped out-of-fold predictions at the chosen number of factors (the curve that chose k). IMPROVE test sites: sites held out of the calibration. Addis test set: FTIR EC vs HIPS Fabs / MAC 10. IMPROVE stages compare FTIR EC with TOR EC. Intercepts are divided by the median reference so stages at 0.1 and 5 µg/m³ can share an axis."
      controls={<><BaselineSeg value={baseline} onChange={setBaseline} /><GroupSelect data={data} value={group} onChange={setGroup} /><Select label="Readout" value={metric} options={[...LADDER_METRICS]} onChange={(v) => setMetric(v as LadderMetric)} /></>}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={y} x={0} label={`Deming ${metric}${unit}`.replace('Deming R²', 'R²').replace('Deming mean bias', 'Mean bias, FTIR − reference')} tickCount={6} gridWidth={iw} labelX={-48} format={(v) => String(+(+v).toFixed(2))} />
            <XAxis scale={xb as any} y={ih} tickCount={4} />
            <line x1={0} x2={iw} y1={y(ref0)} y2={y(ref0)} stroke={INK.axis} strokeOpacity={0.7} />
            <line x1={(xb('IMPROVE test sites')! + xb('Addis test set')!) / 2} x2={(xb('IMPROVE test sites')! + xb('Addis test set')!) / 2} y1={0} y2={ih} stroke={INK.border} strokeDasharray="2 3" />
            {series.map((s) => {
              const pts = s.rows.map((r, i) => (r && get(r) != null && Number.isFinite(get(r)!) ? { x: xb(STAGES[i])!, y: y(get(r)!), r, stage: STAGES[i] } : null))
              const path = d3.line<any>().defined((p) => !!p).x((p) => p.x).y((p) => p.y)(pts as any) ?? ''
              return (
                <g key={s.key}>
                  <path d={path} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash} />
                  {pts.map((p, i) => p && <circle key={i} cx={p.x} cy={p.y} r={5} fill={s.color} stroke="white" strokeWidth={1}
                    onMouseEnter={(e) => tip.show(e, [s.label, p.stage, `n ${p.r.n} · R² ${coef(p.r.R2)} · RMSE ${f3(p.r.RMSE)} µg/m³`, `Deming ${coef(p.r.deming_slope)}x ${coefSigned(p.r.deming_intercept)} (median reference ${f3(p.r.x_median)} µg/m³)`, `mean bias ${signed(p.r.bias, 3)} µg/m³ · PLS factors ${p.r.k}`])} onMouseLeave={tip.hide} />)}
                </g>
              )
            })}
          </g>
        </svg>
        {tip.node}
      </div>
      <Key items={series.map((s) => ({ label: s.label, color: s.color, kind: s.dash ? 'dash' as const : 'line' as const }))} />
      <LadderTable rows={series} />
      <LadderCrossplots data={data} analogId={analogId} baseline={baseline} mask={mm} group={group} />
      <h3 style={{ fontSize: 13.5, margin: '14px 0 4px' }}>Is Addis outside the calibration set?</h3>
      <table className="census-table">
        <thead><tr><th>Calibration</th><th>HIPS-implied EC loading, Addis median (µg)</th><th>Calibration-set loading median / 95th (µg)</th><th>Addis above calibration 95th percentile</th><th>Addis beyond score leverage T² 95th</th><th>Addis beyond spectral residual Q 95th</th><th>IMPROVE test sites beyond T² / Q 95th</th></tr></thead>
        <tbody>{[{ label: `${shortSeason(group)} analog model`, r: ex }, { label: 'Standard model', r: exStd }].map(({ label, r }) => r && <tr key={label}><td>{label}</td><td>{f2(r.addis_loading_median_ug)}</td><td>{f2(r.train_loading_median_ug)} / {f2(r.train_loading_p95_ug)}</td><td>{pct(r.loading_beyond_train_p95)}</td><td>{pct(r.t2_beyond_p95)}</td><td>{pct(r.q_beyond_p95)}</td><td>{pct(r.improve_test_t2_beyond_p95)} / {pct(r.improve_test_q_beyond_p95)}</td></tr>)}</tbody>
      </table>
      <Note>HIPS-implied loading is Fabs / MAC 10 × sampled volume: the EC mass on the filter if MAC were 10. PLS predicts mass on the filter, so this is the range the model must extrapolate over. T² and Q thresholds are the calibration set's own 95th percentiles; the IMPROVE test-site column shows how often genuinely new IMPROVE filters exceed them, as a benchmark.</Note>
    </ChartFrame>
  )
}

function LadderTable({ rows }: { rows: { label: string; color: string; rows: (Row | undefined)[] }[] }) {
  const cell = (r?: Row) => r ? <FitCell slope={r.deming_slope} intercept={r.deming_intercept} r2={r.R2} bias={r.bias} n={r.n} /> : '—'
  return (
    <table className="census-table" style={{ marginTop: 10 }}>
      <thead><tr><th>Model, population</th>{STAGES.map((s) => <th key={s}>{s}</th>)}</tr></thead>
      <tbody>{rows.map((s) => <tr key={s.label}><td style={{ color: s.color === '#b8bfc7' ? INK.muted : s.color, fontWeight: 600 }}>{s.label}</td>{s.rows.map((r, i) => <td key={i}>{cell(r)}</td>)}</tr>)}</tbody>
    </table>
  )
}

/** Cross-validation predictions of the analog model next to the standard model's out-of-sample predictions of the same filters. */
function LadderCrossplots({ data, analogId, baseline, mask, group }: { data: SatoshiData; analogId: string; baseline: string; mask: Mask; group: string }) {
  const lp = data.ladder_points, ep = data.envelope_points
  const own = lp.model_id.map((_, i) => i).filter((i) => lp.model_id[i] === analogId)
  const ids = new Set(own.map((i) => lp.filter_id[i]))
  const std = ep.baseline.map((_, i) => i).filter((i) => ep.baseline[i] === baseline && ids.has(ep.filter_id[i]))
  const a: Pt[] = own.map((i) => ({ x: lp.tor_ec_ugm3[i] as number, y: (lp.role[i] === 'train' ? lp.cv_ugm3[i] : lp.fitted_ugm3[i]) as number, hollow: lp.role[i] !== 'train', id: String(lp.filter_id[i]), site: String(lp.Site[i]), note: lp.role[i] === 'train' ? 'cross-validation (out-of-fold)' : 'IMPROVE test site' }))
  const s: Pt[] = std.map((i) => ({ x: ep.tor_ec_ugm3[i] as number, y: ep.ftir_oos_ugm3[i] as number, hollow: ep.role[i] !== 'cv', id: String(ep.filter_id[i]), site: String(ep.Site[i]), note: ep.role[i] === 'cv' ? 'cross-validation (out-of-fold)' : 'IMPROVE test site' }))
  const all = [...a, ...s].flatMap((p) => [p.x, p.y]).filter(Number.isFinite)
  const hi = (d3.quantile(all.sort(d3.ascending), 0.995) ?? 1) * 1.05
  const lo = Math.min(0, d3.min(all) ?? 0)
  const missing = ids.size - s.length
  return (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14, marginTop: 12 }}>
        <Cross title={`${shortSeason(group)} analog model (${SHORT[baseline]})`} pts={a} color={data.season_colors[group]} dom={[lo, hi]} source={`Satoshi 2 Oct · ${analogId}`} />
        <Cross title={`Standard model, same analogs (${SHORT[baseline]})`} pts={s} color={STANDARD_COLOR} dom={[lo, hi]} source={`Satoshi 2 Oct · standard|${baseline} · analogs of ${group}`} />
      </div>
      <Note>Filled points are out-of-fold cross-validation predictions; open points are IMPROVE test sites. Both panels share axes; the top 0.5 % of values is outside the frame. {missing > 0 ? `${missing} analogs are lot 248 and so outside the lot-251 standard model; they are not in the right panel. ` : ''}Similarity region: {MASK_SHORT[mask]}.</Note>
    </>
  )
}

interface Pt { x: number; y: number; hollow?: boolean; id: string; site: string; note: string }
/** A TOR-vs-FTIR crossplot of saved predictions (nothing is refitted on this page), on axes shared with its neighbour. */
function Cross({ title, pts, color, dom, source }: { title: string; pts: Pt[]; color: string; dom: [number, number]; source: string }) {
  const points: CrossPoint[] = pts.map((p) => ({
    x: p.x, y: p.y, color, hollow: p.hollow, id: `improve:${p.id}`,
    tip: [`IMPROVE ${p.site} · filter ${p.id}`, p.note, `TOR EC ${f3(p.x)} · FTIR EC ${f3(p.y)} µg/m³`],
    rec: improveRecord(p.id, p.site, source, [['prediction', p.note], ['TOR EC (µg/m³)', p.x], ['FTIR EC (µg/m³)', p.y], ['FTIR − TOR (µg/m³)', p.y - p.x]]),
  }))
  return <Crossplot title={title} points={points} domain={dom} fit={false} maxSide={340} xLabel={ecAxis('tor')} yLabel={ecAxis('ftir')} />
}

// ------------------------------------------------------------------ 5. envelope
function Envelope({ data, mask }: { data: SatoshiData; mask: Mask }) {
  const [baseline, setBaseline] = useSearchParam<Baseline>('senvbase', SPLINE, BASELINES)
  const [group, setGroup] = useSearchParam<string>('senvgroup', 'Kiremt (Jun-Sep)', data.groups)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const mm = maskFor(baseline, mask)
  const ep = data.envelope_points, mb = data.members
  const analogIds = useMemo(() => {
    const keep = baseline === SPLINE ? ['spline only', 'shared'] : ['VIBES only', 'shared']
    return new Set(mb.mask.map((_, i) => i).filter((i) => mb.mask[i] === mm && mb.group[i] === group && keep.includes(String(mb.part[i]))).map((i) => mb.filter_id[i]))
  }, [mb, mm, group, baseline])
  const pool = useMemo(() => ep.baseline.map((_, i) => i).filter((i) => ep.baseline[i] === baseline), [ep, baseline])
  const analogs = pool.filter((i) => analogIds.has(ep.filter_id[i]))
  const xsAll = pool.map((i) => ep.tor_ec_ugm3[i] as number)
  const hi = (d3.quantile([...xsAll].sort(d3.ascending), 0.995) ?? 1) * 1.1
  const m = { top: 22, right: 14, bottom: 44, left: 56 }
  const side = Math.max(220, Math.min((width - 40) / 2 - m.left - m.right, 380))
  // one domain for both axes (the shared crossplot is square), a little below zero for negative predictions
  const dom: [number, number] = [Math.min(0, -0.1 * hi), hi]
  const nb = 48, step = (dom[1] - dom[0]) / nb
  // the wider population as a density in data units, drawn under the analogs
  const bins = useMemo(() => {
    const b = new Map<string, number>()
    for (const i of pool) {
      const xv = ep.tor_ec_ugm3[i] as number, yv = ep.ftir_oos_ugm3[i] as number
      if (!(xv >= dom[0] && xv <= dom[1] && yv >= dom[0] && yv <= dom[1])) continue
      const k = `${Math.floor((xv - dom[0]) / step)}|${Math.floor((yv - dom[0]) / step)}`
      b.set(k, (b.get(k) ?? 0) + 1)
    }
    return b
  }, [pool, ep, hi])  // eslint-disable-line react-hooks/exhaustive-deps
  const maxBin = d3.max([...bins.values()]) ?? 1
  const hist = d3.bin().domain([0, 100]).thresholds(d3.range(0, 100, 10))(analogs.map((i) => ep.residual_pct[i] as number).filter(Number.isFinite))
  const hx = d3.scaleLinear().domain([0, 100]).range([0, side])
  const hy = d3.scaleLinear().domain([0, Math.max(0.2, (d3.max(hist, (b) => b.length) ?? 0) / Math.max(1, analogs.length) * 1.15)]).range([side, 0])
  const color = data.season_colors[group]
  const env = data.envelope.filter((e) => e.baseline === baseline && e.mask === mm)
  const years = data.years
  const yearsOf = (set: string) => years.filter((r) => r.set === set)
  const yearList = [...new Set(years.map((r) => r.year))].sort()
  const yearSets = ['IMPROVE pool', ...data.groups.map((g) => `${baseline}|${mm}|${g}`)]
  return (
    <ChartFrame id="satoshi-envelope" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Envelope"
      title="Lower envelope? — where the analogs sit among the standard model's out-of-sample predictions"
      subtitle="Grey: every lot-251 IMPROVE filter, predicted out of sample by the standard model (site-grouped cross-validation for its calibration sites, direct prediction for its test sites). Coloured: the analogs of one Addis group. Right: each analog's residual (FTIR − TOR) as a percentile among all pool filters in the same TOR EC decile; analogs on the lower envelope would pile up on the left, typical filters spread flat at 10 % per bar."
      controls={<><BaselineSeg value={baseline} onChange={setBaseline} /><GroupSelect data={data} value={group} onChange={setGroup} /></>}>
      <div ref={ref} style={{ position: 'relative', display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        <div style={{ width: side + m.left + m.right }}>
          <Crossplot title={`Standard model (${SHORT[baseline]}), out of sample`} fit={false} domain={dom} maxSide={side}
            xLabel={ecAxis('tor')} yLabel="FTIR EC, standard model (µg/m³)"
            background={(x, y) => [...bins.entries()].map(([k, n]) => { const [bx, by] = k.split('|').map(Number)
              const x0 = dom[0] + bx * step, y0 = dom[0] + by * step
              return <rect key={k} x={x(x0)} y={y(y0 + step)} width={x(x0 + step) - x(x0)} height={y(y0) - y(y0 + step)} fill={INK.axis} fillOpacity={0.08 + 0.6 * Math.sqrt(n / maxBin)} /> })}
            points={analogs.map((i) => ({ x: ep.tor_ec_ugm3[i] as number, y: ep.ftir_oos_ugm3[i] as number, color, id: `improve:${ep.filter_id[i]}`,
              tip: [`IMPROVE ${ep.Site[i]} · filter ${ep.filter_id[i]}`, `TOR EC ${f3(ep.tor_ec_ugm3[i])} · FTIR ${f3(ep.ftir_oos_ugm3[i])} µg/m³`, `residual percentile ${f2(ep.residual_pct[i])}`] }))} />
        </div>
        <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
          <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>Analogs' residual percentile at matched TOR EC</text>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={hy} x={0} label="Share of analogs" tickCount={5} gridWidth={side} labelX={-42} format={(v) => `${Math.round(100 * v)} %`} />
            <XAxis scale={hx} y={side} label="Residual percentile (low = lower envelope)" tickCount={5} labelDy={36} />
            {hist.map((b, i) => <rect key={i} x={hx(b.x0!) + 1} width={Math.max(0, hx(b.x1!) - hx(b.x0!) - 2)} y={hy(b.length / Math.max(1, analogs.length))} height={side - hy(b.length / Math.max(1, analogs.length))} fill={color} fillOpacity={0.8}
              onMouseEnter={(e) => tip.show(e, [`percentile ${b.x0}–${b.x1}`, `${b.length} analogs (${pct(b.length / Math.max(1, analogs.length))})`])} onMouseLeave={tip.hide} />)}
            <line x1={0} x2={side} y1={hy(0.1)} y2={hy(0.1)} stroke={INK.text} strokeDasharray="4 3" />
            <text x={side - 4} y={hy(0.1) - 5} textAnchor="end" fontSize={10} fontFamily={FONT.mono} fill={INK.text}>typical: 10 % per bar</text>
          </g>
        </svg>
        {tip.node}
      </div>
      <Key items={[{ label: 'All lot-251 IMPROVE filters (density)', color: INK.axis, kind: 'bar' }, { label: `Analogs of ${group}`, color, kind: 'dot' }]} />
      <table className="census-table" style={{ marginTop: 10 }}>
        <thead><tr><th>Analogs of ({SHORT[baseline]})</th><th>Lot-251 analogs</th><th>Median residual percentile</th><th>Below 10th</th><th>Below 25th</th><th>Above 75th</th><th>Median FTIR / TOR</th><th>Median TOR EC (µg/m³)</th></tr></thead>
        <tbody>{env.map((e) => <tr key={e.group}><td style={{ color: data.season_colors[e.group], fontWeight: 600 }}>{e.group}</td><td>{e.n}</td><td>{Math.round(e.pct_median)}</td><td>{pct(e.below_p10)}</td><td>{pct(e.below_p25)}</td><td>{pct(e.above_p75)}</td><td>{f2(e.ratio_median)}</td><td>{f2(e.tor_ec_median)}</td></tr>)}</tbody>
      </table>
      <Note>A typical filter has median percentile 50, and 10 % / 25 % / 25 % in the three tail columns. Debus et al. (2022, AMT 15, 2685; Fig. 4) calibrated on 2015 IMPROVE filters with second-derivative spectra and separate biomass-burning models. This pool spans {data.summary.improve_dates?.[0]} to {data.summary.improve_dates?.[1]} (lots 248 and 251), so it cannot show whether analog-like samples were present in 2015; the year mix of each analog set is below.</Note>
      <table className="census-table" style={{ marginTop: 6 }}>
        <thead><tr><th>Sampling year</th>{yearList.map((yv) => <th key={yv}>{yv}</th>)}</tr></thead>
        <tbody>{yearSets.map((set) => { const rows = yearsOf(set); const tot = d3.sum(rows, (r) => r.n) || 1; return <tr key={set}><td>{set === 'IMPROVE pool' ? 'IMPROVE pool' : `Analogs of ${set.split('|')[2]}`}</td>{yearList.map((yv) => <td key={yv}>{pct((rows.find((r) => r.year === yv)?.n ?? 0) / tot)}</td>)}</tr> })}</tbody>
      </table>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ methods
function Methods({ data }: { data: SatoshiData }) {
  const s = data.summary
  return (
    <ChartFrame title="How this was computed — and what from the 1 Oct list is not here" exportable={false}>
      <table className="census-table"><tbody>
        <tr><td>Inputs</td><td>The completed rank-70 context the Follow up meeting view uses: {s.improve_n?.toLocaleString()} IMPROVE filters, {s.addis_n} Addis filters, the frozen site-disjoint split. Baselines: {LONG[SPLINE]}; {LONG[VIBES]}.</td></tr>
        <tr><td>Analogs</td><td>Top {s.n_analogs} IMPROVE filters by Pearson r to each group's median Addis spectrum. The "as presented" selections reproduce the Follow up meeting view's exactly.</td></tr>
        <tr><td>Standard model</td><td>{s.standard_model}.</td></tr>
        <tr><td>Swap test</td><td>{s.n_repeats} site-disjoint 80/20 splits per cohort (seed 20260917), each calibrated on both baselines; k by site-grouped cross-validation, first major minimum.</td></tr>
        <tr><td>Deming</td><td>Addis: λ* {f2(s.lambda_addis)} (HIPS uncertainty vs IMPROVE cross-validation RMSE). IMPROVE stages: each calibration's RMSECV against a {(100 * (s.ref_rel_uncertainty ?? 0)).toFixed(1)} % reference uncertainty, so λ is large and Deming sits close to ordinary least squares of FTIR on TOR.</td></tr>
        <tr><td>Reproduce</td><td><code>uv run --no-sync python research/ftir_hips_chem/workflows/run_satoshi_followup_20261002.py</code> ({s.seconds} s), then <code>uv run --no-sync python gallery/data/export_satoshi_followup.py</code>.</td></tr>
        <tr><td>Rest of the 1 Oct list (7 Oct)</td><td>Blank banks and fingerprint groupings: the sections above, from <code>research/ftir_hips_chem/workflows/run_oct1_remaining_20261007.py</code> then <code>gallery/data/export_oct1_remaining.py</code>; the VIBES re-corrections ran on Colab via <code>scripts/vibes_fullrange_run.py --blank-policy</code>. Seasonal slope/intercept toggle and separate-calibration titles: Follow up meeting view. Agreed region: a Similarity tab mask and shaded on the meeting spectra. Excluded and out-of-test-set Addis filters: labelled in filter drawers and similarity tooltips. Closest-match spectra already reach 500 cm⁻¹ on full-range VIBES.</td></tr>
      </tbody></table>
    </ChartFrame>
  )
}


// ================================================================== 1 Oct list: remaining items (7 Oct)
/**
 * The open items from the 1 Oct task list, run 7 Oct 2026: VIBES re-corrected on
 * Colab with every labeled lot-251 blank and with every labeled blank, and the
 * within-season fingerprint groupings of the Addis spectra. Data from
 * research/ftir_hips_chem/workflows/run_oct1_remaining_20261007.py via
 * gallery/data/export_oct1_remaining.py.
 */
const OCT1_PATH = 'satoshi/oct1_remaining_20261007.json'
interface Oct1Data {
  schema_version: number
  summary: Row
  season_colors: Record<string, string>
  groups: string[]
  seasons: string[]
  banks: Row[]
  rank_cv: Cols
  blank_residuals: Row[]
  ladder: Row[]
  vip_bands: Row[]
  overlap: Row[]
  repeats: Row[]
  group_tests: Row[]
  group_summary: Row[]
  group_members: Cols
  group_spectra: { wn: Record<string, number[]>; median: Record<string, number[]>; regions: Record<string, string> }
}
const BANK_ORDER = ['AIRSpec', 'VIBES-full-r70', 'VIBES-lot251', 'VIBES-allblanks', 'VIBES-lot251-x1', 'VIBES-allblanks-x1']
const BANK_SHORT: Record<string, string> = { AIRSpec: 'Spline', 'VIBES-full-r70': 'VIBES, 87 blanks (until now)', 'VIBES-lot251': 'VIBES, every lot-251 blank', 'VIBES-allblanks': 'VIBES, every labeled blank', 'VIBES-lot251-x1': 'VIBES, lot-251 blanks minus LABE1', 'VIBES-allblanks-x1': 'VIBES, all labeled minus LABE1' }
const CLUSTER_COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

const useOct1 = () => useExport<Oct1Data>(OCT1_PATH, schemaV1)

function Oct1Sections({ o }: { o: ReturnType<typeof useOct1> }) {
  if (!o.data) return <ExportState title="1 Oct list — remaining items" what="the 1 Oct remaining-items export" error={o.error}
    command="uv run --no-sync python research/ftir_hips_chem/workflows/run_oct1_remaining_20261007.py && uv run --no-sync python gallery/data/export_oct1_remaining.py" />
  const d = o.data
  return <>
    <BlankBanks d={d} />
    <BlankEffect d={d} />
    <BlankModels d={d} />
    <FingerprintGroups d={d} />
  </>
}

const banksIn = (d: Oct1Data) => BANK_ORDER.filter((b) => d.repeats.some((r) => r.baseline === b))
const ladderAt = (d: Oct1Data, model: string, stage: string, evalGroup?: string) =>
  d.ladder.find((l) => l.model_id === model && l.stage === stage && (evalGroup == null || l.eval_group === evalGroup))

// ------------------------------------------------------------------ where we stand
function Conclusion({ s, o }: { s: SatoshiData; o: Oct1Data | null }) {
  const h = s.headlines
  const banks = o ? o.banks : []
  const bank = (b: string) => banks.find((x) => x.baseline === b)
  const std = (b: string) => o ? ladderAt(o, `standard|${b}`, 'Addis test set', ALL_ADDIS) : undefined
  const newBanks = ['VIBES-lot251', 'VIBES-allblanks'].filter((b) => bank(b))
  // the fair comparison: the big banks without the LABE1 outlier blank (rank ~80)
  const cleanBanks = ['VIBES-lot251-x1', 'VIBES-allblanks-x1'].filter((b) => bank(b))
  const testOf = (b: string) => o ? ladderAt(o, `standard|${b}`, 'IMPROVE test sites') : undefined
  // VIBES standard models split by how many PLS factors cross-validation chose
  const vibesStd = ['VIBES-full-r70', 'VIBES-lot251', 'VIBES-allblanks', ...cleanBanks].filter((b) => bank(b))
  const lowK = vibesStd.filter((b) => (testOf(b)?.k ?? 99) <= 5 && (std(b)?.deming_intercept ?? -99) > -2)
  const highK = vibesStd.filter((b) => (testOf(b)?.k ?? 0) > 5)
  const kir = (b: string) => o?.repeats.find((r) => r.baseline === b && r.group === 'Kiremt (Jun-Sep)')
  const blankRatio = (b: string) => { const q = (m: string) => o?.blank_residuals.find((x) => x.baseline === m && x.source === 'etad' && x.band === 'all (no CO₂)')?.rms_median; const a = q(b), c = q('VIBES-full-r70'); return a && c ? a / c : null }
  // groupings are judged on the cleanest background (87 blanks, rank 70); the low-rank banks leave PTFE residue
  const tests = o ? o.group_tests.filter((t) => t.region === 'fingerprint' && t.baseline === 'VIBES-full-r70') : []
  const dustAll = o ? o.group_tests.filter((t) => t.region === 'fingerprint').every((t) => t.kruskal_p_Al == null || +t.kruskal_p_Al < 0.01) : false
  const realGroups = tests.filter((t) => t.structure_beyond_null)
  const sig = (p: any) => p != null && Number.isFinite(+p) && +p < 0.01
  const ecGroups = tests.filter((t) => sig(t.kruskal_p_ec))
  const dustGroups = tests.filter((t) => sig(t.kruskal_p_Al))
  return (
    <ChartFrame id="satoshi-conclusion" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Conclusion" title="Where we stand, 7 Oct — the current conclusion" exportable={false}>
      <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
        <li><strong>The IMPROVE analogs fit TOR EC well; the bias only appears at Addis.</strong> Seasonal analog calibrations reproduce TOR EC in cross-validation (R² {range(h.analog_cv_r2[SPLINE])} spline) with intercepts within about ±0.2 µg/m³, but predict Addis with negative intercepts. The problem is transfer to Addis, not the fitting.</li>
        <li><strong>The baseline correction, not the choice of analogs, sets the Addis intercept.</strong> Spline calibrations stay near zero whichever analogs they get; VIBES calibrations go strongly negative whichever analogs they get (swap test). Spline and VIBES pick mostly different analogs ({s.seasons.map((g) => `${h.overlap[g].shared}`).join(' / ')} of 500 shared, Dry / Belg / Kiremt), yet that matters less than the baseline.</li>
        <li><strong>With the 87-blank VIBES background, PLS leans on the PTFE region.</strong> {pct(h.vibes_analog_vip2_1050_1425[0])}–{pct(h.vibes_analog_vip2_1050_1425[1])} of the VIBES analog models' VIP² sits in 1050–1425 cm⁻¹, where Teflon residuals live. Spline models use the same bands as the standard model.</li>
        {o && cleanBanks.length > 0 && <li><strong>More blanks, compared fairly, give the cleanest background but do not fix Addis.</strong> The first big-bank runs were dominated by one outlier blank (LABE1, 28 Jul 2022; see the Blanks tab), which forced a rank-{bank('VIBES-lot251')?.rank} background. Without it ({cleanBanks.map((b, i) => <span key={b}>{i ? '; ' : ''}{bank(b)?.n_blanks} blanks, rank {bank(b)?.rank}</span>)}), held-out blanks correct as cleanly or better than with the 87-blank background ({cleanBanks.map((b) => `${f2(blankRatio(b))}×`).join(' / ')} its residue). Standard all-IMPROVE model at Addis: {['VIBES-full-r70', ...cleanBanks, 'VIBES-allblanks'].filter((b) => bank(b)).map((b, i) => <span key={b}>{i ? '; ' : ''}{bank(b)?.n_blanks} blanks / rank {bank(b)?.rank}: {coef(std(b)?.deming_slope)}x {coefSigned(std(b)?.deming_intercept)} with {testOf(b)?.k} PLS factors (IMPROVE test R² {coef(testOf(b)?.R2)})</span>)}; spline {coef(std('AIRSpec')?.deming_slope)}x {coefSigned(std('AIRSpec')?.deming_intercept)}. <em>The Addis intercept follows the number of PLS factors, not the blank bank:</em> the VIBES models that land near spline use only {lowK.map((b) => testOf(b)?.k).join(' / ')} factors and fit IMPROVE test sites worse, while every VIBES model with {d3.min(highK, (b) => testOf(b)?.k as number)}+ factors is strongly negative, including the cleanest-background one. Seasonal analog calibrations stay strongly negative with every bank (Kiremt {['VIBES-full-r70', ...cleanBanks].map((b) => coefSigned(kir(b)?.addis_intercept_median)).join(' / ')} µg/m³ vs spline {coefSigned(kir('AIRSpec')?.addis_intercept_median)}), with IMPROVE cross-validation intercepts near zero.</li>}
        {o && cleanBanks.length === 0 && newBanks.length > 0 && <li><strong>More blanks did not fix VIBES</strong>, but those runs were dominated by one outlier blank that forced a rank-{bank('VIBES-lot251')?.rank} background (see the Blanks tab); the reruns without it are pending.</li>}
        {o && <li><strong>Fingerprint groupings below 1500 cm⁻¹ track dust, not EC.</strong> Within a season the Addis spectra form {realGroups.length ? `distinct clusters only in ${realGroups.map((t) => shortSeason(t.group)).join(', ')}` : 'no clusters more distinct than structureless data'}; the clusters differ in aluminium (dust) in {dustGroups.map((t) => shortSeason(t.group)).join(', ') || 'no season'} and in HIPS EC in {ecGroups.map((t) => shortSeason(t.group)).join(', ') || 'no season'} (Kruskal–Wallis p &lt; 0.01; 87-blank VIBES).{dustAll ? ' The dust separation holds on every blank bank; EC differences do not.' : ''}</li>}
        <li><strong>Not explained by sampling the lower envelope or by loading alone.</strong> Analogs sit mid-distribution of the standard model's residuals (median percentile {range(h.envelope_pct_median[SPLINE]).replace(/\.\d+/g, '')}), and VIBES analogs that cover the Addis loading range still give the most negative intercepts.</li>
      </ol>
      <Note>Working conclusion: the spline-baseline calibrations remain the defensible ones for Addis. A clean, large VIBES blank bank removes the background problems (held-out blanks correct cleanly) but not the Addis bias: VIBES models that use enough PLS factors to fit IMPROVE well predict Addis with large negative intercepts, and every VIBES seasonal analog calibration does too. The remaining VIBES problem is how the extra VIBES spectral information transfers to Addis, not the blank count. The PTFE region should stay out of analog similarity.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ blank banks
function BlankBanks({ d }: { d: Oct1Data }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const cv = d.rank_cv
  const banks = BANK_ORDER.filter((b) => d.banks.some((x) => x.baseline === b))
  const series = banks.map((b) => {
    const pts: { k: number; mse: number }[] = []
    for (let i = 0; i < cv.baseline.length; i++) if (cv.baseline[i] === b) pts.push({ k: cv.components[i] as number, mse: cv.mean_mse[i] as number })
    return { b, pts, rank: d.banks.find((x) => x.baseline === b)!.rank as number }
  })
  const m = { top: 24, right: 16, bottom: 44, left: 64 }, height = 280
  const iw = Math.max(280, width - m.left - m.right), ih = height - m.top - m.bottom
  const all = series.flatMap((s) => s.pts)
  const x = d3.scaleLinear().domain([1, d3.max(all, (p) => p.k) ?? 85]).range([0, iw])
  const y = d3.scaleLog().domain(d3.extent(all, (p) => p.mse) as [number, number]).nice().range([ih, 0])
  const line = d3.line<{ k: number; mse: number }>().x((p) => x(p.k)).y((p) => y(p.mse))
  const res = d.blank_residuals.filter((r) => r.source === 'etad')
  const bands = [...new Set(res.map((r) => r.band))]
  return (
    <ChartFrame id="satoshi-blanks" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › BlankBanks"
      title="Blanks for the VIBES baseline — which labeled blanks trained the background, and the rank each bank supports"
      subtitle="1 Oct: use every labeled lot-251 blank, then every IMPROVE blank. Only lots 248 and 251 have spectra locally, so 'every labeled blank' means both lots, all sites; other MTL lots need a database pull. All banks also include the ETAD training blanks. Each bank re-corrected all 12,808 spectra (3998–500 cm⁻¹) on Colab; rank = blank-only leave-one-out one-SE rule, cap 85.">
      <table className="census-table">
        <thead><tr><th>Bank</th><th>Blanks</th><th>IMPROVE / ETAD</th><th>Lot 251 / 248</th><th>From IMPROVE test sites</th><th>Background rank</th><th>Failed fits</th><th>Colab hours</th></tr></thead>
        <tbody>{d.banks.map((b) => <tr key={b.baseline}><td style={{ color: BANK_COLOR[b.baseline], fontWeight: 600 }}>{BANK_SHORT[b.baseline]}</td><td>{b.n_blanks}</td><td>{b.n_improve} / {b.n_etad}</td><td>{b.n_lot251} / {b.n_lot248}</td><td>{b.n_test_site}</td><td><strong>{b.rank}</strong></td><td>{b.n_failed}</td><td>{f2(b.hours)}</td></tr>)}</tbody>
      </table>
      <Note>The 87-blank bank behind every VIBES result so far was not 10 blanks per lot: it was a random draw of labeled blanks at training sites (at most 50 per lot after holding 25 % out: 50 lot-251 and 9 lot-248 IMPROVE blanks) plus 28 ETAD blanks. Lot numbers are missing for 7 ETAD blanks.</Note>
      <div ref={ref} style={{ position: 'relative', marginTop: 8 }}>
        <svg width={iw + m.left + m.right} height={height} fontFamily={FONT.family}>
          <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>Leave-one-blank-out error vs background rank</text>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={y} x={0} label="Mean squared error" tickCount={5} gridWidth={iw} labelX={-52} format={(v) => (+v).toExponential(0)} />
            <XAxis scale={x} y={ih} label="Background PCA components" tickCount={8} labelDy={36} />
            {series.map((s) => <g key={s.b}>
              <path d={line(s.pts) ?? ''} fill="none" stroke={BANK_COLOR[s.b]} strokeWidth={1.8} />
              {s.pts.filter((p) => p.k === s.rank).map((p) => <circle key={p.k} cx={x(p.k)} cy={y(p.mse)} r={5} fill={BANK_COLOR[s.b]} stroke="white"
                onMouseEnter={(e) => tip.show(e, [BANK_SHORT[s.b], `chosen rank ${s.rank}`, `LOO MSE ${p.mse.toExponential(2)}`])} onMouseLeave={tip.hide} />)}
            </g>)}
          </g>
        </svg>
        {tip.node}
      </div>
      <Key items={[...series.map((s) => ({ label: `${BANK_SHORT[s.b]}: rank ${s.rank}`, color: BANK_COLOR[s.b], kind: 'line' as const }))]} />
      {res.length > 0 && <>
        <table className="census-table" style={{ marginTop: 10 }}>
          <thead><tr><th>Held-out ETAD blanks (n {res[0].n}): RMS corrected absorbance, median [90th pct]</th>{bands.map((b) => <th key={b}>{b}</th>)}</tr></thead>
          <tbody>{BANK_ORDER.filter((b) => res.some((r) => r.baseline === b)).map((b) => <tr key={b}><td style={{ color: BANK_COLOR[b], fontWeight: 600 }}>{BANK_SHORT[b]}</td>
            {bands.map((band) => { const r = res.find((x) => x.baseline === b && x.band === band); return <td key={band}>{r ? `${(1e5 * r.rms_median).toFixed(1)} [${(1e5 * r.rms_p90).toFixed(1)}]` : '—'}</td> })}</tr>)}</tbody>
        </table>
        <Note>×10⁻⁵ absorbance. A held-out blank should correct to zero. These 9 ETAD blanks are the only blanks no bank trained on (the all-labeled bank uses every IMPROVE blank).</Note>
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ effect on calibrations
const OCT1_METRICS = ['intercept', 'slope'] as const
function BlankEffect({ d }: { d: Oct1Data }) {
  const [metric, setMetric] = useSearchParam<(typeof OCT1_METRICS)[number]>('sblankm', 'intercept', OCT1_METRICS)
  const banks = banksIn(d)
  const ref0 = metric === 'intercept' ? 0 : 1
  return (
    <ChartFrame id="satoshi-blank-effect" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › BlankEffect"
      title={`Do more blanks change the answer? One separate calibration per season, built from that season's 500 IMPROVE analogs (agreed 1 Oct region) — Deming ${metric}`}
      subtitle="Each baseline selects its own analogs and is calibrated with 100 site-disjoint 80/20 splits (k by site-grouped cross-validation). Filled dots: Addis test set, FTIR EC vs Fabs / MAC 10. Hollow dots: the held-out IMPROVE analog sites, FTIR EC vs TOR EC. Whiskers: 10th–90th percentile over splits."
      controls={<Segmented label="Readout" value={metric} options={OCT1_METRICS} onChange={setMetric} />}>
      <IntervalStrip labelWidth={220} panels={[{ label: `Deming ${metric}${metric === 'intercept' ? ' (µg/m³)' : ''}`, reference: ref0 }]}
        blocks={d.groups.map((g) => ({ key: g, title: shortSeason(g), color: d.season_colors[g], rows: banks.map((b) => {
          const a = d.repeats.find((x) => x.baseline === b && x.group === g)
          if (!a) return { key: b, label: BANK_SHORT[b], marks: [[]] }
          const A = (p: string, q: string) => a[`${p}_${metric}_${q}`]
          return { key: b, label: BANK_SHORT[b],
            tip: [`${g} · ${BANK_SHORT[b]}`, `Addis ${metric} ${coef(A('addis', 'median'))} [${coef(A('addis', 'p10'))} to ${coef(A('addis', 'p90'))}]`, `IMPROVE held-out ${metric} ${coef(A('improve', 'median'))} [${coef(A('improve', 'p10'))} to ${coef(A('improve', 'p90'))}]`, `PLS factors median ${f2(a.k_median)}`],
            marks: [[{ lo: A('improve', 'p10'), hi: A('improve', 'p90'), mid: A('improve', 'median'), color: BANK_COLOR[b], shape: 'hollow' as const, r: 4, dy: 3, faint: true },
              { lo: A('addis', 'p10'), hi: A('addis', 'p90'), mid: A('addis', 'median'), color: BANK_COLOR[b], r: 5, dy: -3 }]] }
        }) }))} />
      <Key items={[...banks.map((b) => ({ label: BANK_SHORT[b], color: BANK_COLOR[b], kind: 'bar' as const })), { label: 'Addis test set', color: INK.axis, kind: 'dot' }, { label: 'IMPROVE held-out analog sites', color: INK.axis, kind: 'hollow' }]} />
    </ChartFrame>
  )
}

function BlankModels({ d }: { d: Oct1Data }) {
  const banks = banksIn(d)
  const cell = (l?: Row) => l ? <FitCell stacked slope={l.deming_slope} intercept={l.deming_intercept} r2={l.R2} n={l.n} /> : '—'
  const vip = (model: string) => {
    const b = d.vip_bands.filter((v) => v.model_id === model && (v.band === 'PTFE C–F (1050–1250)' || v.band === '1250–1425'))
    return b.length ? pct(d3.sum(b, (v) => v.vip2_share)) : '— (no channels)'
  }
  const ov = (g: string) => d.overlap.filter((o) => o.group === g)
  return (
    <ChartFrame id="satoshi-blank-models" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › BlankModels"
      title="The standard all-IMPROVE model and the protocol-split analog calibrations on each blank bank" exportable={false}>
      <table className="census-table">
        <thead><tr><th>Standard model (lot 251)</th><th>k</th><th>IMPROVE test sites</th>{d.groups.map((g) => <th key={g}>Addis: {g}</th>)}<th>VIP² in 1050–1425</th></tr></thead>
        <tbody>{banks.map((b) => { const id = `standard|${b}`; return <tr key={b}><td style={{ color: BANK_COLOR[b], fontWeight: 600 }}>{BANK_SHORT[b]}</td><td>{ladderAt(d, id, 'IMPROVE test sites')?.k}</td><td>{cell(ladderAt(d, id, 'IMPROVE test sites'))}</td>{d.groups.map((g) => <td key={g}>{cell(ladderAt(d, id, 'Addis test set', g))}</td>)}<td>{vip(id)}</td></tr> })}</tbody>
      </table>
      <table className="census-table" style={{ marginTop: 10 }}>
        <thead><tr><th>Analog calibration (agreed region)</th>{d.groups.map((g) => <th key={g}>{g}: cross-validation → Addis</th>)}</tr></thead>
        <tbody>{banks.map((b) => <tr key={b}><td style={{ color: BANK_COLOR[b], fontWeight: 600 }}>{BANK_SHORT[b]}</td>{d.groups.map((g) => { const id = `analog|${b}|oct1|${g}`; const cv = ladderAt(d, id, 'cross-validation'), a = ladderAt(d, id, 'Addis test set', g); return <td key={g}>{coef(cv?.deming_slope)}x {coefSigned(cv?.deming_intercept)} → <strong>{coef(a?.deming_slope)}x {coefSigned(a?.deming_intercept)}</strong><br /><span style={{ color: INK.muted, fontSize: 11 }}>k {a?.k} · VIP² 1050–1425 {vip(id)}</span></td> })}</tr>)}</tbody>
      </table>
      <table className="census-table" style={{ marginTop: 10 }}>
        <thead><tr><th>Analogs shared (of 500)</th>{d.groups.map((g) => <th key={g}>{g}</th>)}</tr></thead>
        <tbody>{[...new Set(d.overlap.map((o) => `${o.a}|${o.b}`))].map((pair) => { const [a, b] = pair.split('|'); return <tr key={pair}><td>{BANK_SHORT[a]} vs {BANK_SHORT[b]}</td>{d.groups.map((g) => <td key={g}>{ov(g).find((o) => o.a === a && o.b === b)?.shared ?? '—'}</td>)}</tr> })}</tbody>
      </table>
      <Note>Standard model: all-IMPROVE lot 251, frozen site-disjoint split, k by site-grouped cross-validation. Addis: FTIR EC vs Fabs / MAC 10, Deming with the meeting's error ratio. Analog rows: protocol split, so they match the ladder on the 2 Oct sections above for the spline and 87-blank VIBES rows.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ fingerprint groupings
function FingerprintGroups({ d }: { d: Oct1Data }) {
  const baselines = BANK_ORDER.filter((b) => d.group_tests.some((t) => t.baseline === b))
  const regions = Object.keys(d.group_spectra.regions)
  const [baseline, setBaseline] = useSearchParam<string>('sgbase', baselines[baselines.length - 1] ?? baselines[0], baselines)
  const [region, setRegion] = useSearchParam<string>('sgregion', 'fingerprint', regions)
  const [season, setSeason] = useSearchParam<string>('sgseason', 'Kiremt (Jun-Sep)', d.seasons)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const mem = d.group_members
  const pts: { id: string; pc1: number; ec: number; c: number; res: number }[] = []
  for (let i = 0; i < mem.baseline.length; i++) if (mem.baseline[i] === baseline && mem.region[i] === region && mem.group[i] === season)
    pts.push({ id: String(mem.filter_id[i]), pc1: mem.pc1[i] as number, ec: mem.ec_fabs10[i] as number, c: mem.cluster[i] as number, res: mem.residual[i] as number })
  const t = d.group_tests.find((x) => x.baseline === baseline && x.region === region && x.group === season)
  const summ = d.group_summary.filter((x) => x.baseline === baseline && x.region === region && x.group === season)
  const half = Math.max(260, Math.floor((width - 24) / 2))
  const m = { top: 24, right: 12, bottom: 44, left: 56 }, h = 280
  const iw = half - m.left - m.right, ih = h - m.top - m.bottom
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p.pc1) as [number, number]).nice().range([0, iw])
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p.ec) as [number, number]).nice().range([ih, 0])
  const wn = d.group_spectra.wn[baseline] ?? []
  const lo = 500, hi = 1800
  const idx = wn.map((w, i) => [w, i] as const).filter(([w]) => w >= lo && w <= hi)
  const specs = summ.map((s) => ({ c: s.cluster as number, n: s.n as number, v: d.group_spectra.median[`${baseline}|${region}|${season}|${s.cluster}`] ?? [] }))
  const sx = d3.scaleLinear().domain([hi, lo]).range([0, iw])
  const sy = d3.scaleLinear().domain(d3.extent(specs.flatMap((s) => idx.map(([, i]) => s.v[i]))) as [number, number]).nice().range([ih, 0])
  const sline = (v: number[]) => d3.line<readonly [number, number]>().x(([w]) => sx(w)).y(([, i]) => sy(v[i]))(idx) ?? ''
  const pfmt = (p: any) => p == null ? '—' : p < 0.001 ? '<0.001' : f3(p)
  const allTests = d.group_tests.filter((x) => x.baseline === baseline && x.region === region)
  return (
    <ChartFrame id="satoshi-groups" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › FingerprintGroups"
      title="Within-season groupings of the Addis spectra below 1500 cm⁻¹ — do they line up with higher EC?"
      subtitle={`1 Oct: correlate the Addis spectra within each season over the region VIBES newly covers and look for distinct groupings. Raw correlations are ~0.95, so each spectrum is scaled to unit length, the season mean is subtracted, and the shape deviations are clustered (PCA to 90 % variance, Ward, k 2–4 by silhouette, clusters of at least 5). 'Distinct' means a silhouette above the 95th percentile of structureless Gaussian data with the same variances. Region: ${d.group_spectra.regions[region]}.`}
      controls={<>
        <BankPicker label="Baseline" value={baseline} banks={baselines} labels={BANK_SHORT} onChange={setBaseline} />
        <Segmented label="Region" value={region} options={regions} optionLabel={(r) => r === 'fingerprint' ? 'PTFE dropped' : 'PTFE kept'} onChange={setRegion} />
        <SeasonPicker value={season} groups={d.seasons} short onChange={setSeason} />
      </>}>
      <div ref={ref} style={{ position: 'relative', display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        <svg width={half} height={h} fontFamily={FONT.family}>
          <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>Main shape deviation (PC1) vs HIPS EC</text>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={y} x={0} label="Fabs / 10 (µg/m³)" tickCount={5} gridWidth={iw} labelX={-42} />
            <XAxis scale={x} y={ih} label="PC1 score of the shape deviation" tickCount={5} labelDy={36} format={(v) => String(+(+v).toFixed(3))} />
            {pts.map((p) => <circle key={p.id} cx={x(p.pc1)} cy={y(p.ec)} r={4} fill={CLUSTER_COLORS[(p.c - 1) % 4]} fillOpacity={0.8}
              onMouseEnter={(e) => tip.show(e, [p.id, `cluster ${p.c}`, `Fabs/10 ${f2(p.ec)} µg/m³`, `standard spline model error ${signed(p.res)} µg/m³`])} onMouseLeave={tip.hide} />)}
          </g>
        </svg>
        <svg width={half} height={h} fontFamily={FONT.family}>
          <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>Median spectrum of each cluster, 1800–500 cm⁻¹</text>
          <g transform={`translate(${m.left},${m.top})`}>
            <rect x={sx(1250)} width={sx(1050) - sx(1250)} height={ih} fill={region === 'fingerprint' ? '#f1ece0' : 'transparent'} />
            <rect x={sx(1800)} width={sx(1500) - sx(1800)} height={ih} fill="#eef0f3" />
            <rect x={sx(600)} width={sx(500) - sx(600)} height={ih} fill="#eef0f3" />
            <YAxis scale={sy} x={0} label="Absorbance" tickCount={5} gridWidth={iw} labelX={-48} format={(v) => String(+(+v).toFixed(3))} />
            <XAxis scale={sx} y={ih} label="Wavenumber (cm⁻¹)" tickCount={6} labelDy={36} />
            {specs.map((s) => <path key={s.c} d={sline(s.v)} fill="none" stroke={CLUSTER_COLORS[(s.c - 1) % 4]} strokeWidth={1.8} />)}
          </g>
        </svg>
        {tip.node}
      </div>
      <Key items={[...summ.map((s) => ({ label: `cluster ${s.cluster} (n ${s.n}, median Fabs/10 ${f2(s.ec_fabs10_median)})`, color: CLUSTER_COLORS[(s.cluster - 1) % 4] })), { label: 'outside the clustering region', color: '#eef0f3', kind: 'bar' as const }]} />
      {t && <Note>{shortSeason(season)}: {t.k} clusters, silhouette {f2(t.silhouette)} vs {f2(t.silhouette_null_p95)} for structureless data ({t.structure_beyond_null ? 'more distinct than chance' : t.min_cluster < 5 ? `the best split only isolates ${t.min_cluster} outlier filter${t.min_cluster > 1 ? 's' : ''}` : 'no more distinct than chance'}). Clusters differ in HIPS EC p {pfmt(t.kruskal_p_ec)}, in the standard model's error p {pfmt(t.kruskal_p_residual)}, in Al (dust) p {pfmt(t.kruskal_p_Al)}. PC1 vs EC Spearman ρ {f2(t.pc1_spearman_ec)}.</Note>}
      <table className="census-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Cluster ({shortSeason(season)})</th><th>n</th><th>Fabs/10 median</th><th>Percentile in all Addis</th><th>Std. model error</th><th>Nitrate</th><th>Sulfate</th><th>K⁺</th><th>Al (ng/m³)</th><th>PM2.5</th></tr></thead>
        <tbody>{summ.map((s) => <tr key={s.cluster}><td style={{ color: CLUSTER_COLORS[(s.cluster - 1) % 4], fontWeight: 600 }}>{s.cluster}</td><td>{s.n}</td><td>{f2(s.ec_fabs10_median)}</td><td>{Math.round(s.ec_pct_all_addis_median)}</td><td>{signed(s.residual_median)}</td><td>{f2(s.nitrate_median)}</td><td>{f2(s.sulfate_median)}</td><td>{f2(s.K_ion_median)}</td><td>{Math.round(s.Al_median)}</td><td>{f2(s.PM25_median)}</td></tr>)}</tbody>
      </table>
      <table className="census-table" style={{ marginTop: 8 }}>
        <thead><tr><th>All seasons ({BANK_SHORT[baseline]})</th><th>k</th><th>Silhouette / null 95th</th><th>Distinct?</th><th>p: EC</th><th>p: model error</th><th>p: Al</th><th>p: K⁺</th><th>p: nitrate</th><th>PC1–EC ρ</th></tr></thead>
        <tbody>{allTests.map((x) => <tr key={x.group}><td style={{ color: d.season_colors[x.group], fontWeight: 600 }}>{x.group}</td><td>{x.k}</td><td>{f2(x.silhouette)} / {f2(x.silhouette_null_p95)}</td><td>{x.structure_beyond_null ? 'yes' : x.min_cluster < 5 ? `no (only ${x.min_cluster} outlier${x.min_cluster > 1 ? 's' : ''} split off)` : 'no'}</td><td>{pfmt(x.kruskal_p_ec)}</td><td>{pfmt(x.kruskal_p_residual)}</td><td>{pfmt(x.kruskal_p_Al)}</td><td>{pfmt(x.kruskal_p_K_ion)}</td><td>{pfmt(x.kruskal_p_nitrate)}</td><td>{f2(x.pc1_spearman_ec)}</td></tr>)}</tbody>
      </table>
      <Note>SPARTAN chemistry in µg/m³ except Al (ng/m³); Kruskal–Wallis across clusters. Standard model error = all-IMPROVE spline model (lot 251) minus Fabs/10. Dry has no SPARTAN chemistry for most filters.</Note>
    </ChartFrame>
  )
}
