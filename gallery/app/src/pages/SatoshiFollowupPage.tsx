import { useEffect, useMemo, useRef, useState, useId } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, ExplainOnly, Note, Segmented, Select } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { useHighlight, type FilterRecord } from '@/lib/highlight'
import { FONT, INK } from '@/lib/theme'

/**
 * Satoshi's questions after the 1 Oct 2026 meeting (his email of 2 Oct). Every
 * number comes from research/ftir_hips_chem/output/tables/satoshi_followup_20261002
 * via gallery/data/export_satoshi_followup.py; nothing is refitted in the browser.
 */

const URL_PATH = `${import.meta.env.BASE_URL}data/satoshi/followup_20261002.json`
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
const BASE_COLOR: Record<string, string> = { [SPLINE]: '#2C6E9E', [VIBES]: '#d95f0e' }
const SHORT: Record<string, string> = { [SPLINE]: 'Spline', [VIBES]: 'VIBES' }
const LONG: Record<string, string> = { [SPLINE]: 'Spline baseline (4000–1425)', [VIBES]: 'VIBES baseline (4000–500, rank 70)' }
const MASKS = ['presented', 'oct1'] as const
type Mask = (typeof MASKS)[number]
const MASK_SHORT: Record<string, string> = { presented: 'As presented 1 Oct', oct1: 'Agreed 1 Oct region' }
const SHARED_COLOR = '#4b5563'
const STANDARD_COLOR = '#6b7280'

const f2 = (v: any) => (v == null || !Number.isFinite(+v) ? '—' : (+v).toFixed(2))
const f3 = (v: any) => (v == null || !Number.isFinite(+v) ? '—' : (+v).toFixed(3))
const signed = (v: any, d = 2) => (v == null || !Number.isFinite(+v) ? '—' : `${+v >= 0 ? '+' : '−'}${Math.abs(+v).toFixed(d)}`)
const pct = (v: any) => (v == null || !Number.isFinite(+v) ? '—' : `${Math.round(100 * +v)} %`)
const shortSeason = (g: string) => g.split(' ')[0]
/** Log-axis labels at 1, 2 and 5 of each decade only. */
const logTick = (v: number) => { const d = +(v / 10 ** Math.floor(Math.log10(v))).toFixed(6); return [1, 2, 5].includes(d) ? String(+(+v).toPrecision(2)) : '' }
const minus = (t: string) => t.replace(/^-/, '−')
const range = (a: any[]) => `${minus(f2(a?.[0]))} to ${minus(f2(a?.[1]))}`

export function SatoshiFollowupPage() {
  const [data, setData] = useState<SatoshiData | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const c = new AbortController()
    fetch(URL_PATH, { signal: c.signal })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: SatoshiData) => { if (d.schema_version !== 1) throw new Error('unsupported data version'); setData(d) })
      .catch((e) => { if (e.name !== 'AbortError') setError(String(e)) })
    return () => c.abort()
  }, [])
  if (error) return <ChartFrame title="Satoshi's questions" exportable={false}><Empty>Could not load the export: {error}. Run <code>uv run --no-sync python research/ftir_hips_chem/workflows/run_satoshi_followup_20261002.py</code>, then <code>uv run --no-sync python gallery/data/export_satoshi_followup.py</code>.</Empty></ChartFrame>
  if (!data) return <ChartFrame title="Satoshi's questions" exportable={false}><Empty>Loading the 2 Oct follow-up…</Empty></ChartFrame>
  return <SatoshiView data={data} />
}

function SatoshiView({ data }: { data: SatoshiData }) {
  const [mask, setMask] = useSearchParam<Mask>('sregion', 'presented', MASKS)
  const hl = useHighlight()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') hl.setPinned(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [hl])
  return (
    <PageToc>
      <Answers data={data} />
      <ChartFrame title="Similarity region — which spectral region chose the analogs on every chart below" exportable={false}
        controls={<Segmented label="Similarity region" value={mask} options={MASKS} optionLabel={(m) => MASK_SHORT[m]} onChange={setMask} />}>
        <p className="chart-note">{data.summary.masks?.[mask]}. The spline baseline only covers 4000–1425 cm⁻¹, so its analogs are identical under both regions; only the VIBES analogs change. Every PLS model is still fitted on each baseline's full predictor grid (the region only chooses the analogs).</p>
      </ChartFrame>
      <Overlap data={data} mask={mask} />
      <Swap data={data} mask={mask} />
      <Vip data={data} mask={mask} />
      <Ladder data={data} mask={mask} />
      <Envelope data={data} mask={mask} />
      <ExplainOnly><Methods data={data} /></ExplainOnly>
    </PageToc>
  )
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
    <ChartFrame source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Answers" title="Satoshi's questions, 2 Oct — short answers (similarity region as presented 1 Oct; completed rank-70 VIBES)" exportable={false}>
      <table className="census-table">
        <thead><tr><th style={{ width: '24%' }}>Question</th><th>Short answer</th></tr></thead>
        <tbody>
          <tr><td>How different are the analogs the two baselines select?</td>
            <td><strong>Mostly different.</strong> Of 500 analogs per set, spline and VIBES share {data.seasons.map((g, i) => <span key={g}>{i ? ', ' : ''}{ov(g).shared} ({shortSeason(g)})</span>)} and {ov('All Addis').shared} for all Addis. Filters only VIBES picks carry {f2(d3.min(ecRatio))}–{f2(d3.max(ecRatio))}× the median TOR EC of filters only the spline picks, and lower OC/EC{higherOcEc.length ? ` except in ${higherOcEc.map(shortSeason).join(', ')}` : ''}.</td></tr>
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
type KeyItem = { label: string; color: string; kind?: 'dot' | 'square' | 'hollow' | 'line' | 'dash' | 'bar' | 'diamond' }
function Key({ items }: { items: KeyItem[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label} className="legend-item">
          <svg width={20} height={12} aria-hidden="true">
            {(it.kind ?? 'dot') === 'dot' && <circle cx={10} cy={6} r={4} fill={it.color} fillOpacity={0.85} />}
            {it.kind === 'square' && <rect x={6} y={2} width={8} height={8} fill={it.color} />}
            {it.kind === 'hollow' && <circle cx={10} cy={6} r={4} fill="none" stroke={it.color} strokeWidth={1.3} />}
            {it.kind === 'diamond' && <path d="M10 1 L15 6 L10 11 L5 6 Z" fill="white" stroke={it.color} strokeWidth={1.3} />}
            {it.kind === 'line' && <line x1={1} x2={19} y1={6} y2={6} stroke={it.color} strokeWidth={2.2} />}
            {it.kind === 'dash' && <line x1={1} x2={19} y1={6} y2={6} stroke={it.color} strokeWidth={1.6} strokeDasharray="4 3" />}
            {it.kind === 'bar' && <rect x={1} y={3} width={18} height={6} fill={it.color} />}
          </svg>
          {it.label}
        </span>
      ))}
    </div>
  )
}

function GroupSelect({ data, value, onChange, label = 'Analogs of' }: { data: SatoshiData; value: string; onChange: (v: string) => void; label?: string }) {
  return <Select label={label} value={value} options={data.groups} onChange={onChange} />
}
function BaselineSeg({ value, onChange }: { value: string; onChange: (v: Baseline) => void }) {
  return <Segmented label="Baseline" value={value as Baseline} options={BASELINES} optionLabel={(b) => SHORT[b]} onChange={onChange} />
}
/** The analog set a baseline uses under a region: the spline set is the same under both. */
const maskFor = (baseline: string, mask: string) => (baseline === SPLINE ? 'presented' : mask)

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
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const col = metric === 'intercept' ? 'addis_intercept' : 'addis_slope'
  const arm = (g: string, sel: string, cal: string) => data.swap_arms.find((a) => a.group === g && a.selected_on === sel && a.mask === (sel === SPLINE ? 'presented' : mask) && a.calibrated_on === cal)
  const rows = data.groups.flatMap((g) => ARMS.map((a) => ({ g, ...a, r: arm(g, a.sel, a.cal) })))
  const vals = rows.flatMap((r) => [r.r?.[`${col}_p10`], r.r?.[`${col}_p90`], r.r?.[`${col}_protocol`]]).filter((v) => v != null && Number.isFinite(v)) as number[]
  const ref0 = metric === 'intercept' ? 0 : 1
  const m = { top: 10, right: 20, bottom: 44, left: 190 }
  const rowH = 20, gap = 14
  const iw = Math.max(260, width - m.left - m.right)
  const x = d3.scaleLinear().domain(d3.extent([...vals, ref0]) as [number, number]).nice().range([0, iw])
  const yOf = (i: number) => i * rowH + Math.floor(i / 4) * gap
  const ih = yOf(rows.length - 1) + rowH
  const eff = data.swap_effects.filter((e) => e.mask === mask && e.metric === col)
  const effCell = (g: string, effect: string, fixed: string) => {
    const e = eff.find((x) => x.group === g && x.effect === effect && x.held_fixed === fixed)
    return e ? <>{signed(e.median)} <span style={{ color: INK.muted }}>[{signed(e.p10)}, {signed(e.p90)}]</span></> : '—'
  }
  return (
    <ChartFrame id="satoshi-swap" source="gallery/app/src/pages/SatoshiFollowupPage.tsx › Swap"
      title={`Training set or baseline — Addis Deming ${metric} when each analog set is calibrated on each baseline`}
      subtitle="Each analog set (selected on the spline or VIBES baseline) is calibrated on both baselines with the same site-grouped cross-validation. Dots are medians over 100 site-disjoint calibration splits; whiskers span the 10th–90th percentile; open diamonds are the protocol split shown in the meeting tab. Colour is the baseline the PLS model was calibrated on; shape is the baseline that chose the analogs."
      controls={<Segmented label="Readout" value={metric} options={SWAP_METRICS} onChange={setMetric} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            {data.groups.map((g, gi) => (
              <g key={g}>
                <rect x={-m.left + 4} y={yOf(gi * 4) - 3} width={iw + m.left - 4} height={rowH * 4 + 4} fill={gi % 2 ? 'transparent' : INK.page} />
                <text x={-m.left + 8} y={yOf(gi * 4) + rowH * 2} dy="0.32em" fontSize={11.5} fontWeight={700} fill={data.season_colors[g]}>{shortSeason(g)}</text>
              </g>
            ))}
            <XAxis scale={x} y={ih + 4} tickCount={6} label={`Test set (Addis) Deming ${metric}, FTIR EC vs Fabs / MAC 10`} labelDy={36} format={(v) => String(+(+v).toFixed(2))} />
            <line x1={x(ref0)} x2={x(ref0)} y1={-4} y2={ih + 4} stroke={INK.axis} strokeOpacity={0.7} />
            {rows.map((r, i) => {
              const a = r.r
              const cy = yOf(i) + rowH / 2
              if (!a) return null
              const color = BASE_COLOR[r.cal]
              const lo = a[`${col}_p10`], hi = a[`${col}_p90`], med = a[`${col}_median`], prot = a[`${col}_protocol`]
              return (
                <g key={i} onMouseEnter={(e) => tip.show(e, [`${r.g} · ${r.label}`, `median ${f2(med)} [10–90 %: ${f2(lo)} to ${f2(hi)}]`, `protocol split ${f2(prot)} (k ${a.k_protocol})`, `PLS factors median ${f2(a.k_median)} · IMPROVE test R² median ${f2(a.improve_test_R2_median)}`, `Addis R² median ${f2(a.addis_R2_median)}`])} onMouseLeave={tip.hide}>
                  <text x={-10} y={cy} dy="0.32em" textAnchor="end" fontSize={11} fill={INK.text}>{r.label}</text>
                  <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} stroke={color} strokeWidth={1.6} />
                  {r.sel === SPLINE ? <circle cx={x(med)} cy={cy} r={5} fill={color} /> : <rect x={x(med) - 4.5} y={cy - 4.5} width={9} height={9} fill={color} />}
                  <path d={`M${x(prot)} ${cy - 5} L${x(prot) + 5} ${cy} L${x(prot)} ${cy + 5} L${x(prot) - 5} ${cy} Z`} fill="white" stroke={INK.text} strokeWidth={1} />
                  <rect x={-m.left + 4} y={cy - rowH / 2} width={iw + m.left} height={rowH} fill="transparent" />
                </g>
              )
            })}
          </g>
        </svg>
        {tip.node}
      </div>
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
      <Note>Change in the Addis Deming {metric}, median [10th, 90th percentile]. The baseline effect is paired: same cohort and same split, two baselines. The training-set effect compares different cohorts, so its interval is over all pairs of splits and is wider by construction. The diagonal arms reproduce the meeting tab's seasonal calibrations exactly (k, slope and intercept checked to 1e-6).</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. VIP
const SHADE: { lo: number; hi: number; label: string; region: 'both' | 'oct1' | 'presented' }[] = [
  { lo: 1800, hi: 2500, label: 'CO₂, excluded from similarity', region: 'both' },
  { lo: 3600, hi: 4000, label: '>3600, excluded from similarity', region: 'both' },
  { lo: 1050, hi: 1250, label: 'PTFE, excluded from similarity (agreed region)', region: 'oct1' },
  { lo: 400, hi: 600, label: '<600, excluded from similarity (agreed region)', region: 'oct1' },
]

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
  const shades = SHADE.filter((s) => (s.region === 'both' || s.region === mm) && s.hi > x.domain()[1])
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
            {shades.map((s) => <rect key={s.label} x={x(Math.min(s.hi, x.domain()[0]))} width={Math.max(0, x(Math.max(s.lo, x.domain()[1])) - x(Math.min(s.hi, x.domain()[0])))} y={0} height={ih} fill={s.region === 'oct1' ? '#fde68a' : INK.grid} fillOpacity={s.region === 'oct1' ? 0.45 : 0.9} />)}
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
      <Key items={[{ label: `Standard model (all IMPROVE lot 251, k ${std.k})`, color: STANDARD_COLOR, kind: 'line' }, { label: `${group} analog model (k ${ana?.k ?? '—'})`, color: data.season_colors[group], kind: 'line' }, { label: 'Excluded from similarity, both regions', color: '#c9d0d8', kind: 'bar' }, { label: 'Excluded only in the agreed region', color: '#f7d774', kind: 'bar' }]} />
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
                    onMouseEnter={(e) => tip.show(e, [s.label, p.stage, `n ${p.r.n} · R² ${f2(p.r.R2)} · RMSE ${f3(p.r.RMSE)} µg/m³`, `Deming ${f2(p.r.deming_slope)}x ${signed(p.r.deming_intercept)} (median reference ${f3(p.r.x_median)} µg/m³)`, `mean bias ${signed(p.r.bias, 3)} µg/m³ · PLS factors ${p.r.k}`])} onMouseLeave={tip.hide} />)}
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
  const cell = (r?: Row) => r ? <>{f2(r.deming_slope)}x {signed(r.deming_intercept, 3)} <span style={{ color: INK.muted }}>R² {f2(r.R2)}, bias {signed(r.bias, 3)}, n {r.n}</span></> : '—'
  return (
    <table className="census-table" style={{ marginTop: 10 }}>
      <thead><tr><th>Model, population</th>{STAGES.map((s) => <th key={s}>{s}</th>)}</tr></thead>
      <tbody>{rows.map((s) => <tr key={s.label}><td style={{ color: s.color === '#b8bfc7' ? INK.muted : s.color, fontWeight: 600 }}>{s.label}</td>{s.rows.map((r, i) => <td key={i}>{cell(r)}</td>)}</tr>)}</tbody>
    </table>
  )
}

/** Cross-validation predictions of the analog model next to the standard model's out-of-sample predictions of the same filters. */
function LadderCrossplots({ data, analogId, baseline, mask, group }: { data: SatoshiData; analogId: string; baseline: string; mask: string; group: string }) {
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
function Cross({ title, pts, color, dom, source }: { title: string; pts: Pt[]; color: string; dom: [number, number]; source: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const hl = useHighlight()
  const clip = useId()
  const m = { top: 22, right: 12, bottom: 44, left: 54 }
  const side = Math.max(180, Math.min(width - m.left - m.right, 340))
  const x = d3.scaleLinear().domain(dom).range([0, side])
  const y = d3.scaleLinear().domain(dom).range([side, 0])
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
        <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>{title}</text>
        <g transform={`translate(${m.left},${m.top})`}>
          <defs><clipPath id={clip}><rect width={side} height={side} /></clipPath></defs>
          <YAxis scale={y} x={0} label="FTIR EC (µg/m³)" tickCount={5} gridWidth={side} labelX={-40} format={(v) => String(+(+v).toFixed(2))} />
          <XAxis scale={x} y={side} label="TOR EC (µg/m³)" tickCount={5} labelDy={36} format={(v) => String(+(+v).toFixed(2))} />
          <g clipPath={`url(#${clip})`}>
            <rect width={side} height={side} fill="transparent" onClick={() => hl.setPinned(null)} />
            <line x1={x(dom[0])} y1={y(dom[0])} x2={x(dom[1])} y2={y(dom[1])} stroke={INK.identity} strokeDasharray="5 4" strokeWidth={1.5} />
            {pts.map((p, i) => {
              const id = `improve:${p.id}`
              const focus = id === hl.pinnedId || id === hl.hoverId
              return <circle key={i} cx={x(p.x)} cy={y(p.y)} r={focus ? 6 : 3} fill={p.hollow ? 'none' : color} fillOpacity={0.55} stroke={focus ? INK.text : color} strokeWidth={focus ? 2 : p.hollow ? 1 : 0.3} style={{ cursor: 'pointer' }}
                onMouseEnter={(e) => { tip.show(e, [`IMPROVE ${p.site} · filter ${p.id}`, p.note, `TOR EC ${f3(p.x)} · FTIR EC ${f3(p.y)} µg/m³`, 'click to pin and open its record']); hl.setHover(id) }}
                onMouseLeave={() => { tip.hide(); hl.setHover(null) }}
                onClick={() => { hl.setPinned(id); hl.openRecord(improveRecord(p.id, p.site, source, [['prediction', p.note], ['TOR EC (µg/m³)', p.x], ['FTIR EC (µg/m³)', p.y], ['FTIR − TOR (µg/m³)', p.y - p.x]])) }} />
            })}
          </g>
        </g>
      </svg>
      {tip.node}
    </div>
  )
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
  const x = d3.scaleLinear().domain([0, hi]).range([0, side])
  const y = d3.scaleLinear().domain([Math.min(0, -0.1 * hi), hi]).range([side, 0])
  const nb = 48
  const bins = useMemo(() => {
    const b = new Map<string, number>()
    for (const i of pool) {
      const xv = ep.tor_ec_ugm3[i] as number, yv = ep.ftir_oos_ugm3[i] as number
      if (!(xv >= 0 && xv <= hi && yv >= y.domain()[0] && yv <= hi)) continue
      const k = `${Math.floor((xv / hi) * nb)}|${Math.floor(((yv - y.domain()[0]) / (hi - y.domain()[0])) * nb)}`
      b.set(k, (b.get(k) ?? 0) + 1)
    }
    return b
  }, [pool, ep, hi])  // eslint-disable-line react-hooks/exhaustive-deps
  const maxBin = d3.max([...bins.values()]) ?? 1
  const cw = side / nb
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
        <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
          <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>Standard model ({SHORT[baseline]}), out of sample</text>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={y} x={0} label="FTIR EC, standard model (µg/m³)" tickCount={5} gridWidth={side} labelX={-42} format={(v) => String(+(+v).toFixed(2))} />
            <XAxis scale={x} y={side} label="TOR EC (µg/m³)" tickCount={5} labelDy={36} format={(v) => String(+(+v).toFixed(2))} />
            {[...bins.entries()].map(([k, n]) => { const [bx, by] = k.split('|').map(Number); return <rect key={k} x={bx * cw} y={side - (by + 1) * cw} width={cw} height={cw} fill={INK.axis} fillOpacity={0.08 + 0.6 * Math.sqrt(n / maxBin)} /> })}
            <line x1={x(0)} y1={y(0)} x2={x(hi)} y2={y(hi)} stroke={INK.identity} strokeDasharray="5 4" strokeWidth={1.5} />
            {analogs.map((i) => <circle key={i} cx={x(ep.tor_ec_ugm3[i] as number)} cy={y(ep.ftir_oos_ugm3[i] as number)} r={2.6} fill={color} fillOpacity={0.75}
              onMouseEnter={(e) => tip.show(e, [`IMPROVE ${ep.Site[i]} · filter ${ep.filter_id[i]}`, `TOR EC ${f3(ep.tor_ec_ugm3[i])} · FTIR ${f3(ep.ftir_oos_ugm3[i])} µg/m³`, `residual percentile ${f2(ep.residual_pct[i])}`])} onMouseLeave={tip.hide} />)}
          </g>
        </svg>
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
        <tr><td>Inputs</td><td>The completed rank-70 context the Follow up meeting tab uses: {s.improve_n?.toLocaleString()} IMPROVE filters, {s.addis_n} Addis filters, the frozen site-disjoint split. Baselines: {LONG[SPLINE]}; {LONG[VIBES]}.</td></tr>
        <tr><td>Analogs</td><td>Top {s.n_analogs} IMPROVE filters by Pearson r to each group's median Addis spectrum. The "as presented" selections reproduce the meeting tab's exactly.</td></tr>
        <tr><td>Standard model</td><td>{s.standard_model}.</td></tr>
        <tr><td>Swap test</td><td>{s.n_repeats} site-disjoint 80/20 splits per cohort (seed 20260917), each calibrated on both baselines; k by site-grouped cross-validation, first major minimum.</td></tr>
        <tr><td>Deming</td><td>Addis: λ* {f2(s.lambda_addis)} (HIPS uncertainty vs IMPROVE cross-validation RMSE). IMPROVE stages: each calibration's RMSECV against a {(100 * (s.ref_rel_uncertainty ?? 0)).toFixed(1)} % reference uncertainty, so λ is large and Deming sits close to ordinary least squares of FTIR on TOR.</td></tr>
        <tr><td>Reproduce</td><td><code>uv run --no-sync python research/ftir_hips_chem/workflows/run_satoshi_followup_20261002.py</code> ({s.seconds} s), then <code>uv run --no-sync python gallery/data/export_satoshi_followup.py</code>.</td></tr>
        <tr><td>Not in this tab</td><td>From the 1 Oct list: VIBES refits with all lot-251 blanks and with all IMPROVE blanks (need new Colab background fits); the Addis within-season correlation groupings below 1500 cm⁻¹; the full-range closest-match spectra plot; the excluded-sample labels.</td></tr>
      </tbody></table>
    </ChartFrame>
  )
}

