import { useMemo, useRef, useState } from 'react'
import { robustDomain } from '@/lib/axes'
import { Marker } from '@/components/RefLines'
import { useExport } from '@/lib/data'
import { ExportState } from '@/components/ExportState'
import * as d3 from 'd3'
import { ChartFrame, Note, Segmented, Select } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { XAxis, YAxis } from '@/components/Axes'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useSearchParam } from '@/lib/url'
import { BankPicker } from '@/components/Pickers'
import { SpectrumPlot } from '@/components/SpectrumPlot'
import { SweepPlot } from '@/components/SweepPlot'
import { FacetGrid } from '@/components/FacetGrid'
import { BANK_COLOR, FONT, INK, SPECTRAL_REGION } from '@/lib/theme'
import { f2, pct, sci } from '@/lib/format'

/**
 * Blanks analysis: the PTFE field blanks that train the VIBES background.
 * Every number comes from gallery/data/export_blanks.py (bundle raw spectra,
 * the frozen case ledger, the capped run and the two 7 Oct blank-bank runs on
 * Drive); nothing is refitted in the browser.
 */

const BLANKS_PATH = 'blanks/blanks_20261007.json'
type Cols = Record<string, (number | string | boolean | null)[]>
interface Bank {
  n_blanks: number; rank: number; rank_min_loo: number; cap: number
  loo_mean: number[]; loo_se: number[]; one_se_threshold: number
  blank_ids: string[]; per_blank: Record<string, number[]>; top5pct_share: number
  drop_one: { removed: string[]; rank: number; rank_min_loo: number }[]
  worst: { id: string; error_x_median: number; minus_bank_median: number[] }
  mean: number[]; loadings: number[][]; var: number[]; held_corrected: number[][]
  capture: Record<string, { median: number[]; p10: number[]; p90: number[]; n: number }>
}
interface BlanksData {
  schema_version: number
  bin_wn: number[]
  banks: Record<string, Bank>
  bank_labels: Record<string, string>
  inventory: Cols
  raw: number[][]
  groups: Record<string, { n: number; q: number[][]; sd: number[]; sd_detrended: number[] }>
  shared_pca: { var: number[]; scores: number[][]; loadings: number[][]; mean: number[] }
  held_ids: string[]
  held_source: string[]
}

const ALL_BANKS = ['VIBES-full-r70', 'VIBES-lot251', 'VIBES-allblanks', 'VIBES-lot251-x1', 'VIBES-allblanks-x1'] as const
/** Banks present in the export (the minus-LABE1 reruns appear once they are on Drive). */
const banksOf = (d: BlanksData): string[] => ALL_BANKS.filter((b) => d.banks[b])
type BankKey = string
const GROUP_COLOR: Record<string, string> = { 'IMPROVE lot 251': '#2C6E9E', 'IMPROVE lot 248': '#7fb3d5', 'ETAD lot 251': '#c0392b', 'ETAD lot 248': '#f1948a', 'ETAD lot ?': '#999999' }
const ROLE_COLOR: Record<string, string> = { 'trains this background': '#1f2933', 'held out, evaluated': '#0f766e', 'not used': '#c7ccd1' }
/** Regions shaded on every spectrum: CO₂ (never used for similarity) and the PTFE C–F band. */
const SHADES = [SPECTRAL_REGION.ptfe, SPECTRAL_REGION.co2]
const RANGES = ['4000–500', '4000–1500', '1500–500'] as const
const rangeOf = (r: string): [number, number] => r === '4000–1500' ? [4000, 1500] : r === '1500–500' ? [1500, 500] : [4000, 500]

export function BlanksPage() {
  const { data, error } = useExport<BlanksData>(BLANKS_PATH, (d) => d.schema_version === 1 ? null : 'unsupported data version')
  if (!data) return <ExportState title="Blanks" what="the blanks analysis" error={error} command="uv run --no-sync python gallery/data/export_blanks.py" />
  return <BlanksView d={data} />
}

function BlanksView({ d }: { d: BlanksData }) {
  return (
    <PageToc>
      <Findings d={d} />
      <Inventory d={d} />
      <RawSpectra d={d} />
      <Variability d={d} />
      <PcaMap d={d} />
      <Drift d={d} />
      <RankSelection d={d} />
      <PerBlankError d={d} />
      <Components d={d} />
      <AfterCorrection d={d} />
      <Capture d={d} />
    </PageToc>
  )
}

// ------------------------------------------------------------------ helpers
const inv = (d: BlanksData, col: string) => d.inventory[col] as any[]
const nBlanks = (d: BlanksData) => inv(d, 'sample_id').length
function roleIn(d: BlanksData, bank: BankKey, i: number) {
  if (inv(d, `in_${bank}`)[i]) return 'trains this background'
  if (inv(d, 'evaluated_blank')[i]) return 'held out, evaluated'
  return 'not used'
}

/** Channels inside a [high, low] wavenumber window. */
const chans = (wn: number[], [hi, lo]: [number, number]) => d3.range(wn.length).filter((j) => wn[j] <= hi && wn[j] >= lo)

// ------------------------------------------------------------------ 0. findings
function Findings({ d }: { d: BlanksData }) {
  const b = d.banks
  const big = b['VIBES-lot251'], all = b['VIBES-allblanks'], cap = b['VIBES-full-r70']
  // where the worst blank sits in the shared PCA: distance (in SD units, PC1–PC2) to each lot's centroid
  const wi = (inv(d, 'sample_id') as string[]).indexOf(big.worst.id)
  const near = Object.keys(d.groups).filter((g) => d.groups[g].n >= 10).map((g) => {
    const idx = d3.range(nBlanks(d)).filter((i) => inv(d, 'group')[i] === g)
    const z = [0, 1].map((k) => { const v = idx.map((i) => d.shared_pca.scores[k][i]); return (d.shared_pca.scores[k][wi] - d3.mean(v)!) / d3.deviation(v)! })
    return { g, dist: Math.hypot(z[0], z[1]) }
  }).sort((a, c) => a.dist - c.dist)
  return (
    <ChartFrame id="blanks-findings" title="What the blanks show — short answers" exportable={false}>
      <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>
        <li><strong>One blank sets the background rank for the big banks.</strong> {big.worst.id} has a leave-one-out error {Math.round(big.worst.error_x_median).toLocaleString()}× the median blank's. With it, the one-SE rule picks rank {big.rank} ({big.n_blanks} blanks) and {all.rank} ({all.n_blanks}); without it, rank {big.drop_one[0].rank} and {all.drop_one[0].rank}. The worst 5 % of blanks carry {pct(big.top5pct_share)} of the error in the big banks against {pct(cap.top5pct_share)} in the 87-blank bank, which never drew this blank.</li>
        {wi >= 0 && <li><strong>That blank looks like a different lot.</strong> It is labeled {inv(d, 'group')[wi]} ({inv(d, 'site')[wi]}, {inv(d, 'date')[wi]}), but in the PCA map it sits {f2(near[0].dist)} SD from the {near[0].g} centroid and {f2(near.find((x) => x.g === inv(d, 'group')[wi])?.dist)} SD from its own lot's. Worth checking its lot record before reusing it.</li>}
        <li><strong>The "more blanks" runs therefore tested a simpler background, not more blanks.</strong> Held-out blanks correct worse with the rank-{big.rank}/{all.rank} backgrounds than with rank {cap.rank}, mostly in the PTFE band (see "After correction").</li>
        {['VIBES-lot251-x1', 'VIBES-allblanks-x1'].some((k) => b[k]) && <li><strong>Without that blank, the big banks behave.</strong> {['VIBES-lot251-x1', 'VIBES-allblanks-x1'].filter((k) => b[k]).map((k, i) => <span key={k}>{i ? '; ' : ''}{b[k].n_blanks} blanks → rank {b[k].rank}, worst 5 % carry {pct(b[k].top5pct_share)}</span>)}. Their held-out blanks correct as cleanly as the 87-blank background's or better (see "After correction"). The Addis effect of these backgrounds is on the Satoshi 2 Oct tab: cleaner blanks did not remove the Addis bias.</li>}
        <li><strong>Leave-one-blank-out error keeps falling to the rank cap</strong> ({cap.cap}) for every bank; the one-SE rule, not the error curve, decides where to stop.</li>
        <li><strong>Blanks differ mostly by offset and slope;</strong> the shared PCA's first component carries {pct(d.shared_pca.var[0])} of the variance. Lot 248 and lot 251 separate in the PCA map, and ETAD blanks sit with their own lot.</li>
      </ol>
      <Note>{b['VIBES-allblanks-x1'] ? 'The large banks were rerun without the outlier blank (rank 78 / 80), so "more blanks" is now tested at a comparable background complexity.' : 'Next step implied: rerun the large banks without the outlier blank (or at a fixed rank near 70–80), so that "more blanks" is tested at a comparable background complexity.'}</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 1. inventory
function Inventory({ d }: { d: BlanksData }) {
  const [bank, setBank] = useSearchParam<BankKey>('bkbank', 'VIBES-allblanks', banksOf(d))
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const n = nBlanks(d)
  const rows = useMemo(() => d3.range(n).map((i) => ({ i, id: inv(d, 'sample_id')[i] as string, date: inv(d, 'date')[i] ? new Date(inv(d, 'date')[i]) : null, group: inv(d, 'group')[i] as string, site: inv(d, 'site')[i] as string, split: inv(d, 'split')[i] as string })), [d, n])
  const groups = [...new Set(rows.map((r) => r.group))].sort()
  const m = { top: 10, right: 16, bottom: 40, left: 130 }
  const iw = Math.max(300, width - m.left - m.right), rowH = 34, ih = groups.length * rowH
  const dated = rows.filter((r) => r.date)
  const x = d3.scaleTime().domain(d3.extent(dated, (r) => r.date!) as [Date, Date]).nice().range([0, iw])
  const undated = rows.filter((r) => !r.date)
  const ux = d3.scalePoint<number>().domain(undated.map((r) => r.i)).range([0, iw]).padding(0.5)
  const jitter = (i: number) => ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1
  const counts = banksOf(d).map((b) => ({ b, n: d.banks[b].n_blanks, lots: groups.map((g) => rows.filter((r) => r.group === g && inv(d, `in_${b}`)[r.i]).length) }))
  return (
    <ChartFrame id="blanks-inventory" source="gallery/app/src/pages/BlanksPage.tsx › Inventory"
      title="Inventory — every labeled blank, when it was sampled, and which background it trained"
      subtitle={`${n} labeled blanks: IMPROVE blanks carry a sampling date; ETAD blanks have none and are spread out in MediaId order. Colour shows the blank's role in the selected background.`}
      controls={<BankPicker value={bank} banks={banksOf(d)} labels={d.bank_labels} onChange={setBank} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={ih + m.top + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            {groups.map((g, gi) => <g key={g}>
              <rect x={-m.left + 4} y={gi * rowH} width={iw + m.left - 4} height={rowH} fill={gi % 2 ? 'transparent' : INK.page} />
              <text x={-10} y={gi * rowH + rowH / 2} dy="0.32em" textAnchor="end" fontSize={11.5} fontWeight={600} fill={GROUP_COLOR[g] ?? INK.text}>{g}</text>
            </g>)}
            {dated.length > 0 && <XAxis scale={x} y={ih} label="Sampling date (IMPROVE); ETAD rows are in MediaId order" tickCount={8} labelDy={32} format={(v) => d3.timeFormat('%b %Y')(v as Date)} />}
            {rows.map((r) => {
              const gi = groups.indexOf(r.group)
              const cx = r.date ? x(r.date) : ux(r.i)!
              const role = roleIn(d, bank, r.i)
              return <circle key={r.id} cx={cx} cy={gi * rowH + 5 + jitter(r.i) * (rowH - 10)} r={3.2} fill={ROLE_COLOR[role]} fillOpacity={role === 'not used' ? 0.7 : 0.9}
                stroke={r.split === 'test' ? INK.negative : 'none'} strokeWidth={1}
                onMouseEnter={(e) => tip.show(e, [r.id, `${r.site}${r.date ? ' · ' + d3.timeFormat('%Y-%m-%d')(r.date) : ''}`, r.group, `split: ${r.split}`, role])} onMouseLeave={tip.hide} />
            })}
          </g>
        </svg>
        {tip.node}
      </div>
      <Legend items={[...Object.entries(ROLE_COLOR).map(([label, color]) => ({ label, color })), { label: 'IMPROVE test-site blank', color: INK.negative, shape: 'ring' as const }]} />
      <table className="census-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Background</th><th>Blanks</th>{groups.map((g) => <th key={g}>{g}</th>)}<th>Rank</th></tr></thead>
        <tbody>{counts.map((c) => <tr key={c.b}><td style={{ color: BANK_COLOR[c.b], fontWeight: 600 }}>{d.bank_labels[c.b]}</td><td>{c.n}</td>{c.lots.map((v, i) => <td key={i}>{v}</td>)}<td>{d.banks[c.b].rank}</td></tr>)}</tbody>
      </table>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 2. raw spectra
const SPEC_MODES = ['median & IQR', 'every blank'] as const
function RawSpectra({ d }: { d: BlanksData }) {
  const [mode, setMode] = useSearchParam<(typeof SPEC_MODES)[number]>('bkmode', 'median & IQR', SPEC_MODES)
  const [range, setRange] = useSearchParam<(typeof RANGES)[number]>('bkrange', '4000–500', RANGES)
  const lg = useLegend()
  const dom = rangeOf(range)
  const js = chans(d.bin_wn, dom)
  const groups = Object.keys(d.groups)
  const show = (g: string) => lg.show(g)
  const idx = d3.range(nBlanks(d)).filter((i) => show(inv(d, 'group')[i]))
  const vals = mode === 'every blank' ? idx.flatMap((i) => js.map((j) => d.raw[i][j])) : groups.filter(show).flatMap((g) => js.flatMap((j) => [d.groups[g].q[0][j], d.groups[g].q[4][j]]))
  const yDom = robustDomain(vals, 0.002, 0.998).domain
  const every = mode === 'every blank'
  const series = every
    ? idx.map((i) => { const g = inv(d, 'group')[i] as string, id = inv(d, 'sample_id')[i] as string
      return { id, wn: d.bin_wn, y: d.raw[i], color: GROUP_COLOR[g], role: 'trace' as const, width: 0.7, group: g, tip: [id, `${inv(d, 'site')[i]} · ${inv(d, 'date')[i] ?? 'undated'}`, g] } })
    : groups.filter(show).map((g) => ({ id: g, wn: d.bin_wn, y: d.groups[g].q[2], color: GROUP_COLOR[g], width: 1.8, group: g }))
  const bands = every ? [] : groups.filter(show).map((g) => ({ id: g, wn: d.bin_wn, lo: d.groups[g].q[1], hi: d.groups[g].q[3], color: GROUP_COLOR[g], opacity: 0.18, group: g }))
  return (
    <ChartFrame id="blanks-raw" source="gallery/app/src/pages/BlanksPage.tsx › RawSpectra"
      title="Raw blank spectra — what a clean PTFE filter looks like, by lot"
      subtitle="Uncorrected absorbance of every labeled blank (8-channel means). Median and interquartile band per lot and source, or every blank as a faint line; hover a line to name it. Shading: PTFE C–F band and CO₂."
      controls={<><Segmented label="Show" value={mode} options={SPEC_MODES} onChange={setMode} /><Segmented label="Range" value={range} options={RANGES} onChange={setRange} /></>}>
      <SpectrumPlot series={series} bands={bands} window={dom} yDomain={yDom} height={340} yLabel="Absorbance (raw)" regions={SHADES}
        hover={every ? 'nearest' : 'none'} dim={lg.dim} />
      <Legend {...lg.props} items={groups.map((g) => ({ label: g, color: GROUP_COLOR[g], shape: 'line' as const, detail: `n=${d.groups[g].n}` }))} />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 3. variability
const VAR_MODES = ['as measured', 'offset removed'] as const
function Variability({ d }: { d: BlanksData }) {
  const [mode, setMode] = useSearchParam<(typeof VAR_MODES)[number]>('bkvar', 'as measured', VAR_MODES)
  const [range, setRange] = useSearchParam<(typeof RANGES)[number]>('bkvrange', '4000–500', RANGES)
  const lg = useLegend(['ETAD lot ?'])
  const dom = rangeOf(range)
  const js = chans(d.bin_wn, dom)
  const groups = Object.keys(d.groups).filter((g) => d.groups[g].n >= 5)
  const key = mode === 'as measured' ? 'sd' : 'sd_detrended'
  const vals = groups.filter(lg.show).flatMap((g) => js.map((j) => d.groups[g][key][j]))
  return (
    <ChartFrame id="blanks-variability" source="gallery/app/src/pages/BlanksPage.tsx › Variability"
      title="Where blanks vary — standard deviation across blanks at each wavenumber"
      subtitle="This is what the background model has to absorb. 'Offset removed' subtracts each blank's own mean absorbance first, leaving shape differences (PTFE bands, slope). Lots with fewer than 5 blanks are not drawn."
      controls={<><Segmented label="Blanks" value={mode} options={VAR_MODES} onChange={setMode} /><Segmented label="Range" value={range} options={RANGES} onChange={setRange} /></>}>
      <SpectrumPlot series={groups.filter(lg.show).map((g) => ({ id: g, wn: d.bin_wn, y: d.groups[g][key], color: GROUP_COLOR[g], width: 1.7, group: g }))}
        window={dom} yDomain={[0, d3.max(vals) ?? 0.01]} yLabel="SD of absorbance" regions={SHADES} hover="none" dim={lg.dim} />
      <Legend {...lg.props} items={groups.map((g) => ({ label: g, color: GROUP_COLOR[g], shape: 'line' as const, detail: `n=${d.groups[g].n}` }))} />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 4. PCA map
const PCA_COLOR = ['lot / source', 'sampling date', 'role in background'] as const
function PcaMap({ d }: { d: BlanksData }) {
  const pcs = d.shared_pca.var.map((_, i) => `PC${i + 1}`)
  const [px, setPx] = useSearchParam<string>('bkpcx', 'PC1', pcs)
  const [py, setPy] = useSearchParam<string>('bkpcy', 'PC2', pcs)
  const [colorBy, setColorBy] = useSearchParam<string>('bkpccol', 'lot / source', [...PCA_COLOR])
  const [bank, setBank] = useSearchParam<BankKey>('bkpcbank', 'VIBES-allblanks', banksOf(d))
  const [pin, setPin] = useState<number | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const ix = pcs.indexOf(px), iy = pcs.indexOf(py)
  const sx = d.shared_pca.scores[ix], sy = d.shared_pca.scores[iy]
  const n = nBlanks(d)
  const side = Math.min(560, Math.max(260, width * 0.55))
  const m = { top: 12, right: 12, bottom: 44, left: 60 }
  const x = d3.scaleLinear().domain(d3.extent(sx) as [number, number]).nice().range([0, side])
  const y = d3.scaleLinear().domain(d3.extent(sy) as [number, number]).nice().range([side, 0])
  const dates = inv(d, 'date').map((v) => (v ? +new Date(v) : NaN))
  const dateScale = d3.scaleSequential(d3.interpolateViridis).domain(d3.extent(dates.filter(Number.isFinite)) as [number, number])
  const color = (i: number) => colorBy === 'lot / source' ? GROUP_COLOR[inv(d, 'group')[i]] : colorBy === 'sampling date' ? (Number.isFinite(dates[i]) ? dateScale(dates[i]) : INK.neutral) : ROLE_COLOR[roleIn(d, bank, i)]
  const worst = d.banks['VIBES-allblanks'].worst.id
  const devOf = (i: number) => d.raw[i].map((v, j) => v - d.shared_pca.mean[j])
  const legendItems = colorBy === 'lot / source' ? Object.keys(d.groups).map((g) => ({ label: g, color: GROUP_COLOR[g] }))
    : colorBy === 'role in background' ? Object.entries(ROLE_COLOR).map(([label, c]) => ({ label, color: c })) : [{ label: 'earliest', color: dateScale.range()[0] as unknown as string }, { label: 'latest', color: d3.interpolateViridis(1) }, { label: 'undated (ETAD)', color: INK.neutral }]
  return (
    <ChartFrame id="blanks-pca" source="gallery/app/src/pages/BlanksPage.tsx › PcaMap"
      title="PCA map of every blank — which blanks look alike"
      subtitle={`One PCA of all ${n} raw blank spectra (4000–500 cm⁻¹, mean-centred). ${px} carries ${pct(d.shared_pca.var[ix])} of the variance, ${py} ${pct(d.shared_pca.var[iy])}. Click a blank to see how it differs from the average blank. The ringed point is the outlier blank that sets the big banks' rank.`}
      controls={<>
        <Segmented label="x" value={px} options={pcs} onChange={setPx} />
        <Segmented label="y" value={py} options={pcs} onChange={setPy} />
        <Select label="Colour" value={colorBy} options={[...PCA_COLOR]} onChange={setColorBy} />
        {colorBy === 'role in background' && <BankPicker value={bank} banks={banksOf(d)} labels={d.bank_labels} onChange={setBank} />}
      </>}>
      <div ref={ref} style={{ position: 'relative', display: 'flex', flexWrap: 'wrap', gap: 16 }}>
        <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={y} x={0} label={`${py} score`} tickCount={6} gridWidth={side} labelX={-46} />
            <XAxis scale={x} y={side} label={`${px} score`} tickCount={6} labelDy={36} />
            {d3.range(n).map((i) => {
              const id = inv(d, 'sample_id')[i] as string
              return <circle key={id} cx={x(sx[i])} cy={y(sy[i])} r={id === worst ? 6 : 3.4} fill={color(i)} fillOpacity={0.85}
                stroke={id === worst ? INK.negative : pin === i ? INK.text : 'none'} strokeWidth={id === worst || pin === i ? 2 : 0} style={{ cursor: 'pointer' }}
                onClick={() => setPin(pin === i ? null : i)}
                onMouseEnter={(e) => tip.show(e, [id, `${inv(d, 'site')[i]} · ${inv(d, 'date')[i] ?? 'undated'}`, inv(d, 'group')[i], `${px} ${f2(sx[i])} · ${py} ${f2(sy[i])}`, 'click to compare with the average blank'])} onMouseLeave={tip.hide} />
            })}
          </g>
        </svg>
        <div style={{ flex: '1 1 300px', minWidth: 280 }}>
          {pin == null ? <Note>Click a blank in the map to see its spectrum minus the average blank. The loadings of {px} and {py} are below for reference.</Note> : null}
          <SpectrumPlot width={Math.max(300, width - side - m.left - m.right - 24)} height={220} window={[4000, 500]} regions={SHADES} hover="none" zeroLine={pin != null}
            series={pin == null
              ? [{ id: px, wn: d.bin_wn, y: d.shared_pca.loadings[ix], color: INK.text, width: 1.5 }, { id: py, wn: d.bin_wn, y: d.shared_pca.loadings[iy], color: INK.text, width: 1.3, dash: '4 3' }]
              : [{ id: 'dev', wn: d.bin_wn, y: devOf(pin), color: color(pin), width: 1.6 }]}
            yDomain={pin == null ? (d3.extent([...d.shared_pca.loadings[ix], ...d.shared_pca.loadings[iy]]) as [number, number]) : (d3.extent(devOf(pin)) as [number, number])}
            yLabel={pin == null ? 'Loading' : 'Blank − average'} title={pin == null ? `${px} (solid) and ${py} (dashed) loadings` : `${inv(d, 'sample_id')[pin]} minus the average blank`} />
        </div>
        {tip.node}
      </div>
      <Legend items={legendItems} />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 5. drift
function Drift({ d }: { d: BlanksData }) {
  const pcs = d.shared_pca.var.map((_, i) => `PC${i + 1}`)
  const [pc, setPc] = useSearchParam<string>('bkdrift', 'PC1', pcs)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const k = pcs.indexOf(pc)
  const pts = d3.range(nBlanks(d)).filter((i) => inv(d, 'date')[i]).map((i) => ({ i, t: new Date(inv(d, 'date')[i]), v: d.shared_pca.scores[k][i], g: inv(d, 'group')[i] as string }))
  const m = { top: 12, right: 16, bottom: 44, left: 60 }, h = 280
  const iw = Math.max(300, width - m.left - m.right), ih = h - m.top - m.bottom
  const x = d3.scaleTime().domain(d3.extent(pts, (p) => p.t) as [Date, Date]).nice().range([0, iw])
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p.v) as [number, number]).nice().range([ih, 0])
  const months = d3.rollups(pts, (v) => d3.median(v, (p) => p.v)!, (p) => `${p.g}|${d3.timeFormat('%Y-%m')(p.t)}`)
    .map(([key, v]) => ({ g: key.split('|')[0], t: d3.timeParse('%Y-%m')(key.split('|')[1])!, v })).sort((a, b) => +a.t - +b.t)
  return (
    <ChartFrame id="blanks-drift" source="gallery/app/src/pages/BlanksPage.tsx › Drift"
      title="Drift over time — do blanks change across the sampling period?"
      subtitle={`Shared-PCA score of each dated (IMPROVE) blank against its sampling date, with the monthly median per lot. A trend would mean a background trained on one period fits another period worse.`}
      controls={<Segmented label="Score" value={pc} options={pcs} onChange={setPc} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={h} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            <YAxis scale={y} x={0} label={`${pc} score`} tickCount={5} gridWidth={iw} labelX={-46} />
            <XAxis scale={x} y={ih} label="Sampling date" tickCount={8} labelDy={36} format={(v) => d3.timeFormat('%b %Y')(v as Date)} />
            {pts.map((p) => <circle key={p.i} cx={x(p.t)} cy={y(p.v)} r={2.8} fill={GROUP_COLOR[p.g]} fillOpacity={0.55}
              onMouseEnter={(e) => tip.show(e, [inv(d, 'sample_id')[p.i], `${inv(d, 'site')[p.i]} · ${inv(d, 'date')[p.i]}`, `${pc} ${f2(p.v)}`])} onMouseLeave={tip.hide} />)}
            {[...new Set(months.map((mm) => mm.g))].map((g) => <path key={g} d={d3.line<{ t: Date; v: number }>().x((p) => x(p.t)).y((p) => y(p.v))(months.filter((mm) => mm.g === g)) ?? ''} fill="none" stroke={GROUP_COLOR[g]} strokeWidth={2} />)}
          </g>
        </svg>
        {tip.node}
      </div>
      <Legend items={[...new Set(pts.map((p) => p.g))].map((g) => ({ label: g, color: GROUP_COLOR[g] }))} note="lines: monthly median" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 6. rank selection
function RankSelection({ d }: { d: BlanksData }) {
  const lg = useLegend()
  const shown = banksOf(d).filter((b) => lg.show(d.bank_labels[b]))
  const all = shown.flatMap((b) => d.banks[b].loo_mean.map((v, i) => [v - d.banks[b].loo_se[i], v + d.banks[b].loo_se[i]]).flat()).filter((v) => v > 0)
  // the band's lower edge (mean − SE) often goes below zero: floor it at the axis
  const yDom = d3.scaleLog().domain(d3.extent(all) as [number, number]).nice().domain() as [number, number]
  const series = shown.map((b) => {
    const B = d.banks[b], label = d.bank_labels[b]
    return { id: b, label, color: BANK_COLOR[b], group: label, points: B.loo_mean.map((v, i) => ({
      x: i + 1, y: v, lo: Math.max(v - B.loo_se[i], yDom[0]), hi: v + B.loo_se[i],
      tip: i + 1 === B.rank ? [label, `chosen rank ${B.rank} (one-SE rule)`, `error ${sci(v)}, ${f2(v / B.loo_mean[B.rank_min_loo - 1])}× the minimum at ${B.rank_min_loo}`]
        : [label, `rank ${i + 1}`, `error ${sci(v)} ± ${sci(B.loo_se[i])}`] })) }
  })
  return (
    <ChartFrame id="blanks-rank" source="gallery/app/src/pages/BlanksPage.tsx › RankSelection"
      title="Rank selection — how many background components each blank bank supports"
      subtitle="Leave-one-blank-out reconstruction error (mean ± 1 SE over blanks) against the number of PCA components. The one-SE rule takes the smallest rank whose error is within one SE of the minimum (dashed line). A single extreme blank inflates the SE, so the rule stops early even though the error keeps falling.">
      <SweepPlot series={series} xDomain={[1, 85]} yScale="log" yDomain={yDom} height={320} dim={lg.dim}
        xLabel="Background PCA components" yLabel="LOO mean squared error" yFormat={(v) => v.toExponential(0)}
        yMarkers={shown.map((b) => ({ at: d.banks[b].one_se_threshold, color: BANK_COLOR[b] }))}
        chosen={shown.map((b) => ({ series: b, x: d.banks[b].rank }))} />
      <Legend {...lg.props} items={banksOf(d).map((b) => ({ label: d.bank_labels[b], color: BANK_COLOR[b], shape: 'line' as const, detail: `${d.banks[b].n_blanks} blanks → rank ${d.banks[b].rank}` }))} />
      <table className="census-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Background</th><th>Chosen rank</th><th>Error minimum at</th><th>Worst 5 % carry</th><th>Remove the worst blank</th><th>Remove 2</th><th>Remove 3</th></tr></thead>
        <tbody>{banksOf(d).map((b) => { const B = d.banks[b]; return <tr key={b}><td style={{ color: BANK_COLOR[b], fontWeight: 600 }}>{d.bank_labels[b]}</td><td>{B.rank}</td><td>{B.rank_min_loo}</td><td>{pct(B.top5pct_share)}</td>
          {B.drop_one.map((x, i) => <td key={i}><strong>rank {x.rank}</strong><br /><span style={{ color: INK.muted, fontSize: 11 }}>without {x.removed[x.removed.length - 1]}</span></td>)}</tr> })}</tbody>
      </table>
      <Note>Removing blanks recomputes the leave-one-out curve on the remaining blanks only; nothing else changes. The big banks jump from rank 13–15 to about 80 once one blank is left out.</Note>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 7. per-blank error
function PerBlankError({ d }: { d: BlanksData }) {
  const [bank, setBank] = useSearchParam<BankKey>('bkerrbank', 'VIBES-allblanks', banksOf(d))
  const B = d.banks[bank]
  const ranks = Object.keys(B.per_blank).sort((a, b) => +a - +b)
  const [rankSel, setRankSel] = useState<string>(String(B.rank))
  const rk = ranks.includes(rankSel) ? rankSel : String(B.rank)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const e = B.per_blank[rk]
  const idxOf = useMemo(() => new Map((inv(d, 'sample_id') as string[]).map((s, i) => [s, i])), [d])
  const rows = B.blank_ids.map((id, j) => ({ id, e: e[j], i: idxOf.get(id)! })).sort((a, b) => a.e - b.e)
  const med = d3.median(rows, (r) => r.e) ?? 1
  const m = { top: 12, right: 20, bottom: 44, left: 60 }, h = 220
  const iw = Math.max(300, width - m.left - m.right), ih = h - m.top - m.bottom
  const x = d3.scaleLog().domain(d3.extent(rows, (r) => r.e) as [number, number]).nice().range([0, iw])
  const groups = Object.keys(d.groups)
  const yb = d3.scaleBand().domain(groups).range([0, ih]).padding(0.25)
  const jitter = (i: number) => ((Math.sin(i * 78.233) * 43758.5453) % 1 + 1) % 1
  const top = rows.slice(-8).reverse()
  return (
    <ChartFrame id="blanks-outliers" source="gallery/app/src/pages/BlanksPage.tsx › PerBlankError"
      title="Which blanks dominate — each blank's leave-one-out error"
      subtitle="How badly the background built from all the other blanks reconstructs each blank, at the selected rank (log scale). Points far to the right are blanks no other blank resembles; they set the SE in the rank rule."
      controls={<><BankPicker value={bank} banks={banksOf(d)} labels={d.bank_labels} onChange={setBank} /><Segmented label="Rank" value={rk} options={ranks} onChange={setRankSel} /></>}>
      <div ref={ref} style={{ position: 'relative' }}>
        <svg width={iw + m.left + m.right} height={h} fontFamily={FONT.family}>
          <g transform={`translate(${m.left},${m.top})`}>
            {groups.map((g) => <text key={g} x={-6} y={(yb(g) ?? 0) + yb.bandwidth() / 2} dy="0.32em" textAnchor="end" fontSize={10} fill={GROUP_COLOR[g]}>{g.replace('IMPROVE ', 'IMP ')}</text>)}
            <XAxis scale={x} y={ih} label={`LOO squared error at rank ${rk} (log)`} tickCount={6} labelDy={36} format={(v) => (+v).toExponential(0)} />
            <Marker orient="v" at={x(med)} from={0} to={ih} />
            {rows.map((r) => { const g = inv(d, 'group')[r.i]; return <circle key={r.id} cx={x(r.e)} cy={(yb(g) ?? 0) + jitter(r.i) * yb.bandwidth()} r={r.id === B.worst.id ? 6 : 3} fill={GROUP_COLOR[g]} fillOpacity={0.8} stroke={r.id === B.worst.id ? INK.negative : 'none'} strokeWidth={2}
              onMouseEnter={(ev) => tip.show(ev, [r.id, `${inv(d, 'site')[r.i]} · ${inv(d, 'date')[r.i] ?? 'undated'}`, `error ${sci(r.e)} (${Math.round(r.e / med).toLocaleString()}× median)`])} onMouseLeave={tip.hide} />})}
          </g>
        </svg>
        {tip.node}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginTop: 8 }}>
        <table className="census-table" style={{ flex: '1 1 320px' }}>
          <thead><tr><th>Worst blanks (rank {rk})</th><th>Site · date</th><th>× median</th></tr></thead>
          <tbody>{top.map((r) => <tr key={r.id}><td>{r.id}</td><td>{inv(d, 'site')[r.i]} · {inv(d, 'date')[r.i] ?? 'undated'}</td><td>{Math.round(r.e / med).toLocaleString()}</td></tr>)}</tbody>
        </table>
        <div style={{ flex: '1 1 340px' }}>
          <SpectrumPlot width={Math.max(320, width / 2)} height={230} window={[4000, 500]} regions={SHADES} hover="none" zeroLine
            series={[{ id: B.worst.id, wn: d.bin_wn, y: B.worst.minus_bank_median, color: INK.negative, width: 1.6 }]}
            yDomain={d3.extent(B.worst.minus_bank_median) as [number, number]} yLabel="Blank − bank median" title={`${B.worst.id} minus the median blank of this bank`} />
        </div>
      </div>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 8. components
function Components({ d }: { d: BlanksData }) {
  const [bank, setBank] = useSearchParam<BankKey>('bkcompbank', 'VIBES-full-r70', banksOf(d))
  const B = d.banks[bank]
  return (
    <ChartFrame id="blanks-components" source="gallery/app/src/pages/BlanksPage.tsx › Components"
      title="Background components — the shapes VIBES may subtract from every spectrum"
      subtitle={`First ${B.loadings.length} PCA components of the selected bank's blanks (unit-length loadings). The background is the bank's mean blank plus any combination of the first ${B.rank} components; anything a sample shares with these shapes can be absorbed as background.`}
      controls={<BankPicker value={bank} banks={banksOf(d)} labels={d.bank_labels} ranks={Object.fromEntries(banksOf(d).map((b) => [b, d.banks[b].rank]))} onChange={setBank} />}>
      <FacetGrid items={B.loadings} gap={12} render={(l, cw, k) =>
        <SpectrumPlot width={cw} minWidth={0} height={170} window={[4000, 500]} regions={SHADES} hover="none" zeroLine yLabel=""
          series={[{ id: `PC${k + 1}`, wn: d.bin_wn, y: l, color: k >= B.rank ? INK.neutral : BANK_COLOR[bank], width: 1.4 }]}
          yDomain={d3.extent(l) as [number, number]} title={`PC${k + 1} · ${pct(B.var[k])} of blank variance${k >= B.rank ? ' · not used' : ''}`} />} />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 9. after correction
function AfterCorrection({ d }: { d: BlanksData }) {
  const [range, setRange] = useSearchParam<(typeof RANGES)[number]>('bkcrange', '4000–500', RANGES)
  const lg = useLegend()
  const dom = rangeOf(range)
  const js = chans(d.bin_wn, dom)
  const stats = Object.fromEntries(banksOf(d).map((b) => {
    const H = d.banks[b].held_corrected
    const q = (p: number) => d.bin_wn.map((_, j) => d3.quantile(H.map((r) => r[j]).filter((v) => v != null), p) ?? NaN)
    return [b, { lo: q(0.1), med: q(0.5), hi: q(0.9) }]
  }))
  const shown = banksOf(d).filter((b) => lg.show(d.bank_labels[b]))
  const vals = shown.flatMap((b) => js.flatMap((j) => [stats[b].lo[j], stats[b].hi[j]]))
  const bands: [string, number, number][] = [['all (no CO₂)', 4000, 500], ['PTFE 1050–1250', 1250, 1050], ['600–1500', 1500, 600], ['1500–4000', 4000, 1500]]
  const rms = (b: string, hi: number, lo: number) => {
    const js = d.bin_wn.map((w, j) => [w, j] as const).filter(([w]) => w <= hi && w >= lo && !(w > 1800 && w < 2500)).map(([, j]) => j)
    return d3.median(d.banks[b].held_corrected.map((r) => Math.sqrt(d3.mean(js, (j) => r[j] ** 2) ?? NaN))) ?? NaN
  }
  return (
    <ChartFrame id="blanks-after" source="gallery/app/src/pages/BlanksPage.tsx › AfterCorrection"
      title="After correction — what is left on blanks no background trained on"
      subtitle={`VIBES-corrected spectra of the ${d.held_ids.length} blanks held out by every bank (${d.held_source.filter((s) => s === 'IMPROVE').length} IMPROVE, ${d.held_source.filter((s) => s === 'ETAD').length} ETAD). A perfect background leaves zero everywhere; residue is what the background could not explain and would be read as analyte. Median and 10–90 % band per background.`}
      controls={<Segmented label="Range" value={range} options={RANGES} onChange={setRange} />}>
      <SpectrumPlot window={dom} regions={SHADES} hover="none" zeroLine dim={lg.dim} yLabel="Corrected absorbance"
        yDomain={[Math.min(0, d3.min(vals) ?? 0), d3.max(vals) ?? 1e-4]}
        bands={shown.map((b) => ({ id: b, wn: d.bin_wn, lo: stats[b].lo, hi: stats[b].hi, color: BANK_COLOR[b], opacity: 0.15, group: d.bank_labels[b] }))}
        series={shown.map((b) => ({ id: b, wn: d.bin_wn, y: stats[b].med, color: BANK_COLOR[b], width: 1.7, group: d.bank_labels[b] }))} />
      <Legend {...lg.props} items={banksOf(d).map((b) => ({ label: d.bank_labels[b], color: BANK_COLOR[b], shape: 'line' as const, detail: `rank ${d.banks[b].rank}` }))} />
      <table className="census-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Median RMS residue (×10⁻⁵)</th>{bands.map(([l]) => <th key={l}>{l}</th>)}</tr></thead>
        <tbody>{banksOf(d).map((b) => <tr key={b}><td style={{ color: BANK_COLOR[b], fontWeight: 600 }}>{d.bank_labels[b]}</td>{bands.map(([l, hi, lo]) => <td key={l}>{(1e5 * rms(b, hi, lo)).toFixed(1)}</td>)}</tr>)}</tbody>
      </table>
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ 10. capture
const POPS = ['held-out blanks', 'IMPROVE samples', 'Addis samples'] as const
const POP_COLOR: Record<string, string> = { 'held-out blanks': INK.axis, 'IMPROVE samples': '#2C6E9E', 'Addis samples': '#c0392b' }
function Capture({ d }: { d: BlanksData }) {
  const [bank, setBank] = useSearchParam<BankKey>('bkcapbank', 'VIBES-full-r70', banksOf(d))
  const B = d.banks[bank]
  const pops = POPS.filter((p) => B.capture[p])
  const series = pops.map((p) => { const c = B.capture[p]
    return { id: p, label: p, color: POP_COLOR[p], points: c.median.map((v, i) => ({ x: i + 1, y: v, lo: c.p10[i], hi: c.p90[i],
      tip: [p, `n ${c.n}`, `at rank ${i + 1}: median ${pct(v)} [${pct(c.p10[i])}–${pct(c.p90[i])}]`] })) } })
  return (
    <ChartFrame id="blanks-capture" source="gallery/app/src/pages/BlanksPage.tsx › Capture"
      title="What the background could absorb — share of each spectrum inside the blank subspace"
      subtitle="For each raw spectrum, the fraction of its deviation from the bank's mean blank that lies in the span of the first k blank components (median and 10–90 %). Blanks should be near 1; for ambient samples, a high fraction means the background can mimic most of the sample, analyte included. This is a geometric bound, not the VIBES fit itself (VIBES adds a positive analyte term and priors)."
      controls={<BankPicker value={bank} banks={banksOf(d)} labels={d.bank_labels} onChange={setBank} />}>
      <SweepPlot series={series} xDomain={[1, B.cap]} yDomain={[0, 1]} height={300}
        xLabel="Blank components (k)" yLabel="Share inside blank subspace" yFormat={(v) => `${Math.round(100 * v)} %`}
        xMarkers={[{ at: B.rank, label: `chosen rank ${B.rank}`, color: BANK_COLOR[bank] }]}
        chosen={pops.map((p) => ({ series: p, x: B.rank }))} />
      <Legend items={POPS.map((p) => ({ label: p, color: POP_COLOR[p], shape: 'line' as const, detail: `n=${B.capture[p]?.n ?? 0}` }))} />
    </ChartFrame>
  )
}
