import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { histogramSeries, SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { coef, fmt } from '@/lib/format'
import { useEstimator } from '@/lib/regressionPrefs'
import { regression } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { DailyPanels, qLine, type PanelSpec } from './DailyPanels'
import { CHANNELS, INSTRUMENTS, finite, instrumentColor, instrumentName, quantile, type Day, type QualityIndex } from './data'
import { AETH_IR, crossPoint, REFERENCES, slopeFor } from './FlowFixCompare'

export const AETH_IR_SMOOTH = 'IR BCc smoothed (aethalometer)'
const REF_LABEL: Record<string, string> = { 'HIPS BC': 'HIPS', 'EC (FTIR)': 'FTIR EC' }
/** |smoothed − raw| / raw in %, the notebook's smooth_raw_abs_pct */
export const smoothRawPct = (raw: number | null | undefined, smooth: number | null | undefined) =>
  finite(raw) && finite(smooth) && raw !== 0 ? (100 * Math.abs(smooth - raw)) / Math.abs(raw) : null
const rowPct = (r: FilterRow) => smoothRawPct(r[AETH_IR] as number, r[AETH_IR_SMOOTH] as number)
const COLORS = { get raw() { return INK.identity }, get smooth() { return INK.text } }

/** How far the pipeline's smoothing moves each filter's IR BCc: one histogram per site with the thresholds. */
function PctHistogram({ values, color, width, thresholds, title }: { values: number[]; color: string; width: number; thresholds: number[]; title: string }) {
  const top = Math.min(20, Math.max(thresholds[thresholds.length - 1] + 1, d3.max(values) ?? 1))
  const bins = d3.bin().domain([0, top]).thresholds(d3.range(0, top, 0.5))(values.map((v) => Math.min(v, top - 1e-9)))
  const edges = [...bins.map((b) => b.x0!), bins[bins.length - 1].x1!]
  const series = histogramSeries({ id: 'filters', label: 'filters', color, edges, counts: bins.map((b) => b.length), fill: 0.75, width: 1,
    tip: (i, n) => [`${edges[i]}–${edges[i + 1]} %${edges[i + 1] === top ? ' and above' : ''}`, `${n} filters`] })
  return <div>
    <SweepPlot series={[series]} xScale="linear" xDomain={[0, top]} width={width} height={180} title={title} xLabel="|smoothed − raw| / raw (%)" yLabel="Filters"
      include={[0]} yFormat={(v) => (Number.isInteger(v) ? fmt(v, 0) : '')} xMarkers={thresholds.map((t) => ({ at: t, label: `${t}`, color: INK.text }))} />
    <p className="chart-note" style={{ marginTop: 0 }}>kept: {thresholds.map((t) => `≤${t} % ${Math.round((100 * values.filter((v) => v <= t).length) / Math.max(1, values.length))} %`).join(' · ')}</p>
  </div>
}

/**
 * Raw against the smoothed BCc of the daily pickles (the cleaning pipeline's DEMA, src/data/qc/pkl_cleaning.py, which the notebooks call instrument-smoothed), and what that does to the comparison with HIPS
 * (HIPS_Aeth_SmoothRaw_Analysis, smoothing_comparison at daily resolution):
 *  - how far smoothing moves each filter's IR BCc, per site, with config.SMOOTH_RAW_THRESHOLDS;
 *  - the HIPS (or FTIR EC) fit as filters above each threshold are set aside;
 *  - one site's crossplot at a chosen threshold, the set-aside filters drawn as grey crosses;
 *  - raw and smoothed BCc day by day, and which channels were smoothed at all.
 */
export function SmoothRaw({ rows, meta, index }: { rows: FilterRow[]; meta: MetaFile; index: QualityIndex }) {
  const method = useEstimator()
  const thresholds = index.smooth_raw_thresholds
  const [ref, setRef] = useSearchParam<string>('aq.srref', 'HIPS BC', [...REFERENCES])
  const levels = ['all', ...[...thresholds].sort((a, b) => b - a).map(String)]
  const [level, setLevel] = useSearchParam<string>('aq.srlevel', String(thresholds[0]), levels)
  const withPct = useMemo(() => rows.filter((r) => finite(rowPct(r))), [rows])
  const sites = useMemo(() => [...new Set(withPct.map((r) => r.code))].sort((a, b) => Object.keys(index.sites).indexOf(a) - Object.keys(index.sites).indexOf(b)), [withPct, index])
  const [picked, setSite] = useSearchParam<string>('aq.srsite', 'USPA', [...INSTRUMENTS])
  const site = sites.includes(picked) ? picked : sites[0] ?? picked

  const sweep = useMemo(() => {
    const r2: SweepSeries[] = [], slope: SweepSeries[] = []
    for (const code of sites) {
      const pairs = withPct.filter((r) => r.code === code && finite(r[ref] as number))
      const color = instrumentColor(code, meta), label = instrumentName(code, meta)
      const pts = levels.map((lv) => {
        const kept = lv === 'all' ? pairs : pairs.filter((r) => rowPct(r)! <= +lv)
        const s = regression(kept.map((r) => r[AETH_IR] as number), kept.map((r) => r[ref] as number), { errorsInVariables: true, lambda: 1 })
        const x = lv === 'all' ? 'All' : `≤${lv} %`
        const tip = [`${label} · ${x}`, `${kept.length} of ${pairs.length} filters kept`, `r² ${coef(s?.r2)} · Deming slope ${coef(slopeFor(s, method))}`]
        return { x, r2: s?.r2 ?? null, slope: slopeFor(s, method), tip }
      })
      r2.push({ id: code, label, color, points: pts.map((p) => ({ x: p.x, y: p.r2, tip: p.tip })) })
      slope.push({ id: code, label, color, points: pts.map((p) => ({ x: p.x, y: p.slope, tip: p.tip })) })
    }
    return { r2, slope }
  }, [sites, withPct, ref, levels, meta, method])

  const pairs = withPct.filter((r) => r.code === site && finite(r[ref] as number))
  const kept = pairs.filter((r) => level === 'all' || rowPct(r)! <= +level)
  const removed = pairs.filter((r) => !kept.includes(r))
  const points = kept.map((r) => crossPoint(r, AETH_IR, ref, instrumentColor(site, meta), 'Aethalometer smoothing threshold', meta))
  const xLabel = withUnit(AETH_IR, meta.field_units), yLabel = withUnit(ref, meta.field_units)

  return (<>
    <ChartFrame id="aq-smooth-hist" title="How far smoothing moves each filter — |smoothed − raw| / raw of IR BCc, per site"
      subtitle={`For every filter with both values: the smoothed IR BCc (the cleaning pipeline's DEMA, src/data/qc/pkl_cleaning.py, which the notebooks call instrument-smoothed) against the raw IR BCc over the filter's sampling window, as an absolute percentage. The dashed lines are config.SMOOTH_RAW_THRESHOLDS (${thresholds.join(', ')} %), with the share of filters at or under each listed below the panel. The axis stops at 20 %; larger differences sit in the last bar.`}
      provenance="HIPS_Aeth_SmoothRaw_Analysis §4 · plotting/distributions.smooth_raw_histogram · config.SMOOTH_RAW_THRESHOLDS">
      {sites.length ? <FacetGrid items={sites} minCell={280} maxCols={4}
        render={(code, cell) => {
          const v = withPct.filter((r) => r.code === code).map((r) => rowPct(r)!)
          return <PctHistogram values={v} color={instrumentColor(code, meta)} width={cell} thresholds={thresholds}
            title={`${instrumentName(code, meta)} · ${v.length} filters, median ${fmt(quantile(v, 0.5), 2)} %`} />
        }} /> : <Empty>No filter in the current subset has both raw and smoothed IR BCc.</Empty>}
    </ChartFrame>

    <ChartFrame id="aq-smooth-sweep" title={`The fit as filters are set aside by their smoothing — r² and slope against ${REF_LABEL[ref]}`}
      subtitle={`The fit of ${REF_LABEL[ref]} on raw IR BCc per site, first on every paired filter, then keeping only filters whose smoothing moved them by at most each threshold. A rise in r² with a stricter threshold is descriptive: it can come from dropping noisy days as much as from smoothing errors, and the thresholds authorise no exclusions.`}
      provenance="HIPS_Aeth_SmoothRaw_Analysis §6–9 (threshold analysis, R² and slope against threshold, the FTIR EC comparison) · plotting/comparisons.threshold_analysis"
      controls={<Segmented label="Against" value={ref} options={[...REFERENCES]} onChange={setRef} optionLabel={(c) => REF_LABEL[c]} />}>
      {sweep.r2.length ? <>
        <FacetGrid items={['r2', 'slope'] as const} minCell={320} maxCols={2}
          render={(k, cell) => <SweepPlot series={k === 'r2' ? sweep.r2 : sweep.slope} xScale="point" xLabel="Filters kept" yLabel={k === 'r2' ? 'r²' : 'Deming slope'}
            yDomain={k === 'r2' ? [0, 1] : undefined} reference={k === 'slope' ? 1 : undefined} include={k === 'slope' ? [0] : []} width={cell} height={230} onPoint={(s) => setSite(s.id)} />} />
        <Key items={sweep.r2.map((s) => ({ label: s.label, color: s.color, kind: 'line' as const }))} />
      </> : <Empty>No paired filters.</Empty>}
    </ChartFrame>

    <ChartFrame id="aq-smooth-cross" title={`${instrumentName(site, meta)} · the comparison at one threshold — set-aside filters as grey crosses`}
      subtitle={`Raw IR BCc against ${REF_LABEL[ref]}, keeping filters whose smoothing moved them by at most the chosen threshold; the fit uses the kept filters only. The crosses are the filters set aside, drawn so the reader can see what the threshold removes (plotting_gaps_scenarios scenario 7).`}
      provenance="HIPS_Aeth_SmoothRaw_Analysis §5–6 (baseline and threshold crossplots) · plotting_gaps_scenarios scenario 7 (excluded-sample overlay)"
      controls={<>
        <Segmented label="Site" value={site} options={sites} onChange={setSite} optionLabel={(c) => instrumentName(c, meta)} />
        <Segmented label="Keep" value={level} options={levels} onChange={setLevel} optionLabel={(l) => (l === 'all' ? 'All' : `≤${l} %`)} />
      </>}>
      {pairs.length ? <Crossplot points={points} title={`${kept.length} kept, ${removed.length} set aside`} xLabel={xLabel} yLabel={yLabel}
        domain={undefined} background={(x, y) => removed.map((r) => {
          const cx = x(r[AETH_IR] as number), cy = y(r[ref] as number)
          return <path key={r.id} d={`M${cx - 4} ${cy - 4}L${cx + 4} ${cy + 4}M${cx - 4} ${cy + 4}L${cx + 4} ${cy - 4}`} stroke={INK.muted} strokeWidth={1.6} strokeOpacity={0.7} />
        })} /> : <Empty>No paired filters at {instrumentName(site, meta)}.</Empty>}
    </ChartFrame>
  </>)
}

/** raw and pipeline-smoothed BCc for one instrument, day by day, and their difference */
export function SmoothTimeline({ code, all, meta, index }: { code: string; all: Record<string, Day[]>; meta: MetaFile; index: QualityIndex }) {
  const days = all[code] ?? []
  const has = (ch: string) => index.sites[code]?.series.includes(`${ch}.smoothed`)
  const channels = CHANNELS.filter(has)
  const panels: PanelSpec[] = channels.filter((c) => c === 'IR' || c === 'Blue').map((c) => ({
    key: c, title: `${c} BCc (µg/m³)`,
    lines: [qLine(`${c}.smoothed`, 'Smoothed BCc (DEMA)', COLORS.raw), { key: `${c}raw`, label: 'Raw BCc', color: COLORS.smooth, value: (d: Day) => d.bc[c], digits: 3, dash: '3 2' }],
  }))
  panels.push({ key: 'pct', title: 'IR smoothed − raw (%)', refs: [{ at: 0, label: '0' }],
    lines: [{ key: 'pct', label: 'IR (smoothed − raw) / raw', color: INK.deming, value: (d) => (finite(d.bc.IR) && finite(d.q('IR.smoothed')) && d.bc.IR !== 0 ? (100 * (d.q('IR.smoothed')! - d.bc.IR!)) / d.bc.IR! : null), digits: 2 }] })
  // the smoothing per channel and site: the share of IR days that also have a smoothed value
  const table = Object.entries(all).map(([c, ds]) => {
    const ir = ds.filter((d) => finite(d.bc.IR))
    const listed = index.sites[c]?.series ?? []
    return { c, n: ir.length, cells: CHANNELS.map((ch) => (listed.includes(`${ch}.smoothed`) ? ir.filter((d) => finite(d.q(`${ch}.smoothed`))).length / Math.max(1, ir.length) : null)) }
  })
  return (
    <ChartFrame id="aq-smooth-timeline" title={`${instrumentName(code, meta)} · raw and smoothed BCc, day by day`}
      subtitle={`Daily means of the raw BCc and of the pickles' smoothed BCc (${channels.length ? channels.join(', ') : 'none'} at this instrument), and the IR difference in percent. The smoothed column is the cleaning pipeline's DEMA, src/data/qc/pkl_cleaning.py, which the notebooks call instrument-smoothed (a 10-minute DEMA per tape spot, so its daily mean sits on the raw one). The notebook's own rolling medians, diurnal profiles and IQR widths are in the minute-level charts below, for Addis and Pasadena.`}
      provenance="smoothing_comparison (raw vs smoothed; availability table) · HIPS_Aeth_SmoothRaw_Analysis · processed_sites/*.pkl">
      {panels.length > 1 ? <DailyPanels days={days} panels={panels} periods={index.flow_periods[code]} panelHeight={100} /> : <Empty>This instrument has no smoothed BCc.</Empty>}
      <div style={{ overflowX: 'auto' }}><table className="census-table">
        <caption className="chart-note" style={{ captionSide: 'top', textAlign: 'left' }}>Smoothing by channel: the share of each instrument's days with IR BCc that also have a smoothed value</caption>
        <thead><tr><th>Instrument</th><th>Days with IR BCc</th>{CHANNELS.map((ch) => <th key={ch}>{ch}</th>)}</tr></thead>
        <tbody>{table.map((t) => <tr key={t.c} className={t.c === code ? 'selected' : undefined}><td>{instrumentName(t.c, meta)}</td><td>{t.n}</td>
          {t.cells.map((v, i) => <td key={i}>{v === null ? '—' : `${Math.round(100 * v)} %`}</td>)}</tr>)}</tbody>
      </table></div>
    </ChartFrame>
  )
}
