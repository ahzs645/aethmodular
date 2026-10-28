import { useMemo, useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { ColorLegend } from '@/components/ColorLegend'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key, type KeyItem } from '@/components/Key'
import { IdentityLine } from '@/components/RefLines'
import { histogramSeries, SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt, pct } from '@/lib/format'
import { FONT, INK, themed } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { finite, localDate, quantile } from './data'
import { CHANNEL_COLOR } from './DualSpot'
import type { Hist, Hist2d, MinuteSite } from './minute'

const SOURCE = 'gallery/data/export_aeth_quality_minute.py · Drive "Aethalometry Data/Raw" MA350 1-min CSVs (Time (UTC) → local)'
/** the raw series pale, the pipeline's DEMA in ink (both theme colours), the rolling medians in category colours */
const WINDOW_COLOR: Record<string, string> = { get raw() { return INK.identity }, get dema() { return INK.text }, rm5: '#1f77b4', rm10: '#ff7f0e', rm15: '#2ca02c', rm30: '#d62728' }
const VARIANT_LABEL = (k: string) => (k === 'raw' ? 'Raw BCc' : k === 'dema' ? 'DEMA (pipeline smoothing)' : `${k.slice(2)}-min rolling median`)

// ------------------------------------------------------------------------------------------ primitives

/** Pre-binned histograms on one axis: the first series as filled bars, the others as outlines (a fraction of each series when `share`). */
export function BinHist({ series, width, height = 200, xLabel, share = false, markers = [], bands = [], title }: {
  series: { label: string; color: string; hist: Hist }[]
  width: number; height?: number; xLabel: string; share?: boolean
  markers?: { at: number; label?: string; color?: string }[]
  bands?: { lo: number; hi: number; color: string }[]
  title?: string
}) {
  if (!series[0]?.hist) return null
  const lo = d3.min(series, (s) => s.hist.lo)!, hi = d3.max(series, (s) => s.hist.hi)!
  const sweep = series.map(({ label, color, hist: h }, k) => {
    const edges = d3.range(h.counts.length + 1).map((i) => h.lo + ((h.hi - h.lo) * i) / h.counts.length)
    const last = h.counts.length - 1
    return histogramSeries({ id: label, label, color, edges, counts: h.counts, total: share ? d3.sum(h.counts) + h.below + h.above : undefined,
      fill: k === 0 ? 0.55 : 0, width: k === 0 ? 1 : 1.5,
      tip: (i, v) => [label, `${fmt(edges[i], 2)} to ${fmt(edges[i + 1], 2)}${i === 0 && h.below ? ' (and below)' : i === last && h.above ? ' (and above)' : ''}`, share ? pct(v) : `${v}`] })
  })
  return <SweepPlot series={sweep} xScale="linear" xDomain={[lo, hi]} width={width} height={height} title={title} xLabel={xLabel} yLabel={share ? 'Share' : 'Count'}
    include={[0]} yFormat={(v) => (share ? `${Math.round(100 * v)}%` : Number.isInteger(v) ? d3.format('~s')(v) : '')}
    xMarkers={markers.map((mk) => ({ ...mk, color: mk.color ?? INK.text }))}
    background={({ x, ih }) => bands.map((b, i) => { const sx = x as (v: number) => number
      return <rect key={i} x={sx(b.lo)} width={Math.max(0, sx(b.hi) - sx(b.lo))} height={ih} fill={b.color} fillOpacity={0.14} /> })} />
}

/** A 2-D histogram (counts on a log colour scale) with reference lines through the origin. */
export function Heat2d({ h, width, xLabel, yLabel, lines = [], identity = false, title }: {
  h: Hist2d; width: number; xLabel: string; yLabel: string; identity?: boolean; title?: string
  lines?: { slope: number; color: string; dash?: string }[]
}) {
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  const colours = useColourScheme('aq-heat2d', 'sequential')
  const m = { top: title ? 22 : 8, right: 12, bottom: 42, left: 52 }
  const side = Math.max(160, Math.min(320, width - m.left - m.right))
  const nx = h.counts[0]?.length ?? 1, ny = h.counts.length
  const x = d3.scaleLinear().domain(h.x).range([0, side]), y = d3.scaleLinear().domain(h.y).range([side, 0])
  const max = d3.max(h.counts.flat()) ?? 1
  const color = d3.scaleSequentialLog(d3.interpolateRgbBasis(colours.stops.slice(1))).domain([1, Math.max(2, max)])
  const cw = side / nx, ch = side / ny
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={side + m.left + m.right} height={side + m.top + m.bottom} fontFamily={FONT.family}>
        {title && <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>{title}</text>}
        <g transform={`translate(${m.left},${m.top})`}>
          <defs><clipPath id={`aq-h2-${xLabel.length}-${yLabel.length}`}><rect width={side} height={side} /></clipPath></defs>
          <rect width={side} height={side} fill={INK.empty} />
          {h.counts.map((row, j) => row.map((c, i) => c > 0 && <rect key={`${i}-${j}`} x={i * cw} y={side - (j + 1) * ch} width={cw + 0.3} height={ch + 0.3} fill={color(c)}
            onMouseMove={(e) => tip.show(e, [`${xLabel}: ${fmt(h.x[0] + (i * (h.x[1] - h.x[0])) / nx, 2)}–${fmt(h.x[0] + ((i + 1) * (h.x[1] - h.x[0])) / nx, 2)}`,
              `${yLabel}: ${fmt(h.y[0] + (j * (h.y[1] - h.y[0])) / ny, 2)}–${fmt(h.y[0] + ((j + 1) * (h.y[1] - h.y[0])) / ny, 2)}`, `${c} minutes`])} onMouseLeave={tip.hide} />))}
          <g clipPath={`url(#aq-h2-${xLabel.length}-${yLabel.length})`} pointerEvents="none">
            {identity && <IdentityLine x1={x(h.x[0])} y1={y(h.x[0])} x2={x(h.x[1])} y2={y(h.x[1])} />}
            {lines.map((l, k) => <line key={k} x1={x(h.x[0])} y1={y(l.slope * h.x[0])} x2={x(h.x[1])} y2={y(l.slope * h.x[1])} stroke={l.color} strokeWidth={1.5} strokeDasharray={l.dash} />)}
          </g>
          <YAxis scale={y} x={0} tickCount={5} label={yLabel} labelX={-40} />
          <XAxis scale={x} y={side} tickCount={5} label={xLabel} labelDy={34} />
        </g>
      </svg>
      <div className="legend"><ColorLegend scale={color} label="Minutes" width={120} scheme={colours.scheme} format={(v) => d3.format('~s')(v)} ticks={3} /></div>
      {tip.node}
    </div>
  )
}

function Table({ head, rows, caption }: { head: string[]; rows: ReactNode[][]; caption?: string }) {
  return <div style={{ overflowX: 'auto' }}><table className="census-table">
    {caption && <caption className="chart-note" style={{ captionSide: 'top', textAlign: 'left' }}>{caption}</caption>}
    <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
    <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
  </table></div>
}

const NoMinute = ({ name }: { name: string }) => <Empty>No minute record for {name} in the repo's Drive mirror: only Addis (MA350-0238, filed as Jacros) and Pasadena (JPL, MA350-0229) have one.</Empty>

// ------------------------------------------------------------------------------------------ flow QC

/** a trailing 7-day mean of daily fractions, as % (the 7-day average of flow_qc_analysis), on days with at least 3 values in the week */
function rolling7(days: Date[], v: (number | null)[]) {
  return days.map((t, i) => {
    const xs: number[] = []
    for (let j = i; j >= 0 && +t - +days[j] < 7 * 864e5; j--) if (finite(v[j])) xs.push(v[j]!)
    return { x: t, y: xs.length >= 3 ? (100 * xs.reduce((a, b) => a + b, 0)) / xs.length : null }
  })
}

/** notebooks/qc/flow_qc_analysis on the full minute record */
export function MinuteFlowQC({ site, name }: { site: MinuteSite | null; name: string }) {
  const f = site?.flow_qc
  const days = useMemo(() => f?.daily.day.map(localDate) ?? [], [f])
  const id = 'aq-min-flow'
  const head = (
    <ChartFrame id={id} title={`${name} · minute flow QC — pass rates, the flow ratio and the two spot flows`}
      subtitle={`flow_qc_analysis on every minute with all four flows: R = Flow1/Flow2, f1 = Flow1/total, f2 = Flow2/total, S = (Flow1 + Flow2)/total, T = total/setpoint. R passes within 5 % of the instrument's own median R (warns within 10 %), S and T within 0.02 of 1; a minute passes QC when all three do. Below: the daily median R with its 10–90 % band and the daily pass rates; the distribution of R; and Flow1 against Flow2 as a 2-D histogram with the median ratio (solid) and ±5/10 % (dashed). The notebook's nominal 3.0 is not drawn: neither instrument runs near it.`}
      provenance={`notebooks/qc/flow_qc_analysis (summary statistics, R over time with QC bands, pass rates, distribution, Flow1 vs Flow2 scatter and hexbin) · ${SOURCE}`}>
      {!f ? <NoMinute name={name} /> : <>
        <Table head={['Metric', 'Mean', 'Median', 'SD', '5 %', '95 %', 'IQR']}
          rows={(['R', 'f1', 'f2', 'S', 'T'] as const).map((k) => [k, fmt(f.stats[k].mean, 3), fmt(f.stats[k].median, 3), fmt(f.stats[k].std, 4), fmt(f.stats[k].p5, 3), fmt(f.stats[k].p95, 3), fmt(f.stats[k].iqr, 4)])}
          caption={`${f.n.toLocaleString()} minutes · R pass ${pct(f.rates.R_pass)}, warn ${pct(f.rates.R_warn)}, fail ${pct(f.rates.R_fail)} · S pass ${pct(f.rates.S_pass)} · T pass ${pct(f.rates.T_pass)} · all three ${pct(f.rates.QC_pass)} · r(Flow1, Flow2) ${fmt(f.corr_F1_F2, 3)}`} />
        <SweepPlot height={200} xScale="time" yLabel="Flow1 / Flow2 (daily)" yMarkers={[{ at: f.R_target, label: `median ${fmt(f.R_target, 3)}` }, { at: f.R_target * 1.05, label: '+5 %', color: INK.good }, { at: f.R_target * 0.95, label: '−5 %', color: INK.good }, { at: f.R_target * 1.1, label: '+10 %', color: INK.warn }, { at: f.R_target * 0.9, label: '−10 %', color: INK.warn }]}
          series={[{ id: 'r', label: 'daily median R', color: INK.text, width: 1.3, marks: false, points: days.map((t, i) => ({ x: t, y: f.daily.R_med[i], lo: f.daily.R_p10[i], hi: f.daily.R_p90[i] })) }]} />
        <SweepPlot height={180} xScale="time" yLabel="Minutes passing (%)" yDomain={[0, 100]}
          series={[
            { id: 'rpd', label: 'R pass, daily', color: '#41ab5d', width: 0, marks: true, r: 1.5, points: days.map((t, i) => ({ x: t, y: finite(f.daily.R_pass[i]) ? 100 * f.daily.R_pass[i]! : null })) },
            { id: 'rp', label: 'R pass, 7-day mean', color: '#41ab5d', width: 1.6, marks: false, points: rolling7(days, f.daily.R_pass) },
            { id: 'qc', label: 'QC pass (R, S and T), 7-day mean', color: '#2c7fb8', width: 1.4, marks: false, dash: '4 2', points: rolling7(days, f.daily.QC_pass) }]} />
        <Key items={[{ label: 'daily median R, 10–90 % band', color: INK.text, kind: 'line' }, { label: 'R pass, daily', color: '#41ab5d', kind: 'dot' }, { label: 'R pass, 7-day mean', color: '#41ab5d', kind: 'line' }, { label: 'QC pass (R, S and T), 7-day mean', color: '#2c7fb8', kind: 'dash' }]} />
        <FacetGrid items={['hist', 'flows'] as const} minCell={330} maxCols={2} render={(k, cell) => k === 'hist'
          ? <BinHist title="Distribution of R (every minute)" width={cell} height={260} xLabel="R = Flow1 / Flow2" series={[{ label: 'minutes', color: '#2c7fb8', hist: f.R_hist }]}
            bands={[{ lo: f.R_target * 0.95, hi: f.R_target * 1.05, color: INK.good }]} markers={[{ at: f.R_target, label: `median ${fmt(f.R_target, 3)}`, color: INK.negative }, { at: f.stats.R.p5, label: '5 %' , color: INK.muted }, { at: f.stats.R.p95, label: '95 %', color: INK.muted }]} />
          : <Heat2d title="Flow1 against Flow2" h={f.flows_2d} width={cell} xLabel="Flow2 (mL/min)" yLabel="Flow1 (mL/min)"
            lines={[{ slope: f.R_target, color: INK.accent }, { slope: f.R_target * 0.95, color: INK.good, dash: '5 3' }, { slope: f.R_target * 1.05, color: INK.good, dash: '5 3' }, { slope: f.R_target * 0.9, color: INK.warn, dash: '2 3' }, { slope: f.R_target * 1.1, color: INK.warn, dash: '2 3' }]} />} />
      </>}
    </ChartFrame>
  )
  return head
}

// ------------------------------------------------------------------------------------------ smoothing

/** research/ftir_hips_chem/smoothing_comparison on the minute record */
export function MinuteSmoothing({ site, name }: { site: MinuteSite | null; name: string }) {
  const s = site?.smoothing
  const variants = s ? ['raw', 'dema', ...s.windows.map((w) => `rm${w}`)] : []
  const best = useMemo(() => {
    const out = new Map<string, number>()
    for (const r of s?.stats ?? []) if (!out.has(r.channel) || r.rmse < s!.stats.find((q) => q.channel === r.channel && q.window === out.get(r.channel))!.rmse) out.set(r.channel, r.window)
    return out
  }, [s])
  const keyItems: KeyItem[] = variants.map((v) => ({ label: VARIANT_LABEL(v), color: WINDOW_COLOR[v], kind: 'line' }))
  return (<>
    <ChartFrame id="aq-min-smooth" title={`${name} · rolling medians against the pipeline's smoothing — which window matches, and does it change the day?`}
      subtitle={`Raw BCc (negatives dropped), its centred 5, 10, 15 and 30-minute rolling medians, and the smoothed BCc of the daily pickles. That "smoothed" column is not the instrument's: it is the cleaning pipeline's DEMA (src/data/qc/pkl_cleaning.py dema_bc_and_atn: 2·EMA − EMA(EMA), span ${s?.dema_span ?? 10} minutes, restarted on each tape spot), recomputed here on the same minutes. The table scores each window against the DEMA (the best per channel, by RMSE, in bold); the example day is ${s?.example?.day ?? '—'} (the middle of the days with at least 1,300 IR minutes, as the notebook chose).`}
      provenance={`research/ftir_hips_chem/smoothing_comparison §2–4 (time series, r and RMSE against the smoothed series, best-window scatter) · ${SOURCE}`}>
      {!s ? <NoMinute name={name} /> : <>
        {s.example && <SweepPlot height={240} xScale="linear" xLabel={`Local hour, ${s.example.day}`} yLabel="IR BCc (µg/m³)" xDomain={[0, 24]} yClip={[0, 0.995]} include={[0]}
          series={variants.map((v) => ({ id: v, label: VARIANT_LABEL(v), color: WINDOW_COLOR[v], width: v === 'raw' ? 0.8 : v === 'dema' ? 2 : 1.3, marks: false,
            points: s.example!.minute.map((mm, i) => ({ x: mm / 60, y: s.example![v]?.[i] ?? null })) }))} />}
        <Key items={keyItems} />
        <Table head={['Channel', 'Window', 'Minutes', 'r', 'RMSE (µg/m³)', 'MAE (µg/m³)', 'Bias (µg/m³)']}
          rows={s.stats.map((r) => {
            const b = best.get(r.channel) === r.window
            const w = (v: ReactNode) => (b ? <b>{v}</b> : v)
            return [r.channel, w(`${r.window} min`), r.n.toLocaleString(), w(fmt(r.r, 4)), w(fmt(r.rmse, 3)), fmt(r.mae, 3), fmt(r.bias, 3)]
          })} caption="Each rolling median against the DEMA-smoothed series, minute by minute" />
        {s.scatter_IR && <FacetGrid items={[s.scatter_IR]} minCell={320} maxCols={1} render={(h, cell) => <Heat2d h={h} width={cell} identity title={`IR: the ${h.window}-min rolling median against the DEMA`} xLabel="DEMA-smoothed IR BCc (µg/m³)" yLabel={`${h.window}-min rolling median (µg/m³)`} />} />}
      </>}
    </ChartFrame>
    <ChartFrame id="aq-min-smooth-diurnal" title={`${name} · the day by smoothing method — hourly medians and IQR width`}
      subtitle="Per local hour: the median of each version of IR and Blue BCc over every minute in that hour of the day, and the width of its interquartile range. If the medians lie on one another the choice of smoothing does not move the diurnal cycle; a narrower IQR is the noise it removes."
      provenance={`research/ftir_hips_chem/smoothing_comparison §5–6 (diurnal patterns, IQR widths) · ${SOURCE}`}>
      {!s ? <NoMinute name={name} /> : <>
        <FacetGrid items={Object.keys(s.diurnal).flatMap((ch) => [{ ch, k: 'q50' as const }, { ch, k: 'iqr' as const }])} minCell={300} maxCols={4}
          render={({ ch, k }, cell) => <SweepPlot width={cell} height={220} xScale="linear" xDomain={[0, 23]} xLabel="Local hour" title={`${ch} · ${k === 'q50' ? 'hourly median' : 'IQR width'}`}
            yLabel={k === 'q50' ? `${ch} BCc (µg/m³)` : 'IQR (µg/m³)'} include={[0]}
            series={variants.filter((v) => s.diurnal[ch][v]).map((v): SweepSeries => {
              const d = s.diurnal[ch][v]
              return { id: v, label: VARIANT_LABEL(v), color: WINDOW_COLOR[v], width: v === 'dema' ? 2 : 1.3, marks: false, dash: v.startsWith('rm') ? '4 2' : undefined,
                points: d3.range(24).map((hr) => ({ x: hr, y: k === 'q50' ? d.q50[hr] : finite(d.q75[hr]) && finite(d.q25[hr]) ? d.q75[hr]! - d.q25[hr]! : null })) }
            })} />} />
        <Key items={keyItems} />
      </>}
    </ChartFrame>
  </>)
}

// ------------------------------------------------------------------------------------------ dual spot (ftir_49 A)

/** ftir_49 A2: the BCc step across each tape advance against random times */
export function MinuteTape({ site, name }: { site: MinuteSite | null; name: string }) {
  const t = site?.tape
  const pts = t?.ir_points
  return (
    <ChartFrame id="aq-min-tape" title={`${name} · tape-advance steps — does BCc jump when the tape moves?`}
      subtitle={`For each of the ${t?.advances ?? '—'} advances (a change of tape position or session; median spot life ${t ? Math.round(t.segments_median_min / 60) : '—'} h): mean BCc over the 10 minutes ending 5 minutes before the advance against the 10 minutes starting 5 minutes after (8 valid minutes each), and the same at random times at least 60 minutes from any advance as the control. A real advance artefact must stand out from the control's 20-minute variability. The histograms are (after − before)/before per channel, the grey fill the control at IR; the scatter is the IR step against the level before it.`}
      provenance={`research/ftir_ec_phase3/ftir_49_ma350_raw_chain_diagnostics A2 · RNG seed 20260901 · ${SOURCE}`}>
      {!t ? <NoMinute name={name} /> : <>
        <Table head={['Channel', 'Advances', 'Median step (µg/m³)', 'Step MAD', 'Control MAD', 'Median step (%)', 'IQR (%)', '|step| > 25 %', 'control > 25 %', 'Positive', 'control positive']}
          rows={t.table.map((r) => [r.channel, r.n, fmt(r.step_median, 3), fmt(r.step_mad, 3), fmt(r.control_mad, 3), fmt(r.rel_median, 1), `${fmt(r.rel_q25, 1)} to ${fmt(r.rel_q75, 1)}`,
            `${fmt(r.rel_over25, 1)} %`, `${fmt(r.control_rel_over25, 1)} %`, pct(r.positive), pct(r.control_positive)])} />
        <FacetGrid items={['hist', 'scatter'] as const} minCell={340} maxCols={2} render={(k, cell) => k === 'hist'
          ? <BinHist title="Step across the advance, per channel" width={cell} height={250} share xLabel="(after − before) / before (%)" markers={[{ at: 0 }]}
            series={[{ label: 'control (IR, random times)', color: INK.neutral, hist: t.hist.control }, ...t.table.map((r) => ({ label: r.channel, color: CHANNEL_COLOR[r.channel as keyof typeof CHANNEL_COLOR], hist: t.hist[r.channel] }))]} />
          : pts ? <SweepPlot width={cell} height={250} xScale="linear" title="IR step against the level before it" xLabel="IR BCc before the advance (µg/m³)" yLabel="IR step (µg/m³)" reference={0} yClip={[0.01, 0.99]}
            series={[{ id: 'p', label: 'advance', color: CHANNEL_COLOR.IR, width: 0, marks: true, r: 2.4, points: pts.pre.map((p, i) => ({ x: p ?? NaN, y: pts.step[i] })).filter((p) => finite(p.x)) }]} /> : null} />
        <Key items={[{ label: 'control (IR, random times)', color: INK.neutral, kind: 'bar' }, ...t.table.map((r) => ({ label: r.channel, color: CHANNEL_COLOR[r.channel as keyof typeof CHANNEL_COLOR], kind: 'line' as const }))]} />
      </>}
    </ChartFrame>
  )
}

/** ftir_49 A1 at minute resolution: K against the closure K* by loading and spot age */
export function MinuteKstar({ site, name }: { site: MinuteSite | null; name: string }) {
  const rows = site?.kstar ?? []
  const series = (by: string, what: 'K' | 'Kstar'): SweepSeries[] => ['UV', 'Blue', 'Green', 'Red', 'IR'].map((c) => ({
    id: `${c}${what}`, label: `${c} ${what === 'K' ? 'K' : 'K*'}`, color: CHANNEL_COLOR[c as keyof typeof CHANNEL_COLOR], dash: what === 'K' ? '4 3' : undefined, width: what === 'K' ? 1.2 : 1.8, marks: true, r: what === 'K' ? 2 : 3,
    interval: what === 'Kstar' ? 'whisker' : 'none',
    points: rows.filter((r) => r.channel === c && r.by === by).map((r) => ({ x: r.bin, y: r[what], lo: what === 'Kstar' ? r.Kstar_q25 : null, hi: what === 'Kstar' ? r.Kstar_q75 : null,
      tip: [`${c} · ${by} ${r.bin} · ${r.n.toLocaleString()} minutes`, `K ${fmt(r.K, 5)} · K* ${fmt(r.Kstar, 5)} (IQR ${fmt(r.Kstar_q25, 5)}–${fmt(r.Kstar_q75, 5)})`, `spot 2 under K ÷ BCc ${fmt(r.spot2_over_bcc, 4)} · BCc(K*)/BCc ${fmt(r.bcc_star_over_bcc, 4)}`] })),
  }))
  const s2 = rows.filter((r) => r.by === 'ATN1').map((r) => r.spot2_over_bcc).filter(finite)
  return (
    <ChartFrame id="aq-min-kstar" title={`${name} · the instrument K against the closure K*, minute by minute`}
      subtitle={`On minutes with ATN1 ≥ 5 and both spot BCs positive: the instrument's K (dashed) and K* = (BC1 − BC2)/(BC1·ATN2 − BC2·ATN1), the K that makes the two spots agree that minute (solid, IQR whiskers), by spot-1 attenuation and by the spot's age. Spot 2 compensated with the instrument's K over the reported BCc has a median of ${fmt(quantile(s2, 0), 4)}–${fmt(quantile(s2, 1), 4)} across the bins: at minute resolution the reported K closes the two spots exactly, so the daily-mean K* on the chart above (and ftir_49's daily fallback) measures the averaging, not a mis-set K.`}
      provenance={`research/ftir_ec_phase3/ftir_49_ma350_raw_chain_diagnostics A1 (kstar_vs_K_by_atn_and_age) · ${SOURCE}`}>
      {!rows.length ? <NoMinute name={name} /> : <FacetGrid items={['ATN1', 'spot age']} minCell={360} maxCols={2}
        render={(by, cell) => <SweepPlot width={cell} height={260} xScale="point" title={`by ${by === 'ATN1' ? 'spot-1 attenuation' : 'spot age'}`} xLabel={by === 'ATN1' ? 'ATN1 bin' : 'Time on the spot'} yLabel="K"
          series={[...series(by, 'Kstar'), ...series(by, 'K')]} />} />}
      {rows.length > 0 && <Key items={[...(['UV', 'Blue', 'Green', 'Red', 'IR'] as const).map((c) => ({ label: c, color: CHANNEL_COLOR[c], kind: 'dot' as const })), { label: 'K* (closure), IQR whiskers', color: INK.text, kind: 'line' }, { label: 'K (instrument)', color: INK.text, kind: 'dash' }]} />}
    </ChartFrame>
  )
}

/** ftir_49 A3: the clean mask and AAE by environment */
export function MinuteClean({ site, name }: { site: MinuteSite | null; name: string }) {
  const c = site?.clean
  const keys = c ? Object.keys(c.env) : []
  const [env, setEnv] = useSearchParam<string>('aq.minenv', 'IR ATN1', [])
  const pick = keys.includes(env) ? env : keys[0]
  const blocks: IntervalBlock[] = c ? [{ key: 'p', rows: [...Object.entries(c.parts), ['all five (the clean mask)', c.fraction] as [string, number]].map(([k, v]) => ({
    key: k, label: k, strong: k.startsWith('all'), note: pct(v), marks: [[{ mid: 100 * v, color: k.startsWith('all') ? INK.accent : INK.muted }]] })) }] : []
  const sum = c?.daily_aae ?? []
  return (
    <ChartFrame id="aq-min-clean" title={`${name} · the clean mask — how many minutes pass, and does the AAE care?`}
      subtitle={`ftir_49's five conditions on minutes with all five channels: nominal flow (total within 5 % of setpoint and Flow1/Flow2 within 10 % of its median ${c ? fmt(c.nominal_ratio, 3) : ''}), |dT/dt| and |dRH/dt| (10-minute means) under their medians, UV ATN1 under 30, and more than 30 minutes from a tape advance. Left: how many minutes pass each. Right: AAE from the summed absorption of each bin of the chosen variable (BCc × the MA350 cross-section, the ratio of means). Below: daily AAE(625, 880) on all minutes against the clean ones, on days with ≥ 720 minutes and ≥ 180 clean ones.`}
      provenance={`research/ftir_ec_phase3/ftir_49_ma350_raw_chain_diagnostics A3 (optics_by_environment_bins, clean_mask_daily_aae) · ${SOURCE}`}
      controls={c ? <Segmented label="Binned by" value={pick} options={keys} onChange={setEnv} /> : undefined}>
      {!c ? <NoMinute name={name} /> : <>
        <FacetGrid items={['parts', 'env'] as const} minCell={360} maxCols={2} render={(k, cell) => k === 'parts'
          ? <div style={{ width: cell }}><IntervalStrip blocks={blocks} labelWidth={170} noteWidth={44} minPanelWidth={140} panels={[{ label: 'Minutes passing (%)', domain: [0, 100], stems: true, format: (v) => v.toFixed(0) }]} /></div>
          : <SweepPlot width={cell} height={250} xScale="point" xLabel={pick} yLabel="AAE (ratio of bin sums)" reference={1}
            series={[{ id: 'red', label: 'AAE(625, 880)', color: CHANNEL_COLOR.Red }, { id: 'blue', label: 'AAE(470, 880)', color: CHANNEL_COLOR.Blue }].map((sr) => ({ ...sr, marks: true,
              points: c.env[pick].map((b) => ({ x: b.bin, y: sr.id === 'red' ? b.aae_red : b.aae_blue, tip: [`${pick} ${b.bin}`, `${b.n.toLocaleString()} minutes`, `AAE(625,880) ${fmt(b.aae_red, 3)} · AAE(470,880) ${fmt(b.aae_blue, 3)}`, `Red/IR BCc ${fmt(b.red_ir, 3)} · mean IR ${fmt(b.ir_mean, 2)} µg/m³`] })) }))} />} />
        <Key items={[{ label: 'AAE(625, 880)', color: CHANNEL_COLOR.Red, kind: 'line' }, { label: 'AAE(470, 880)', color: CHANNEL_COLOR.Blue, kind: 'line' }]} />
        <FacetGrid items={['hist', 'table'] as const} minCell={360} maxCols={2} render={(k, cell) => k === 'hist'
          ? <BinHist title={`Daily AAE(625, 880), ${sum[0]?.n_days ?? 0} days`} width={cell} height={230} xLabel="AAE(625, 880)" markers={[{ at: 1, label: 'AAE 1' }]}
            series={[{ label: 'all minutes', color: INK.neutral, hist: c.hist_red_all }, { label: 'clean mask', color: CHANNEL_COLOR.Red, hist: c.hist_red_clean }]} />
          : <Table head={['Minutes', 'Days', 'AAE(625,880)', 'AAE(528,880)', 'AAE(470,880)', 'Days with AAE(625,880) > 1']}
            rows={[...sum.map((s) => [s.minutes, s.n_days, `${fmt(s.red_mean, 3)} ± ${fmt(s.red_sd, 3)}`, `${fmt(s.green_mean, 3)} ± ${fmt(s.green_sd, 3)}`, `${fmt(s.blue_mean, 3)} ± ${fmt(s.blue_sd, 3)}`, `${fmt(s.red_over1, 1)} %`]),
              ['paired difference (clean − all)', sum[0]?.n_days ?? '—', `${fmt(c.paired_red_mean, 3)} ± ${fmt(c.paired_red_sd, 3)}`, '', '', '']]}
            caption={`Clean minutes: ${pct(c.fraction)} of ${c.n.toLocaleString()}`} />} />
        <Key items={[{ label: 'all minutes', color: INK.neutral, kind: 'bar' }, { label: 'clean mask', color: CHANNEL_COLOR.Red, kind: 'line' }]} />
      </>}
    </ChartFrame>
  )
}

/** ftir_49 B5: the order of averaging for AAE */
export function MinuteAaeOrder({ site, name }: { site: MinuteSite | null; name: string }) {
  const a = site?.aae_order
  const pairs = a ? Object.keys(a) : []
  const color = (k: string) => (k.includes('470') ? CHANNEL_COLOR.Blue : CHANNEL_COLOR.Red)
  return (
    <ChartFrame id="aq-min-aae-order" title={`${name} · order of operations — the daily mean of minute AAEs is not the AAE of the daily mean`}
      subtitle="On near-complete 9am days (≥ 95 % of 1,440 minutes): each day's mean of the per-minute AAE (undefined where either absorption is ≤ 0) minus the AAE of the day's mean absorption. A spread around zero wider than ±0.1 is as large as the source signal AAE is used for."
      provenance={`research/ftir_ec_phase3/ftir_49_ma350_raw_chain_diagnostics B5 (aae_aggregation_order) · ${SOURCE}`}>
      {!a ? <NoMinute name={name} /> : <>
        <FacetGrid items={['hist', 'table'] as const} minCell={360} maxCols={2} render={(k, cell) => k === 'hist'
          ? <BinHist width={cell} height={230} share xLabel="mean of minute AAEs − AAE of the mean" markers={[{ at: 0 }, { at: -0.1, color: INK.muted }, { at: 0.1, color: INK.muted }]}
            series={pairs.map((p) => ({ label: p, color: color(p), hist: a[p].hist }))} />
          : <Table head={['Pair', 'Days', 'Undefined minutes', 'AAE of the mean', 'Median difference', 'IQR', '|diff| > 0.1']}
            rows={pairs.map((p) => [p, a[p].n_days, `${fmt(a[p].undefined_pct, 1)} %`, `${fmt(a[p].agg_mean, 3)} ± ${fmt(a[p].agg_sd, 3)}`, fmt(a[p].diff_median, 3), `${fmt(a[p].diff_q25, 3)} to ${fmt(a[p].diff_q75, 3)}`, `${fmt(a[p].over01, 1)} % of days`])} />} />
        <Key items={pairs.map((p, i) => ({ label: p, color: color(p), kind: i === 0 ? 'bar' as const : 'line' as const }))} />
      </>}
    </ChartFrame>
  )
}

// ------------------------------------------------------------------------------------------ coverage (ftir_49 B)

/** ftir_49 B4: masked daily means on near-complete days */
export function MinuteMissing({ sites, names }: { sites: Record<string, MinuteSite>; names: Record<string, string> }) {
  const codes = Object.keys(sites)
  const [code, setCode] = useSearchParam<string>('aq.minmiss', 'ETAD-JACROS', [])
  const c = codes.includes(code) ? code : codes[0]
  const m = c ? sites[c].missing : null
  // the heaviest loss in the darkest blue; on the dark page it turns to the palest, so it stays the strongest mark
  const LOSS_COLOR: Record<number, string> = { 10: '#9ecae1', 20: '#4292c6', 40: themed('#08306b', '#c6dbef') }
  const blocks: IntervalBlock[] = m ? ['random', 'contiguous', 'rush-hour', 'high-pollution', 'low-pollution'].map((kind) => ({
    key: kind, title: kind,
    rows: [10, 20, 40].map((loss) => {
      const r = (ch: string) => m.table.find((x) => x.channel === ch && x.mask === kind && x.loss === loss)!
      const mk = (ch: string) => [{ mid: r(ch).median, lo: r(ch).p2_5, hi: r(ch).p97_5, color: LOSS_COLOR[loss], faint: true, r: 0 }, { mid: r(ch).median, lo: r(ch).q25, hi: r(ch).q75, color: LOSS_COLOR[loss] }]
      return { key: `${kind}${loss}`, label: `${loss} % of minutes lost`, note: `${fmt(r('IR').over25, 1)} %`, marks: [mk('IR'), mk('Red')],
        tip: [`${kind}, ${loss} % lost`, ...['IR', 'Red'].map((ch) => `${ch}: bias ${fmt(r(ch).bias, 2)} %, median ${fmt(r(ch).median, 2)} %, 2.5–97.5 % ${fmt(r(ch).p2_5, 1)} to ${fmt(r(ch).p97_5, 1)} %, |err| > 25 % on ${fmt(r(ch).over25, 1)} % of day-draws`)] }
    }),
  })) : []
  return (
    <ChartFrame id="aq-min-missing" title="Missingness benchmark — which 20 % of a day is missing matters more than how much"
      subtitle={`${c ? `${names[c]}: ${m?.n_days} near-complete 9am days (≥ 95 % of 1,440 minutes; ${m?.first} to ${m?.last}; mean IR BCc ${fmt(m?.ir_mean, 2)} µg/m³), gaps of ≤ 5 % filled by interpolation so every mask removes an exact share. ` : ''}Five ways to lose 10, 20 or 40 % of a day's minutes: at random, as one contiguous outage, from the rush hours (07–10 and 17–20, then at random), from the most polluted minutes, or from the cleanest. Each row is the error of the recovered daily mean in percent of the true one (thick: IQR; thin: 2.5–97.5 % over days and 20 draws); the note is the share of IR day-draws off by more than 25 %.`}
      provenance={`research/ftir_ec_phase3/ftir_49_ma350_raw_chain_diagnostics B4 (missingness_benchmark) · RNG seed 20260901 · ${SOURCE}`}
      controls={codes.length > 1 ? <Segmented label="Instrument" value={c} options={codes} onChange={setCode} optionLabel={(k) => names[k]} /> : undefined}>
      {!m ? <Empty>No minute record found.</Empty> : <>
        <IntervalStrip blocks={blocks} labelWidth={150} noteWidth={56} minPanelWidth={240}
          panels={[{ label: 'IR: masked − true daily mean (%)', reference: 0, format: (v) => v.toFixed(0) }, { label: 'Red: masked − true daily mean (%)', reference: 0, format: (v) => v.toFixed(0) }]} />
        <Key items={[10, 20, 40].map((l) => ({ label: `${l} % lost`, color: LOSS_COLOR[l], kind: 'line' as const }))} />
      </>}
    </ChartFrame>
  )
}

const WINDOW_ALTS = ['calendar', 'tolerant', 'window'] as const
const ALT_LABEL: Record<string, string> = { calendar: 'calendar day', tolerant: '±1-day mean', window: "the filter's sampling window", '9am': '9am day' }
/** ftir_49 B6: the 9am day against the calendar day, the ±1-day mean and the filter's own sampling window */
export function MinuteWindows({ sites, names, meta, rows }: { sites: Record<string, MinuteSite>; names: Record<string, string>; meta: MetaFile; rows: FilterRow[] }) {
  const [alt, setAlt] = useSearchParam<string>('aq.minwin', 'tolerant', [...WINDOW_ALTS])
  const [ch, setCh] = useSearchParam<string>('aq.minwinch', 'IR', ['IR', 'Red'])
  const inView = useMemo(() => new Set(rows.map((r) => r.id)), [rows])
  const blocks: IntervalBlock[] = Object.entries(sites).map(([code, s]) => {
    const w = s.windows
    return {
      key: code, title: names[code],
      rows: WINDOW_ALTS.map((a) => {
        const d = w.id.flatMap((id, i) => {
          const base = w[`${ch}_9am`][i], v = w[`${ch}_${a}`][i]
          return inView.has(id) && finite(base) && finite(v) && base !== 0 ? [(100 * (v - base)) / base] : []
        })
        return { key: a, label: ALT_LABEL[a], note: `n=${d.length}`, marks: [[{ mid: quantile(d, 0.5), lo: quantile(d, 0.025), hi: quantile(d, 0.975), color: INK.muted, faint: true, r: 0 },
          { mid: quantile(d, 0.5), lo: quantile(d, 0.25), hi: quantile(d, 0.75), color: a === 'window' ? INK.accent : INK.text }]],
          tip: [`${names[code]} · ${ALT_LABEL[a]} against the 9am day (${ch})`, `${d.length} filters · median ${fmt(quantile(d, 0.5), 1)} %, IQR ${fmt(quantile(d, 0.25), 1)} to ${fmt(quantile(d, 0.75), 1)} %`,
            `|Δ| > 10 %: ${d.filter((x) => Math.abs(x) > 10).length} · > 25 %: ${d.filter((x) => Math.abs(x) > 25).length}`] }
      }),
    }
  })
  const points: CrossPoint[] = Object.entries(sites).flatMap(([code, s]) => s.windows.id.flatMap((id, i) => {
    const x = s.windows[`${ch}_9am`][i], y = s.windows[`${ch}_${alt}`][i]
    if (!inView.has(id) || !finite(x) || !finite(y)) return []
    const color = code === 'USPA' ? meta.sites.find((m) => m.code === 'USPA')?.color ?? INK.accent : meta.sites.find((m) => m.code === 'ETAD')?.color ?? INK.warn
    return [{ x, y, color, id, tip: [id, s.windows.date[i], `${ch} BCc, 9am day ${fmt(x, 2)} · ${ALT_LABEL[alt]} ${fmt(y, 2)} µg/m³`, `window ${fmt(s.windows.hours[i], 1)} h`],
      rec: { id, source: 'Aethalometer matching window', date: s.windows.date[i], fields: [[`${ch} BCc, 9am day (µg/m³)`, x], [`${ch} BCc, ${ALT_LABEL[alt]} (µg/m³)`, y]] } }]
  }))
  return (
    <ChartFrame id="aq-min-windows" title="Which day is the filter's day? — the 9am day against the calendar day, the ±1-day mean and the sampling window"
      subtitle={`On every filter of the current subset that falls inside a minute record (Addis filters on the Addis record, JPL filters on Pasadena): ${ch} BCc over the 9am day starting on the filter's start date, the calendar day, the mean of the 9am days either side (data_matching's legacy ±1-day match), and the filter's own sampling window (gallery weather export; at least half its minutes). Day means need 720 minutes. Where a filter's window runs 09:00 to 09:00 (most Addis filters) it is the 9am day itself. The strip is each alternative against the 9am day in percent (thick IQR, thin 2.5–97.5 %); the crossplot draws one alternative against it.`}
      provenance={`research/ftir_ec_phase3/ftir_49_ma350_raw_chain_diagnostics B6 (filter_day_window_comparison), with the sampling window added · ${SOURCE}`}
      controls={<>
        <Segmented label="Channel" value={ch} options={['IR', 'Red']} onChange={setCh} />
        <Segmented label="Against the 9am day" value={alt} options={[...WINDOW_ALTS]} onChange={setAlt} optionLabel={(a) => ALT_LABEL[a]} />
      </>}>
      <IntervalStrip blocks={blocks} labelWidth={190} noteWidth={56} minPanelWidth={260} panels={[{ label: 'alternative − 9am day (% of the 9am day)', reference: 0, format: (v) => v.toFixed(0) }]} />
      {points.length ? <Crossplot points={points} title={`${points.length} filters`} xLabel={`${ch} BCc, 9am day (µg/m³)`} yLabel={`${ch} BCc, ${ALT_LABEL[alt]} (µg/m³)`} maxSide={340} fit={false} />
        : <Empty>No filter in the current subset falls inside a minute record (pick Addis Ababa or JPL in the Filters panel).</Empty>}
    </ChartFrame>
  )
}
