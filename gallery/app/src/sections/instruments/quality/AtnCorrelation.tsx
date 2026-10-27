import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { FacetGrid } from '@/components/FacetGrid'
import { HeatCell } from '@/components/HeatCell'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt } from '@/lib/format'
import { pearson } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { CHANNELS, finite, instrumentColor, instrumentName, type Channel, type Day } from './data'

type Matrix = (number | null)[][]
/** pairwise Pearson r between channels (pandas DataFrame.corr: each pair on the days both have) */
export function channelCorr(days: Day[], value: (d: Day, c: Channel) => number | null): Matrix {
  return CHANNELS.map((a) => CHANNELS.map((b) => {
    const xs: number[] = [], ys: number[] = []
    for (const d of days) { const x = value(d, a), y = value(d, b); if (finite(x) && finite(y)) { xs.push(x); ys.push(y) } }
    return xs.length >= 10 ? pearson(xs, ys) : null
  }))
}
const offDiagonal = (m: Matrix) => m.flatMap((row, i) => row.filter((v, j) => j < i && finite(v))) as number[]

function Triangle({ code, atn, bcc, width, meta, color }: { code: string; atn: Matrix; bcc: Matrix; width: number; meta: MetaFile; color: (v: number) => string }) {
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  const n = CHANNELS.length, left = 44, top = 24
  const cell = Math.max(24, Math.min(44, (width - left - 8) / n))
  const a = offDiagonal(atn), b = offDiagonal(bcc)
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={left + n * cell + 8} height={top + n * cell + 44} fontFamily={FONT.family}>
        <text x={0} y={14} fontSize={12.5} fontWeight={600} fill={instrumentColor(code, meta)}>{instrumentName(code, meta)}</text>
        <g transform={`translate(${left},${top})`}>
          {CHANNELS.map((ci, i) => CHANNELS.map((cj, j) => {
            if (i === j) return <HeatCell key={`${i}${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1} fill={INK.zebra} label={ci} fontSize={9} />
            const below = j < i, v = (below ? atn : bcc)[i][j]
            return <HeatCell key={`${i}${j}`} x={j * cell} y={i * cell} width={cell - 1} height={cell - 1} fill={finite(v) ? color(v) : null} label={finite(v) ? v.toFixed(2).replace(/^0/, '').replace(/^-0/, '−') : '—'} fontSize={cell > 34 ? 10 : 8.5}
              onEnter={(e) => tip.show(e, [`${instrumentName(code, meta)} · ${ci} with ${cj}`, `${below ? 'raw attenuation' : 'BCc'}: r = ${fmt(v, 3)}`, below ? `BCc: r = ${fmt(bcc[i][j], 3)}` : `raw attenuation: r = ${fmt(atn[i][j], 3)}`])} onLeave={tip.hide} />
          }))}
          {CHANNELS.map((c, i) => <text key={c} x={-6} y={i * cell + cell / 2} dy="0.32em" textAnchor="end" fontSize={10} fill={INK.muted}>{c}</text>)}
          <text x={0} y={n * cell + 14} fontSize={10.5} fill={INK.muted}>below: raw ATN · above: BCc</text>
          <text x={0} y={n * cell + 28} fontSize={10.5} fill={INK.text}>mean r {fmt(d3.mean(a), 2)} (ATN) · {fmt(d3.mean(b), 2)} (BCc)</text>
        </g>
      </svg>
      {tip.node}
    </div>
  )
}

/**
 * Are the channels' high correlations in the aerosol or in the processing? Inter-wavelength correlation of the
 * raw attenuation (one spot) below the diagonal and of the processed BCc above it, per instrument
 * (Raw_ATN_Correlation_Analysis §4–8). Correlations are over daily means. Attenuation is the loaded spot's
 * cumulative optical depth, a sawtooth reset at each tape advance, so its correlation across channels mostly
 * follows the shared loading cycle rather than the day's aerosol; the notebook's reading of the difference is
 * a prompt for the minute record, not a result.
 */
export function AtnCorrelation({ all, meta }: { all: Record<string, Day[]>; meta: MetaFile }) {
  const [spot, setSpot] = useSearchParam<string>('aq.atnspot', 'ATN1', ['ATN1', 'ATN2'])
  const colours = useColourScheme('aq-atn-corr', 'sequential')
  const color = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([0.5, 1]).clamp(true)
  const mats = useMemo(() => Object.entries(all).map(([code, days]) => ({
    code,
    atn: channelCorr(days, (d, c) => d.q(`${c}.${spot}`)),
    bcc: channelCorr(days, (d, c) => d.bc[c]),
  })), [all, spot])
  return (
    <ChartFrame id="aq-atn-corr" title={`Raw attenuation against BCc — do the channels agree before processing? (${spot})`}
      subtitle="Pearson r between the five channels over each instrument's days: below the diagonal the raw attenuation of the chosen spot, above it the processed BCc (compensated for loading by the dual-spot method). If BCc correlated much more tightly than raw attenuation, the processing would be tying the channels together. Raw attenuation is a sawtooth that resets at each tape advance, so its daily means share the loading cycle across channels; read the comparison as the notebook did, as a prompt rather than a verdict. Colours run from r = 0.5 (and below) to 1."
      provenance="Raw_ATN_Correlation_Analysis (correlation matrices, the BCc − ATN difference, UV–IR) · processed_sites/*.pkl"
      controls={<Segmented label="Spot" value={spot} options={['ATN1', 'ATN2']} onChange={setSpot} optionLabel={(s) => (s === 'ATN1' ? 'Spot 1' : 'Spot 2')} />}>
      {mats.length ? <FacetGrid items={mats} minCell={240} maxCols={5} render={(m, cell) => <Triangle {...m} width={cell} meta={meta} color={color} />} />
        : <Empty>No attenuation columns loaded.</Empty>}
      <div className="legend"><ColorLegend scale={color} label="Pearson r" width={150} scheme={colours.scheme} format={(v) => v.toFixed(1)} /></div>
    </ChartFrame>
  )
}
