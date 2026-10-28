import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { histogramSeries, SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt, signed } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { useHighlightActions } from '@/lib/highlight'
import type { FilterRow } from '@/lib/types'
import type { BrcFile } from './data'


const NB = 'research/ftir_ec_phase3/ftir_28_ma350_brc_falsification.ipynb'
const BC880 = 'IR BCc (aethalometer)'
const EC_FTIR = 'EC (FTIR)'

/** a histogram of filter-days in equal bins over [lo, hi] (values outside are left out) */
function filterDayHistogram(id: string, values: number[], lo: number, hi: number, bins: number, color: string, unit: string): SweepSeries {
  const b = d3.bin().domain([lo, hi]).thresholds(d3.range(bins + 1).map((i) => lo + (i * (hi - lo)) / bins))(values.filter((v) => v >= lo && v <= hi))
  const edges = [...b.map((x) => x.x0!), b[b.length - 1].x1!]
  return histogramSeries({ id, label: id, color, edges, counts: b.map((x) => x.length), fill: 0.25, width: 1.6,
    tip: (i, n) => [`${fmt(edges[i], 2)} to ${fmt(edges[i + 1], 2)}${unit}`, `${n} filter-days`] })
}

/**
 * ftir_28: can the Addis MA350 measure a brown-carbon share at the HIPS wavelength? Anchor black carbon
 * at AAE 1 on IR and call the 625 nm excess brown carbon; the measured AAE(625, 880) sits below the
 * anchor, so the implied brown-carbon absorption is negative, and three channels fail checks no aerosol
 * can fail. An instrument statement, not one about the atmosphere.
 */
export function BrcTest({ brc }: { brc: BrcFile }) {
  const s = brc.summary
  const aae = brc.aae.Red.filter((v): v is number => v != null)
  const babs = brc.brc.filter((v): v is number => v != null)
  const closingLo = d3.min(s.closing_aae.per_setup)!, closingHi = d3.max(s.closing_aae.per_setup)!
  const panels = [
    { key: 'aae', title: 'AAE(625, 880) from b_ATN, per filter-day', xLabel: 'AAE(625, 880)',
      series: [filterDayHistogram('AAE(625, 880)', aae, 0.1, 1.3, 48, INK.accent, '')],
      markers: [{ at: 1, label: 'anchor AAE_BC = 1', color: INK.text, labelSide: 'left' as const }, { at: s.closing_aae.mean_target, label: `needed ${fmt(s.closing_aae.mean_target, 2)}`, color: INK.warn, labelSide: 'right' as const }],
      shade: [closingLo, closingHi] as [number, number] },
    { key: 'brc', title: 'Implied Babs_BrC(625), per filter-day', xLabel: 'Babs_BrC (Mm⁻¹, b_ATN scale)',
      series: [filterDayHistogram('Babs_BrC', babs, -10, 28, 48, INK.accent, ' Mm⁻¹')],
      markers: [{ at: 0, label: '0', color: INK.text }, { at: s.target.mean, label: `target ${fmt(s.target.mean, 1)}`, color: INK.warn, labelSide: 'left' as const }],
      shade: [s.target.min, s.target.max] as [number, number] },
  ]
  return (
    <ChartFrame id="diurnal-brc" title="Addis · can the MA350 measure a brown-carbon share? — no: the red channel sits below the black-carbon anchor"
      subtitle={`MA350-0238, ${s.days} ETAD filter-days (the daily record). Absorption b_ATN = BCc × σ_ATN; brown carbon would be the 625 nm absorption above an AAE = 1 black carbon anchored at 880 nm, Babs_BrC = b(625) − b(880)·(880/625). Measured AAE(625, 880) is ${fmt(s.aae_red_ir.mean, 3)} ± ${fmt(s.aae_red_ir.sd, 3)}, above 1 on only ${fmt(s.aae_red_ir.above_anchor_pct, 1)} % of days, so Babs_BrC averages ${signed(s.brc.mean, 2)} Mm⁻¹ and is negative on ${fmt(s.brc.negative_pct, 1)} %, never reaching the +${fmt(s.target.mean, 1)} Mm⁻¹ the HIPS intercept needs (|intercept| × MAC / slope over six calibration setups, shaded ${fmt(s.target.min, 1)}–${fmt(s.target.max, 1)}). Closing it would need AAE_BC ≈ ${fmt(s.closing_aae.mean_target, 2)} (setups ${fmt(closingLo, 2)}–${fmt(closingHi, 2)}, shaded): black carbon absorbing more at 880 than at 625 nm. This says the instrument cannot answer, not that Addis has no brown carbon.`}
      provenance={`${NB}; ${brc.source}`}>
      <FacetGrid items={panels} minCell={340} maxCols={2} render={(p, w) => (
        <SweepPlot title={p.title} width={w} height={240} series={p.series} xScale="linear" xLabel={p.xLabel} yLabel="Filter-days" include={[0]}
          xMarkers={p.markers} yFormat={(v) => fmt(v, 0)} xFormat={(v) => fmt(+v, 1)}
          background={({ x, ih }) => { const sx = x as (v: number) => number
            return <rect x={sx(p.shade[0])} width={Math.max(1, sx(p.shade[1]) - sx(p.shade[0]))} y={0} height={ih} fill={INK.warn} fillOpacity={0.12} /> }} />
      )} />
    </ChartFrame>
  )
}

/** ftir_28 part 2: which channels pass checks no real aerosol can fail. */
export function ChannelHealth({ brc }: { brc: BrcFile }) {
  const s = brc.summary
  const rows = [
    { ch: 'UV (375 nm)', check: 'BCc ≤ 0, or less absorption at 375 than at 880 nm', value: `${fmt(s.uv.out_of_range_pct, 1)} % of days out of range (${s.uv.nonpositive} ≤ 0, ${s.uv.below_ir} more below IR)`, r: s.uv.r_ir, verdict: 'fails' },
    { ch: 'Green (528 nm)', check: 'AAE(528, 880); a negative one is unphysical', value: `median ${fmt(s.green.aae_median, 3)}, negative on ${fmt(s.green.negative_pct, 1)} %; ${fmt(s.green.deficit_pct, 0)} % below its own Blue–Red power law`, r: s.green.r_ir, verdict: 'fails' },
    { ch: 'Red (625 nm)', check: 'b(625)/b(880) spread about its median', value: `IQR ±${fmt(s.red.iqr_halfwidth_pct, 1)} %; median BCc ratio Red/IR ${fmt(s.red.bcc_ratio_median, 4)}`, r: s.red.r_ir, verdict: 'carries nothing IR does not' },
    { ch: 'Blue (470 nm)', check: 'AAE(470, 880)', value: `mean ${fmt(s.blue.aae_mean, 3)}`, r: s.blue.r_ir, verdict: 'behaves' },
  ]
  return (
    <ChartFrame id="diurnal-channel-health" title="Addis · MA350 channel health — only IR is trustworthy" exportable={false}
      subtitle="Each check is a property no real aerosol can violate, so failing it is a statement about the instrument. r is the day-to-day correlation of the channel's BCc with IR's. A channel-to-channel error of 45 % (Green) is shown on this instrument; the red excess the brown-carbon method looks for is a 24 % shift."
      provenance={`${NB} (part 2)`}>
      <div style={{ overflowX: 'auto' }}>
        <table className="census-table">
          <thead><tr><th>Channel</th><th>Check</th><th>Result</th><th>r with IR</th><th>Verdict</th></tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.ch}><td>{r.ch}</td><td>{r.check}</td><td>{r.value}</td><td style={{ fontFamily: FONT.mono, textAlign: 'right' }}>{fmt(r.r, 3)}</td>
              <td style={{ color: r.verdict === 'behaves' ? INK.good : r.verdict === 'fails' ? INK.negative : INK.warn }}>{r.verdict}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </ChartFrame>
  )
}

/**
 * ftir_28's positive result: FTIR EC against MA350 BC at 880 nm, where brown carbon barely absorbs. An
 * intercept near zero puts the HIPS intercept's offset on the 633 nm optics, not the FTIR side.
 */
export function EcAgainstBc880({ rows }: { rows: FilterRow[] }) {
  const { openSample } = useHighlightActions()
  // gallery filters: a click opens the filter's own record, as on the gallery's charts
  const points = useMemo<CrossPoint[]>(() => rows.flatMap((r) => {
    const x = r[BC880], y = r[EC_FTIR]
    return typeof x === 'number' && typeof y === 'number' ? [{ x, y, color: r.color, id: r.id, open: () => openSample(r.id), tip: [r.id, r.date, `${BC880}: ${fmt(x, 2)}`, `${EC_FTIR}: ${fmt(y, 2)}`] }] : []
  }), [rows, openSample])
  return (
    <ChartFrame id="diurnal-ec-bc880" title="Addis · FTIR EC against MA350 BC at 880 nm — the FTIR axis is not where the offset lives"
      subtitle={`Each Addis filter in view with both: FTIR EC, and the MA350 IR BCc averaged over the filter's sampling window. ftir_28 found intercept +0.32 µg/m³ (95 % CI +0.01 to +0.63), R² 0.868, n = 173 on exact 9 am days, about 13× smaller than the −4.17 against HIPS and of the opposite sign. The fit here is the gallery's live refit of the points drawn, so dates, seasons and hand edits apply.`}
      provenance={`${NB} (the related positive result)`}>
      {points.length >= 3 ? <Crossplot points={points} title="" xLabel="MA350 IR BCc, 880 nm (µg/m³)" yLabel="FTIR EC (µg/m³)" />
        : <Empty>Fewer than three Addis filters in view have both {EC_FTIR} and {BC880}.</Empty>}
    </ChartFrame>
  )
}
