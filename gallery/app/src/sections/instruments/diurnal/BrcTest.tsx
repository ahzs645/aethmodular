import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Crossplot, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { histogramSeries, SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt, signed } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { useHighlightActions } from '@/lib/highlight'
import { ExcludedNote, useMarkedPoints } from '@/lib/markedRows'
import type { FilterRow } from '@/lib/types'
import { Key } from '@/components/Key'
import { CHANNEL_COLOUR, type BrcFile, type Channel } from './data'


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
 * ftir_28 Figure 2: each suspect channel's absorption against IR's, day by day, with the locus an
 * AAE = 1 aerosol would follow (b(λ) = b(880) · 880/λ) and equality with IR. Green below its locus is a
 * negative AAE; Red on IR carries nothing IR does not; UV needs its own axis to fit at all.
 */
export function ChannelHealthScatter({ brc }: { brc: BrcFile }) {
  const s = brc.summary, nm = brc.wavelengths_nm, b = brc.b
  const panels = useMemo(() => {
    const hi = (d3.max(b.IR, (v) => v ?? undefined) ?? 1) * 1.04
    // Blue, Green and Red share a y range (to Blue's AAE = 1 locus at the largest IR, as the notebook drew); UV its own, which is the diagnosis
    const shared: [number, number] = [0, hi * (nm.IR / nm.Blue) * 1.11]
    const uvRange: [number, number] = [-3000, 3000]
    const notes: Record<'UV' | 'Green' | 'Red' | 'Blue', string> = {
      UV: `${fmt(s.uv.out_of_range_pct, 0)} % of days out of range: BCc ≤ 0 on ${s.uv.nonpositive}, b(375) < b(880) on ${s.uv.below_ir} more; r with IR ${fmt(s.uv.r_ir, 3)}`,
      Green: `AAE(528, 880) median ${fmt(s.green.aae_median, 3)}, negative on ${fmt(s.green.negative_pct, 0)} % of days; ${fmt(s.green.deficit_pct, 0)} % below its own Blue–Red power law`,
      Red: `b(625)/b(880) IQR ±${fmt(s.red.iqr_halfwidth_pct, 1)} %; r with IR ${fmt(s.red.r_ir, 4)}; AAE(625, 880) ${fmt(s.aae_red_ir.mean, 3)} ± ${fmt(s.aae_red_ir.sd, 3)}`,
      Blue: `AAE(470, 880) mean ${fmt(s.blue.aae_mean, 3)}; r with IR ${fmt(s.blue.r_ir, 4)}`,
    }
    const titles = { UV: 'UV (375 nm) — out of range', Green: 'Green (528 nm) — unphysical negative AAE', Red: 'Red (625 nm) — indistinguishable from IR', Blue: 'Blue (470 nm) — for contrast, behaves' }
    return (['UV', 'Green', 'Red', 'Blue'] as const).map((c: Channel & keyof typeof notes) => {
      const yDomain = c === 'UV' ? uvRange : shared
      const points: CrossPoint[] = brc.date.flatMap((d, i) => {
        const x = b.IR[i], y = b[c][i]
        return x == null || y == null ? [] : [{ x, y, color: CHANNEL_COLOUR[c], tip: [d, `b(880) ${fmt(x, 1)} Mm⁻¹`, `b(${nm[c]}) ${fmt(y, 1)} Mm⁻¹`, `ratio ${fmt(y / x, 3)}`] }]
      })
      const off = points.filter((p) => p.y < yDomain[0] || p.y > yDomain[1] || p.x > hi).length
      const lines: CrossLine[] = [
        { slope: nm.IR / nm[c], intercept: 0, color: INK.text, label: 'AAE = 1', method: 'ols', stats: null },
        { slope: 1, intercept: 0, color: INK.muted, label: '1:1', dash: '2 3', method: 'ols', stats: null },
      ]
      return { c, title: titles[c], note: notes[c], points, off, lines, yDomain, x: [0, hi] as [number, number] }
    })
  }, [brc, b, nm, s])
  return (
    <ChartFrame id="diurnal-channel-health-scatter" title="Addis · the suspect channels against IR, day by day — only IR is trustworthy"
      subtitle={`Each point is one of the ${s.days} ETAD filter-days in the daily MA350 record: a channel's absorption b_ATN = BCc × σ_ATN against IR's. Solid: where an AAE = 1 aerosol would sit, b(λ) = b(880) × 880/λ; dotted: equal to IR. Green sits far below its locus (a negative AAE, which no aerosol produces); Red sits just below its own and on IR, the same statement as "no red excess"; Blue is the one short channel that behaves. The UV panel has its own y-axis, running ${fmt(3000 / (panels[1]?.yDomain[1] ?? 1), 0)}× taller and well below zero, and still loses points off its edges. An instrument diagnosis, not a measurement of brown carbon at Addis.`}
      provenance={`${NB} (Figure 2, channel_health_panel.png)`}>
      <FacetGrid items={panels} minCell={260} maxCols={4} maxCell={360}
        title={(p) => p.title} note={(p) => (p.off ? `${p.off} off scale` : '')}
        render={(p) => (
          <div>
            <Crossplot points={p.points} title="" xLabel="b_ATN(880 nm), Mm⁻¹" yLabel={`b_ATN(${nm[p.c]} nm), Mm⁻¹`} fit={false} grouping="none" sameScale={false}
              domain={p.x} yDomain={p.yDomain} extraLines={p.lines} maxSide={300} />
            <p className="chart-note" style={{ margin: '4px 8px 0' }}>{p.note}</p>
          </div>
        )} />
      <Key items={[{ label: 'what an AAE = 1 aerosol would give', color: INK.text, kind: 'line' }, { label: '1:1 (equal to IR)', color: INK.muted, kind: 'dash' }]} />
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
  const pointsOf = (rs: FilterRow[]): CrossPoint[] => rs.flatMap((r) => {
    const x = r[BC880], y = r[EC_FTIR]
    return typeof x === 'number' && typeof y === 'number' ? [{ x, y, color: r.color, id: r.id, open: () => openSample(r.id), tip: [r.id, r.date, `${BC880}: ${fmt(x, 2)}`, `${EC_FTIR}: ${fmt(y, 2)}`] }] : []
  })
  const points = useMemo(() => pointsOf(rows), [rows, openSample]) // eslint-disable-line react-hooks/exhaustive-deps
  // Filters → excluded: Mark (lib/markedRows): the Addis filters excluded in view, × only, in no refit
  const marked = useMarkedPoints((rs) => pointsOf(rs.filter((r) => r.code === 'ETAD')), [openSample])
  return (
    <ChartFrame id="diurnal-ec-bc880" title="Addis · FTIR EC against MA350 BC at 880 nm — the FTIR axis is not where the offset lives"
      subtitle={`Each Addis filter in view with both: FTIR EC, and the MA350 IR BCc averaged over the filter's sampling window. ftir_28 found intercept +0.32 µg/m³ (95 % CI +0.01 to +0.63), R² 0.868, n = 173 on exact 9 am days, about 13× smaller than the −4.17 against HIPS and of the opposite sign. The fit here is the gallery's live refit of the points drawn, so dates, seasons and hand edits apply.`}
      provenance={`${NB} (the related positive result)`}>
      {points.length >= 3 ? <><Crossplot points={points} marked={marked} title="" xLabel="MA350 IR BCc, 880 nm (µg/m³)" yLabel="FTIR EC (µg/m³)" />
        <ExcludedNote count={marked.length} drawn /></>
        : <Empty>Fewer than three Addis filters in view have both {EC_FTIR} and {BC880}.</Empty>}
    </ChartFrame>
  )
}
