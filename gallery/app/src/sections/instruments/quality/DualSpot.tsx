import { useMemo } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { fmt } from '@/lib/format'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { DailyPanels, qLine, type PanelSpec } from './DailyPanels'
import { CHANNELS, finite, instrumentColor, instrumentName, quantile, ratio, type Channel, type Day, type QualityIndex } from './data'

const SPOT = { one: '#2c7fb8', two: '#d95f0e' }
/** ftir_49's channel colours */
export const CHANNEL_COLOR: Record<Channel, string> = { UV: '#7D3C98', Blue: '#2E86C1', Green: '#28B463', Red: '#C0392B', IR: '#34495E' }

/**
 * The dual-spot chain day by day for one instrument and channel: attenuation on both spots (a sawtooth that
 * resets at each tape advance), their rates, the leak ratio, the loading compensation BCc/BC1 and the spot ratio
 * BC1/BC2, and the roughness metric the minute cleaning flags high-noise periods with; at Addis, where the raw
 * MA350 export kept it, the instrument's K. All are daily means of the minute values the cleaning computed
 * (src/data/qc/pkl_cleaning.py), so a ratio is a mean of minute ratios.
 */
export function DualSpotTimeline({ code, days, meta, index }: { code: string; days: Day[]; meta: MetaFile; index: QualityIndex }) {
  const info = index.sites[code]
  const channels = info?.channels ?? []
  const [ch, setCh] = useSearchParam<string>('aq.dsch', 'IR', [...CHANNELS])
  const c = (channels.includes(ch as Channel) ? ch : channels[channels.length - 1] ?? 'IR') as Channel
  const has = (k: string) => info?.series.includes(`${c}.${k}`)
  const b = index.cleaning_bounds
  const panels: PanelSpec[] = [
    { key: 'atn', title: 'Attenuation', lines: [qLine(`${c}.ATN1`, 'Spot 1', SPOT.one, { digits: 1 }), qLine(`${c}.ATN2`, 'Spot 2', SPOT.two, { digits: 1 })] },
    { key: 'datn', title: 'dATN per minute', lines: [qLine(`${c}.dATN1`, 'Spot 1', SPOT.one, { digits: 4 }), qLine(`${c}.dATN2`, 'Spot 2', SPOT.two, { digits: 4 })] },
    ...(has('leak') ? [{ key: 'leak', title: 'Leak ratio', refs: [{ at: 1, label: '1' }], lines: [qLine(`${c}.leak`, '(dATN1/F1)/(dATN2/F2)', '#6a51a3')] }] : []),
    ...(has('BCc/BC1') ? [{ key: 'ratios', title: 'Spot ratios', refs: [{ at: 1, label: '1' }], lines: [qLine(`${c}.BCc/BC1`, 'BCc / BC1 (smoothed)', '#8c2d04'), qLine(`${c}.BC1/BC2`, 'BC1 / BC2 (smoothed)', '#41ab5d')] }] : []),
    ...(has('rough1') ? [{ key: 'rough', title: 'Roughness', lines: [qLine(`${c}.rough1`, 'Spot 1', SPOT.one, { digits: 5 }), qLine(`${c}.rough2`, 'Spot 2', SPOT.two, { dash: '4 3', digits: 5 })] }] : []),
    ...(has('K') ? [{ key: 'k', title: 'Instrument K', refs: [{ at: 0, label: '0' }], lines: [qLine(`${c}.K`, 'K (loading compensation)', '#111827', { digits: 4 })] }] : []),
  ]
  return (
    <ChartFrame id="aq-dualspot" title={`${instrumentName(code, meta)} · the dual-spot chain — attenuation, its rate, the spot ratios and roughness`}
      subtitle={`${c} channel, daily means. Attenuation rises as each spot loads and drops at a tape advance; dATN is its per-minute change (rolling mean). The leak ratio compares the spots' attenuation rates per unit flow ((dATN1/Flow1)/(dATN2/Flow2); the minute cleaning keeps ${b.leak_ratio.lower_bound}–${b.leak_ratio.upper_bound}). BCc/BC1 is how much the loading compensation raises spot 1 (the cleaning keeps BCc/BC1 or BCc/BC2 within ${b.bcc_ratio.lower_bound}–${b.bcc_ratio.upper_bound}); BC1/BC2 compares the uncompensated spots. Roughness is the 60-minute mean of |dATN − its rolling mean|, the noise the cleaning flags high-roughness periods by.${has('K') ? ' K is the instrument\'s loading parameter (BCc = BC1 / (1 − K·ATN1)).' : ''}`}
      provenance="ftir_49_ma350_raw_chain_diagnostics (A1, daily fallback) · Raw_ATN_Correlation_Analysis · src/data/qc/pkl_cleaning.py (leak ratio, BCc ratio, roughness)"
      controls={<Segmented label="Channel" value={c} options={channels} onChange={setCh} />}>
      {channels.length ? <DailyPanels days={days} panels={panels} periods={index.flow_periods[code]} panelHeight={84} /> : <Empty>No attenuation columns for this instrument.</Empty>}
    </ChartFrame>
  )
}

/**
 * ftir_49's daily fallback of the correction diagnostics at Jacros, per channel: the instrument K, the loading
 * factor BCc/BC1 and the compensation share 1 − BC1/BCc, the K* that would reconcile the two spots on daily means,
 * and spot 2 compensated with the instrument's K against the reported BCc. Ratios of daily means are not daily
 * means of minute ratios, so these are indicative.
 */
export function JacrosChain({ days }: { days: Day[] }) {
  const blocks = useMemo<IntervalBlock[]>(() => {
    const rows = CHANNELS.map((c) => {
      const v = (f: (d: Day) => number | null) => days.map(f).filter(finite)
      const K = v((d) => d.q(`${c}.K`))
      const load = v((d) => ratio(d.bc[c], d.q(`${c}.BC1`)))
      const share = v((d) => { const r = ratio(d.q(`${c}.BC1`), d.bc[c]); return r === null ? null : 1 - r })
      const kstar = v((d) => {
        const bc1 = d.q(`${c}.BC1`), bc2 = d.q(`${c}.BC2`), a1 = d.q(`${c}.ATN1`), a2 = d.q(`${c}.ATN2`)
        return finite(bc1) && finite(bc2) && finite(a1) && finite(a2) ? ratio(bc1 - bc2, bc1 * a2 - bc2 * a1) : null
      })
      const spot2 = v((d) => {
        const k = d.q(`${c}.K`), bc2 = d.q(`${c}.BC2`), a2 = d.q(`${c}.ATN2`)
        return finite(k) && finite(bc2) && finite(a2) ? ratio(ratio(bc2, 1 - k * a2), d.bc[c]) : null
      })
      const s = (x: number[], lo = 0.05, hi = 0.95) => ({ mid: quantile(x, 0.5), lo: quantile(x, lo), hi: quantile(x, hi) })
      const col = CHANNEL_COLOR[c]
      const k = s(K), l = s(load, 0.25, 0.75), sh = s(share, 0.25, 0.75), ks = s(kstar, 0.25, 0.75), s2 = s(spot2, 0.25, 0.75)
      return {
        key: c, label: c, labelColor: col, note: `n=${K.length}`,
        marks: [[{ ...l, color: col }], [{ ...sh, color: col }], [{ ...k, color: col }, { ...ks, color: col, shape: 'hollow' as const, dy: 5, faint: true }], [{ ...s2, color: col }]],
        tip: [`${c} · ${K.length} days`, `BCc/BC1 ${fmt(l.mid, 2)} (IQR ${fmt(l.lo, 2)}–${fmt(l.hi, 2)})`, `compensation share ${fmt(sh.mid, 3)}`,
          `K ${fmt(k.mid, 4)} (5–95 %: ${fmt(k.lo, 4)}–${fmt(k.hi, 4)}) · K* ${fmt(ks.mid, 4)} (IQR ${fmt(ks.lo, 4)}–${fmt(ks.hi, 4)})`,
          `BC2/(1−K·ATN2) ÷ BCc ${fmt(s2.mid, 3)} (IQR ${fmt(s2.lo, 3)}–${fmt(s2.hi, 3)})`],
      }
    })
    return [{ key: 'jacros', rows }]
  }, [days])
  return (
    <ChartFrame id="aq-jacros-chain" title="Addis · how much of BCc is the loading compensation — K, BCc/BC1 and the spot-2 check per channel"
      subtitle="The raw MA350 export at Addis (MA350-0238, its Jacros file: 1,047 daily means): the loading factor BCc/BC1 and the compensation share 1 − BC1/BCc (median, interquartile range of days); the instrument's K (filled, 5–95 %) beside K* = (BC1 − BC2)/(BC1·ATN2 − BC2·ATN1), the K that would make the two spots agree on daily means (hollow, IQR); and spot 2 compensated with the instrument's K over the reported BCc (IQR), 1 if the spots agreed. ftir_49 reads these as: the compensation is 37–48 % of BCc at the channels that matter, so it is not a small correction."
      provenance="ftir_49_ma350_raw_chain_diagnostics (A1 daily fallback: k_distribution_daily_fallback, k_sensitivity_daily_fallback) · processed_sites/df_Jacros_9am_resampled.pkl">
      {days.length ? <IntervalStrip blocks={blocks} labelWidth={70} noteWidth={56} minPanelWidth={170}
        panels={[{ label: 'BCc / BC1', reference: 1, format: (v) => v.toFixed(1) }, { label: '1 − BC1 / BCc', reference: 0, format: (v) => v.toFixed(1) },
          { label: 'K (filled), K* (hollow)', reference: 0, format: (v) => v.toFixed(3) }, { label: 'BC2/(1−K·ATN2) ÷ BCc', reference: 1, format: (v) => v.toFixed(2) }]} />
        : <Empty>Addis's daily bins did not load.</Empty>}
    </ChartFrame>
  )
}

/**
 * The legacy ±1-day matching against a single bin, on each instrument's filter dates (ftir_49 B6, daily
 * fallback, for every instrument): data_matching.match_aeth_filter_data averaged the bins labelled within a day
 * of the filter's sample date; this is how far that moves the value from the one bin labelled on the date.
 */
export function WindowEffect({ all, rows, meta }: { all: Record<string, Day[]>; rows: FilterRow[]; meta: MetaFile }) {
  const blocks = useMemo<IntervalBlock[]>(() => {
    const rowsOut = Object.entries(all).flatMap(([code, days]) => {
      const dates = [...new Set(rows.filter((r) => r.code === code).map((r) => r.date.slice(0, 10)))]
      const ir = new Map(days.map((d) => [d.start.slice(0, 10), d.bc.IR]))
      const shift = (d: string, k: number) => new Date(Date.parse(`${d}T00:00:00Z`) + k * 864e5).toISOString().slice(0, 10)
      const diffs = dates.flatMap((d) => {
        const one = ir.get(d)
        const tol = [-1, 0, 1].map((k) => ir.get(shift(d, k))).filter(finite)
        return finite(one) && one !== 0 && tol.length ? [(100 * (tol.reduce((a, b) => a + b, 0) / tol.length - one)) / one] : []
      })
      if (!diffs.length) return []
      const color = instrumentColor(code, meta)
      const over = (t: number) => diffs.filter((x) => Math.abs(x) > t).length
      return [{
        key: code, label: instrumentName(code, meta), labelColor: color, note: `n=${diffs.length}`,
        marks: [[{ mid: quantile(diffs, 0.5), lo: quantile(diffs, 0.025), hi: quantile(diffs, 0.975), color, faint: true, r: 0 },
          { mid: quantile(diffs, 0.5), lo: quantile(diffs, 0.25), hi: quantile(diffs, 0.75), color }]],
        tip: [`${instrumentName(code, meta)} · ${diffs.length} filter dates`,
          `median ${fmt(quantile(diffs, 0.5), 1)} % · IQR ${fmt(quantile(diffs, 0.25), 1)} to ${fmt(quantile(diffs, 0.75), 1)} %`,
          `|Δ| > 10 %: ${over(10)} dates · > 25 %: ${over(25)}`],
      }]
    })
    return [{ key: 'all', rows: rowsOut }]
  }, [all, rows, meta])
  return (
    <ChartFrame id="aq-window-effect" title="The ±1-day matching window — how far it moves a filter day's IR BCc"
      subtitle="On every filter date in the current subset the mean of the bins labelled the day before, the day and the day after (data_matching's legacy ±1-day match) against the single bin labelled on the date, in percent of the single bin. Thick: interquartile range; thin: 2.5–97.5 %. The gallery's own per-filter values use the overlap of the bins with each filter's sampling window instead (export_aethalometer.py)."
      provenance="ftir_49_ma350_raw_chain_diagnostics B6 (daily fallback), generalised to every instrument · data_matching.match_aeth_filter_data">
      {blocks[0].rows.length ? <IntervalStrip blocks={blocks} labelWidth={130} noteWidth={56} minPanelWidth={260}
        panels={[{ label: '±1-day mean − single bin (% of the single bin)', reference: 0, format: (v) => v.toFixed(0) }]} />
        : <Empty>No filter dates in the current subset fall on a bin with IR BCc.</Empty>}
    </ChartFrame>
  )
}
