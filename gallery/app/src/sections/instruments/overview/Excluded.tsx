import { useMemo, useRef } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock, type IntervalMark } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { useDimensions } from '@/hooks/useGalleryData'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { CHANNELS, channelColor, instColor, type Exclusion, type Series } from './data'
import { SIGMA, summarize, type Day } from './stats'

/** AAE(UV, IR) of a day's absorption (BCc × σ), or null */
function aaeUvIr(d: Day, lambda: Record<string, number>): number | null {
  const uv = d.bc.UV, ir = d.bc.IR
  if (uv === null || ir === null || uv <= 0 || ir <= 0) return null
  return -Math.log((uv * SIGMA.UV) / (ir * SIGMA.IR)) / Math.log(lambda.UV / lambda.IR)
}
const shift = (date: string, k: number) => new Date(Date.parse(`${date}T00:00:00Z`) + k * 864e5).toISOString().slice(0, 10)

/**
 * Multi_Site_Analysis_FollowUp 7b: the samples outliers.EXCLUDED_SAMPLES leaves out, at all five
 * wavelengths. For each, the daily BCc of the day it names (dot) and the days either side (bar),
 * each channel divided by that channel's median day, so a value that is high at every wavelength
 * reads as a real event and one high at a single wavelength as that channel; and the day's AAE
 * against the instrument's median (open).
 */
export function Excluded({ series, meta, exclusions }: { series: Series[]; meta: MetaFile; exclusions: Record<string, Exclusion[]> }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const lambda = meta.wavelengths_nm
  const { blocks, missing } = useMemo(() => {
    let missing = 0
    const blocks: IntervalBlock[] = series.filter((s) => s.inst.code === s.inst.siteCode && exclusions[s.inst.siteCode]?.length).map((s) => {
      const med = Object.fromEntries(CHANNELS.map((c) => [c, summarize(s.days.map((d) => d.bc[c]))?.median ?? null]))
      const aaeMed = summarize(s.days.map((d) => aaeUvIr(d, lambda)))?.median ?? null
      const byDate = new Map(s.days.map((d) => [d.date, d]))
      return {
        key: s.inst.code, title: s.inst.site, color: instColor(meta, s.inst),
        rows: exclusions[s.inst.siteCode].map((x) => {
          const day = byDate.get(x.date)
          const near = [-1, 0, 1].map((k) => byDate.get(shift(x.date, k))).filter((d): d is Day => !!d)
          if (!near.length) missing++
          const chMarks: IntervalMark[] = CHANNELS.flatMap((c) => {
            const m = med[c]
            const vs = near.map((d) => d.bc[c]).filter((v): v is number => v !== null)
            if (!m || !vs.length) return []
            const v = day?.bc[c] ?? null
            return [{ mid: v === null ? null : v / m, lo: Math.min(...vs) / m, hi: Math.max(...vs) / m, color: channelColor(c), r: 3.5, dy: (CHANNELS.indexOf(c) - 2) * 2.4 }]
          })
          const a = day ? aaeUvIr(day, lambda) : null
          return {
            key: `${x.date}|${x.filter_id ?? ''}`, label: `${x.date}${x.filter_id ? ` · ${x.filter_id}` : ''}`,
            note: !near.length ? 'no daily bin' : chMarks.length ? '' : 'no BCc then',
            marks: [chMarks, [...(aaeMed !== null ? [{ mid: aaeMed, color: INK.muted, shape: 'hollow' as const, r: 3.5 }] : []), ...(a !== null ? [{ mid: a, color: instColor(meta, s.inst) }] : [])]],
            tip: [`${s.inst.site} · ${x.date}${x.filter_id ? ` · ${x.filter_id}` : ''}`, x.reason,
              ...(near.length ? CHANNELS.map((c) => `${c}: ${fmt(day?.bc[c], 2)} µg/m³ that day (${fmt(med[c] && day?.bc[c] != null ? day.bc[c]! / med[c]! : null, 2)}× its median); ±1 day ${near.map((d) => fmt(d.bc[c], 2)).join(' / ')}`)
                : ['no daily bin within a day of this date in the span']),
              `AAE UV–IR (absorption) ${fmt(a, 2)} against the instrument's median ${fmt(aaeMed, 2)}`],
          }
        }),
      }
    })
    return { blocks, missing }
  }, [series, meta, exclusions, lambda])
  const n = blocks.reduce((k, b) => k + b.rows.length, 0)
  return (
    <ChartFrame id="aeth-overview-excluded" title="The excluded samples at five wavelengths — is the aethalometer odd at every channel, or one?"
      subtitle={`The ${n} samples research/ftir_hips_chem/scripts/outliers.py EXCLUDED_SAMPLES leaves out at the sites in view, looked up in the daily file. Left: each channel's daily BCc on the named day (dot) and its range over the day either side (bar; the notebook averaged ±1 day), divided by that channel's median day in the span, so 1 is an ordinary day. Right: the day's AAE from UV and IR absorption (filled) against the instrument's median day (open). The dated exclusions predate the matched-sample audit; ${missing} of them have no daily bin within a day.`}
      provenance="Multi_Site_Analysis_FollowUp.ipynb §7b (excluded dates by wavelength) · outliers.EXCLUDED_SAMPLES via gallery/data/export_aeth_overview.py · daily files: gallery/data/export_aethalometer.py">
      <div ref={ref} className="chart-wrap">
        {!blocks.length ? <Empty>No excluded samples at the sites in view.</Empty>
          : <IntervalStrip blocks={blocks} labelWidth={width < 600 ? 130 : 170} rowHeight={24} noteWidth={width < 600 ? 0 : 80} minPanelWidth={200}
            panels={[{ label: "Daily BCc / the channel's median day", reference: 1, heading: 'Five wavelengths', format: (v) => `${fmt(v, 1)}×` }, { label: 'AAE UV–IR (absorption)', heading: 'Spectral shape', reference: 1 }]} />}
        <Key items={[...CHANNELS.map((c) => ({ label: `${c} (${lambda[c]} nm)`, color: channelColor(c) })), { label: "instrument's median AAE", color: INK.muted, kind: 'hollow' as const }]} />
      </div>
    </ChartFrame>
  )
}
