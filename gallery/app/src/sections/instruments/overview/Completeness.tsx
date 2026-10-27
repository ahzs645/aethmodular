import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis } from '@/components/Axes'
import { ColorLegend } from '@/components/ColorLegend'
import { HeatCell } from '@/components/HeatCell'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK, RAMP_SEQUENTIAL } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { instColor, type Series } from './data'

type Metric = 'ir' | 'bins' | 'cov'
const METRIC_LABEL: Record<Metric, string> = { ir: 'Days with IR BCc', bins: 'Bins in the file', cov: 'Minute coverage' }
const M = { top: 8, right: 70, bottom: 34, left: 150 }
const ROW = 26

const monthKey = (date: string) => date.slice(0, 7)
const daysIn = (key: string) => new Date(Date.UTC(+key.slice(0, 4), +key.slice(5, 7), 0)).getUTCDate()

/**
 * How much of each month the daily files cover, per instrument (Multi_Site_Analysis §4 drew the
 * minute coverage day by day; §7 tabulated days with BC). The legacy files keep only some days
 * (Beijing, Delhi and JPL: those near a filter sample), so "bins in the file" is part of the story.
 */
export function Completeness({ series, meta }: { series: Series[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [metric, setMetric] = useSearchParam<Metric>('aeoc', 'ir', ['ir', 'bins', 'cov'])
  const withDays = series.filter((s) => s.days.length)
  const months = useMemo(() => {
    const all = withDays.flatMap((s) => [s.days[0].date, s.days[s.days.length - 1].date]).sort()
    if (!all.length) return []
    const out: string[] = []
    for (let t = new Date(`${monthKey(all[0])}-01T00:00:00Z`); monthKey(t.toISOString()) <= monthKey(all[all.length - 1]); t = new Date(Date.UTC(t.getUTCFullYear(), t.getUTCMonth() + 1, 1))) out.push(monthKey(t.toISOString()))
    return out
  }, [withDays])
  const cells = useMemo(() => withDays.map((s) => {
    const by = d3.group(s.days, (d) => monthKey(d.date))
    return {
      s,
      vals: months.map((m) => {
        const ds = by.get(m) ?? []
        const ir = ds.filter((d) => d.bc.IR !== null).length
        const cov = ds.map((d) => d.coverage).filter((v): v is number => v !== null)
        return { m, bins: ds.length, ir, cov: cov.length ? d3.mean(cov)! : null, days: daysIn(m) }
      }),
      ir: s.days.filter((d) => d.bc.IR !== null).length,
    }
  }), [withDays, months])
  const value = (c: (typeof cells)[number]['vals'][number]) =>
    metric === 'ir' ? (100 * c.ir) / c.days : metric === 'bins' ? (100 * c.bins) / c.days : c.cov
  const iw = Math.max(200, width - M.left - M.right)
  const x = d3.scaleUtc().domain(months.length ? [new Date(`${months[0]}-01T00:00:00Z`), new Date(Date.UTC(+months[months.length - 1].slice(0, 4), +months[months.length - 1].slice(5, 7), 1))] : [new Date(0), new Date(1)]).range([0, iw])
  const color = d3.scaleSequential(d3.interpolateRgbBasis(RAMP_SEQUENTIAL)).domain([0, 100]).clamp(true)
  const h = M.top + cells.length * ROW + M.bottom

  return (
    <ChartFrame id="aeth-overview-completeness" title="Data completeness — how much of each month the daily files cover"
      subtitle={`Each cell is a calendar month at one instrument: ${metric === 'ir' ? 'the days with an IR BCc value, as a % of the days in the month' : metric === 'bins' ? 'the daily bins the legacy file kept, as a % of the days in the month (Beijing, Delhi and JPL keep only days near a filter sample)' : "the mean of the bins' minute coverage (minutes with data, % of the day; the Addis files have no such column, so their row is empty)"}. Right: days with IR BCc / bins in the span. Grey: no bins that month.`}
      provenance="Multi_Site_Analysis.ipynb §4 (data completeness over time) and §7 (days with BC) · processed_sites/README.md · gallery/data/export_aethalometer.py"
      controls={<Segmented label="Show" value={metric} options={['ir', 'bins', 'cov'] as Metric[]} onChange={setMetric} optionLabel={(o) => METRIC_LABEL[o]} />}>
      <div ref={wrapRef} className="chart-wrap">
        {!cells.length ? <Empty>No daily bins in this span.</Empty> : (
          <svg width={width} height={h} fontFamily={FONT.family}>
            <g transform={`translate(${M.left},${M.top})`}>
              {cells.map((row, i) => <g key={row.s.inst.code} transform={`translate(0,${i * ROW})`}>
                <circle cx={-M.left + 10} cy={ROW / 2} r={5} fill={instColor(meta, row.s.inst)} />
                <text x={-M.left + 20} y={ROW / 2} dy="0.32em" fontSize={11.5} fill={INK.text}>{row.s.inst.code === row.s.inst.siteCode ? row.s.inst.site : row.s.inst.info.site}</text>
                {row.vals.map((c) => {
                  const a = x(new Date(`${c.m}-01T00:00:00Z`)), b = x(new Date(Date.UTC(+c.m.slice(0, 4), +c.m.slice(5, 7), 1)))
                  const v = c.bins ? value(c) : null
                  return <HeatCell key={c.m} x={a} y={2} width={b - a - 0.5} height={ROW - 4} fill={v === null ? null : color(v)}
                    onEnter={(e) => tip.show(e, [`${row.s.inst.label} · ${c.m}`, `${c.bins} bins in the file, ${c.ir} with IR BCc, of ${c.days} days`,
                      `minute coverage ${c.cov === null ? 'not in the file' : `${fmt(c.cov, 0)} % (mean of the bins)`}`])} onLeave={tip.hide} />
                })}
                <text x={iw + 6} y={ROW / 2} dy="0.32em" fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>{row.ir}/{row.s.days.length}</text>
              </g>)}
              <XAxis scale={x} y={cells.length * ROW + 2} tickCount={8} format={d3.utcFormat('%b %Y')} />
            </g>
          </svg>
        )}
        <div className="legend"><ColorLegend scale={color} width={160} label={`${METRIC_LABEL[metric]} (%)`} format={(v) => fmt(v, 0)} /></div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
