import { useMemo } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import type { MatrixItem } from '@/components/HeatMatrix'
import { SiteMonthMatrix } from '@/components/SiteMonthMatrix'
import { fmt } from '@/lib/format'
import { calendarMonths } from '@/lib/notebookAvailability'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { instColor, type Series } from './data'

type Metric = 'ir' | 'bins' | 'cov'
const METRIC_LABEL: Record<Metric, string> = { ir: 'Days with IR BCc', bins: 'Bins in the file', cov: 'Minute coverage' }
const ROW = 26

const monthKey = (date: string) => date.slice(0, 7)
const daysIn = (key: string) => new Date(Date.UTC(+key.slice(0, 4), +key.slice(5, 7), 0)).getUTCDate()

/**
 * How much of each month the daily files cover, per instrument (Multi_Site_Analysis §4 drew the
 * minute coverage day by day; §7 tabulated days with BC). The legacy files keep only some days
 * (Beijing, Delhi and JPL: those near a filter sample), so "bins in the file" is part of the story.
 */
export function Completeness({ series, meta }: { series: Series[]; meta: MetaFile }) {
  const [metric, setMetric] = useSearchParam<Metric>('aeoc', 'ir', ['ir', 'bins', 'cov'])
  const withDays = series.filter((s) => s.days.length)
  const months = useMemo(() => {
    const all = withDays.flatMap((s) => [s.days[0].date, s.days[s.days.length - 1].date]).sort()
    return all.length ? calendarMonths(monthKey(all[0]), monthKey(all[all.length - 1])) : []
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
  const colours = useColourScheme('aeth-overview-completeness', 'sequential')
  const color = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([0, 100]).clamp(true)
  const items = useMemo<MatrixItem[]>(() => cells.map((row) => ({ key: row.s.inst.code, dot: instColor(meta, row.s.inst),
    label: row.s.inst.code === row.s.inst.siteCode ? row.s.inst.site : row.s.inst.info.site })), [cells, meta])

  return (
    <ChartFrame id="aeth-overview-completeness" title="Data completeness — how much of each month the daily files cover"
      subtitle={`Each cell is a calendar month at one instrument: ${metric === 'ir' ? 'the days with an IR BCc value, as a % of the days in the month' : metric === 'bins' ? 'the daily bins the legacy file kept, as a % of the days in the month (Beijing, Delhi and JPL keep only days near a filter sample)' : "the mean of the bins' minute coverage (minutes with data, % of the day; the Addis files have no such column, so their row is empty)"}. Right: days with IR BCc / bins in the span. Grey: no bins that month.`}
      provenance="Multi_Site_Analysis.ipynb §4 (data completeness over time) and §7 (days with BC) · processed_sites/README.md · gallery/data/export_aethalometer.py"
      controls={<Segmented label="Show" value={metric} options={['ir', 'bins', 'cov'] as Metric[]} onChange={setMetric} optionLabel={(o) => METRIC_LABEL[o]} />}>
      {!cells.length ? <Empty>No daily bins in this span.</Empty> : (
        <SiteMonthMatrix rows={items} months={months} colLabelPosition="bottom" cellWidth={[1, 1000]} cellHeight={ROW} gap={[0.5, 4]} rowLabelSize={11.5}
          cellLabels="never" color={color} maxLabelWidth={130}
          extraColumns={[{ key: 'ir', side: 'right', text: (i) => `${cells[i].ir}/${cells[i].s.days.length}` }]}
          cell={(i, _m, j) => {
            const row = cells[i], c = row.vals[j]
            return { value: c.bins ? value(c) : null, tip: [`${row.s.inst.label} · ${c.m}`, `${c.bins} bins in the file, ${c.ir} with IR BCc, of ${c.days} days`,
              `minute coverage ${c.cov === null ? 'not in the file' : `${fmt(c.cov, 0)} % (mean of the bins)`}`] }
          }}
          legend={{ scale: color, scheme: colours.scheme, width: 160, label: `${METRIC_LABEL[metric]} (%)`, format: (v) => fmt(v, 0) }} />
      )}
    </ChartFrame>
  )
}
