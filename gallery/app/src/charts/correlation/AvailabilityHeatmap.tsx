import { useMemo, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Segmented } from '@/components/ChartFrame'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useColourScheme } from '@/lib/colourSchemes'
import { SiteMonthMatrix } from '@/components/SiteMonthMatrix'
import { calendarMonths } from '@/lib/notebookAvailability'
import { fidsProps, useFilterEdits } from '@/lib/filterEdits'
import { ExcludedNote } from '@/lib/markedRows'

/**
 * Heatmap — sample coverage by site and month. The matplotlib equivalents are
 * the data-completeness / availability strip charts
 * (notebooks/analysis/data_availability/, timeseries.data_completeness).
 *
 * "both" is the one that matters for the crossplot above: a filter only
 * enters the regression if it carries x *and* y, and the two instruments'
 * coverage windows do not line up.
 */
export function AvailabilityHeatmap({ rows, meta, xField, yField }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string }) {
  const [which, setWhich] = useState<'x' | 'y' | 'both'>('both')
  const colours = useColourScheme('coverage', 'sequential')
  const edits = useFilterEdits()

  const has = (r: FilterRow, f: string) => typeof r[f] === 'number' && Number.isFinite(r[f] as number)
  const label = which === 'x' ? xField : which === 'y' ? yField : `both ${xField} and ${yField}`

  const { byKey, periods, sites, maxCount } = useMemo(() => {
    const sites = meta.sites.map((s) => s.name)
    const byKey = new Map<string, string[]>()
    for (const r of rows) {
      const ok = which === 'x' ? has(r, xField) : which === 'y' ? has(r, yField) : has(r, xField) && has(r, yField)
      if (!ok) continue
      const k = `${r.site}|${r.date.slice(0, 7)}` // YYYY-MM
      const ids = byKey.get(k)
      if (ids) ids.push(r.id)
      else byKey.set(k, [r.id])
    }
    // every month from the first to the last, so a gap in sampling is a run of blank columns
    const present = [...byKey.keys()].map((k) => k.split('|')[1]).sort()
    const periods = present.length ? calendarMonths(present[0], present[present.length - 1]) : []
    return { byKey, periods, sites, maxCount: Math.max(1, ...[...byKey.values()].map((ids) => ids.length)) }
  }, [rows, which, xField, yField, meta.sites])

  const color = d3.scaleQuantize<string>().domain([0, maxCount]).range(colours.stops)

  return (
    <ChartFrame
      id="coverage"
      title="Heatmap — sample coverage by site and month"
      subtitle="How many filters carry the measurement per site per month. Gaps here are the reason several notebooks exist at all: Beijing reaches back to 2013, Delhi only starts mid-2022. 'Both' shows what the crossplot can actually use."
      provenance="stands in for the data-availability strip charts · react-graph-gallery.com/heatmap"
      controls={<>
        <Segmented
          label="Count filters with"
          value={which}
          options={['x', 'y', 'both'] as const}
          optionLabel={(o) => (o === 'both' ? 'Both' : `${o} field`)}
          onChange={setWhich}
          title={`x = ${xField} · y = ${yField}`}
        />
      </>}
    >
        <SiteMonthMatrix rows={sites} months={periods} colLabelPosition="bottom" cellWidth={[1, 1000]} cellHeight={34} gap={[0.6, 4]} radius={1.5}
          rowLabelSize={12} cellLabels="never" color={color} fitRows={[10, 34]}
          cell={(i, month) => {
            const ids = byKey.get(`${sites[i]}|${month}`)
            if (!ids) return undefined
            // a cell is a site's month of filters: click lists them in the side panel, right-click excludes or keeps them
            return { value: ids.length, props: fidsProps(ids, `${sites[i]} · ${month}`),
              tip: [sites[i], month, `${ids.length} filters with ${label}`, 'click to list them · right-click to exclude or keep'] }
          }}
          onCell={(i, j) => {
            const ids = byKey.get(`${sites[i]}|${periods[j]}`)
            if (ids) edits.openGroup({ title: `${sites[i]} · ${periods[j]}`, ids, fields: [xField, yField] })
          }}
          legend={{ scale: color, scheme: colours.scheme, label: 'filters per month', width: 160, format: (v) => String(Math.round(v)), note: `counting ${label}` }}
          footer={<ExcludedNote fields={[xField, yField]} />} />
    </ChartFrame>
  )
}
