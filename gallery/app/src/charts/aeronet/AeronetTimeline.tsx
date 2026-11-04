import { useCallback, useMemo } from 'react'
import { LoadState } from '@/components/ExportState'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { useHighlightActions } from '@/lib/highlight'
import { fmt } from '@/lib/stats'
import { withUnit, VARIABLE_COLOR } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { loadDailyAeronet, L2_AOD440, type AeronetLevel } from '@/lib/aeronet'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { ExcludedNote } from '@/lib/markedRows'
import { Key } from '@/components/Key'
import { DailyLanes, FieldLane, dailyDomain, dailyLine, dailyReadout, fieldDomain, fieldPoints, useSiteDaily } from '@/components/DailyLanes'
import type { StackPanel } from '@/components/StackedPanels'

/**
 * The column over one site's filters: AERONET's daily AOD at 500 nm and its fine-mode fraction,
 * the inversion's absorption AOD at 675 nm and single-scattering albedo at 440 nm (open circles
 * where the inversion's AOD 440 is under 0.4, AERONET's Level 2 threshold for absorption), above
 * the chosen measurement for each filter on the same axis, with the site's seasons shaded.
 * The lanes, cursor and season shading are components/DailyLanes (shared with the weather timeline).
 */
export function AeronetTimeline({ rows, meta, field, site, station, level = '15' }: {
  rows: FilterRow[]; meta: MetaFile; field: string; site: { name: string; code: string } | null; station?: string; level?: AeronetLevel
}) {
  // the brush only: the filter points read the hover themselves (components/ScatterLayer)
  const hl = useHighlightActions()
  const load = useCallback((code: string) => loadDailyAeronet(code, level), [level])
  const { daily, error, domain, days } = useSiteDaily(rows, site, load, (name) => `No daily AERONET exported for ${name}.`)
  const pts = useMemo(() => fieldPoints(rows, site?.name, field), [rows, site, field])
  const seasons = useMemo(() => (site ? seasonsForSite(meta, site.name) : []), [meta, site])
  const val = (arr: (number | null)[] | undefined, i: number) => arr?.[i] ?? null
  // the inversion's own AOD 440 is not in the daily series; the daily AOD 440 stands in for the threshold
  const weak = (i: number) => (val(daily?.aod440, i) ?? 0) < L2_AOD440
  const dots = (arr: (number | null)[], c: string) => ({ x, y }: { x: d3.ScaleTime<number, number>; y: d3.ScaleLinear<number, number> }) => days.map(({ t, i }) => {
    const v = val(arr, i)
    return v === null ? null : <circle key={i} cx={x(t)} cy={y(v)} r={2.2} fill={weak(i) ? 'none' : c} stroke={c} strokeWidth={0.8} opacity={0.8} />
  })
  const lanes: StackPanel<d3.ScaleTime<number, number>>[] = !daily ? [] : [
    { key: 'aod', y: { domain: dailyDomain(days, [daily.aod500], { lo: 0 }), nice: true, label: 'AOD 500', labelX: -50 },
      children: ({ x, y }) => <path d={dailyLine(days, daily.aod500, x, y)} fill="none" stroke={VARIABLE_COLOR.aod} strokeWidth={1.1} /> },
    { key: 'fmf', y: { domain: [0, 1], label: 'Fine fraction', labelX: -50 },
      children: ({ x, y }) => <path d={dailyLine(days, daily.fmf, x, y)} fill="none" stroke={VARIABLE_COLOR.fineModeFraction} strokeWidth={1.1} /> },
    { key: 'aaod', y: { domain: dailyDomain(days, [daily.aaod675], { lo: 0 }), nice: true, label: 'AAOD 675', labelX: -50 }, children: dots(daily.aaod675, '#8c2d04') },
    { key: 'ssa', y: { domain: dailyDomain(days, [daily.ssa440], { hi: 1 }), nice: true, label: 'SSA 440', labelX: -50 }, children: dots(daily.ssa440, '#6a51a3') },
    { key: 'field', y: { domain: fieldDomain(pts), nice: true, label: withUnit(field, meta.field_units), labelX: -50, pick: 'field' }, pointer: false,
      children: ({ x, y, width, height, tip }) => <FieldLane pts={pts} field={field} site={site?.name} x={x} y={y} width={width} height={height}
        seasons={meta.seasons} selected={hl.selected} tip={tip}
        tipOf={(p) => [p.row.id, `${p.row.date} · ${seasonOnly(String(p.row.season ?? ''))}`, `${field}: ${fmt(p.v)}`, 'click for details · right-click to exclude']} /> },
  ]
  const readout = dailyReadout(days, (d) => !daily ? [] : [d3.utcFormat('%d %b %Y')(d.t), `AOD 500 ${fmt(daily.aod500[d.i], 3)} · AE ${fmt(daily.ae[d.i], 2)} · fine fraction ${fmt(daily.fmf[d.i], 2)}`,
    `AAOD 675 ${fmt(daily.aaod675[d.i], 4)} · AAE ${fmt(daily.aae[d.i], 2)} · SSA 440 ${fmt(daily.ssa440[d.i], 3)}`, `precipitable water ${fmt(daily.pw[d.i], 2)} cm`])

  return (
    <ChartFrame id="aeronet-timeline" title={`${site ? `${site.name} · ` : ''}AERONET timeline`}
      subtitle={`AERONET's ${level === '20' ? 'Level 2.0' : 'Level 1.5'} daily means${station ? ` at ${station}` : ''}: AOD at 500 nm and its fine-mode fraction (direct sun and SDA), absorption AOD at 675 nm and single-scattering albedo at 440 nm (almucantar inversions; open circles on days with AOD 440 under ${L2_AOD440}, where absorption is uncertain), above the chosen measurement for each filter. Seasons are the site's local calendar.`}
      provenance="addis_04_aeronet and aeronet_addis_diagnostics generalised to every SPARTAN site · NASA AERONET Version 3">
      <div className="chart-wrap">
        {error ? <Empty>{error}</Empty> : !site || !domain ? <Empty>No filters at this site in the current subset.</Empty> : !daily ? <LoadState what="the daily AERONET" /> : (
          <DailyLanes domain={domain} lanes={lanes} seasons={seasons} readout={readout} />
        )}
        <Key items={[...seasons.map((s) => ({ label: s.name.replace(`${site?.name} · `, ''), color: s.color, kind: 'faded' as const })),
          { label: `inversion, AOD 440 ≥ ${L2_AOD440}`, color: '#8c2d04' }, { label: `AOD 440 < ${L2_AOD440}`, color: '#8c2d04', kind: 'hollow' as const }]} />
        <ExcludedNote fields={[field]} of={(r) => r.site === site?.name} drawn />
      </div>
    </ChartFrame>
  )
}
