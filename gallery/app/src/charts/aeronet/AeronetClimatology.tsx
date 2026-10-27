import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { FONT, INK, inkOn } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { levelLabel, loadDailyAeronet, type AeronetLevel, type DailyAeronet } from '@/lib/aeronet'
import { monthlyMeans } from '@/lib/aeronetMath'

const QUANTITIES = [
  { key: 'aod500', label: 'AOD 500', digits: 2, reverse: false },
  { key: 'aaod675', label: 'AAOD 675', digits: 3, reverse: false },
  { key: 'ssa440', label: 'SSA 440', digits: 2, reverse: true },
  { key: 'fmf', label: 'Fine-mode fraction', digits: 2, reverse: false },
  { key: 'ae', label: 'Ångström exponent 440–870', digits: 2, reverse: false },
  { key: 'pw', label: 'Precipitable water (cm)', digits: 1, reverse: false },
] as const
type QKey = (typeof QUANTITIES)[number]['key']
const MONTHS = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split(' ')
/** a month with fewer days than this is drawn faint: its mean rests on a handful of clear days */
const FEW = 5

/**
 * AERONET's own climatology at each picked site: the mean of its daily values by calendar month, over every
 * day the station has in the export (the filter period, padded), site by month. Daily means of every
 * retrieval, not of the filter windows: the column's seasonal cycle, against which the filters' seasons read.
 */
export function AeronetClimatology({ sites, level }: { sites: { name: string; code: string }[]; level: AeronetLevel }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [qKey, setQ] = useSearchParam<QKey>('aeronet.clim', 'aod500', QUANTITIES.map((q) => q.key))
  const q = QUANTITIES.find((x) => x.key === qKey) ?? QUANTITIES[0]
  const [data, setData] = useState<Record<string, DailyAeronet | null>>({})
  const key = `${level}:${sites.map((s) => s.code).join(',')}`
  useEffect(() => {
    let live = true
    Promise.all(sites.map((s) => loadDailyAeronet(s.code, level).then((d) => [s.code, d] as const).catch(() => [s.code, null] as const)))
      .then((pairs) => { if (live) setData(Object.fromEntries(pairs)) })
    return () => { live = false }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps

  const grid = useMemo(() => sites.map((s) => {
    const d = data[s.code]
    return { site: s, months: d ? monthlyMeans(d.date, d[q.key]) : null }
  }).filter((g) => g.months && g.months.some((m) => m.mean !== null)), [data, sites, q])
  const values = grid.flatMap((g) => g.months!.map((m) => m.mean)).filter((v): v is number => v !== null)
  const [lo, hi] = d3.extent(values) as [number, number]
  const color = d3.scaleSequential(d3.interpolateYlOrBr).domain(q.reverse ? [hi, lo] : [lo, hi])

  const labelW = Math.min(150, Math.max(90, width * 0.22)), cellH = 22, head = 22
  const cellW = Math.max(18, Math.min(56, (width - labelW - 8) / 12))
  const loading = sites.length > 0 && Object.keys(data).length === 0

  return (
    <ChartFrame id="aeronet-climatology" title={`AERONET monthly climatology — ${q.label}, site by month`}
      subtitle={`Mean of AERONET's ${levelLabel(level)} daily ${q.label} by calendar month at each picked site's station, over every day in the export (the site's filter period). Faint cells rest on fewer than ${FEW} days. ${q.reverse ? 'Darker is lower SSA, i.e. more absorbing.' : 'Darker is higher.'}`}
      provenance="addis_04_aeronet_daily (monthly cycle) for every SPARTAN site with a station · NASA AERONET Version 3 daily means"
      controls={<Select label="Quantity" value={q.key} options={QUANTITIES.map((x) => x.key)} optionLabel={(k) => QUANTITIES.find((x) => x.key === k)?.label ?? k} onChange={(v) => setQ(v as QKey)} />}>
      <div ref={wrapRef} className="chart-wrap">
        {loading ? <Empty>Loading the daily AERONET…</Empty> : grid.length === 0 ? <Empty>No picked site has daily {q.label} in this export.</Empty> : (
          <svg width={labelW + cellW * 12 + 4} height={head + grid.length * cellH + 4} fontFamily={FONT.family}>
            {MONTHS.map((m, i) => <text key={m} x={labelW + i * cellW + cellW / 2} y={14} textAnchor="middle" fontSize={10.5} fill={INK.muted}>{cellW < 28 ? m[0] : m}</text>)}
            {grid.map((g, j) => (
              <g key={g.site.code} transform={`translate(0,${head + j * cellH})`}>
                <text x={labelW - 6} y={cellH / 2} dy="0.32em" textAnchor="end" fontSize={11.5} fill={INK.text}>{g.site.name.length > 22 ? `${g.site.name.slice(0, 21)}…` : g.site.name}</text>
                {g.months!.map((m, i) => {
                  const fill = m.mean === null ? INK.empty : color(m.mean)
                  return <g key={i} transform={`translate(${labelW + i * cellW},0)`}
                    onMouseEnter={(e) => tip.show(e, [`${g.site.name} · ${MONTHS[i]}`, m.mean === null ? 'no days' : `${q.label}: ${fmt(m.mean, q.digits + 1)}`, `${m.n} days`])} onMouseLeave={tip.hide}>
                    <rect width={cellW - 1} height={cellH - 1} fill={fill} fillOpacity={m.mean !== null && m.n < FEW ? 0.35 : 1} />
                    {m.mean !== null && cellW >= 34 && <text x={(cellW - 1) / 2} y={cellH / 2} dy="0.32em" textAnchor="middle" fontSize={9.5} fontFamily={FONT.mono}
                      fill={m.n < FEW ? INK.muted : inkOn(fill)}>{fmt(m.mean, q.digits)}</text>}
                  </g>
                })}
              </g>
            ))}
          </svg>
        )}
        {values.length > 0 && <div className="legend">
          <span className="legend-item">{fmt(q.reverse ? hi : lo, q.digits)}</span>
          <span className="legend-item" style={{ width: 120, height: 10, background: `linear-gradient(to right, ${d3.range(0, 1.01, 0.25).map((t) => d3.interpolateYlOrBr(t)).join(',')})` }} />
          <span className="legend-item">{fmt(q.reverse ? lo : hi, q.digits)}</span>
        </div>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
