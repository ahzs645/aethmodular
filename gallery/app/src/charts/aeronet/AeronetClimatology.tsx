import { useEffect, useMemo, useState } from 'react'
import { useColourScheme } from '@/lib/colourSchemes'
import { SiteMonthMatrix } from '@/components/SiteMonthMatrix'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { fmt } from '@/lib/stats'
import { INK } from '@/lib/theme'
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
const MONTH_NUMBERS = MONTHS.map((_, i) => i + 1)
/** a month with fewer days than this is drawn faint: its mean rests on a handful of clear days */
const FEW = 5

/**
 * AERONET's own climatology at each picked site: the mean of its daily values by calendar month, over every
 * day the station has in the export (the filter period, padded), site by month. Daily means of every
 * retrieval, not of the filter windows: the column's seasonal cycle, against which the filters' seasons read.
 */
export function AeronetClimatology({ sites, level }: { sites: { name: string; code: string }[]; level: AeronetLevel }) {
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
  const colours = useColourScheme('aeronet-climatology', 'sequential', 'Oranges')
  const color = d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain(q.reverse ? [hi, lo] : [lo, hi])

  const loading = sites.length > 0 && Object.keys(data).length === 0
  const rowItems = useMemo(() => grid.map((g) => ({ key: g.site.code, label: g.site.name, title: g.site.name })), [grid])

  return (
    <ChartFrame id="aeronet-climatology" title={`AERONET monthly climatology — ${q.label}, site by month`}
      subtitle={`Mean of AERONET's ${levelLabel(level)} daily ${q.label} by calendar month at each picked site's station, over every day in the export (the site's filter period). Faint cells rest on fewer than ${FEW} days. ${q.reverse ? 'Darker is lower SSA, i.e. more absorbing.' : 'Darker is higher.'}`}
      provenance="addis_04_aeronet_daily (monthly cycle) for every SPARTAN site with a station · NASA AERONET Version 3 daily means"
      controls={<Select label="Quantity" value={q.key} options={QUANTITIES.map((x) => x.key)} optionLabel={(k) => QUANTITIES.find((x) => x.key === k)?.label ?? k} onChange={(v) => setQ(v as QKey)} />}>
      {loading ? <Empty>Loading the daily AERONET…</Empty> : grid.length === 0 ? <Empty>No picked site has daily {q.label} in this export.</Empty> : (
        <SiteMonthMatrix rows={rowItems} months={MONTH_NUMBERS} gap={1} cellHeight={22} cellWidth={[18, 56]} maxLabelWidth={150} rowLabelSize={11.5} labelSize={9.5} colLabelColor={INK.muted}
          color={color} format={(v) => fmt(v, q.digits)}
          cell={(i, _m, j) => {
            const g = grid[i], m = g.months![j], faint = m.mean !== null && m.n < FEW
            return { value: m.mean, opacity: faint ? 0.35 : undefined, ink: faint ? INK.muted : undefined,
              tip: [`${g.site.name} · ${MONTHS[j]}`, m.mean === null ? 'no days' : `${q.label}: ${fmt(m.mean, q.digits + 1)}`, `${m.n} days`] }
          }}
          legend={values.length > 0 ? { scale: d3.scaleSequential(d3.interpolateRgbBasis(colours.stops)).domain([lo, hi]), scheme: colours.scheme,
            label: `${q.label}, monthly mean${q.reverse ? ' (darker = lower)' : ''}`, format: (v) => fmt(v, q.digits) } : null} />
      )}
    </ChartFrame>
  )
}
