import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { useChartOptions } from '@/lib/chartOptions'
import { IntervalStrip, type IntervalBlock, type IntervalRow } from '@/components/IntervalStrip'
import { fmt } from '@/lib/stats'
import { seasonOnly } from '@/lib/seasons'
import { boxStats, quantileSorted, spearman } from '@/lib/stats'
import { useNarrowScreen } from '@/hooks/useMediaQuery'
import { eta, maiaLine, useAeronetSites } from '@/lib/aeronetSites'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonColor, siteColor } from '@/lib/siteColours'

const MIN_N = 5
const VIEWS = ['Site', 'Site × season'] as const

interface Group { key: string; label: string; site: string; code: string; color: string; etas: number[]; pm: number[]; aod: number[]; strong?: boolean }

/**
 * η = PM2.5 / AOD 500 by site and season: the conversion from column AOD to surface PM2.5 that every
 * satellite PM2.5 retrieval (MAIA's among them) has to model. Per group, the median and interquartile
 * range of the filters' own ratios (filter PM2.5 over the mean AOD 500 inside its sampling window), and
 * beside it how closely PM2.5 follows AOD there (Spearman ρ): a site whose PM2.5 does not move with
 * AOD has no useful η, whatever its median. Sites in a MAIA target area are marked ◎.
 */
export function EtaBySite({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const summary = useAeronetSites().data
  const nSites = useMemo(() => new Set(rows.map((r) => r.site)).size, [rows])
  // one site has only its seasons to show; old links' `etaview` still opens the same rows
  const o = useChartOptions('aeronet-eta', [{ key: 'view', label: 'Rows', choices: VIEWS, default: nSites <= 6 ? 'Site × season' : 'Site',
    hidden: 'Site × season', when: (c) => (c.sites as number) > 1, legacy: ['etaview'] }], { sites: nSites })
  const view = o.values.view as (typeof VIEWS)[number]
  // a phone: shorter labels' room and narrower panels, so both panels fit without clipping
  const narrow = useNarrowScreen()

  const blocks = useMemo<IntervalBlock[]>(() => {
    const order = meta.seasons.map((s) => s.name)
    const group = (key: string, label: string, rs: FilterRow[], color: string, strong = false): Group | null => {
      const pairs = rs.map((r) => ({ e: eta(r), r })).filter((p): p is { e: number; r: FilterRow } => p.e !== null)
      if (pairs.length < MIN_N) return null
      return { key, label, site: rs[0].site, code: rs[0].code, color, strong, etas: pairs.map((p) => p.e).sort(d3.ascending),
        pm: pairs.map((p) => p.r['PM2.5 mass'] as number), aod: pairs.map((p) => p.r['AOD 500 nm (AERONET)'] as number) }
    }
    const toRow = (g: Group): IntervalRow => {
      const box = boxStats(g.etas, { minN: 1 }), mid = box?.median ?? null, lo = box?.q1 ?? null, hi = box?.q3 ?? null
      const rho = spearman(g.pm, g.aod, { minN: 5 })
      const maia = maiaLine(summary?.maia[g.code])
      return {
        key: g.key, label: g.label, strong: g.strong, note: `n=${g.etas.length}`,
        marks: [[{ mid, lo, hi, color: g.color }], [{ mid: rho, color: g.color, shape: rho !== null && rho < 0.3 ? 'hollow' : 'dot' }]],
        tip: [g.key === g.site ? g.site : `${g.site} · ${g.label}`,
          `median η ${fmt(mid, 0)} µg/m³ per unit AOD (IQR ${fmt(lo, 0)}–${fmt(hi, 0)})`,
          `PM2.5 against AOD 500: Spearman ρ ${fmt(rho, 2)}`,
          `${g.etas.length} filters with PM2.5 and an AOD 500 in the window`, ...(maia ? [maia] : [])],
      }
    }
    const bySite = d3.groups(rows, (r) => r.site)
    const mark = (site: string, code: string) => (summary?.maia[code] ? `${site} ◎` : site)
    if (view === 'Site') {
      const groups = bySite.map(([site, rs]) => group(site, mark(site, rs[0].code), rs, siteColor(meta, site))).filter((g): g is Group => g !== null)
      groups.sort((a, b) => (quantileSorted(a.etas, 0.5) || 0) - (quantileSorted(b.etas, 0.5) || 0))
      return [{ key: 'sites', rows: groups.map(toRow) }]
    }
    return bySite.map(([site, rs]) => {
      const all = group(site, 'All seasons', rs, siteColor(meta, site), true)
      const seasons = d3.groups(rs, (r) => String(r.season ?? '')).filter(([s]) => s)
        .sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]))
        .map(([s, srs]) => group(`${site}|${s}`, seasonOnly(s), srs, seasonColor(meta, s)))
        .filter((g): g is Group => g !== null)
      return { key: site, title: nSites > 1 ? mark(site, rs[0]?.code ?? '') : undefined, color: siteColor(meta, site), rows: [all, ...seasons].filter((g): g is Group => g !== null).map(toRow) }
    }).filter((b) => b.rows.length > 0).sort((a, b) => a.key.localeCompare(b.key))
  }, [rows, meta, view, summary, nSites])

  const any = blocks.some((b) => b.rows.length > 0)
  const maiaHere = [...new Set(rows.map((r) => r.code))].filter((c) => summary?.maia[c])
  return (
    <ChartFrame id="aeronet-eta" source="gallery/app/src/charts/aeronet/EtaBySite.tsx"
      title={`PM2.5 / AOD by ${view === 'Site' ? 'site' : nSites === 1 ? 'season' : 'site and season'} — η, the conversion from column AOD to surface PM2.5`}
      subtitle={`η = filter PM2.5 ÷ AOD 500 (the mean of the AERONET retrievals inside the filter's sampling window), µg/m³ per unit AOD: median and interquartile range, at least ${MIN_N} filters each. Satellite PM2.5 retrievals, MAIA's included, turn AOD into surface PM2.5 through this ratio, so a site or season where it moves is where they have to model it. Right: how closely PM2.5 follows AOD (Spearman ρ; hollow below 0.3, where η says little). AOD is a daylight, clear-sky mean; PM2.5 covers the whole window.`}
      provenance={`AERONET Version 3 ${summary ? `Level ${summary.level}` : ''} (export_aeronet.py) over SPARTAN filter PM2.5; the same ratio per site and season, all filters, in aeronet/sites_summary.json (export_aeronet_sites.py). ◎: a SPARTAN site inside a MAIA target area (research/maia/inventory/spartan_sites.csv).`}
      controls={o.controls()}>
      {any ? <IntervalStrip blocks={blocks} labelWidth={narrow ? 130 : view === 'Site' ? 190 : 210} minPanelWidth={narrow ? 150 : 260} noteWidth={narrow ? 42 : 50} rowHeight={view === 'Site' ? 22 : 20}
        panels={[
          { label: 'η = PM2.5 / AOD 500 (µg/m³ per unit AOD)', heading: 'Conversion η', format: (v) => fmt(v, 0) },
          { label: 'Spearman ρ, PM2.5 against AOD 500', heading: 'Does PM2.5 follow AOD?', reference: 0, domain: [-1, 1], stems: false, format: (v) => fmt(v, 1) },
        ]} />
        : <Empty>No site or season in view has {MIN_N} filters with both a PM2.5 mass and an AERONET AOD 500 inside the window.</Empty>}
      {maiaHere.length > 0 && <Note>◎ In a MAIA target area: {maiaHere.map((c) => `${meta.sites.find((s) => s.code === c)?.name ?? c} (${summary!.maia[c].target}, ${summary!.maia[c].type})`).join(' · ')}.</Note>}
    </ChartFrame>
  )
}
