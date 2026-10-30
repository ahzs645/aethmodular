import { useEffect, useMemo, useRef, useState } from 'react'
import { MapMarker, MarkerContent, type LngLatPoint } from '@pgmaps/geo-toolkit/map'
import { GeoMap, type GeoMapHandle } from '@/components/GeoMap'
import { SitePicker } from '@/components/SitePicker'
import { loadJson } from '@/lib/data'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { SizeLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { maiaLine, useAeronetSites, type AeronetSitesFile } from '@/lib/aeronetSites'

const STATS = ['Median', 'Mean', 'Sample count'] as const
const PROJECTIONS = ['Flat', 'Globe'] as const
/** the contiguous US, where IMPROVE opens (Fit sites takes in Alaska, Hawaii and overseas sites) */
const US_MAINLAND: LngLatPoint[] = [[-125, 24], [-66, 50]]
// Selected sites: the sites picked in the Filters panel (the same pick as the chart tabs)
const SCOPES = ['Selected sites', 'All SPARTAN sites', 'All IMPROVE sites'] as const
// old links: the four focus sites (and a single one) are now the Filters panel's pick, which starts on them
const LEGACY_SCOPES = [...SCOPES, 'Four focus sites', 'All study sites', 'Focus site', 'Single focus site'] as const
interface Summary { n: number; median: number | null; mean: number | null }
interface NetworkMap {
  note: string
  field_units: Record<string, string>
  sites: { code: string; name: string; location: string; lat: number | null; lon: number | null;
    n_filters: number; date_min: string; date_max: string; stats: Record<string, Summary> }[]
}

/**
 * The SPARTAN inventory with the AERONET quantities joined on (aeronet/sites_summary.json,
 * export_aeronet_sites.py): each one's median, mean and n over the site's filter windows, so the
 * SPARTAN measurement picker offers them beside the public chemistry. Sites without a station stay
 * on the map as hollow markers.
 */
function withAeronet(map: NetworkMap, aeronet: AeronetSitesFile | null): NetworkMap {
  if (!aeronet) return map
  const units = Object.fromEntries(aeronet.quantities.map((q) => [q.label, q.unit]))
  return {
    ...map,
    field_units: { ...map.field_units, ...units },
    sites: map.sites.map((s) => ({ ...s, stats: { ...s.stats, ...Object.fromEntries(aeronet.quantities.map((q) => {
      const v = aeronet.sites[s.code]?.all[q.key]
      return [q.label, { n: v?.n ?? 0, median: v?.median ?? null, mean: v?.mean ?? null }]
    })) } })),
  }
}

/**
 * Offline bubble map. Selected sites: the Filters panel's pick (`meta.sites`, summarised from `rows`, so
 * its dates, seasons and exclusions apply); All SPARTAN / All IMPROVE: every network site's exported summary.
 */
export function SiteBubbleMap({ rows, meta, field, onSite }: {
  rows: FilterRow[]; meta: MetaFile; field: string
  /** a click on a site's bubble (the Dashboard opens that site's own page) */
  onSite?: (code: string) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const tip = useTooltip(wrapRef)
  const fitted = useFitHeight()
  const height = fitted ?? 400
  const geo = useRef<GeoMapHandle>(null)
  const [projection, setProjection] = useSearchParam<(typeof PROJECTIONS)[number]>('mapproj', 'Flat', PROJECTIONS)
  const [stat, setStat] = useSearchParam<(typeof STATS)[number]>('mapstat', 'Median', STATS)
  const [savedScope, setScope] = useSearchParam<(typeof LEGACY_SCOPES)[number]>('mapscope', 'Selected sites', LEGACY_SCOPES)
  const scope: (typeof SCOPES)[number] = (SCOPES as readonly string[]).includes(savedScope) ? savedScope as (typeof SCOPES)[number] : 'Selected sites'
  const [improveField, setImproveField] = useSearchParam<string>('mapfield', 'EC (TOR)')
  const [improve, setImprove] = useState<NetworkMap | null>(null)
  const [spartanBase, setSpartan] = useState<NetworkMap | null>(null)
  // the AERONET column quantities per site, and the MAIA target areas (both optional)
  const aeronet = useAeronetSites().data
  const spartan = useMemo(() => (spartanBase ? withAeronet(spartanBase, aeronet) : null), [spartanBase, aeronet])
  const [spartanField, setSpartanField] = useSearchParam<string>('mapspartanfield', 'PM2.5 mass')
  const [error, setError] = useState('')
  const lg = useLegend()
  const isImprove = scope === 'All IMPROVE sites'
  const isSpartan = scope === 'All SPARTAN sites'
  const isNetwork = isImprove || isSpartan
  const network = isImprove ? improve : isSpartan ? spartan : null
  const selectedField = isImprove ? improveField : spartanField
  const metric = isNetwork ? (network && selectedField in network.field_units ? selectedField : isImprove ? 'EC (TOR)' : 'PM2.5 mass') : field
  const units = isNetwork ? network?.field_units ?? {} : meta.field_units
  // an AERONET quantity picked in the SPARTAN scope: its tooltip adds the station, ρ and the seasons
  const aeronetQ = isSpartan ? aeronet?.quantities.find((q) => q.label === metric) : undefined
  const fmtQ = (v: number | null | undefined) => fmt(v, aeronetQ?.digits)
  const unlocated = network?.sites.filter(s => s.lat == null || s.lon == null) ?? []

  useEffect(() => {
    if (!isNetwork || network) return
    let live = true
    setError('')
    loadJson<any>(`${isImprove ? 'improve' : 'spartan'}_map.json`)
      .then((d) => { if (live) (isImprove ? setImprove : setSpartan)(d) })
      .catch(e => { if (live) setError(String(e)) })
    return () => { live = false }
  }, [isImprove, isNetwork, network])

  const sites = useMemo(() => {
    const bySite = d3.group(rows, r => r.site)
    const source = isNetwork
      ? (network?.sites ?? []).map(s => ({ ...s, label: s.code,
          color: isSpartan ? meta.sites.find(f => f.code === s.code)?.color ?? INK.accent : INK.accent,
          summary: s.stats[metric] ?? { n: 0, mean: null, median: null } }))
      : meta.sites.map(s => {
          const subset = bySite.get(s.name) ?? []
          const vals = subset.map(r => r[metric]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
          return { ...s, n_filters: subset.length, label: s.name,
            date_min: d3.min(subset, r => r.date) ?? '—', date_max: d3.max(subset, r => r.date) ?? '—',
            summary: { n: vals.length, mean: d3.mean(vals) ?? null, median: d3.median(vals) ?? null } }
        })
    return source.map(s => ({ ...s, value: stat === 'Sample count' ? s.n_filters : s.summary[stat === 'Mean' ? 'mean' : 'median'] }))
      .filter((s): s is typeof s & { lat: number; lon: number } =>
        s.lat != null && s.lon != null && Number.isFinite(s.lat) && Number.isFinite(s.lon)
        && (isNetwork || (s.value != null && Number.isFinite(s.value) && s.n_filters > 0)))
  }, [rows, meta.sites, metric, stat, scope, isNetwork, isSpartan, network])
  // the site list beside the map picks which sites are drawn (the legend's hidden set); a dashboard
  // tile, too narrow for the list, keeps the legend
  const sidePanel = fitted === null
  const drawnSites = sites.filter((s) => lg.show(s.label))

  // the camera's home: IMPROVE opens on the US mainland, a fitted dashboard tile on the sites in view
  // (at least 40° across, so one site is a region rather than a street), the rest on all their sites
  const scopeSites = useMemo(() => sites.map((s) => [s.lon, s.lat] as const), [sites])
  const home = isImprove ? US_MAINLAND : scopeSites
  const homeOptions = useMemo(() => ({ pad: 0.1, minSpan: isImprove ? 4 : fitted !== null ? 40 : 60, maxZoom: 6 }), [isImprove, fitted])
  // sized to the largest value in view, however small (AAOD tops out near 0.05), not to at least 1
  const vmax = d3.max(sites, s => s.value) ?? 0
  const rScale = d3.scaleSqrt().domain([0, vmax > 0 ? vmax : 1]).range([0, isNetwork || sites.length > 8 ? 16 : 26])
  const fitSites = () => geo.current?.fit(drawnSites.map((s) => [s.lon, s.lat]), { pad: 0.15, minSpan: 6, maxZoom: 6 })

  return (
    <ChartFrame id="map" source="gallery/app/src/charts/map/SiteBubbleMap.tsx"
      title={`Bubble map — ${drawnSites.length === 1 ? drawnSites[0].name : scope.toLowerCase()}`}
      subtitle={`${stat === 'Sample count' ? 'Sample counts' : `${stat} ${withUnit(metric, units)}`} at ${isNetwork ? `the available ${isImprove ? 'IMPROVE' : 'SPARTAN'} sites` : `the ${meta.sites.length === 1 ? 'site' : `${meta.sites.length} sites`} picked in the Filters panel`}; pick or search sites in the list beside the map. Drag to pan; scroll or use +/− to zoom (Globe: drag with the right button to turn). Bubble sizes stay fixed while zooming. Selected sites follows the Filters panel's sites, seasons, dates and exclusions; the All SPARTAN and All IMPROVE views summarise every filter.`}
      provenance={isNetwork && network ? `${network.sites.length} sites in the inventory.${unlocated.length ? ` Missing coordinates: ${unlocated.map(s => s.code).join(', ')}.` : ''} ${aeronetQ && aeronet ? `${aeronet.source}, Level ${aeronet.level}: ${aeronet.note} (aeronet/sites_summary.json, export_aeronet_sites.py)` : network.note}${isSpartan ? ' MAIA target areas: research/maia/inventory/spartan_sites.csv.' : ''}` : `Local site inventory${aeronet ? '. MAIA target areas: research/maia/inventory/spartan_sites.csv' : ''}`}
      controls={<>
        <Segmented label="Sites" value={scope} options={SCOPES} onChange={setScope} />
        {isNetwork && network && stat !== 'Sample count' && <Select label={`${isImprove ? 'IMPROVE' : 'SPARTAN'} measurement`} value={metric} options={Object.keys(network.field_units)} onChange={isImprove ? setImproveField : setSpartanField} />}
        <Segmented label="Size by" value={stat} options={STATS} onChange={setStat} />
        <Segmented label="Projection" value={projection} options={PROJECTIONS} onChange={setProjection} />
        <span className="control">View
          <button className="btn quiet" type="button" onClick={() => geo.current?.fit(home, homeOptions)}>{isImprove ? 'US mainland' : 'Reset view'}</button>
          <button className="btn quiet" type="button" title="Fit the sites ticked in the list" onClick={fitSites}>Fit sites</button>
        </span>
      </>}
    >
      <div className={sidePanel ? 'map-with-sites' : undefined}>
      <div ref={wrapRef} className="chart-wrap">
      {isNetwork && !network ? <Empty>{error ? <>Could not load site summaries ({error}). Run <code>uv run python gallery/data/export_{isImprove ? 'improve' : 'spartan'}_map.py</code> and reopen the map.</> : 'Loading network sites…'}</Empty> : <>
        <GeoMap ref={geo} height={height} globe={projection === 'Globe'} frame={home} frameOptions={homeOptions}
          frameKey={`${scope}|${fitted === null ? 'page' : 'tile'}|${network ? 'loaded' : ''}`} label={`Bubble map: ${scope}, ${stat.toLowerCase()}`}>
          {/* biggest first, so a small bubble stays clickable on top of a large neighbour */}
          {[...drawnSites].sort((a, b) => (b.value ?? 0) - (a.value ?? 0)).map(s => {
            const available = s.value != null && s.n_filters > 0
            const r = available ? Math.max(4, rScale(Math.max(0, s.value!))) : 4
            const maia = aeronet?.maia[s.code]
            const col = aeronetQ ? aeronet?.sites[s.code] : undefined
            const box = Math.ceil(r + 5)
            // every name while few sites are drawn; with many, the hovered one (the rest crowd into each other)
            const labelled = (!isNetwork && sites.length <= 8) || lg.hover === s.label || (isSpartan && meta.sites.length <= 8 && meta.sites.some(f => f.code === s.code))
            return <MapMarker key={s.code} longitude={s.lon} latitude={s.lat} onClick={onSite ? () => onSite(s.code) : undefined}>
              <MarkerContent className="bubble-marker" style={{ cursor: onSite ? 'pointer' : 'default', opacity: lg.dim(s.label, 0.2) }}>
                <svg width={2 * box} height={2 * box} viewBox={`${-box} ${-box} ${2 * box} ${2 * box}`} data-export-site={s.name} data-export-color={s.color}
                  onMouseEnter={e => { lg.setHover(s.label); tip.show(e, [
              `${s.name} (${s.code})`, s.location,
              `${s.n_filters.toLocaleString()} samples · ${s.date_min} → ${s.date_max}`,
              ...(stat === 'Sample count' ? [] : [
                `${withUnit(metric, units)}: median ${fmtQ(s.summary.median)}, mean ${fmtQ(s.summary.mean)}`,
                aeronetQ ? `${s.summary.n.toLocaleString()} filter windows with a value` : `n with ${metric} = ${s.summary.n.toLocaleString()}`,
              ]),
              ...(aeronetQ && stat !== 'Sample count' ? col ? [
                `AERONET station ${col.station.replace(/_/g, ' ')}, ${fmt(col.km, 1)} km away`,
                ...(aeronet!.seasons[s.code] ?? []).filter((n) => col.seasons[n]?.[aeronetQ.key]).map((n) =>
                  `  ${n}: median ${fmtQ(col.seasons[n][aeronetQ.key]!.median)} (n=${col.seasons[n][aeronetQ.key]!.n})`),
                ...(aeronetQ.key === 'eta' ? [`PM2.5 against AOD 500: Spearman ρ ${fmt(col.rho_pm_aod, 2)}`] : []),
              ] : ['No AERONET station within 50 km with retrievals in its filter windows'] : []),
              ...(maia ? [maiaLine(maia)!] : []),
            ]) }} onMouseLeave={() => { lg.setHover(null); tip.hide() }}>
                  {/* a dashed ring: the site sits in a MAIA target area */}
                  {maia && <circle r={r + 3.5} fill="none" stroke={INK.text} strokeWidth={1.1} strokeDasharray="2.5 2" />}
                  <circle r={r} fill={available ? s.color : INK.panel} fillOpacity={available ? 0.55 : 1} stroke={s.color} strokeWidth={1.6} />
                  {available && <circle r={2} fill={s.color} />}
                </svg>
                {labelled && <span data-export="text" className="map-marker-label" style={{ bottom: `calc(50% + ${r + 4}px)` }}>{isSpartan ? meta.sites.find(f => f.code === s.code)?.name ?? s.label : s.label}</span>}
              </MarkerContent>
            </MapMarker>
          })}
        </GeoMap>
        <div className="legend"><SizeLegend scale={rScale} label={stat === 'Sample count' ? 'sample count' : `${stat.toLowerCase()} ${withUnit(metric, units)}`} format={v => stat === 'Sample count' ? String(Math.round(v)) : fmt(v, aeronetQ ? Math.min(aeronetQ.digits, 3) : 1)} /></div>
        {!sidePanel && <div style={isNetwork ? { maxHeight: 140, overflowY: 'auto' } : undefined}>
          <Legend items={sites.map(s => ({ label: s.label, color: s.color, detail: stat === 'Sample count' ? String(s.n_filters) : fmt(s.value, aeronetQ?.digits ?? 2) }))} {...lg.props} />
        </div>}
        {!sites.length && <Empty>No samples with coordinates{stat === 'Sample count' ? '' : ` and ${metric}`} in this selection.</Empty>}
        <Note>Drag to pan · scroll or use +/− to zoom · Fit sites frames the ticked sites · Globe shows the network without Mercator's stretched poles.{isImprove && ' IMPROVE opens on the US mainland; Fit sites also takes in Alaska, Hawaii and overseas sites.'}</Note>
        {!isImprove && aeronet && sites.some((s) => aeronet.maia[s.code]) && <Note>Dashed ring: a SPARTAN site inside a MAIA target area ({sites.filter((s) => aeronet.maia[s.code]).length} here; hover for the area). MAIA turns its AOD into surface PM2.5 there, so the AERONET η = PM2.5 / AOD 500 is the quantity it has to get right.</Note>}
        {aeronetQ && <Note>{aeronetQ.label}: {aeronetQ.about}. Median over each site's non-excluded filters of the AERONET means inside the filter's sampling window (all dates; the sidebar's dates and seasons do not apply here; hover a site for its seasons). Hollow: no station within 50 km with retrievals in the windows.{aeronetQ.key === 'ssa440' ? ' SSA varies over a narrow range, so bubble sizes differ little: read the values in the list or the tooltip.' : ''}</Note>}

      </>}
        {tip.node}
      </div>
      {sidePanel && <aside className="map-sites" aria-label="Sites on the map">
        <SitePicker mapPicker={false} noun={isImprove ? 'IMPROVE sites' : isSpartan ? 'SPARTAN sites' : 'sites'}
          items={sites.map(s => ({ key: s.label, label: isSpartan ? meta.sites.find(f => f.code === s.code)?.name ?? s.name ?? s.label : s.name ?? s.label,
            sub: s.code, lat: s.lat, lon: s.lon, color: s.color, n: s.n_filters, selected: lg.show(s.label) }))}
          onToggle={lg.toggle} onAll={() => lg.setHidden(new Set())} allSelected={lg.hidden.size === 0}
          onClear={() => lg.setHidden(new Set(sites.map(s => s.label)))} />
      </aside>}
      </div>
    </ChartFrame>
  )
}
