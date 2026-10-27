import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { SitePicker } from '@/components/SitePicker'
import { loadJson } from '@/lib/data'
import * as d3 from 'd3'
import type { FeatureCollection } from 'geojson'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { SizeLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT, withUnit } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useFitHeight } from '@/lib/chartFit'
import { maiaLine, useAeronetSites, type AeronetSitesFile } from '@/lib/aeronetSites'

const STATS = ['Median', 'Mean', 'Sample count'] as const
const SCOPES = ['Four focus sites', 'All SPARTAN sites', 'All IMPROVE sites'] as const
// old links: a single focus site is now a pick in the site list beside the map
const LEGACY_SCOPES = [...SCOPES, 'All study sites', 'Focus site', 'Single focus site'] as const
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

/** Offline bubble map with independent study-site and IMPROVE scopes. */
export function SiteBubbleMap({ rows, meta, field, onSite }: {
  rows: FilterRow[]; meta: MetaFile; field: string; focusSite?: string
  /** a click on a site's bubble (the Dashboard opens that site's own page) */
  onSite?: (code: string) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const fitted = useFitHeight()
  const height = fitted ?? 400
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomBehavior = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [zoom, setZoom] = useState(d3.zoomIdentity)
  const clipId = useId().replace(/:/g, '')
  const [stat, setStat] = useSearchParam<(typeof STATS)[number]>('mapstat', 'Median', STATS)
  const [savedScope, setScope] = useSearchParam<(typeof LEGACY_SCOPES)[number]>('mapscope', 'Four focus sites', LEGACY_SCOPES)
  const scope: (typeof SCOPES)[number] = (SCOPES as readonly string[]).includes(savedScope) ? savedScope as (typeof SCOPES)[number] : 'Four focus sites'
  const [improveField, setImproveField] = useSearchParam<string>('mapfield', 'EC (TOR)')
  const [world, setWorld] = useState<FeatureCollection | null>(null)
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
    let live = true
    loadJson<any>('world.geojson').then((w) => { if (live) setWorld(w) }).catch(() => {})
    return () => { live = false }
  }, [])

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
    const source = isNetwork
      ? (network?.sites ?? []).map(s => ({ ...s, label: s.code,
          color: isSpartan ? meta.sites.find(f => f.code === s.code)?.color ?? INK.accent : INK.accent,
          summary: s.stats[metric] ?? { n: 0, mean: null, median: null } }))
      : meta.sites.map(s => {
          const subset = rows.filter(r => r.site === s.name)
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

  // in a fitted dashboard tile the frame is the sites in view (20° around them), not the whole
  // globe, which a short wide tile would shrink to a sliver; elsewhere the world as before
  const frame = useMemo(() => {
    if (fitted === null || !sites.length) return null
    const lons = sites.map((s) => s.lon), lats = sites.map((s) => s.lat)
    const lo = (v: number[], pad: number, min: number) => Math.max(min, d3.min(v)! - pad)
    const hi = (v: number[], pad: number, max: number) => Math.min(max, d3.max(v)! + pad)
    const [w0, w1, s0, n0] = [lo(lons, 20, -180), hi(lons, 20, 180), lo(lats, 20, -60), hi(lats, 20, 80)]
    return { type: 'MultiPoint', coordinates: [[w0, s0], [w0, n0], [w1, s0], [w1, n0], [(w0 + w1) / 2, s0], [(w0 + w1) / 2, n0]] } as GeoJSON.MultiPoint
  }, [fitted, sites])
  const projection = useMemo(() => {
    return d3.geoNaturalEarth1().fitExtent([[10, 10], [Math.max(60, width - 10), height - 10]], frame ?? { type: 'Sphere' })
  }, [width, height, frame])
  const path = useMemo(() => d3.geoPath(projection), [projection])
  // sized to the largest value in view, however small (AAOD tops out near 0.05), not to at least 1
  const vmax = d3.max(sites, s => s.value) ?? 0
  const rScale = d3.scaleSqrt().domain([0, vmax > 0 ? vmax : 1]).range([0, isNetwork ? 16 : 26])

  const fit = (coords: number[][]) => {
    const points = coords.map(p => projection([p[0], p[1]])).filter((p): p is [number, number] => p != null)
    if (!points.length) return d3.zoomIdentity
    const x0 = d3.min(points, p => p[0])!, x1 = d3.max(points, p => p[0])!
    const y0 = d3.min(points, p => p[1])!, y1 = d3.max(points, p => p[1])!
    // one site has no extent: a regional view around it rather than the deepest zoom
    const k = points.length === 1 ? 5 : Math.max(1, Math.min(16, Math.min((width - 60) / Math.max(1, x1 - x0), (height - 60) / Math.max(1, y1 - y0))))
    return d3.zoomIdentity.translate(width / 2 - k * (x0 + x1) / 2, height / 2 - k * (y0 + y1) / 2).scale(k)
  }
  const home = () => isImprove ? fit([[-125, 24], [-125, 50], [-66, 24], [-66, 50]]) : d3.zoomIdentity
  const move = (transform: d3.ZoomTransform) => {
    if (svgRef.current && zoomBehavior.current) d3.select(svgRef.current).call(zoomBehavior.current.transform, transform)
  }
  useEffect(() => {
    if (!svgRef.current) return
    const svg = d3.select(svgRef.current)
    const behavior = d3.zoom<SVGSVGElement, unknown>().extent([[0, 0], [width, height]])
      .scaleExtent([1, 16]).on('zoom', event => setZoom(event.transform))
    zoomBehavior.current = behavior
    svg.call(behavior).call(behavior.transform, home())
    return () => { svg.on('.zoom', null) }
  }, [width, scope, projection, network])

  return (
    <ChartFrame id="map" source="gallery/app/src/charts/map/SiteBubbleMap.tsx"
      title={`Bubble map — ${drawnSites.length === 1 ? drawnSites[0].name : scope.toLowerCase()}`}
      subtitle={`${stat === 'Sample count' ? 'Sample counts' : `${stat} ${withUnit(metric, units)}`} at ${isNetwork ? `the available ${isImprove ? 'IMPROVE' : 'SPARTAN'} sites` : 'the four focus sites: Beijing, Delhi, JPL and Addis Ababa'}; pick or search sites in the list beside the map. Drag to pan; scroll or use +/− to zoom. Bubble sizes stay fixed while zooming. The four-focus-site views retain sidebar season, date and exclusion choices.`}
      provenance={isNetwork && network ? `${network.sites.length} sites in the inventory.${unlocated.length ? ` Missing coordinates: ${unlocated.map(s => s.code).join(', ')}.` : ''} ${aeronetQ && aeronet ? `${aeronet.source}, Level ${aeronet.level}: ${aeronet.note} (aeronet/sites_summary.json, export_aeronet_sites.py)` : network.note}${isSpartan ? ' MAIA target areas: research/maia/inventory/spartan_sites.csv.' : ''}` : `Local site inventory${aeronet ? '. MAIA target areas: research/maia/inventory/spartan_sites.csv' : ''}`}
      controls={<>
        <Segmented label="Sites" value={scope} options={SCOPES} onChange={setScope} />
        {isNetwork && network && stat !== 'Sample count' && <Select label={`${isImprove ? 'IMPROVE' : 'SPARTAN'} measurement`} value={metric} options={Object.keys(network.field_units)} onChange={isImprove ? setImproveField : setSpartanField} />}
        <Segmented label="Size by" value={stat} options={STATS} onChange={setStat} />
        <span className="control">View
          <button className="btn quiet" type="button" aria-label="Zoom out" onClick={() => { if (svgRef.current && zoomBehavior.current) d3.select(svgRef.current).call(zoomBehavior.current.scaleBy, 0.5) }}>−</button>
          <button className="btn quiet" type="button" aria-label="Zoom in" onClick={() => { if (svgRef.current && zoomBehavior.current) d3.select(svgRef.current).call(zoomBehavior.current.scaleBy, 2) }}>+</button>
          <span>{zoom.k.toFixed(1)}×</span>
          <button className="btn quiet" type="button" onClick={() => move(home())}>Reset view</button>
          {isImprove && <button className="btn quiet" type="button" onClick={() => move(home())}>US mainland</button>}
          <button className="btn quiet" type="button" title="Fit the sites ticked in the list" onClick={() => move(fit(drawnSites.map(s => [s.lon, s.lat])))}>Fit sites</button>
        </span>
      </>}
    >
      <div className={sidePanel ? 'map-with-sites' : undefined}>
      <div ref={wrapRef} className="chart-wrap">
      {isNetwork && !network ? <Empty>{error ? <>Could not load site summaries ({error}). Run <code>uv run python gallery/data/export_{isImprove ? 'improve' : 'spartan'}_map.py</code> and reopen the map.</> : 'Loading network sites…'}</Empty> : <>
        <svg ref={svgRef} width={width} height={height} role="img" aria-label={`Bubble map: ${scope}, ${stat.toLowerCase()}`} style={{ cursor: 'grab', touchAction: 'none' }}>
          <defs><clipPath id={clipId}><rect width={width} height={height} /></clipPath></defs>
          <g clipPath={`url(#${clipId})`}>
          <rect width={width} height={height} fill={INK.sea} />
          <g transform={zoom.toString()}>
          <path d={path({ type: 'Sphere' }) ?? ''} fill={INK.sea} stroke={INK.border} />
          {world?.features.map((f, i) => <path key={i} d={path(f) ?? ''} fill={INK.land} stroke={INK.panel} strokeWidth={0.5} vectorEffect="non-scaling-stroke" />)}
          </g>
          {sites.map(s => {
            const projected = projection([s.lon, s.lat])
            if (!projected || !lg.show(s.label)) return null
            const p = zoom.apply(projected)
            const available = s.value != null && s.n_filters > 0
            const r = available ? Math.max(4, rScale(Math.max(0, s.value!))) : 4
            const maia = aeronet?.maia[s.code]
            const col = aeronetQ ? aeronet?.sites[s.code] : undefined
            return <g key={s.code} data-export-site={s.name} data-export-color={s.color} opacity={lg.dim(s.label, 0.2)} onClick={onSite ? () => onSite(s.code) : undefined} style={onSite ? { cursor: 'pointer' } : undefined} onMouseEnter={e => tip.show(e, [
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
            ])} onMouseLeave={tip.hide}>
              {/* a dashed ring: the site sits in a MAIA target area */}
              {maia && <circle cx={p[0]} cy={p[1]} r={r + 3.5} fill="none" stroke={INK.text} strokeWidth={1.1} strokeDasharray="2.5 2" />}
              <circle cx={p[0]} cy={p[1]} r={r} fill={available ? s.color : INK.panel} fillOpacity={available ? 0.55 : 1} stroke={s.color} strokeWidth={1.6} />
              {available && <circle cx={p[0]} cy={p[1]} r={2} fill={s.color} />}
              {/* every name while few sites are drawn; with many, the hovered one (the rest crowd into each other) */}
              {((!isNetwork && sites.length <= 8) || lg.hover === s.label || (isSpartan && meta.sites.some(f => f.code === s.code))) && <text x={p[0]} y={p[1] - r - 6} textAnchor="middle" fontSize={11} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>{isSpartan ? meta.sites.find(f => f.code === s.code)?.name ?? s.label : s.label}</text>}
            </g>
          })}
          </g>
        </svg>
        <div className="legend"><SizeLegend scale={rScale} label={stat === 'Sample count' ? 'sample count' : `${stat.toLowerCase()} ${withUnit(metric, units)}`} format={v => stat === 'Sample count' ? String(Math.round(v)) : fmt(v, aeronetQ ? Math.min(aeronetQ.digits, 3) : 1)} /></div>
        {!sidePanel && <div style={isNetwork ? { maxHeight: 140, overflowY: 'auto' } : undefined}>
          <Legend items={sites.map(s => ({ label: s.label, color: s.color, detail: stat === 'Sample count' ? String(s.n_filters) : fmt(s.value, aeronetQ?.digits ?? 2) }))} {...lg.props} />
        </div>}
        {!sites.length && <Empty>No samples with coordinates{stat === 'Sample count' ? '' : ` and ${metric}`} in this selection.</Empty>}
        <Note>Drag to pan · scroll or use +/− to zoom · Fit sites frames the ticked sites.{isImprove && ' IMPROVE opens on the US mainland; Fit sites also takes in Alaska, Hawaii and overseas sites.'}</Note>
        {!isImprove && aeronet && sites.some((s) => aeronet.maia[s.code]) && <Note>Dashed ring: a SPARTAN site inside a MAIA target area ({sites.filter((s) => aeronet.maia[s.code]).length} here; hover for the area). MAIA turns its AOD into surface PM2.5 there, so the AERONET η = PM2.5 / AOD 500 is the quantity it has to get right.</Note>}
        {aeronetQ && <Note>{aeronetQ.label}: {aeronetQ.about}. Median over each site's non-excluded filters of the AERONET means inside the filter's sampling window (all dates; the sidebar's dates and seasons do not apply here; hover a site for its seasons). Hollow: no station within 50 km with retrievals in the windows.{aeronetQ.key === 'ssa440' ? ' SSA varies over a narrow range, so bubble sizes differ little: read the values in the list or the tooltip.' : ''}</Note>}

        {!world && <Note>Basemap unavailable; bubbles remain positioned by site coordinates.</Note>}
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
