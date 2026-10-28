import { useEffect, useMemo, useRef, useState, type SVGProps } from 'react'
import * as d3 from 'd3'
import type { Feature, FeatureCollection } from 'geojson'
import { MapDonutMarker } from '@pgmaps/geo-toolkit/map'
import { DonutRingMap, type DonutRingItem } from '@pgmaps/geo-toolkit/visualizations/react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { GeoMap, asReactMouse } from '@/components/GeoMap'
import { useSearchParam } from '@/lib/url'
import { Legend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { loadJson } from '@/lib/data'
import { useFitHeight } from '@/lib/chartFit'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'
import { categoricalScale } from '@/lib/categoricalColours'

/** gallery/data/export_spartan_composition.py */
interface CompositionFile {
  generated: string
  source: string
  note: string
  slices: { key: string; label: string }[]
  columns: string[]
  sites: { code: string; name: string; lat: number | null; lon: number | null; first: string | null; last: string | null
    n: number; missing: { oc: number; bc: number; k: number; other: number } }[]
  filters: (string | number)[][]
}

/** "active" here is data recency, not SPARTAN's operating status: a public sample within this many days of the newest */
const RECENT_DAYS = 730
const ACTIVE = '#1f2e86', RETIRED = '#8a8a8a'

const city = (name: string) => name.split(',')[0].trim()
const VIEWS = ['Map', 'Figure'] as const

/**
 * SPARTAN's network PM2.5 composition, recreated: a donut per site (the centre is the mean PM2.5, µg/m³)
 * around a world map, each joined to its site. The slices are SPARTAN's own mass reconstruction
 * (the public ReconstrPM25 product), averaged over every filter with its mass balance.
 */
export function SpartanCompositionMap() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const fitted = useFitHeight()
  const tip = useTooltip(wrapRef)
  const [data, setData] = useState<CompositionFile | null>(null)
  const [world, setWorld] = useState<FeatureCollection | null>(null)
  const [error, setError] = useState('')
  const [hover, setHover] = useState<string | null>(null)
  // Map: a pannable world (geo-toolkit) with each donut on its site; Figure: SPARTAN's network figure, donuts in a ring
  const [view, setView] = useSearchParam<(typeof VIEWS)[number]>('spartanview', 'Map', VIEWS)
  const colours = useColourScheme('spartan-composition-map', 'categorical', 'SPARTAN composition')
  const sliceColour = useMemo(() => categoricalScale(colours.name, data?.slices.map(s => s.key) ?? [], colours.reverse), [data, colours.name, colours.reverse])
  useEffect(() => {
    let live = true
    loadJson<CompositionFile>('spartan_composition.json').then((d) => { if (live) setData(d) }).catch((e) => { if (live) setError(String(e)) })
    loadJson<FeatureCollection>('world.geojson').then((w) => { if (live) setWorld(w) }).catch(() => {})
    return () => { live = false }
  }, [])

  // per site: the mean of each slice over its filters (so the slices still sum to the mean PM2.5)
  const sites = useMemo(() => {
    if (!data) return []
    const col = Object.fromEntries(data.columns.map((c, i) => [c, i]))
    const bySite = d3.group(data.filters, (f) => f[col.site] as string)
    const newest = d3.max(data.sites, (s) => s.last ?? '') ?? ''
    const cutoff = d3.timeDay.offset(new Date(newest), -RECENT_DAYS).toISOString().slice(0, 10)
    return data.sites.flatMap((s) => {
      const rows = bySite.get(s.code)
      if (!rows?.length || s.lat == null || s.lon == null) return []
      const mean = (k: string) => d3.mean(rows, (r) => r[col[k]] as number) ?? 0
      return [{ ...s, lat: s.lat, lon: s.lon, pm: mean('pm'), recent: (s.last ?? '') >= cutoff,
        slices: data.slices.map((sl) => ({ ...sl, value: mean(sl.key), color: sliceColour(sl.key) })) }]
    })
  }, [data, sliceColour])

  // the Figure view's land: Antarctica holds no site and would only squash the rest
  const land = useMemo<FeatureCollection | null>(() => world && { ...world, features: world.features.filter((f: Feature) => f.properties?.name !== 'Antarctica') }, [world])
  const ringItems = useMemo<DonutRingItem[]>(() => sites.map((s) => ({ id: s.code, longitude: s.lon, latitude: s.lat,
    values: s.slices.map((sl) => sl.value), centerLabel: fmt(s.pm, 0), name: city(s.name), muted: !s.recent })), [sites])
  const bySite = useMemo(() => new Map(sites.map((s) => [s.code, s])), [sites])

  const tipLines = (s: (typeof sites)[number]) => [
    `${s.name} (${s.code})`,
    `${s.n} filters with a mass balance · ${s.first} → ${s.last}`,
    `PM2.5 ${fmt(s.pm, 1)} µg/m³ (mean)`,
    ...s.slices.map((sl) => `  ${sl.label}: ${fmt(sl.value, 2)} µg/m³ · ${fmt(sl.value / s.pm * 100, 1)}%`),
    ...(s.missing.oc ? [`${s.missing.oc} filters without FTIR OC: their organic carbon is in the residual`] : []),
    ...(s.missing.bc ? [`${s.missing.bc} filters without BC: counted in the residual`] : []),
  ]
  // the map's donuts: area grows with mean PM2.5, so Delhi's outweighs Halifax's at a glance
  const pmMax = d3.max(sites, (s) => s.pm) ?? 1
  const mapRadius = (pm: number) => 11 + 13 * Math.sqrt(Math.max(0, pm) / pmMax)
  const points = useMemo(() => sites.map((s) => [s.lon, s.lat] as const), [sites])

  const total = d3.sum(sites, (s) => s.n)

  return (
    <ChartFrame id="spartan-composition-map" source="gallery/app/src/charts/map/SpartanCompositionMap.tsx"
      title="SPARTAN PM2.5 composition — every site's mean reconstructed mass, as in SPARTAN's network figure"
      subtitle="Each donut is one site: the centre is its mean PM2.5 (µg/m³ at 35% RH) over the filters with SPARTAN's mass balance, the slices are SPARTAN's reconstruction of that mass (counter-clockwise from the top in the legend's order). Map: each donut sits on its site and grows with PM2.5; pan and zoom into crowded regions. Figure: the donuts ring the world as in SPARTAN's figure, a line running to each site. Hover a donut for the values; point at a legend entry to pick out that component everywhere."
      provenance={data ? `${data.source}. ${data.note} (spartan_composition.json, gallery/data/export_spartan_composition.py)` : undefined}
      controls={<><Segmented label="View" value={view} options={VIEWS} onChange={setView} />{colours.picker}</>}
      tip={<>Water is not published by SPARTAN: it is what remains of SPARTAN's &lsquo;Residual Matter with OC&rsquo; balance once the other slices are taken out (median 2% of PM2.5). Sodium and chloride split SPARTAN&rsquo;s sea salt (2.54 × sea-salt sodium). Older filters without FTIR OC keep that mass in the residual, as in SPARTAN&rsquo;s balance. Blue and grey dots mark data recency here, not SPARTAN&rsquo;s operating status.</>}>
      <div ref={wrapRef} className="chart-wrap" data-export-fixed-colours={sliceColour.range().join(' ')}>
        {error ? <Empty>Could not load the composition ({error}). Run <code>uv run python gallery/data/export_spartan_composition.py</code>.</Empty>
          : !sites.length ? <Empty>Loading SPARTAN composition…</Empty> : view === 'Map' ? <>
          <GeoMap height={fitted ?? 520} frame={points} frameOptions={{ pad: 0.05, minSpan: 60 }} frameKey="network"
            label={`SPARTAN PM2.5 composition at ${sites.length} sites`}>
            {/* smallest last, so it stays clickable over a big neighbour */}
            {[...sites].sort((a, b) => b.pm - a.pm).map((s) => {
              const reversed = s.slices.slice().reverse()
              const emphasized = hover ? reversed.findIndex((sl) => sl.label === hover) : -1
              return <MapDonutMarker key={s.code} longitude={s.lon} latitude={s.lat}
                counts={reversed.map((sl) => sl.value)} colors={reversed.map((sl) => sl.color)}
                radius={mapRadius(s.pm)} innerRatio={0.56} label={fmt(s.pm, 0)} title={`${s.name}: PM2.5 ${fmt(s.pm, 1)} µg/m³`}
                centerColor={INK.panel} textColor={INK.text} shadow={false}
                emphasizedBand={emphasized >= 0 ? emphasized : null}
                onMouseEnter={(e) => tip.show(asReactMouse(e), tipLines(s))} onMouseLeave={tip.hide}
                className="donut-marker"
                content={<span data-export="text" className="map-marker-label when-near" style={{ top: `calc(50% + ${mapRadius(s.pm) + 2}px)` }} data-export-site={s.name}>
                  <span className="status-dot" style={{ background: s.recent ? ACTIVE : RETIRED }} aria-hidden="true" />{city(s.name)}
                </span>} />
            })}
          </GeoMap>
          <Legend items={[
            ...(data?.slices ?? []).map((sl) => ({ label: sl.label, color: sliceColour(sl.key), shape: 'square' as const })),
            { label: 'Public sample in the last 2 years', color: ACTIVE, shape: 'dot' as const },
            { label: 'No public sample in 2 years', color: RETIRED, shape: 'dot' as const },
          ]} onHover={setHover} highlighted={hover} />
          <Note>{sites.length} sites, {total.toLocaleString()} filters. Donut area grows with mean PM2.5 (a fixed size in the Figure view); zoom in for the site names. A site whose mean slice is negative (Abu Dhabi&rsquo;s residual) draws it as zero; the tooltip keeps the value.</Note>
        </> : !land ? <Empty>Loading SPARTAN composition…</Empty> : <>
          {/* SPARTAN's own figure: geo-toolkit's DonutRingMap, drawn with d3 as one SVG */}
          <DonutRingMap width={width || 900} height={fitted ?? undefined} land={land} items={ringItems}
            colors={data?.slices.map((sl) => sliceColour(sl.key)) ?? []}
            emphasized={hover ? data?.slices.findIndex((sl) => sl.label === hover) ?? null : null}
            landColor={INK.panel} landStroke={INK.axis} panelColor={INK.panel} textColor={INK.text} leaderColor={ACTIVE} mutedColor={RETIRED}
            fontFamily={FONT.family} label={`SPARTAN PM2.5 composition at ${sites.length} sites`}
            empty={<Empty>Too many sites to ring this map at this width.</Empty>}
            itemProps={(it) => { const s = bySite.get(it.id)!; return { 'data-export-site': s.name, onMouseLeave: tip.hide, onMouseEnter: (e) => tip.show(e, tipLines(s)) } as SVGProps<SVGGElement> }} />
          <Legend items={[
            ...(data?.slices ?? []).map((sl) => ({ label: sl.label, color: sliceColour(sl.key), shape: 'square' as const })),
            { label: 'Public sample in the last 2 years', color: ACTIVE, shape: 'dot' as const },
            { label: 'No public sample in 2 years', color: RETIRED, shape: 'dot' as const },
          ]} onHover={setHover} highlighted={hover} />
          <Note>{sites.length} sites, {total.toLocaleString()} filters. A site whose mean slice is negative (Abu Dhabi&rsquo;s residual) draws it as zero; the tooltip keeps the value.</Note>
        </>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
