import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { Feature, FeatureCollection } from 'geojson'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { Legend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { loadJson } from '@/lib/data'
import { useFitHeight } from '@/lib/chartFit'
import { assignRing, ringSlots } from '@/lib/donutRing'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'

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

/**
 * The colours of SPARTAN's network composition figure (spartan-network.org), matched by eye from the
 * October 2026 figure; legend order as there. Fixed in both themes, as in the figure.
 */
const SLICE_COLOR: Record<string, string> = {
  bc: '#1a1a1a', teo: '#8c8c8c', dust: '#f9e81b', na: '#2f4ea0', cl: '#2aa198', k: '#b48ad8',
  no3: '#f47c20', nh4: '#ee5fa7', so4: '#ec2227', water: '#6dd0f5', oc: '#2e6b3c', rm: '#4cb847',
}
/** "active" here is data recency, not SPARTAN's operating status: a public sample within this many days of the newest */
const RECENT_DAYS = 730
const ACTIVE = '#1f2e86', RETIRED = '#8a8a8a'

const city = (name: string) => name.split(',')[0].trim()

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
        slices: data.slices.map((sl) => ({ ...sl, value: mean(sl.key), color: SLICE_COLOR[sl.key] ?? INK.neutral })) }]
    })
  }, [data])

  const layout = useMemo(() => {
    const W = Math.max(320, width || 900)
    const land: FeatureCollection | null = world && { ...world, features: world.features.filter((f: Feature) => f.properties?.name !== 'Antarctica') }
    if (!sites.length || !land) return null
    // the largest donut whose ring of slots still holds every site; the map fills the box inside the ring
    for (let r = Math.min(40, W / 26); r >= 11; r -= 1) {
      const font = Math.max(8.5, Math.min(12.5, r * 0.42))
      const cell = { w: Math.max(2 * r + 12, 2.6 * r + 6), h: 2 * r + font + 12 }
      const innerW = W - 2 * cell.w
      const projection = d3.geoEquirectangular().fitWidth(innerW, land)
      const [[, y0], [, y1]] = d3.geoPath(projection).bounds(land)
      const H = Math.round(y1 - y0 + 2 * cell.h)
      const slots = ringSlots(W, H, cell)
      if (slots.length < sites.length) continue
      projection.fitExtent([[cell.w, cell.h], [W - cell.w, H - cell.h]], land)
      const points = sites.map((s) => { const [x, y] = projection([s.lon, s.lat])!; return { x, y } })
      const pick = assignRing(points, slots, { x: W / 2, y: H / 2 })
      const placed = pick.map((i) => slots[i])
      const shift = (font + 6) / 2
      const centres = placed.map((sl) => ({ x: sl.x, y: sl.y + (sl.side === 'bottom' ? -shift : shift) }))
      return { W, H, r, font, cellW: cell.w, projection, path: d3.geoPath(projection), land, points, slots: placed, centres }
    }
    return null
  }, [sites, world, width])

  const arcs = useMemo(() => d3.pie<{ value: number }>().sort(null).value((d) => Math.max(0, d.value)), [])
  const total = d3.sum(sites, (s) => s.n)

  return (
    <ChartFrame id="spartan-composition-map" source="gallery/app/src/charts/map/SpartanCompositionMap.tsx"
      title="SPARTAN PM2.5 composition — every site's mean reconstructed mass, as in SPARTAN's network figure"
      subtitle="Each donut is one site: the centre is its mean PM2.5 (µg/m³ at 35% RH) over the filters with SPARTAN's mass balance, the slices are SPARTAN's reconstruction of that mass (counter-clockwise from the top in the legend's order), and the line runs to the site. Hover a donut for the values; point at a legend entry to pick out that component everywhere."
      provenance={data ? `${data.source}. ${data.note} (spartan_composition.json, gallery/data/export_spartan_composition.py)` : undefined}
      tip={<>Water is not published by SPARTAN: it is what remains of SPARTAN's &lsquo;Residual Matter with OC&rsquo; balance once the other slices are taken out (median 2% of PM2.5). Sodium and chloride split SPARTAN&rsquo;s sea salt (2.54 × sea-salt sodium). Older filters without FTIR OC keep that mass in the residual, as in SPARTAN&rsquo;s balance. Blue and grey dots mark data recency here, not SPARTAN&rsquo;s operating status.</>}>
      <div ref={wrapRef} className="chart-wrap">
        {error ? <Empty>Could not load the composition ({error}). Run <code>uv run python gallery/data/export_spartan_composition.py</code>.</Empty>
          : !layout ? <Empty>Loading SPARTAN composition…</Empty> : <>
          <svg width="100%" height={fitted ?? layout.H} viewBox={`0 0 ${layout.W} ${layout.H}`} role="img" fontFamily={FONT.family}
            aria-label={`SPARTAN PM2.5 composition at ${sites.length} sites`}>
            <g>{layout.land.features.map((f, i) => <path key={i} d={layout.path(f) ?? ''} fill={INK.panel} stroke={INK.axis} strokeWidth={0.5} />)}</g>
            {sites.map((s, i) => {
              const p = layout.points[i], c = layout.centres[i], r = layout.r
              // the leader stops at the donut's edge
              const dx = p.x - c.x, dy = p.y - c.y, len = Math.hypot(dx, dy) || 1
              return <line key={`l-${s.code}`} x1={c.x + dx / len * (r + 1)} y1={c.y + dy / len * (r + 1)} x2={p.x} y2={p.y}
                stroke={ACTIVE} strokeOpacity={0.75} strokeWidth={0.9} />
            })}
            {sites.map((s, i) => {
              const p = layout.points[i]
              return s.recent ? <circle key={`p-${s.code}`} cx={p.x} cy={p.y} r={4.2} fill={ACTIVE} stroke={INK.panel} strokeWidth={0.8} />
                : <g key={`p-${s.code}`}><circle cx={p.x} cy={p.y} r={5} fill={RETIRED} opacity={0.55} /><circle cx={p.x} cy={p.y} r={3} fill="#333" /></g>
            })}
            {sites.map((s, i) => {
              const slot = layout.slots[i], r = layout.r
              // the name goes below a bottom-row donut and above the rest, inside the slot
              const below = slot.side === 'bottom', shift = (layout.font + 6) / 2
              const c = { x: slot.x, y: slot.y + (below ? -shift : shift) }
              const name = city(s.name)
              // a long name shrinks to its slot's width rather than running into the next one
              const nameSize = Math.min(layout.font, (layout.cellW - 6) / (0.58 * name.length))
              const pie = arcs(s.slices.slice().reverse())
              const arc = d3.arc<d3.PieArcDatum<{ value: number }>>().innerRadius(r * 0.56).outerRadius(r)
              const labelY = below ? c.y + r + nameSize + 2 : c.y - r - 5
              return <g key={s.code} data-export-site={s.name} onMouseLeave={tip.hide} onMouseEnter={(e) => tip.show(e, [
                `${s.name} (${s.code})`,
                `${s.n} filters with a mass balance · ${s.first} → ${s.last}`,
                `PM2.5 ${fmt(s.pm, 1)} µg/m³ (mean)`,
                ...s.slices.map((sl) => `  ${sl.label}: ${fmt(sl.value, 2)} µg/m³ · ${fmt(sl.value / s.pm * 100, 1)}%`),
                ...(s.missing.oc ? [`${s.missing.oc} filters without FTIR OC: their organic carbon is in the residual`] : []),
                ...(s.missing.bc ? [`${s.missing.bc} filters without BC: counted in the residual`] : []),
              ])}>
                <g transform={`translate(${c.x},${c.y})`}>
                  <circle r={r} fill={INK.panel} />
                  {pie.map((a, k) => {
                    const sl = s.slices[s.slices.length - 1 - k]
                    return <path key={sl.key} d={arc(a) ?? ''} fill={sl.color} stroke={INK.panel} strokeWidth={0.4}
                      opacity={hover && hover !== sl.label ? 0.18 : 1} />
                  })}
                  <text textAnchor="middle" dy="0.36em" fontSize={Math.max(9, r * 0.52)} fontWeight={700} fill={INK.text}>{fmt(s.pm, 0)}</text>
                </g>
                <text x={c.x} y={labelY} textAnchor="middle" fontSize={nameSize} fontWeight={600} fill={INK.text}>{name}</text>
              </g>
            })}
          </svg>
          <Legend items={[
            ...(data?.slices ?? []).map((sl) => ({ label: sl.label, color: SLICE_COLOR[sl.key], shape: 'square' as const })),
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
