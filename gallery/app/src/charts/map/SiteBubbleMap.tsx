import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { FeatureCollection } from 'geojson'
import { ChartFrame, Note, Segmented } from '@/components/ChartFrame'
import { SizeLegend } from '@/components/ColorLegend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const STATS = ['Median', 'Mean', 'Sample count'] as const

/**
 * Bubble map — the four sampling sites, sized by a chosen statistic.
 * Basemap is Natural Earth 110m, shipped locally by
 * gallery/data/fetch_basemap.py so this renders offline.
 */
export function SiteBubbleMap({ rows, meta, field }: { rows: FilterRow[]; meta: MetaFile; field: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const height = 400

  const [stat, setStat] = useState<(typeof STATS)[number]>('Median')
  const [world, setWorld] = useState<FeatureCollection | null>(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/world.geojson`)
      .then((r) => r.json())
      .then(setWorld)
      .catch(() => setWorld(null))
  }, [])

  const sites = useMemo(
    () =>
      meta.sites
        .map((s) => {
          const vals = rows
            .filter((r) => r.site === s.name)
            .map((r) => r[field])
            .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
          const sorted = vals.slice().sort(d3.ascending)
          const value = stat === 'Sample count' ? vals.length : stat === 'Mean' ? (d3.mean(vals) ?? 0) : (d3.quantile(sorted, 0.5) ?? 0)
          return { ...s, value, n: vals.length, median: d3.quantile(sorted, 0.5) ?? 0, mean: d3.mean(vals) ?? 0 }
        })
        .filter((s) => s.n > 0),
    [rows, meta.sites, field, stat]
  )

  const projection = useMemo(() => d3.geoNaturalEarth1().fitExtent([[4, 4], [width - 4, height - 4]], { type: 'Sphere' } as any), [width])
  const path = useMemo(() => d3.geoPath(projection), [projection])
  const rScale = d3.scaleSqrt().domain([0, d3.max(sites, (s) => s.value) ?? 1]).range([0, 26])

  return (
    <ChartFrame
      id="map"
      title="Bubble map — the four sampling sites"
      subtitle={`Beijing, Delhi, Pasadena and Addis Ababa, sized by the ${stat.toLowerCase()} of ${field}. The estate has one map notebook; this makes the network geography a first-class filter rather than a static inset.`}
      provenance="stands in for notebooks/analysis/meteorology/map.ipynb · react-graph-gallery.com/bubble-map"
      controls={<Segmented label="size by" value={stat} options={STATS} onChange={setStat} />}
    >
      <div ref={wrapRef} className="chart-wrap">
        <svg width={width} height={height} className="animated">
          <path d={path({ type: 'Sphere' } as any) ?? ''} fill="#f2f6fa" stroke={INK.border} />
          {world && world.features.map((f, i) => <path key={i} d={path(f as any) ?? ''} fill="#e4e9ef" stroke="#fff" strokeWidth={0.5} />)}
          {sites.map((s) => {
            const p = projection([s.lon, s.lat])
            if (!p) return null
            const r = Math.max(4, rScale(s.value))
            return (
              <g
                key={s.code}
                onMouseEnter={(e) =>
                  tip.show(e, [
                    `${s.name} (${s.code})`,
                    s.location,
                    `${s.n_filters} filters · ${s.date_min} → ${s.date_max}`,
                    `${field}: median ${fmt(s.median)}, mean ${fmt(s.mean)}`,
                    `n with ${field} in subset = ${s.n}`,
                  ])
                }
                onMouseLeave={tip.hide}
              >
                <circle cx={p[0]} cy={p[1]} r={r} fill={s.color} fillOpacity={0.55} stroke={s.color} strokeWidth={1.6} />
                <circle cx={p[0]} cy={p[1]} r={2} fill={s.color} />
                <text x={p[0]} y={p[1] - r - 6} textAnchor="middle" fontSize={11} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>
                  {s.name}
                </text>
              </g>
            )
          })}
        </svg>
        <div className="legend">
          <SizeLegend scale={rScale} label={`${stat.toLowerCase()} of ${stat === 'Sample count' ? 'filters' : withUnit(field, meta.field_units)}`} format={(v) => (stat === 'Sample count' ? String(Math.round(v)) : fmt(v, 1))} />
          {sites.map((s) => (
            <span key={s.code} className="legend-item">
              <span className="swatch" style={{ background: s.color }} />
              {s.name}
              <span style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{stat === 'Sample count' ? s.n : fmt(s.value, 2)}</span>
            </span>
          ))}
        </div>
        {!world && (
          <Note>
            Basemap missing — run <code>python gallery/data/fetch_basemap.py</code>. Bubbles are still positioned correctly.
          </Note>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
