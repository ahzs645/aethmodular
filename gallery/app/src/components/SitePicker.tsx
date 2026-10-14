import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import * as d3 from 'd3'
import type { FeatureCollection } from 'geojson'
import { useDimensions } from '@/hooks/useGalleryData'
import { loadJson } from '@/lib/data'
import { FONT, INK } from '@/lib/theme'

/** One pickable site: a row in the search box's list and a dot on the map. */
export interface SiteItem {
  key: string
  label: string
  /** state, country or site code under the name */
  sub?: string
  lat: number | null
  lon: number | null
  color: string
  /** filters in the current subset (or in the site file before it is loaded) */
  n: number
  selected: boolean
}

/**
 * Sites for the Filters sidebar: a search box that filters a scrolling checkbox list (picked
 * sites first), quick buttons (focus sites, all, none) and **Map…**, a large map dialog for
 * picking by clicking. All edit one selection. With only the four focus sites, plain chips.
 */
export function SitePicker({ items, onToggle, onAll, onClear, onFocus, allSelected = false, chipsOnly = false, limitNote, noun = 'sites' }: {
  items: SiteItem[]
  onToggle: (key: string) => void
  /** select every site (absent where sites load one by one, e.g. IMPROVE) */
  onAll?: () => void
  /** deselect every site: the charts then show nothing until sites are picked */
  onClear?: () => void
  /** back to the four focus sites (SPARTAN's default) */
  onFocus?: () => void
  allSelected?: boolean
  /** few sites (the focus-sites view): toggle chips, no list */
  chipsOnly?: boolean
  /** e.g. "3 of 10 IMPROVE sites" */
  limitNote?: string
  noun?: string
}) {
  const [mapOpen, setMapOpen] = useState(false)
  const [query, setQuery] = useState('')
  const selected = items.filter((it) => it.selected)
  const mapButton = <button type="button" className="btn quiet" onClick={() => setMapOpen(true)} title="Pick sites on a large map">Map…</button>
  const q = query.trim().toLowerCase()
  const shown = useMemo(() => {
    const hit = q ? items.filter((it) => `${it.label} ${it.sub ?? ''} ${it.key}`.toLowerCase().includes(q)) : items
    // picked sites first, each block in its own order, so the selection is visible without scrolling
    return [...hit.filter((it) => it.selected), ...hit.filter((it) => !it.selected)]
  }, [items, q])
  return (
    <div className="site-picker">
      {chipsOnly ? (
        <div className="season-chips">
          {onAll && <button type="button" className={`chip ${allSelected ? 'active' : ''}`} style={{ ['--chip' as string]: 'var(--ink-muted)' }} onClick={onAll}>All</button>}
          {items.map((it) => (
            <button key={it.key} type="button" className={`chip ${it.selected ? 'active' : ''} ${it.n === 0 ? 'disabled' : ''}`}
              style={{ ['--chip' as string]: it.color }} disabled={it.n === 0} onClick={() => onToggle(it.key)}>
              {it.label} · {it.n}
            </button>
          ))}
          {mapButton}
        </div>
      ) : (
        <>
          <div className="site-picker-bar">
            {onFocus && <button type="button" className="btn quiet" onClick={onFocus} title="Beijing, Delhi, JPL and Addis Ababa">Focus sites</button>}
            {onAll && <button type="button" className="btn quiet" onClick={onAll} disabled={allSelected}>Select all</button>}
            {onClear && <button type="button" className="btn quiet" onClick={onClear} disabled={selected.length === 0}>Deselect all</button>}
            <span className="site-picker-count">{limitNote ?? `${selected.length} of ${items.length}`}</span>
          </div>
          <div className="site-picker-bar">
            <input className="select site-filter" type="search" placeholder={`Search ${items.length} ${noun}…`} value={query}
              onChange={(e) => setQuery(e.target.value)} aria-label={`Search ${noun}`} />
            {mapButton}
          </div>
          <ul className="site-list" aria-label={noun}>
            {shown.map((it) => (
              <li key={it.key}>
                <label className={`site-row ${it.selected ? 'on' : ''}`}>
                  <input type="checkbox" checked={it.selected} onChange={() => onToggle(it.key)} />
                  <span className="site-check" style={{ ['--chip' as string]: it.color }} aria-hidden="true" />
                  <span className="site-name">{it.label}{it.sub && <span className="site-sub"> · {it.sub}</span>}</span>
                  <span className="site-n">{it.n.toLocaleString()}</span>
                </label>
              </li>
            ))}
            {shown.length === 0 && <li className="site-empty">No site matches “{query}”.</li>}
          </ul>
          {selected.length === 0 && <span className="site-empty">No {noun} selected: the charts are empty until you pick some.</span>}
        </>
      )}
      {mapOpen && <MapDialog items={items} onToggle={onToggle} onClose={() => setMapOpen(false)} limitNote={limitNote} noun={noun} />}
    </div>
  )
}

/** A large map in a dialog: click dots to add or remove sites; scroll, drag or +/− to move. */
function MapDialog({ items, onToggle, onClose, limitNote, noun }: { items: SiteItem[]; onToggle: (key: string) => void; onClose: () => void; limitNote?: string; noun: string }) {
  const dialog = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    dialog.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const n = items.filter((it) => it.selected).length
  return createPortal(
    <div className="map-dialog-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="map-dialog" role="dialog" aria-modal="true" aria-label={`Pick ${noun} on the map`} tabIndex={-1} ref={dialog}>
        <header>
          <strong>Pick {noun} on the map</strong>
          <span className="site-picker-count">{limitNote ?? `${n} of ${items.length} selected`}</span>
          <button type="button" className="btn" onClick={onClose}>Done</button>
        </header>
        <SiteMap items={items} onToggle={onToggle} />
      </div>
    </div>,
    document.body,
  )
}

function SiteMap({ items, onToggle }: { items: SiteItem[]; onToggle: (key: string) => void }) {
  const wrap = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const { width } = useDimensions(wrap)
  const height = Math.max(320, Math.round(Math.min(window.innerHeight * 0.68, width * 0.6)))
  const [world, setWorld] = useState<FeatureCollection | null>(null)
  const [zoom, setZoom] = useState(d3.zoomIdentity)
  const [hover, setHover] = useState<SiteItem | null>(null)
  useEffect(() => { loadJson<FeatureCollection>('world.geojson').then(setWorld).catch(() => {}) }, [])

  const located = items.filter((it) => it.lat != null && it.lon != null && Number.isFinite(it.lat) && Number.isFinite(it.lon))
  // open on the middle 90 % of sites, so a few overseas IMPROVE sites do not shrink the US to a speck
  const frameKey = located.map((it) => it.key).join(',')
  const projection = useMemo(() => {
    const lons = located.map((it) => it.lon as number).sort(d3.ascending)
    const lats = located.map((it) => it.lat as number).sort(d3.ascending)
    const q = (a: number[], p: number) => d3.quantileSorted(a, p) ?? 0
    const [x0, x1] = located.length > 8 ? [q(lons, 0.05), q(lons, 0.95)] : [lons[0] ?? -180, lons[lons.length - 1] ?? 180]
    const [y0, y1] = located.length > 8 ? [q(lats, 0.05), q(lats, 0.95)] : [lats[0] ?? -60, lats[lats.length - 1] ?? 75]
    const pad = Math.max(4, 0.12 * Math.max(x1 - x0, y1 - y0))
    const box = { type: 'Feature', properties: {}, geometry: { type: 'MultiPoint', coordinates: [[x0 - pad, y0 - pad], [x1 + pad, y1 + pad], [x0 - pad, y1 + pad], [x1 + pad, y0 - pad]] } } as const
    return d3.geoNaturalEarth1().fitExtent([[12, 12], [Math.max(120, width - 12), height - 12]], box as any)
    // the frame follows the set of sites, not their selection
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameKey, width, height])
  const path = useMemo(() => d3.geoPath(projection), [projection])

  useEffect(() => {
    if (!svgRef.current) return
    const svg = d3.select(svgRef.current)
    const behavior = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 32]).on('zoom', (e) => setZoom(e.transform))
    zoomRef.current = behavior
    svg.call(behavior).on('dblclick.zoom', null)
    svg.call(behavior.transform, d3.zoomIdentity)
    return () => { svg.on('.zoom', null) }
  }, [projection])
  const zoomBy = (k: number) => { if (svgRef.current && zoomRef.current) d3.select(svgRef.current).transition().duration(200).call(zoomRef.current.scaleBy, k) }
  const reset = () => { if (svgRef.current && zoomRef.current) d3.select(svgRef.current).transition().duration(200).call(zoomRef.current.transform, d3.zoomIdentity) }

  return (
    <div ref={wrap} className="site-map">
      <div className="site-map-tools">
        <button type="button" className="btn quiet" aria-label="Zoom out" onClick={() => zoomBy(0.5)}>−</button>
        <button type="button" className="btn quiet" aria-label="Zoom in" onClick={() => zoomBy(2)}>+</button>
        <button type="button" className="btn quiet" onClick={reset}>Reset view</button>
      </div>
      <svg ref={svgRef} width={width} height={height} fontFamily={FONT.family} role="img" aria-label="Map of the sites; click a dot to select or deselect it">
        <rect width={width} height={height} fill="#f3f6fa" />
        <g transform={zoom.toString()}>
          {world && <path d={path(world) ?? ''} fill="#e1e6ed" stroke="#ffffff" strokeWidth={0.7 / zoom.k} />}
        </g>
        {located.map((it) => {
          const p = projection([it.lon as number, it.lat as number])
          if (!p) return null
          const [x, y] = zoom.apply(p)
          if (x < -12 || x > width + 12 || y < -12 || y > height + 12) return null
          const on = it.selected, hot = hover?.key === it.key
          return (
            <g key={it.key} transform={`translate(${x},${y})`} style={{ cursor: 'pointer' }}
              onClick={(e) => { e.stopPropagation(); onToggle(it.key) }} onMouseEnter={() => setHover(it)} onMouseLeave={() => setHover(null)}>
              {/* a generous invisible hit area, so small dots are easy to click */}
              <circle r={12} fill="transparent" />
              <circle r={on ? 7 : hot ? 6.5 : 5} fill={on ? it.color : '#ffffff'} stroke={on || hot ? INK.text : '#7d8796'} strokeWidth={on || hot ? 1.6 : 1.2} />
              {(on || hot) && <text x={10} y={4} fontSize={12} fontWeight={on ? 600 : 400} fill={INK.text} stroke="#ffffff" strokeWidth={3} paintOrder="stroke">{it.label}</text>}
            </g>
          )
        })}
      </svg>
      <div className="site-map-note">{hover ? `${hover.label}${hover.sub ? ` · ${hover.sub}` : ''} · ${hover.n.toLocaleString()} · click to ${hover.selected ? 'remove' : 'add'}` : 'Click a dot to add or remove a site · scroll or drag to move, or use − / +'}</div>
    </div>
  )
}
