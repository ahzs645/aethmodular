import { useMemo, useRef, useState } from 'react'
import { MapMarker, MarkerContent } from '@pgmaps/geo-toolkit/map'
import { GeoMap, type GeoMapHandle } from '@/components/GeoMap'
import type { SiteItem } from '@/components/SitePicker'
import { INK } from '@/lib/theme'

/** The Map… dialog's map (components/SitePicker): click a site's dot to add or remove it. */
export function SiteMapPicker({ items, onToggle }: { items: SiteItem[]; onToggle: (key: string) => void }) {
  const geo = useRef<GeoMapHandle>(null)
  const height = Math.max(320, Math.round(Math.min(window.innerHeight * 0.68, window.innerWidth * 0.55)))
  const [hover, setHover] = useState<SiteItem | null>(null)
  const located = items.filter((it) => it.lat != null && it.lon != null && Number.isFinite(it.lat) && Number.isFinite(it.lon))
  // open on the middle 90 % of sites, so a few overseas IMPROVE sites do not shrink the US to a speck;
  // the frame follows the set of sites, not their selection
  const frameKey = located.map((it) => it.key).join(',')
  const points = useMemo(() => located.map((it) => [it.lon as number, it.lat as number] as const),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [frameKey])
  const frame = { trim: points.length > 8 ? 0.05 : 0, pad: 0.12 }
  const selected = located.filter((it) => it.selected)

  return (
    <div className="site-map">
      <div className="site-map-tools">
        <button type="button" className="btn quiet" onClick={() => geo.current?.fit(points, frame)}>Reset view</button>
        <button type="button" className="btn quiet" disabled={!selected.length} title="Frame the selected sites"
          onClick={() => geo.current?.fit(selected.map((it) => [it.lon as number, it.lat as number]), { pad: 0.2 })}>Fit selected</button>
      </div>
      <GeoMap ref={geo} height={height} frame={points} frameOptions={frame} frameKey={frameKey} label="Map of the sites; click a dot to select or deselect it">
        {located.map((it) => {
          const on = it.selected, hot = hover?.key === it.key
          return (
            <MapMarker key={it.key} longitude={it.lon as number} latitude={it.lat as number} onClick={it.disabled ? undefined : () => onToggle(it.key)}>
              <MarkerContent className="site-map-dot">
                {/* a generous hit area, so small dots are easy to click */}
                <svg width={24} height={24} viewBox="-12 -12 24 24" onMouseEnter={() => setHover(it)} onMouseLeave={() => setHover(null)}
                  role="checkbox" aria-checked={on} aria-label={`${it.label}${it.sub ? `, ${it.sub}` : ''}`}>
                  <circle r={12} fill="transparent" />
                  <circle r={on ? 7 : hot ? 6.5 : 5} fill={on ? it.color : INK.panel} stroke={on || hot ? INK.text : '#7d8796'} strokeWidth={on || hot ? 1.6 : 1.2} />
                </svg>
                {(on || hot) && <span className="site-map-label" style={{ fontWeight: on ? 600 : 400 }}>{it.label}</span>}
              </MarkerContent>
            </MapMarker>
          )
        })}
      </GeoMap>
      <div className="site-map-note">{hover ? `${hover.label}${hover.sub ? ` · ${hover.sub}` : ''} · ${hover.n.toLocaleString()} · click to ${hover.selected ? 'remove' : 'add'}` : 'Click a dot to add or remove a site · drag to pan, scroll or use + / − to zoom'}</div>
    </div>
  )
}
