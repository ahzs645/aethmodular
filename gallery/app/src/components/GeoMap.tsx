import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode, type RefObject } from 'react'
import { Map as ToolkitMap, MapControls, fitMapToPoints, geojsonBasemapStyle, useMap, type FitMapToPointsOptions, type LngLatPoint, type MapRef } from '@pgmaps/geo-toolkit/map'
import '@pgmaps/geo-toolkit/styles.css'
import { dataUrl } from '@/lib/data'
import { INK_DARK, INK_LIGHT, type Ink } from '@/lib/theme'
import { useTheme } from '@/lib/colorTheme'

/**
 * The gallery's map: @pgmaps/geo-toolkit's MapLibre <Map> (gallery/packages/geo-toolkit) over the offline
 * Natural Earth outlines in public/data/world.geojson (gallery/data/fetch_basemap.py), so a map needs no
 * tile server, like the rest of the static site. Sea and land take the theme's ink; markers drawn in light
 * colours are recoloured for the dark page by lib/themeCss like any other chart mark.
 *
 * The canvas keeps its drawing buffer, so lib/export can copy the basemap into a saved figure under the
 * markers' SVG.
 */
const basemap = (ink: Ink, globe: boolean) => geojsonBasemapStyle({
  land: new URL(dataUrl('world.geojson'), window.location.href).href,
  water: ink.sea, landColor: ink.land, borderColor: ink.panel, borderWidth: 0.6,
  // Mercator stretches the poles, so the graticule is there for the globe's sake
  graticule: globe ? 30 : false, graticuleColor: ink.grid,
})
const STYLES = { flat: { light: basemap(INK_LIGHT, false), dark: basemap(INK_DARK, false) },
  globe: { light: basemap(INK_LIGHT, true), dark: basemap(INK_DARK, true) } }

export interface GeoMapHandle {
  map: MapRef | null
  /** frame these [lon, lat] points (lib: pointBounds); false when none are finite */
  fit: (points: readonly LngLatPoint[], options?: FitMapToPointsOptions) => boolean
  zoomBy: (factor: number) => void
}

export const GeoMap = forwardRef<GeoMapHandle, {
  children?: ReactNode
  height: number
  /** the points the camera frames on load, and again whenever `frameKey` changes (a new scope) */
  frame?: readonly LngLatPoint[]
  frameOptions?: FitMapToPointsOptions
  /** defaults to the points themselves; pass e.g. the scope name so a new selection does not move the camera */
  frameKey?: string
  label: string
  /** a 3-D globe instead of Web Mercator */
  globe?: boolean
  className?: string
}>(function GeoMap({ children, height, frame, frameOptions, frameKey, label, globe = false, className }, ref) {
  const theme = useTheme()
  const mapRef = useRef<MapRef | null>(null)
  const bandRef = useRef<HTMLDivElement>(null)
  useImperativeHandle(ref, () => ({
    get map() { return mapRef.current },
    fit: (points, options) => (mapRef.current ? fitMapToPoints(mapRef.current, points, { padding: 36, ...options }) : false),
    zoomBy: (factor) => { const m = mapRef.current; if (m) m.easeTo({ zoom: m.getZoom() + Math.log2(factor), duration: 200 }) },
  }), [])
  return (
    <div ref={bandRef} className={`geo-map ${className ?? ''}`} style={{ height }} role="region" aria-label={label} data-zoom="far">
      <ToolkitMap ref={mapRef} theme={theme} styles={globe ? STYLES.globe : STYLES.flat} loader="spinner"
        controls={<MapControls position="top-right" showZoom showCompass={globe} />}
        center={[15, 15]} zoom={0.4} minZoom={0} maxZoom={11}
        dragRotate={globe} pitchWithRotate={false} touchPitch={false} attributionControl={false}
        canvasContextAttributes={{ preserveDrawingBuffer: true }}>
        <Projection globe={globe} />
        <ZoomBand target={bandRef} />
        {frame && <Frame points={frame} options={frameOptions} frameKey={frameKey ?? frame.map((p) => p.join(',')).join(';')} />}
        {children}
      </ToolkitMap>
    </div>
  )
})

/** MapLibre sets a projection per style; re-apply it whenever the style (theme) or the toggle changes */
function Projection({ globe }: { globe: boolean }) {
  const { map, isLoaded } = useMap()
  useEffect(() => {
    if (!map || !isLoaded) return
    map.setProjection({ type: globe ? 'globe' : 'mercator' })
  }, [map, isLoaded, globe])
  return null
}

/**
 * data-zoom="near" on the map once it is zoomed past a continent: labels marked .when-near (styles.css) show
 * only then, so a world view of 40 sites is not a wall of names. Set on the element, not in state, so zooming
 * does not re-render the markers.
 */
function ZoomBand({ target }: { target: RefObject<HTMLDivElement> }) {
  const { map } = useMap()
  useEffect(() => {
    if (!map) return
    const set = () => target.current?.setAttribute('data-zoom', map.getZoom() >= 2.6 ? 'near' : 'far')
    set()
    map.on('zoom', set)
    return () => { map.off('zoom', set) }
  }, [map, target])
  return null
}

/** frame the points once the style has loaded (a jump the first time), and on each new frameKey */
function Frame({ points, options, frameKey }: { points: readonly LngLatPoint[]; options?: FitMapToPointsOptions; frameKey: string }) {
  const { map, isLoaded } = useMap()
  const [framed, setFramed] = useState<string | null>(null)
  useEffect(() => {
    if (!map || !isLoaded || framed === frameKey || !points.length) return
    if (fitMapToPoints(map, points, { padding: 36, ...options, duration: framed === null ? 0 : 400 })) setFramed(frameKey)
  }, [map, isLoaded, framed, frameKey, points, options])
  return null
}

/** a toolkit marker's native mouse event in the shape hooks/useTooltip reads */
export const asReactMouse = (e: MouseEvent) =>
  ({ clientX: e.clientX, clientY: e.clientY, nativeEvent: e, currentTarget: e.currentTarget, target: e.target }) as unknown as ReactMouseEvent
