import { useMemo, type ReactNode } from 'react'
import { useExport } from '@/lib/data'
import { FONT, INK } from '@/lib/theme'
import { cellsFromRose8, cellsFromSparse, ringLabelBearing, type WindCell } from '@/lib/windRose'
import type { RoseFile, WeatherSource } from '@/lib/weather'

/**
 * What the Weather tab's roses share (WindRoses, PollutionRose, PolarPlot): one panel layout with
 * its title above, the compass letters a fixed gap outside the outer ring, the ring labels along
 * a gap between sectors (not up the N axis, where they sat under "N") with a halo over the wedges,
 * and the footnote on its own line under "S"; the size rule for a row of them; and each filter's
 * hours by wind direction and speed.
 */

/** above the rose: its title; below: its footnote; around the outer ring: the compass letters */
const TITLE = 18, FOOT = 16, PAD = 24, COMPASS_GAP = 12
/** the least height between two ring labels */
const LABEL_GAP = 10
export const ROSE_SIZE = 210
/** the outer ring's radius in a rose `size` wide */
export const roseRadius = (size: number) => size / 2 - PAD

/**
 * The side of each rose: in a fitted tile, the column count that gives the largest rose fitting
 * both the width and the height; on a page, roses grow to share the row (210 to 300 px each).
 */
export function roseSize(n: number, width: number, fit: number | null) {
  if (!n) return ROSE_SIZE
  if (fit === null) return Math.round(Math.max(ROSE_SIZE, Math.min(300, (width - (n - 1) * 10) / n)))
  let best = 0
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols)
    best = Math.max(best, Math.min((width - (cols - 1) * 10) / cols, (height(fit, rows))))
  }
  return Math.round(Math.max(110, Math.min(280, best)))
}
// a row of roses in `fit` px: 10 px gaps, and each rose's title and footnote lines
const height = (fit: number, rows: number) => (fit - (rows - 1) * 10) / rows - TITLE - FOOT

/** a ring label's halo, so it reads over a wedge */
const HALO = { stroke: INK.panel, strokeWidth: 3, strokeLinejoin: 'round' as const, paintOrder: 'stroke' }

/**
 * One rose: the title, the rings and compass under the marks (`children`, drawn about the centre),
 * the ring labels over them, and the footnote. `nSectors` puts the ring labels in a gap between
 * sectors (lib/windRose ringLabelBearing).
 */
export function RosePanel({ size, title, titleColor, rings, nSectors, foot, children }: {
  size: number; title: string; titleColor?: string
  /** radius and label of each ring */
  rings: { r: number; label: string }[]
  nSectors: number; foot: ReactNode; children: ReactNode
}) {
  const R = roseRadius(size)
  const b = (ringLabelBearing(nSectors) * Math.PI) / 180
  // a label too close (in height) to the one inside it is left out: a square-root scale crowds its outer rings
  const labelled: typeof rings = []
  for (const t of rings.slice().sort((x, y) => x.r - y.r)) {
    const prev = labelled[labelled.length - 1]
    if (!prev || (t.r - prev.r) * Math.cos(b) >= LABEL_GAP) labelled.push(t)
  }
  return (
    <svg width={size} height={TITLE + size + FOOT} fontFamily={FONT.family}>
      <text x={size / 2} y={13} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={titleColor ?? INK.text}>{title}</text>
      <g transform={`translate(${size / 2},${TITLE + size / 2})`}>
        {/* the outer ring always, so the compass letters sit the same gap outside a drawn ring */}
        <circle r={R} fill="none" stroke={INK.grid} />
        {rings.map((t) => <circle key={t.label} r={t.r} fill="none" stroke={INK.grid} />)}
        {['N', 'E', 'S', 'W'].map((d, i) => <text key={d} x={Math.sin((i * Math.PI) / 2) * (R + COMPASS_GAP)} y={-Math.cos((i * Math.PI) / 2) * (R + COMPASS_GAP)}
          dy="0.35em" textAnchor="middle" fontSize={10} fill={INK.muted}>{d}</text>)}
        {children}
        {labelled.map((t) => <text key={t.label} x={Math.sin(b) * t.r + 2} y={-Math.cos(b) * t.r} dy="0.35em" fontSize={8.5} fill={INK.muted} {...HALO}
          style={{ pointerEvents: 'none' }}>{t.label}</text>)}
      </g>
      <text x={size / 2} y={TITLE + size + FOOT - 4} textAnchor="middle" fontSize={9} fill={INK.muted}>{foot}</text>
    </svg>
  )
}

/** the wind of each filter's window for the roses: hours by direction sector and speed bin */
export interface WindHours {
  /** 16 sectors from wind.json; 8 from roses.json until export_weather.py has written wind.json */
  sectors: string[]
  /** the speed bins' lower edges in m/s, the last bin open (wind.json only) */
  speedEdges: number[] | null
  calm: string
  of: (id: string) => { cells: WindCell[]; calm: number } | null
}

/**
 * The finest wind the export has: wind.json (16 sectors × 0.5 m/s bins) where export_weather.py has
 * written it, else the wind roses' 8 sectors × 3 speed classes. `fine` says which.
 */
export function useWindHours(source: WeatherSource): { wind: WindHours | null; fine: boolean; loading: boolean } {
  const fine = useExport<RoseFile & { speed_edges: number[]; era5: Record<string, number[]>; station: Record<string, number[]> }>('weather/wind.json')
  const coarse = useExport<RoseFile>('weather/roses.json')
  const wind = useMemo((): WindHours | null => {
    if (fine.data) {
      const table = source === 'station' ? fine.data.station : fine.data.era5, n = fine.data.speed_edges.length
      return { sectors: fine.data.sectors, speedEdges: fine.data.speed_edges, calm: fine.data.calm, of: (id) => (table[id] ? cellsFromSparse(table[id], n) : null) }
    }
    if (fine.loading || !coarse.data) return null
    const table = source === 'station' ? coarse.data.station : coarse.data.era5
    return { sectors: coarse.data.sectors, speedEdges: null, calm: coarse.data.calm, of: (id) => (table[id] ? cellsFromRose8(table[id]) : null) }
  }, [fine.data, fine.loading, coarse.data, source])
  return { wind, fine: !!fine.data, loading: fine.loading || (!fine.data && coarse.loading) }
}
