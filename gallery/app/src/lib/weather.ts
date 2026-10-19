/**
 * Weather over each filter's sampling window (gallery/data/export_weather.py): ERA5 for
 * every SPARTAN filter, and the Addis airport station (Meteostat, WMO 63450) for the Addis
 * filters it covers. App merges the per-filter values into the rows as ordinary fields
 * ("Temperature (ERA5)" …), so every chart can plot them; the Weather tab also reads the
 * daily series, the wind roses and the station comparison from here.
 */
import { useEffect, useState } from 'react'
import { loadJson, useExport } from './data.ts'

export interface WeatherFile {
  source: string
  note: string
  fields: string[]
  field_units: Record<string, string>
  era5: string[]
  station: string[]
  /** id -> values in `fields` order */
  rows: Record<string, (number | null)[]>
  /** id -> [local start, local end, hours sampled, window assumed] */
  windows: Record<string, [string, string, number | null, boolean]>
}

/** weather/index.json: the fields, units and sources, and how many filters each site's file holds */
export interface WeatherIndex extends Omit<WeatherFile, 'rows' | 'windows'> { sites: Record<string, number> }

/**
 * The per-filter weather for the sites in view: the small index once, then one file per site
 * (weather/filters/<CODE>.json, export_weather.py), so a page loads only the sites it shows.
 * While a newly picked site loads, the previous result stays, so the rows do not lose their
 * weather fields in between. Weather is optional: a site without a file simply has none.
 */
export function useWeather(codes: readonly string[]): WeatherFile | null {
  const index = useExport<WeatherIndex>('weather/index.json').data
  const key = index ? [...new Set(codes)].filter((c) => index.sites[c]).sort().join(',') : ''
  const [state, setState] = useState<{ key: string; data: WeatherFile } | null>(null)
  useEffect(() => {
    if (!index) return
    let live = true
    const want = key ? key.split(',') : []
    Promise.all(want.map((c) => loadJson<Pick<WeatherFile, 'rows' | 'windows'>>(`weather/filters/${c}.json`)))
      .then((parts) => {
        if (!live) return
        const { sites: _sites, ...rest } = index
        setState({ key, data: { ...rest, rows: Object.assign({}, ...parts.map((p) => p.rows)), windows: Object.assign({}, ...parts.map((p) => p.windows)) } })
      })
      .catch(() => { /* weather is optional; the rows go without it */ })
    return () => { live = false }
  }, [index, key])
  return state?.data ?? null
}

export interface RoseFile {
  sectors: string[]
  speed_classes: string[]
  calm: string
  /** id -> 8 sectors x 3 speed classes (row-major), then calm hours */
  era5: Record<string, number[]>
  station: Record<string, number[]>
}

export interface DailyWeather {
  date: string[]
  t: (number | null)[]
  tmin: (number | null)[]
  tmax: (number | null)[]
  p: (number | null)[]
  rh: (number | null)[]
  ws: (number | null)[]
  blh: (number | null)[]
}

export interface StationCompare {
  date: string[]
  station: Record<'t' | 'tmin' | 'tmax' | 'p' | 'rh' | 'ws', (number | null)[]>
  era5: Record<'t' | 'tmin' | 'tmax' | 'p' | 'rh' | 'ws', (number | null)[]>
}

export const WEATHER_GROUP = 'Weather (sampling window)'
export const loadDaily = (code: string) => loadJson<DailyWeather>(`weather/daily/${code}.json`)

export type WeatherSource = 'ERA5' | 'station'
/** The quantities the tab's charts offer, and the field each source keeps them under. */
export const WEATHER_VARS = [
  { key: 'temp', label: 'Temperature', era5: 'Temperature (ERA5)', station: 'Temperature (station)' },
  { key: 'range', label: 'Daily temperature range', era5: 'Temperature range (ERA5)', station: 'Temperature range (station)' },
  { key: 'prcp', label: 'Precipitation', era5: 'Precipitation (ERA5)', station: 'Precipitation (station)' },
  { key: 'rh', label: 'Relative humidity', era5: 'Relative humidity (ERA5)', station: 'Relative humidity (station)' },
  { key: 'dew', label: 'Dew point', era5: 'Dew point (ERA5)', station: null },
  { key: 'wspd', label: 'Wind speed', era5: 'Wind speed (ERA5)', station: 'Wind speed (station)' },
  { key: 'pres', label: 'Pressure', era5: 'Surface pressure (ERA5)', station: 'Sea-level pressure (station)' },
  { key: 'blh', label: 'Boundary-layer height', era5: 'Boundary-layer height (ERA5)', station: null },
] as const
export type WeatherVarKey = (typeof WEATHER_VARS)[number]['key']
export const weatherField = (key: WeatherVarKey, source: WeatherSource): string | null => {
  const v = WEATHER_VARS.find((x) => x.key === key)!
  return source === 'station' ? v.station : v.era5
}

/**
 * Rain classes for a filter window's mean rate (mm/day), the bins addis_03_meteorology used
 * for daily totals: none, light (< 2.5), moderate (2.5–10), heavy (≥ 10). A 24 h window's
 * mean rate is its daily total.
 */
export const RAIN_CLASSES = [
  { key: 'No rain', max: 0.1, color: '#d9c9a3' },
  { key: 'Light (< 2.5 mm/day)', max: 2.5, color: '#9ecae1' },
  { key: 'Moderate (2.5–10)', max: 10, color: '#4292c6' },
  { key: 'Heavy (≥ 10)', max: Infinity, color: '#084594' },
] as const
export const rainClass = (mmPerDay: number) => RAIN_CLASSES.find((c) => mmPerDay < c.max)!.key

/** Spearman rank correlation (average ranks for ties); null below 5 pairs. */
export function spearman(xs: number[], ys: number[]): number | null {
  const pairs = xs.map((x, i) => [x, ys[i]]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b))
  if (pairs.length < 5) return null
  const rank = (v: number[]) => {
    const idx = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0])
    const r = new Array<number>(v.length)
    for (let i = 0; i < idx.length;) {
      let j = i
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1
      i = j + 1
    }
    return r
  }
  const rx = rank(pairs.map((p) => p[0])), ry = rank(pairs.map((p) => p[1]))
  const n = rx.length, mx = (n + 1) / 2
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) { const a = rx[i] - mx, b = ry[i] - mx; sxy += a * b; sxx += a * a; syy += b * b }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null
}

/** Unit-vector mean of directions in degrees (0 = N, clockwise). */
export function meanDirection(degrees: number[], weights?: number[]): number | null {
  let u = 0, v = 0
  degrees.forEach((d, i) => { const w = weights?.[i] ?? 1; u += w * Math.sin((d * Math.PI) / 180); v += w * Math.cos((d * Math.PI) / 180) })
  if (!u && !v) return null
  return ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360
}
