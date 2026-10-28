import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { fidProps } from '@/lib/filterEdits'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { spearman } from '@/lib/weather'
import type { FilterRow, MetaFile } from '@/lib/types'

const PW = 'Precipitable water (AERONET)'
const MEASURES = ['Dew point', 'Relative humidity'] as const
type Measure = (typeof MEASURES)[number]
const MIN_N = 8
const STATION = '#5b6470'

/**
 * Dew point (°C) from temperature (°C) and relative humidity (%), Magnus form with Alduchov and
 * Eskridge's (1996) constants. Applied to the station's window means, so it is approximate.
 */
export function dewPoint(t: number, rh: number): number | null {
  if (!(rh > 0) || rh > 100 || !Number.isFinite(t)) return null
  const a = 17.625, b = 243.04
  const g = Math.log(rh / 100) + (a * t) / (b + t)
  return (b * g) / (a - g)
}

interface Point { row: FilterRow; x: number; y: number }
interface Series { key: string; label: string; color: string; hollow: boolean; pts: Point[]; rho: number | null; fit: { a: number; b: number } | null }

/** OLS of ln(PW) on dew point: PW = exp(a + b·Td), the near-exponential rise of column water with surface dew point. */
function logFit(pts: Point[]): { a: number; b: number } | null {
  const ok = pts.filter((p) => p.y > 0)
  if (ok.length < MIN_N) return null
  const xs = ok.map((p) => p.x), ys = ok.map((p) => Math.log(p.y))
  const mx = d3.mean(xs)!, my = d3.mean(ys)!
  const sxx = d3.sum(xs, (x) => (x - mx) ** 2)
  if (!sxx) return null
  const b = d3.sum(xs, (x, i) => (x - mx) * (ys[i] - my)) / sxx
  return { a: my - b * mx, b }
}

/**
 * The column's water against the surface's, over the same filter windows: AERONET precipitable
 * water against ERA5 dew point or relative humidity, and at Addis Ababa also against the Bole airport
 * station. The MAC–PW comparison (notebooks/analysis/meteorology/pw_mac_calibration) reads moisture
 * from the photometer, which only retrieves in clear daylight; if PW tracks the reanalysis' surface
 * moisture over the windows, that moisture is the one the filters saw.
 */
export function PwMoisture({ rows, meta, onePick }: { rows: FilterRow[]; meta: MetaFile; onePick: boolean }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const [measure, setMeasure] = useSearchParam<Measure>('pwmoist', 'Dew point', MEASURES)
  const isDew = measure === 'Dew point'

  const series = useMemo<Series[]>(() => {
    const out: Series[] = []
    const make = (key: string, label: string, color: string, hollow: boolean, pts: Point[]) => {
      if (pts.length < MIN_N) return
      out.push({ key, label, color, hollow, pts, rho: spearman(pts.map((p) => p.x), pts.map((p) => p.y)), fit: isDew ? logFit(pts) : null })
    }
    const num = (r: FilterRow, f: string) => (typeof r[f] === 'number' && Number.isFinite(r[f]) ? (r[f] as number) : null)
    for (const [site, rs] of d3.groups(rows.filter((r) => num(r, PW) !== null), (r) => r.site)) {
      const color = meta.sites.find((s) => s.name === site)?.color ?? INK.accent
      make(`${site}|era5`, `${site} · ERA5`, color, false, rs.flatMap((r) => {
        const x = num(r, isDew ? 'Dew point (ERA5)' : 'Relative humidity (ERA5)')
        return x === null ? [] : [{ row: r, x, y: num(r, PW)! }]
      }))
      make(`${site}|station`, `${site} · airport station${isDew ? ' (from T, RH)' : ''}`, onePick ? STATION : color, true, rs.flatMap((r) => {
        const t = num(r, 'Temperature (station)'), rh = num(r, 'Relative humidity (station)')
        const x = isDew ? (t !== null && rh !== null ? dewPoint(t, rh) : null) : rh
        return x === null ? [] : [{ row: r, x, y: num(r, PW)! }]
      }))
    }
    return out
  }, [rows, meta.sites, isDew, onePick])

  // at Addis: how far ERA5's window dew point sits from the station's
  const stationGap = useMemo(() => {
    const d = rows.flatMap((r) => {
      const e = r['Dew point (ERA5)'], t = r['Temperature (station)'], rh = r['Relative humidity (station)']
      if (typeof e !== 'number' || typeof t !== 'number' || typeof rh !== 'number') return []
      const s = dewPoint(t, rh)
      return s === null ? [] : [e - s]
    })
    d.sort(d3.ascending)
    return d.length >= MIN_N ? { n: d.length, median: d3.quantileSorted(d, 0.5)!, p25: d3.quantileSorted(d, 0.25)!, p75: d3.quantileSorted(d, 0.75)! } : null
  }, [rows])

  const all = series.flatMap((s) => s.pts)
  const sq = Math.max(240, Math.min(440, width - 80))
  const M = { top: 12, left: 56, bottom: 44, right: 14 }
  const x = d3.scaleLinear().domain(d3.extent(all, (p) => p.x) as [number, number]).nice().range([0, sq])
  const y = d3.scaleLinear().domain([0, (d3.max(all, (p) => p.y) ?? 1) * 1.05]).nice().range([sq, 0])
  const curve = (s: Series) => {
    if (!s.fit) return ''
    const [lo, hi] = d3.extent(s.pts, (p) => p.x) as [number, number]
    return d3.line<number>().x((v) => x(v)).y((v) => y(Math.exp(s.fit!.a + s.fit!.b * v)))(d3.range(lo, hi, (hi - lo) / 40).concat(hi)) ?? ''
  }
  const xLabel = isDew ? 'Surface dew point over the window (°C)' : 'Surface relative humidity over the window (%)'

  return (
    <ChartFrame id="weather-pw-moisture" source="gallery/app/src/charts/weather/PwMoisture.tsx"
      title={`Column water against surface moisture — AERONET precipitable water against ${isDew ? 'dew point' : 'relative humidity'}`}
      subtitle={`Each filter: the mean AERONET precipitable water inside its sampling window against the ${measure.toLowerCase()} averaged over the same window, from ERA5 at every site and, at Addis Ababa, from the Bole airport station (hollow${isDew ? '; its dew point computed from the window-mean temperature and humidity' : ''}). Spearman ρ per series${isDew ? '; the curves fit ln PW = a + b·Td, the near-exponential rise of column water with surface dew point' : ''}. The photometer averages only clear daylight retrievals while ERA5 and the station cover every hour, so the cross-check also shows how representative the clear-sky PW is of the whole window. This is the moisture behind the MAC–PW comparison.`}
      provenance="AERONET Version 3 precipitable water (export_aeronet.py); ERA5 and Bole airport (Meteostat, WMO 63450) over the same windows (export_weather.py); notebooks/analysis/meteorology/pw_mac_calibration"
      controls={<Segmented label="Surface moisture" value={measure} options={MEASURES} onChange={setMeasure} />}>
      <div ref={wrapRef} className="chart-wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'flex-start' }}>
        {series.length === 0 ? <Empty>No site in view has {MIN_N} filters with an AERONET precipitable water and ERA5 moisture over the window.</Empty> : <>
          <svg width={sq + M.left + M.right} height={sq + M.top + M.bottom} fontFamily={FONT.family} role="img" aria-label={`Precipitable water against ${measure.toLowerCase()}`}>
            <g transform={`translate(${M.left},${M.top})`}>
              <YAxis scale={y} x={0} label="Precipitable water, AERONET (cm)" gridWidth={sq} labelX={-42} />
              <XAxis scale={x} y={sq} label={xLabel} />
              {series.map((s) => <g key={s.key}>
                {s.pts.map((p) => {
                  const st = focusStyle(p.row.id, hl.focusId, { r: 3 }, hl.selected)
                  return <circle key={p.row.id} cx={x(p.x)} cy={y(p.y)} r={st.r} fill={s.hollow ? 'none' : s.color} fillOpacity={st.opacity}
                    stroke={s.hollow ? s.color : st.stroke} strokeWidth={s.hollow ? 1.2 : st.strokeWidth} strokeOpacity={s.hollow ? 0.8 : 1} data-dot={s.hollow ? undefined : st.dot}
                    style={{ cursor: 'pointer' }} {...fidProps(p.row.id)}
                    onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${s.label} · ${p.row.date}`, `PW ${fmt(p.y, 2)} cm`, `${measure} ${fmt(p.x, 1)} ${isDew ? '°C' : '%'}`]) }}
                    onMouseLeave={() => { hl.setHover(null); tip.hide() }} onClick={(e) => { e.stopPropagation(); hl.openSample(p.row.id) }} />
                })}
              </g>)}
              {series.map((s) => s.fit && <path key={`${s.key}-fit`} d={curve(s)} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.hollow ? '5 3' : undefined} />)}
            </g>
          </svg>
          <div style={{ minWidth: 0, flex: '1 1 260px' }}>
            <div className="chart-scroll" tabIndex={0} role="region" aria-label="Precipitable water against surface moisture, per series">
            <table className="census-table">
              <thead><tr><th>Series</th><th>n</th><th style={{ textTransform: 'none' }}>Spearman ρ</th>{isDew && <th>PW per °C</th>}</tr></thead>
              <tbody>{series.map((s) => <tr key={s.key}>
                <td><span aria-hidden="true" style={{ display: 'inline-block', width: 9, height: 9, borderRadius: '50%', marginRight: 6, background: s.hollow ? 'transparent' : s.color, border: `1.5px solid ${s.color}` }} />{s.label}</td>
                <td style={{ whiteSpace: 'nowrap' }}>{s.pts.length}</td><td style={{ whiteSpace: 'nowrap' }}>{fmt(s.rho, 2)}</td>
                {isDew && <td style={{ whiteSpace: 'nowrap' }}>{s.fit ? `${s.fit.b >= 0 ? '+' : '−'}${fmt(Math.abs(Math.expm1(s.fit.b) * 100), 1)} %` : '—'}</td>}
              </tr>)}</tbody>
            </table>
            </div>
            {isDew && <p className="chart-note">PW per °C: the fitted growth of column water per degree of surface dew point (Reitan’s 1963 empirical relation gives about 6 % per °C).</p>}
          </div>
        </>}
        {tip.node}
      </div>
      {stationGap && <Note>At Addis Ababa, ERA5's window dew point minus the station's (from its temperature and humidity): median {fmt(stationGap.median, 1)} °C, interquartile range {fmt(stationGap.p25, 1)} to {fmt(stationGap.p75, 1)} °C, over {stationGap.n} filters with both (whether or not the photometer saw them).</Note>}
    </ChartFrame>
  )
}
