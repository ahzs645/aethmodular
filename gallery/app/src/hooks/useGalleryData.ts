import { useEffect, useState } from 'react'
import { loadDerived, loadJson } from '@/lib/data'
import type { CalibFile, CalibRunsFile, CensusFile, FiltersFile, MetaFile, PmfFile } from '@/lib/types'
import { useTileLegend } from '@/lib/chartFit'

export interface GalleryData {
  filters: FiltersFile
  meta: MetaFile
  census: CensusFile
  pmf: PmfFile | null
  /** phase-3 calibration grid from the explorer's batch results; optional */
  calibration: CalibFile | null
  /** per-filter readouts and cohort diagnostics from the explorer; optional */
  calibrationRuns: CalibRunsFile | null
}


/**
 * Loads every exported dataset once and shares it across the app.
 * Regenerate the files with `python gallery/data/export_data.py`.
 */
export function useGalleryData(includeCalibration = false) {
  const [data, setData] = useState<GalleryData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [calibrationLoaded, setCalibrationLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    // cached in lib/data, so a remount (or React's development double effect) does not download twice
    const get = (name: string) => loadDerived(`gallery:${name}`, async () => {
      const value = await loadJson<unknown>(`${name}.json`)
      if (!name.startsWith('calibration')) return value
      // The calibration explorer names USPA "Pasadena"; the filter data (config.SITES)
      // call it JPL. Relabel on load so one site has one name on every tab.
      return JSON.parse(JSON.stringify(value).replace(/\bPasadena\b/g, 'JPL'))
    })
    // pmf.json is ETAD-only and optional — a checkout without the factor CSVs
    // should still render everything else.
    Promise.all([
      get('filters'),
      get('meta'),
      get('census'),
      get('pmf').catch(() => null),
    ])
      .then(([filters, meta, census, pmf]) => {
        if (!cancelled) setData({ filters, meta, census, pmf, calibration: null, calibrationRuns: null })
      })
      .catch((e) => !cancelled && setError(String(e)))
    return () => {
      cancelled = true
    }
  }, [])

  // Large calibration exports are needed only by their pages/dashboard tiles.
  // They never delay the first render of the filter charts.
  const ready = !!data
  useEffect(() => {
    if (!includeCalibration || !ready) return
    let live = true
    const get = <T,>(name: string) => loadDerived<T>(`gallery:${name}`, async () =>
      JSON.parse(JSON.stringify(await loadJson(`${name}.json`)).replace(/\bPasadena\b/g, 'JPL')))
    Promise.all([get<CalibFile>('calibration').catch(() => null), get<CalibRunsFile>('calibration_runs').catch(() => null)])
      .then(([calibration, calibrationRuns]) => {
        if (live) {
          setData(d => d ? { ...d, calibration, calibrationRuns } : d)
          setCalibrationLoaded(true)
        }
      })
    return () => { live = false }
  }, [includeCalibration, ready])

  return { data, error, calibrationLoading: includeCalibration && !calibrationLoaded }
}

/** Observes an element and reports its pixel size, so charts stay responsive. */
export function useDimensions<T extends HTMLElement>(ref: React.RefObject<T>) {
  // a Dashboard tile with its legend on the right keeps that column out of the chart's width
  const legend = useTileLegend()
  const [size, setSize] = useState({ width: 720, height: 460 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    // measure once now: the observer's first callback can lag a frame (or never
    // come in a background tab), and until then every chart drew at 720 px
    const w0 = el.getBoundingClientRect().width
    if (w0 > 0) setSize((s) => (s.width === w0 ? s : { ...s, width: w0 }))
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect
      if (r.width > 0) setSize((s) => (s.width === r.width ? s : { ...s, width: r.width }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return legend?.mode === 'right' ? { ...size, width: Math.max(120, size.width - legend.reserve) } : size
}
