import { useCallback, useEffect, useState } from 'react'
import { loadDerived, loadJson } from '@/lib/data'
import { normalizeSiteIdentity } from '@/lib/metricDefinition'
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
export function useGalleryData(includeCalibration: boolean | 'runs' = false) {
  const [data, setData] = useState<GalleryData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState({ grid: false, runs: false })
  const [calibrationError, setCalibrationError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const retryCalibration = useCallback(() => setAttempt(n => n + 1), [])

  useEffect(() => {
    let cancelled = false
    // cached in lib/data, so a remount (or React's development double effect) does not download twice
    // pmf.json is ETAD-only and optional — a checkout without the factor CSVs
    // should still render everything else.
    Promise.all([
      loadJson<FiltersFile>('filters.json'),
      loadJson<MetaFile>('meta.json'),
      loadJson<CensusFile>('census.json'),
      loadJson<PmfFile>('pmf.json').catch(() => null),
    ])
      .then(([filters, meta, census, pmf]) => {
        if (!cancelled) setData({ filters, meta, census, pmf, calibration: null, calibrationRuns: null })
      })
      .catch((e) => !cancelled && setError(String(e)))
    return () => {
      cancelled = true
    }
  }, [])

  // A dashboard asks only for the runs used by its calibration tiles. Full explorer pages
  // also need the grid. The two files share the existing request/derived caches.
  const ready = !!data
  useEffect(() => {
    if (!includeCalibration || !ready) return
    let live = true
    setCalibrationError('')
    const get = <T,>(name: string) => loadDerived<T>(`gallery:${name}`, async () =>
      normalizeSiteIdentity(await loadJson<T>(`${name}.json`)))
    const grid = includeCalibration === true
    Promise.allSettled([grid ? get<CalibFile>('calibration') : Promise.resolve(null), get<CalibRunsFile>('calibration_runs')])
      .then(([calibration, runs]) => {
        if (!live) return
        setData(d => d ? { ...d,
          calibration: calibration.status === 'fulfilled' ? calibration.value ?? d.calibration : d.calibration,
          calibrationRuns: runs.status === 'fulfilled' ? runs.value : d.calibrationRuns } : d)
        setLoaded(old => ({ grid: old.grid || (grid && calibration.status === 'fulfilled'), runs: old.runs || runs.status === 'fulfilled' }))
        const failures = [calibration, runs].flatMap(r => r.status === 'rejected' ? [String(r.reason)] : [])
        setCalibrationError(failures.join('; '))
      })
    return () => { live = false }
  }, [includeCalibration, ready, attempt])

  return { data, error, calibrationError, retryCalibration,
    calibrationLoading: !!includeCalibration && !calibrationError && (!loaded.runs || (includeCalibration === true && !loaded.grid)) }
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
