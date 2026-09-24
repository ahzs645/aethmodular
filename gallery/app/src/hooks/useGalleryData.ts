import { useEffect, useState } from 'react'
import type { CalibFile, CalibRunsFile, CensusFile, FiltersFile, MetaFile, PmfFile } from '@/lib/types'

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

const BASE = `${import.meta.env.BASE_URL}data`

/**
 * Loads every exported dataset once and shares it across the app.
 * Regenerate the files with `python gallery/data/export_data.py`.
 */
export function useGalleryData() {
  const [data, setData] = useState<GalleryData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const get = async (name: string) => {
      const res = await fetch(`${BASE}/${name}.json`)
      if (!res.ok) throw new Error(`${name}.json -> HTTP ${res.status}`)
      if (!name.startsWith('calibration')) return res.json()
      // The calibration explorer names USPA "Pasadena"; the filter data (config.SITES)
      // call it JPL. Relabel on load so one site has one name on every tab.
      return JSON.parse((await res.text()).replace(/\bPasadena\b/g, 'JPL'))
    }
    // pmf.json is ETAD-only and optional — a checkout without the factor CSVs
    // should still render everything else.
    Promise.all([
      get('filters'),
      get('meta'),
      get('census'),
      get('pmf').catch(() => null),
      get('calibration').catch(() => null),
      get('calibration_runs').catch(() => null),
    ])
      .then(([filters, meta, census, pmf, calibration, calibrationRuns]) => {
        if (!cancelled) setData({ filters, meta, census, pmf, calibration, calibrationRuns })
      })
      .catch((e) => !cancelled && setError(String(e)))
    return () => {
      cancelled = true
    }
  }, [])

  return { data, error }
}

/** Observes an element and reports its pixel size, so charts stay responsive. */
export function useDimensions<T extends HTMLElement>(ref: React.RefObject<T>) {
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
  return size
}
