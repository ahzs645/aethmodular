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
      return res.json()
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
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect
      if (r.width > 0) setSize((s) => (s.width === r.width ? s : { ...s, width: r.width }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}
