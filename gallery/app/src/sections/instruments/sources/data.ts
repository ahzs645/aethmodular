/**
 * What the Sources (AAE) page loads beyond the rows: each site's daily bins (sections/instruments/lib.ts)
 * and the instruments' own apportionment (gallery/data/export_aeth_sources.py), plus the PMF solution's
 * µg/m³ contributions for the K_F dominance the rows do not carry.
 */
import { useEffect, useMemo, useState } from 'react'
import { useExport } from '@/lib/data'
import type { MetaFile, PmfFile } from '@/lib/types'
import { loadAethDaily, type AethDaily } from '../lib'

/**
 * The Jacros file (df_Jacros_9am_resampled.pkl): the same MA350-0238 as the Addis filter-site file, over a
 * longer span (Apr 2022 to Jun 2025 against Dec 2022 to Sep 2024). On the 515 bins both files hold, every
 * channel is identical, so it stands in for Addis as "the whole record", never beside it as a second instrument.
 */
export const JACROS = 'ETAD-JACROS'

export interface DailySite { code: string; name: string; color: string; daily: AethDaily }

/** a daily file's site: the gallery's site, or the Jacros file as Addis's whole record (in Addis's colour) */
export function siteOf(code: string, meta: MetaFile): { name: string; color: string; site: string } | null {
  const own = meta.sites.find((s) => s.code === code)
  if (own) return { name: own.name, color: own.color, site: own.name }
  if (code === JACROS) {
    const addis = meta.sites.find((s) => s.code === 'ETAD')
    return addis ? { name: `${addis.name} (whole record)`, color: addis.color, site: addis.name } : null
  }
  return null
}

/** the daily bins of these site codes, in the order given (a site with no file is left out) */
export function useDailySites(codes: readonly string[], meta: MetaFile): DailySite[] | null {
  const key = codes.join(',')
  const [state, setState] = useState<{ key: string; sites: DailySite[] } | null>(null)
  useEffect(() => {
    let live = true
    const want = key ? key.split(',') : []
    Promise.all(want.map((c) => loadAethDaily(c).then((daily) => ({ c, daily })).catch(() => null)))
      .then((parts) => {
        if (!live) return
        const sites = parts.flatMap((p) => {
          const s = p && siteOf(p.c, meta)
          return p && s ? [{ code: p.c, name: s.name, color: s.color, daily: p.daily }] : []
        })
        setState({ key, sites })
      })
    return () => { live = false }
  }, [key, meta])
  return state?.key === key ? state.sites : null
}

/**
 * sources/instrument.json: the instrument's own AAE and biomass share per daily bin, from two places:
 * the processed pickles' columns (`daily`) and the MAIA hourly exports averaged over the bin (`hourly`).
 */
export interface InstrumentSeries { start: string[]; aae: (number | null)[]; bb_pct: (number | null)[] }
export interface InstrumentSite {
  site: string
  daily: InstrumentSeries & { n: number; aae_fossil: number | null; aae_biomass: number | null; bb_bcc: (number | null)[]; ff_bcc: (number | null)[] }
  hourly?: InstrumentSeries & { n: number; file: string; hours: number[] }
}
export interface InstrumentFile { source: string; note: string; sites: Record<string, InstrumentSite> }
export const useInstrument = () => useExport<InstrumentFile>('aethalometer/sources/instrument.json').data

/** the PMF solution (ETAD only): the same file the PMF sources tab reads */
export const usePmf = () => useExport<PmfFile>('pmf.json').data

/** a filter's PMF day, as lib/pmfGroups withPmf joins it: by filter id, else by sampling date */
export function usePmfByRow(pmf: PmfFile | null | undefined) {
  return useMemo(() => {
    const byId = new Map((pmf?.rows ?? []).map((r) => [r.id, r]))
    const byDate = new Map((pmf?.rows ?? []).map((r) => [r.date, r]))
    return (r: { id: string; code?: string; date?: string }) =>
      !pmf || r.code !== pmf.site ? undefined : byId.get(r.id) ?? (r.date ? byDate.get(r.date.slice(0, 10)) : undefined)
  }, [pmf])
}
