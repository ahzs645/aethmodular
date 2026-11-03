import { useMemo } from 'react'
import { Select, Segmented } from '@/components/Controls'
import { useExport } from '@/lib/data'
import { useExclusionSplit, useKnownFilter, useMarkedRows, usePlaceFilter } from '@/lib/markedRows'
import { useSearchParam } from '@/lib/url'
import { useHighlightActions } from '@/lib/highlight'
import { siteCodeColor } from '@/lib/siteColours'
import { baselineColor, INK } from '@/lib/theme'
import { RESEARCH_SERIES } from '@/lib/palettes'
import type { FilterRow, MetaFile } from '@/lib/types'
import { LAB_FILE, labOpticalSourceIncluded, validateLabData, validateLabSpectrum, type BaselineLabData, type LabFilter, type LabSpectra } from '../../common/baselineLab'

export interface LabChartProps { rows: FilterRow[]; meta: MetaFile }

export const useLabExport = () => useExport<BaselineLabData>(LAB_FILE, validateLabData)

export function useLabSpectrum(data: BaselineLabData | null, id: string) {
  const path = data?.spectra_ids.includes(id) ? `${data.spectra_base}${encodeURIComponent(id)}.json` : null
  return useExport<LabSpectra>(path, d => validateLabSpectrum(d, id))
}

const NONE: LabFilter[] = []

/**
 * The live filter plots follow Samples site/date/season settings and the gallery's shared exclusions.
 * Newer exports with no gallery row use the same placement rule as the PTFE lot page. Undated native
 * samples and field blanks remain available in the spectrum picker, never invented as concentration pairs.
 */
export function useLabFilters(data: BaselineLabData | null, rows: FilterRow[]) {
  const known = useKnownFilter()
  const place = usePlaceFilter()
  const marked = useMarkedRows()
  const inView = useMemo(() => new Set([...rows, ...marked].map(r => r.id)), [rows, marked])
  const candidates = useMemo(() => {
    if (!data) return NONE
    const names = new Map(data.sites.map(s => [s.code, s.name]))
    return data.filters.filter(r => {
      if (known(r.id)) return inView.has(r.id)
      if (!r.date) return false
      const month = Number(r.date.slice(5, 7))
      return !!place(names.get(r.code) ?? r.code, r.code, r.date, month)
    })
  }, [data, known, place, inView])
  return useExclusionSplit(candidates, r => r.id)
}

export function useLabSelection(): [string, (id: string) => void] {
  const [id, set] = useSearchParam<string>('blab.filter', 'ETAD-0037')
  const hl = useHighlightActions()
  return [id, next => { set(next); hl.setPinned(next) }]
}

export function useLabSite(key = 'blab.site', fallback = 'all') {
  return useSearchParam(key, fallback)
}

export function LabSiteSelect({ data, value, onChange, all = true }: {
  data: BaselineLabData | null; value: string; onChange: (value: string) => void; all?: boolean
}) {
  const choices = [...(all ? ['all'] : []), ...(data?.sites.map(s => s.code) ?? [])]
  return <Select label="Site" value={value} options={choices} onChange={onChange}
    optionLabel={code => code === 'all' ? 'All selected sites' : data?.sites.find(s => s.code === code)?.name ?? code} />
}

export const SOURCE_CHOICES = ['direct', 'combined'] as const
export type LabSource = typeof SOURCE_CHOICES[number]
export const sourceIncludes = labOpticalSourceIncluded
export function LabSourceSelect({ value, onChange }: { value: LabSource; onChange: (value: LabSource) => void }) {
  return <Segmented label="Optical pairs" value={value} options={SOURCE_CHOICES} onChange={onChange}
    optionLabel={v => v === 'direct' ? 'Direct Fabs' : 'All HIPS'} />
}

export const labSiteColor = (data: BaselineLabData, code: string) => siteCodeColor(data, code)
export const labSiteName = (data: BaselineLabData, code: string) => data.sites.find(s => s.code === code)?.name ?? code
export const labMethodLabel = (data: BaselineLabData, method: string) => data.methods.find(m => m.id === method)?.label
  ?? ({ spline: 'Spline baseline', raw: 'Raw', neutral: 'Neutral robust baseline', removed: 'Removed component' }[method] ?? method.replace(/_/g, ' '))
export const labMethodColor = (data: BaselineLabData, method: string) => {
  const family = method.replace(/^cal_/, '')
  if (family === 'raw' || family.startsWith('raw_')) return INK.text
  if (/spline|airspec/i.test(family)) return baselineColor('AIRSpec')
  if (/neutral|arpls/i.test(family)) return RESEARCH_SERIES[1]
  if (/vibes_local3/.test(family)) return RESEARCH_SERIES[4]
  if (/vibes_local8/.test(family)) return RESEARCH_SERIES[2]
  if (method === 'removed') return INK.muted
  const index = data.methods.findIndex(m => m.id === method)
  return RESEARCH_SERIES[(Math.max(0, index) + 1) % RESEARCH_SERIES.length]
}

export const spectrumWindows = {
  band: { label: '1450–1800', range: [1800, 1450] as [number, number] },
  common: { label: '1425–4000', range: [4000, 1425] as [number, number] },
  full: { label: 'Full raw range', range: [4000, 500] as [number, number] },
  ch: { label: 'C–H 2800–3000', range: [3000, 2800] as [number, number] },
  oh: { label: 'O–H 3000–3600', range: [3600, 3000] as [number, number] },
}
export type LabWindow = keyof typeof spectrumWindows
export const WINDOW_KEYS = Object.keys(spectrumWindows) as LabWindow[]
