/**
 * AERONET per SPARTAN site (gallery/data/export_aeronet_sites.py → aeronet/sites_summary.json): the
 * median, mean, quartiles and n of the column quantities over each site's filter windows, all year and
 * by the site's season calendar, the AOD-to-surface-PM conversion η = PM2.5 / AOD 500, the absorption
 * scale height, and which sites sit in a MAIA target area. The Map's SPARTAN scope plots these; the
 * AERONET tab's η chart reads the MAIA targets from here.
 */
import { useExport } from './data.ts'
import type { FilterRow } from './types.ts'

export interface QuantitySummary { n: number; median: number | null; mean?: number | null; p25?: number | null; p75?: number | null }
export type AeronetQuantityKey = 'aod500' | 'aaod675' | 'aae' | 'ssa440' | 'eta' | 'h'
export interface AeronetSiteSummary {
  station: string
  km: number
  n_filters: number
  /** Spearman ρ of filter PM2.5 with AOD 500 over the windows; null below 8 pairs */
  rho_pm_aod: number | null
  all: Partial<Record<AeronetQuantityKey, QuantitySummary>>
  seasons: Record<string, Partial<Record<AeronetQuantityKey, QuantitySummary>> & { n_filters: number; rho_pm_aod: number | null }>
}
export interface MaiaTarget { target: string; type: string }
export interface AeronetSitesFile {
  generated: string
  level: string
  source: string
  note: string
  quantities: { key: AeronetQuantityKey; label: string; unit: string; digits: number; about: string }[]
  /** code -> its season names in calendar order */
  seasons: Record<string, string[]>
  /** every SPARTAN site in a MAIA target area (research/maia/inventory/spartan_sites.csv), with or without AERONET */
  maia: Record<string, MaiaTarget>
  sites: Record<string, AeronetSiteSummary>
}

export const useAeronetSites = () => useExport<AeronetSitesFile>('aeronet/sites_summary.json')

/** MAIA's target-area types, as the mission names them */
export const MAIA_TYPE: Record<string, string> = { PTA: 'primary target area', STA: 'secondary target area', CVTA: 'calibration/validation area' }
export const maiaLine = (t: MaiaTarget | undefined) => (t ? `MAIA ${MAIA_TYPE[t.type] ?? t.type}: ${t.target}` : null)

/** η = filter PM2.5 / AOD 500 (µg/m³ per unit AOD) for one filter; null without both positive. */
export function eta(r: FilterRow): number | null {
  const pm = r['PM2.5 mass'], aod = r['AOD 500 nm (AERONET)']
  return typeof pm === 'number' && typeof aod === 'number' && pm > 0 && aod > 0 ? pm / aod : null
}
