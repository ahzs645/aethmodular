/**
 * The raw HIPS optics export (gallery/data/export_research_hips_optics.py → research/hips_optics/filters.json)
 * and what the operating-envelope pages derive from it, joined to the gallery's rows by id.
 */
import { useMemo } from 'react'
import { useExport } from '@/lib/data'
import type { FilterRow } from '@/lib/types'
import { quantile } from './stats'

export const OPTICS_PATH = 'research/hips_optics/filters.json'
export const OPTICS_COMMAND = 'uv run python gallery/data/export_research_hips_optics.py'

interface OpticsFile { generated: string; source: string; note: string; columns: string[]; rows: (string | number | null)[][] }
export interface Optics {
  id: string; filter_id: string; code: string; type: string; date: string | null; lot: string | null; analysis_date: string | null
  t: number | null; r: number | null; tau: number | null; T1: number | null; R1: number | null
  cal_intercept: number | null; cal_slope: number | null; fabs: number | null; mdl: number | null
  area_cm2: number | null; volume_m3: number | null; ec_ug: number | null; oc_ug: number | null; om_ug: number | null
}

export function useOptics(): { rows: Optics[] | null; error: string; generated?: string } {
  const { data, error } = useExport<OpticsFile>(OPTICS_PATH)
  const rows = useMemo(() => data ? data.rows.map((r) => Object.fromEntries(data.columns.map((c, i) => [c, r[i]])) as unknown as Optics) : null, [data])
  return { rows, error, generated: data?.generated }
}

/** the per-filter envelope quantities, from a gallery row and its optics row (µg/cm² loadings, residuals) */
export interface Envelope {
  row: FilterRow; o: Optics
  hipsBc: number; ec: number
  ecLoad: number | null; ocLoad: number | null; omLoad: number | null; pmLoad: number | null; bcLoad: number | null; ironLoad: number | null
  resid: number; relResid: number; ocEc: number | null; omEc: number | null; pmEc: number | null; ironEc: number | null
}

const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const per = (mass: number | null, area: number | null) => (mass !== null && area ? mass / area : null)

/**
 * addis_06_hips_ptfe_operating_envelope's derived columns: HIPS BC = Fabs / MAC; loadings = µg on the filter /
 * deposit area, FTIR mass from MassLoading_ug, else concentration × volume; residual HIPS BC − FTIR EC and its
 * relative form. PM2.5 and iron come from the gallery row (date-matched ChemSpec), times the filter's volume.
 */
export function envelopeOf(row: FilterRow, o: Optics, mac: number): Envelope | null {
  const ec = n(row['EC (FTIR)']), fabs = o.fabs ?? n(row['HIPS Fabs'])
  if (ec === null || fabs === null || ec <= 0 || fabs <= 0) return null
  const hipsBc = fabs / mac, v = o.volume_m3, a = o.area_cm2
  const conc = (f: string) => { const c = n(row[f]); return c !== null && v ? c * v : null }
  const pm = n(row['PM2.5 mass']), iron = n(row['Iron']), oc = n(row['OC (FTIR)']), om = n(row['OM (FTIR)'])
  return {
    row, o, hipsBc, ec,
    ecLoad: per(o.ec_ug ?? conc('EC (FTIR)'), a), ocLoad: per(o.oc_ug ?? conc('OC (FTIR)'), a), omLoad: per(o.om_ug ?? conc('OM (FTIR)'), a),
    pmLoad: per(conc('PM2.5 mass'), a), bcLoad: per(v ? hipsBc * v : null, a), ironLoad: per(conc('Iron'), a),
    resid: hipsBc - ec, relResid: (hipsBc - ec) / ec,
    ocEc: oc !== null ? oc / ec : null, omEc: om !== null ? om / ec : null, pmEc: pm !== null ? pm / ec : null, ironEc: iron !== null ? iron / ec : null,
  }
}

/** p05 … p95 of a list, as the IMPROVE notebooks' Addis windows */
export const windowOf = (values: (number | null)[]) => {
  const v = values.filter((x): x is number => x !== null && Number.isFinite(x))
  return { n: v.length, min: Math.min(...v), p05: quantile(v, 0.05), p25: quantile(v, 0.25), median: quantile(v, 0.5), p75: quantile(v, 0.75), p95: quantile(v, 0.95), max: Math.max(...v) }
}
