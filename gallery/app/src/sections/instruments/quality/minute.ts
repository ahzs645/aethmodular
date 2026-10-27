/**
 * The minute-level QC summaries (gallery/data/export_aeth_quality_minute.py → aethalometer/quality/minute/): the raw
 * MA350 records at Jacros (MA350-0238, also the instrument behind the Addis daily record) and Pasadena (MA350-0229,
 * JPL's). Each file is a set of small tables; the charts in MinuteCharts.tsx draw them as exported.
 */
import { useExport } from '@/lib/data'
import { JACROS } from './data'

export interface Hist { lo: number; hi: number; counts: number[]; below: number; above: number }
export interface Hist2d { x: [number, number]; y: [number, number]; counts: number[][]; n: number }
type Stats = { mean: number; median: number; std: number; p5: number; p95: number; iqr: number }
type Quartiles = { q25: (number | null)[]; q50: (number | null)[]; q75: (number | null)[] }

export interface MinuteSite {
  minutes: number; span: [string, string]; segments: number
  flow_qc: {
    n: number; R_target: number; corr_F1_F2: number
    stats: Record<'R' | 'f1' | 'f2' | 'S' | 'T', Stats>
    rates: Record<'R_pass' | 'R_warn' | 'R_fail' | 'S_pass' | 'T_pass' | 'QC_pass', number>
    daily: { day: string[]; R_med: (number | null)[]; R_p10: (number | null)[]; R_p90: (number | null)[]; R_pass: (number | null)[]; QC_pass: (number | null)[]
      f1: (number | null)[]; f2: (number | null)[]; sum_diff: (number | null)[]; set_diff: (number | null)[]; n: number[] }
    R_hist: Hist; flows_2d: Hist2d
  }
  smoothing: {
    windows: number[]; dema_span: number
    stats: { channel: string; window: number; n: number; r: number; rmse: number; mae: number; bias: number }[]
    diurnal: Record<string, Record<string, Quartiles>>
    example?: { day: string; minute: number[] } & Record<string, (number | null)[]>
    scatter_IR?: Hist2d & { window: number }
  }
  tape: {
    advances: number; segments_median_min: number
    table: { channel: string; n: number; n_control: number; step_median: number; step_mad: number; control_mad: number; rel_median: number; rel_q25: number; rel_q75: number
      rel_over25: number; control_rel_over25: number; positive: number; control_positive: number }[]
    hist: Record<string, Hist>
    ir_points: { pre: (number | null)[]; step: (number | null)[] } | null
  }
  kstar: { channel: string; by: 'ATN1' | 'spot age'; bin: string; n: number; K: number; Kstar: number; Kstar_q25: number; Kstar_q75: number; spot2_over_bcc: number; bcc_star_over_bcc: number }[]
  clean: {
    fraction: number; n: number; median_dT: number; median_dRH: number; nominal_ratio: number
    parts: Record<string, number>
    env: Record<string, { bin: string; n: number; aae_blue: number | null; aae_red: number | null; red_ir: number | null; uv_ir: number | null; ir_mean: number | null }[]>
    daily_aae: ({ minutes: string; n_days: number; red_over1: number } & Record<string, number | string>)[]
    paired_red_mean: number; paired_red_sd: number; hist_red_all: Hist; hist_red_clean: Hist
  }
  missing: { n_days: number; first: string | null; last: string | null; ir_mean: number | null
    table: { channel: string; mask: string; loss: number; bias: number | null; median: number | null; q25: number | null; q75: number | null; p2_5: number | null; p97_5: number | null; over25: number | null }[] }
  aae_order: Record<string, { n_days: number; undefined_pct: number; agg_mean: number; agg_sd: number; diff_median: number; diff_q25: number; diff_q75: number
    diff_p2_5: number; diff_p97_5: number; over01: number; median_diff_median: number; hist: Hist }>
  windows: { id: string[]; date: string[]; hours: (number | null)[] } & Record<string, (number | null)[]>
}
export interface MinuteIndex { generated: string; seed: number; sites: Record<string, { name: string; file: string; zone: string; minutes: number; span: [string, string]; filter_site: string }> }

/** the minute record behind an instrument code: Jacros's for Jacros and for the Addis daily record (the same MA350) */
export const minuteCode = (code: string) => (code === 'ETAD' ? JACROS : code)

export const useMinuteIndex = () => useExport<MinuteIndex>('aethalometer/quality/minute/index.json')

/** both minute records (Jacros, Pasadena), keyed by code, once the index has loaded; sites without a file are left out */
export function useMinuteSites(index: MinuteIndex | null): Record<string, MinuteSite> {
  const jacros = useExport<MinuteSite>(index?.sites[JACROS] ? `aethalometer/quality/minute/${JACROS}.json` : null).data
  const uspa = useExport<MinuteSite>(index?.sites.USPA ? 'aethalometer/quality/minute/USPA.json' : null).data
  return Object.fromEntries([[JACROS, jacros], ['USPA', uspa]].filter(([, d]) => d)) as Record<string, MinuteSite>
}
export function useMinute(code: string, index: MinuteIndex | null) {
  const c = minuteCode(code)
  return { code: c, ...useExport<MinuteSite>(index?.sites[c] ? `aethalometer/quality/minute/${c}.json` : null) }
}
