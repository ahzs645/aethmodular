/**
 * Research → Python PMF (Oct 2026): the export (gallery/data/export_python_pmf.py → research/python_pmf/), its types,
 * the page's case picker and the joins every chart shares. The package's factors are "Factor 1…5" only: the report
 * forbids source names, so the archived KF factor a factor resembles is given as descriptive context, never a label.
 */
import { useMemo } from 'react'
import { Segmented, Select } from '@/components/ChartFrame'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { seasonOfDate } from '@/lib/seasons'
import { rowsOf, type Table } from '@/lib/table'
import type { MetaFile, PmfFile, SeasonMeta } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import type { SpectralExport } from '../spectralCommon'
import { baseId, type CaseContributions } from '../../common/pythonPmf'

export const EXPORTER = 'gallery/data/export_python_pmf.py'
export const REPORT = 'python_pmf/Addis_Python_PMF_Analysis_Report.md'
/** the five factors' colours: a neutral categorical ramp, deliberately not the archived sources' colours */
export const FACTOR_COLOR = ['#2c7fb8', '#d08c2c', '#4f9d69', '#b5446e', '#6a51a3']
export const FACTORS = [1, 2, 3, 4, 5] as const
export const factorName = (k: number) => `Factor ${k}`

export interface PmfCase {
  key: string; cohort: string; cohort_label: string; prep: string; prep_label: string; folder: string; grid: 'etad' | 'etbi'
  n: number; features: number; cells: number; seed: number; converged_starts: number; total_starts: number; converged: boolean; within_1pct: number
  objective: number; loss_per_cell: number; q_true: number; rmse: number; frac_gt3: number; frac_negative: number
  negative_q_lower_bound: number; negative_rmse_lower_bound: number; lower_bound_share_of_q: number
  near_best_min_cosine: number | null; near_best_min_r: number | null; relative_objective_range: number
}
type Metrics = Record<string, number | null>
export interface PythonPmfExport extends SpectralExport {
  package: { report: string; report_date: string; created_utc: string; fits: number; converged_fits: number; physical_filters: number; native_epa_parity: boolean; factor_count_finalized: boolean }
  kf_labels: Record<string, string>
  comparison_filters: string[]
  cases: PmfCase[]
  contributions: Record<string, CaseContributions>
  heldout_pooled: Metrics
  heldout_baseline: Metrics
  conditional: { heldout: Metrics; baseline: Metrics; in_sample: Metrics; coefficients: number[]; outside_reference_dates: number; reference_dates: [string, string]; allocated_filters: number }
  bootstrap: Record<string, { accepted: number[]; eligible: number[]; resamples: number; converged: number }>
  tables: Record<string, Table>
}
export interface ProfilesExport {
  schema_version: number
  grids: Record<'etad' | 'etbi', number[]>
  profiles: Record<string, number[][]>
  residuals: Record<string, { mean: number[]; rmse: number[]; negative: number[] }>
  bootstrap: Record<string, { p05: number[][]; p50: number[][]; p95: number[][] }>
}

const v1 = (d: { schema_version: number }) => (d.schema_version === 1 ? null : 'unsupported data version')
export const usePythonPmf = () => useExport<PythonPmfExport>('research/python_pmf/python_pmf.json', v1)
export const useProfiles = () => useExport<ProfilesExport>('research/python_pmf/profiles.json', v1)
/** the archived (KF) solution the gallery already carries: 102 ETAD filters, µg/m³ per historical factor */
export const useArchived = () => useExport<PmfFile>('pmf.json').data

export const tableRows = (d: PythonPmfExport, name: string) => rowsOf(d.tables[name])

// --------------------------------------------------------------------------------------------- the case picker

const COHORT_ORDER = ['etad246', 'etad102', 'etbi53', 'etad244']
const PREP_ORDER = ['huber', 'quadratic', 'blanksub_huber', 'df4_3_huber', 'df8_5_huber']
export const PRIMARY = 'etad246_huber'

/** the page's one case (URL `ppmf.case`), shared by every chart that draws a single fit */
export function useCase(data: PythonPmfExport | null, etadOnly = false): [PmfCase | null, (key: string) => void] {
  const keys = useMemo(() => (data?.cases ?? []).map((c) => c.key), [data])
  const [key, setKey] = useSearchParam<string>('ppmf.case', PRIMARY, keys)
  const c = data?.cases.find((x) => x.key === key) ?? data?.cases.find((x) => x.key === PRIMARY) ?? null
  return [c && etadOnly && c.grid !== 'etad' ? null : c, setKey]
}

/** cohort × loss/preparation, offering only the preparations the package ran for the chosen cohort */
export function CasePick({ data, value, onChange, etadOnly = false }: { data: PythonPmfExport; value: PmfCase | null; onChange: (key: string) => void; etadOnly?: boolean }) {
  const cohorts = COHORT_ORDER.filter((c) => data.cases.some((x) => x.cohort === c && (!etadOnly || x.grid === 'etad')))
  const cohort = value?.cohort ?? 'etbi53'
  const preps = PREP_ORDER.filter((p) => data.cases.some((x) => x.cohort === cohort && x.prep === p))
  const label = (k: string) => data.cases.find((x) => x.cohort === k)?.cohort_label ?? k
  const pick = (co: string, prep: string) => {
    const hit = data.cases.find((x) => x.cohort === co && x.prep === prep) ?? data.cases.find((x) => x.cohort === co && x.prep === 'huber')
    if (hit) onChange(hit.key)
  }
  return <>
    <Segmented label="Cohort" value={cohorts.includes(cohort) ? cohort : ''} options={cohorts} optionLabel={label} onChange={(co) => pick(co, value?.prep ?? 'huber')} />
    {value && preps.length > 1 && <Select label="Loss / preparation" value={value.prep} options={preps}
      optionLabel={(p) => data.cases.find((x) => x.prep === p)?.prep_label ?? p} onChange={(p) => pick(cohort, p)} />}
  </>
}

/** "ETAD 246 · Robust (Huber)" */
export const caseLabel = (c: PmfCase) => `${c.cohort_label} · ${c.prep_label}`

// --------------------------------------------------------------------------------------------- shared joins

/** the archived KF factor each new factor was matched to (descriptive; analysis/historical_factor_pattern_matches.csv) */
export function archivedFor(data: PythonPmfExport, caseKey: string): Map<number, { kf: number; label: string; r: number }> {
  const out = new Map<number, { kf: number; label: string; r: number }>()
  for (const r of tableRows(data, 'historical')) if (r.case === caseKey) {
    const kf = Number(r.historical_factor)
    out.set(Number(r.new_factor), { kf, label: data.kf_labels[String(kf)] ?? `KF${kf}`, r: Number(r.contribution_pearson_r) })
  }
  return out
}
/** "closest archived: Charcoal, r = 0.94" */
export const closest = (m: { label: string; r: number } | undefined) => (m ? `closest archived: ${m.label}, r = ${fmt(m.r, 2)}` : '')

/** one modelled filter of a case: its contributions, scaled RMSE and the season of its date under the reader's calendar */
export interface CaseFilter { id: string; base: string; date: string; g: number[]; rmse: number | null; season: string | null }

/** the Addis calendar in use (Bishoftu, 45 km away, has no gallery calendar of its own and takes the same one) */
export const addisCalendar = (meta: MetaFile): SeasonMeta[] => meta.seasons.filter((s) => !s.site || s.site === 'Addis Ababa')

export function caseFilters(data: PythonPmfExport, key: string, calendar: readonly SeasonMeta[]): CaseFilter[] {
  const c = data.contributions[key]
  if (!c) return []
  return c.id.map((id, i) => ({
    id, base: baseId(id), date: c.date[i], g: c.g.map((col) => col[i] ?? NaN), rmse: c.rmse[i],
    season: seasonOfDate(c.date[i], Number(c.date[i].slice(5, 7)), calendar)?.name ?? null,
  }))
}
