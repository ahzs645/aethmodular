/**
 * What every Filter-only chart shares: the export (gallery/data/export_research_filter_only.py), the page's
 * settings in the link (population, minimum training n, HIPS units), site colours, and the refits per site.
 */
import { useMemo, type ReactNode } from 'react'
import * as d3 from 'd3'
import { Segmented, Select } from '@/components/ChartFrame'
import { useExport } from '@/lib/data'
import { useHandEdits } from '@/lib/filterEdits'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { MIN_TRAIN, POPULATIONS, evaluate, inPopulation, type Block, type FPoint, type Population, type Scheme } from '../../common/filterOnly'

export const COMMAND = 'uv run python gallery/data/export_research_filter_only.py'
export const NB_DIAG = 'research/ftir_hips_chem/filter_only_diagnostics.ipynb'
export const NB_PROP = 'research/ftir_hips_chem/filter_proportionality.ipynb'
export const NB_STAB = 'research/ftir_hips_chem/filter_relationship_stability.ipynb'
export const SOURCE = 'gallery/app/src/sections/research/pages/filterOnly'
export const SITES = ['Addis Ababa', 'Beijing', 'Delhi', 'JPL']
const FALLBACK: Record<string, string> = { 'Addis Ababa': '#F39C12', Beijing: '#E74C3C', Delhi: '#3498DB', JPL: '#2ECC71' }

interface PointsFile { generated: string; points: FPoint[]; omitted_default: string }
/** the release's own tables, columnar (name -> values) */
export type Columns = Record<string, (string | number | boolean | null)[]>
export interface ReleaseFile { generated: string; blocks: Columns; id11: Columns; performance: Columns; proportionality: Columns; coefficients: Columns; influence: Columns; cohorts: Columns; denominator: Columns }
export const rowsOf = (t: Columns): Record<string, any>[] => (t[Object.keys(t)[0]] ?? []).map((_, i) => Object.fromEntries(Object.keys(t).map((k) => [k, t[k][i]])))

export interface FoData {
  /** the diagnostic filters, less any the reader excluded by hand */
  points: FPoint[]
  all: FPoint[]
  dropped: number
  omitDefault: string
  note: string | null
}

export function useFilterOnly(): { data: FoData | null; error: string } {
  const { data, error } = useExport<PointsFile>('research/filter_only/points.json', (d) => (Array.isArray(d.points) ? null : 'no points'))
  const hand = useHandEdits()
  const out = useMemo(() => {
    if (!data) return null
    const points = data.points.filter((p) => !hand.isOut(p.id))
    const dropped = data.points.length - points.length
    return { points, all: data.points, dropped, omitDefault: data.omitted_default, note: hand.note(dropped, { refit: true }) }
  }, [data, hand])
  return { data: out, error }
}

export const useRelease = () => useExport<ReleaseFile>('research/filter_only/release.json', (d) => (d.blocks ? null : 'no blocks'))

export const siteColor = (meta: MetaFile, site: string) => meta.sites.find((s) => s.name === site)?.color ?? FALLBACK[site] ?? INK.accent

export type Unit = 'fabs' | 'bc'
export interface Settings {
  population: Population
  minTrain: number
  unit: Unit
  /** multiply an Fabs-unit value by this for display (1, or 1 / MAC) */
  k: number
  /** the y unit as written on axes */
  u: string
  mac: number
}

const POP_KEYS = POPULATIONS.map((p) => p.key)
const MIN_OPTIONS = ['5', '10', '20']

/** the page-wide settings, in the link as fo.pop, fo.min and fo.unit */
export function useSettings(): Settings {
  const [population] = useSearchParam<Population>('fo.pop', 'diagnostic', POP_KEYS)
  const [min] = useSearchParam<string>('fo.min', String(MIN_TRAIN), MIN_OPTIONS)
  const [unit] = useSearchParam<Unit>('fo.unit', 'fabs', ['fabs', 'bc'])
  const { mac } = useMac()
  return { population, minTrain: Number(min), unit, k: unit === 'bc' ? 1 / mac : 1, u: unit === 'bc' ? 'µg/m³' : 'Mm⁻¹', mac }
}

export function SettingsControls() {
  const [population, setPopulation] = useSearchParam<Population>('fo.pop', 'diagnostic', POP_KEYS)
  const [min, setMin] = useSearchParam<string>('fo.min', String(MIN_TRAIN), MIN_OPTIONS)
  const [unit, setUnit] = useSearchParam<Unit>('fo.unit', 'fabs', ['fabs', 'bc'])
  const { mac } = useMac()
  return <>
    <Select label="Population" value={population} options={POP_KEYS} onChange={(v) => setPopulation(v as Population)} optionLabel={(o) => POPULATIONS.find((p) => p.key === o)!.label} />
    <Segmented label="Min training n" value={min} options={MIN_OPTIONS} onChange={setMin} optionLabel={(o) => (o === String(MIN_TRAIN) ? `${o} (release)` : o)} />
    <Segmented label="HIPS as" value={unit} options={['fabs', 'bc'] as const} onChange={setUnit} optionLabel={(o) => (o === 'fabs' ? 'Fabs (Mm⁻¹)' : `BC at MAC ${fmt(mac, 1)}`)} />
  </>
}

export interface SiteRun { site: string; diagnostic: FPoint[]; cohort: FPoint[]; lqo: Block[]; later: Block[] }

/** both schemes at every site for the chosen population (and with one filter left out, when `omit` is given) */
export function useRuns(points: FPoint[], s: Settings, omit?: string | null, population = s.population): SiteRun[] {
  return useMemo(() => SITES.map((site) => {
    const diagnostic = points.filter((p) => p.site === site)
    const cohort = diagnostic.filter((p) => inPopulation(p, population) && p.id !== omit)
    return { site, diagnostic, cohort, lqo: evaluate(diagnostic, cohort, 'leave_quarter_out', s.minTrain), later: evaluate(diagnostic, cohort, 'later_period', s.minTrain) }
  }), [points, population, s.minTrain, omit])
}

export const blocksOf = (r: SiteRun, scheme: Scheme) => (scheme === 'leave_quarter_out' ? r.lqo : r.later)
export const SCHEMES = ['leave_quarter_out', 'later_period'] as const
export const schemeLabel = (s: Scheme) => (s === 'leave_quarter_out' ? 'Leave quarter out' : 'Later period')
export const popLabel = (p: Population) => POPULATIONS.find((q) => q.key === p)!.label
/** '2023Q1' -> '23Q1', so eight or nine quarters fit a facet's axis */
export const shortQ = (v: unknown) => (/^\d{4}Q\d$/.test(String(v)) ? String(v).slice(2) : String(v))
export const MODEL_LABEL = { median: 'Training median', proportional: 'Proportional (through 0)', ols: 'OLS with intercept' } as const

/**
 * Vertical bars at the categories of a SweepPlot's point scale, from zero (its `background`): the notebooks' bar
 * panels. A series with the same values carries the hover.
 */
export function barLayer(bars: { x: string; y: number | null; color: string; faint?: boolean }[]) {
  return ({ x, y }: { x: unknown; y: d3.ScaleContinuousNumeric<number, number> }): ReactNode => {
    const sx = x as d3.ScalePoint<string>
    const w = Math.max(4, Math.min(28, sx.step() * 0.55))
    const [lo, hi] = y.domain(), zero = y(Math.max(Math.min(lo, hi), Math.min(Math.max(lo, hi), 0)))
    return <g>{bars.map((b) => {
      if (b.y === null || !Number.isFinite(b.y)) return null
      const at = sx(b.x)
      if (at === undefined) return null
      const top = y(Math.max(Math.min(lo, hi), Math.min(Math.max(lo, hi), b.y)))
      return <rect key={b.x} x={at - w / 2} width={w} y={Math.min(top, zero)} height={Math.max(1, Math.abs(zero - top))} fill={b.color} fillOpacity={b.faint ? 0.3 : 0.8} />
    })}</g>
  }
}

/** a status in words, for tooltips and NA marks */
export const statusText = (s: string) => ({ evaluated: 'evaluated', insufficient_training_n: 'fewer training filters than the minimum', constant_training_ec: 'no spread in training EC', empty_test_block: 'no filters this quarter' } as Record<string, string>)[s] ?? s.replace(/_/g, ' ')
