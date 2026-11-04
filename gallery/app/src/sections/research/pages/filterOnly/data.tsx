/**
 * What every Filter-only chart shares: the export (gallery/data/export_research_filter_only.py), the page's
 * settings in the link (population, minimum training n, HIPS units), site colours, and the refits per site.
 */
import { createContext, useContext, useMemo, type ReactNode } from 'react'
import * as d3 from 'd3'
import { Segmented, Select } from '@/components/ChartFrame'
import { useExport } from '@/lib/data'
import { useMac } from '@/lib/mac'
import { ExcludedNote, useExclMode, useExclusionSplit, useMarkedRows } from '@/lib/markedRows'
import { INK } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { rowsOf as tableRows } from '@/lib/table'
import { useSearchParam } from '@/lib/url'
import { PlotControls, SidebarControls } from '@/lib/pageSidebar'
import { Row } from '@/components/SubsetBar'
import { MIN_TRAIN, POPULATIONS, evaluate, inPopulation, type Block, type FPoint, type Population, type Scheme } from '../../common/filterOnly'
import { siteColor as metaSiteColor } from '@/lib/siteColours'

export const COMMAND = 'uv run python gallery/data/export_research_filter_only.py'
export const NB_DIAG = 'research/ftir_hips_chem/filter_only_diagnostics.ipynb'
export const NB_PROP = 'research/ftir_hips_chem/filter_proportionality.ipynb'
export const NB_STAB = 'research/ftir_hips_chem/filter_relationship_stability.ipynb'
export const SOURCE = 'gallery/app/src/sections/research/pages/filterOnly'
export const SITES = ['Addis Ababa', 'Beijing', 'Delhi', 'JPL']
/** what a chart says when Samples leaves it no release filter */
export const NO_FILTERS = `No filter of the release is in these Samples: pick ${SITES.slice(0, -1).join(', ')} or ${SITES[SITES.length - 1]}, or widen the dates and seasons.`
const FALLBACK: Record<string, string> = { 'Addis Ababa': '#F39C12', Beijing: '#E74C3C', Delhi: '#3498DB', JPL: '#2ECC71' }

interface PointsFile { generated: string; points: FPoint[]; omitted_default: string }
/** the release's own tables, columnar (name -> values) */
export type Columns = Record<string, (string | number | boolean | null)[]>
export interface ReleaseFile { generated: string; blocks: Columns; id11: Columns; performance: Columns; proportionality: Columns; coefficients: Columns; influence: Columns; cohorts: Columns; denominator: Columns }
/** the release's columns as rows (lib/table rowsOf), loosely typed as its pages read them */
export const rowsOf = (t: Columns): Record<string, any>[] => tableRows<Record<string, any>>(t)

export interface FoData {
  /** the diagnostic filters that count under Filters → excluded (lib/markedRows useCountedFilter) */
  points: FPoint[]
  /** under Filters → excluded: Mark, the ones that do not count: drawn as × on the held-out scatter, in no refit */
  marked: FPoint[]
  all: FPoint[]
  dropped: number
  omitDefault: string
  note: string | null
}

const NO_POINTS: FPoint[] = []

/**
 * The release's 545 diagnostic filters under Samples and Filters → excluded, as every chart's filters: every one has
 * a gallery row (filters.json), so the rows the page is given (and the excluded ones Mark draws) say which are in view:
 * the sites, dates, calendar, season chips and PMF pick narrow the refits, as the Baseline lab's useLabFilters does.
 * The data export's own exclusions (ETAD-0248, the PM10 filter the export excludes, is one of the 545) and the reader's
 * hand exclusions leave every refit, Include puts the export's back (as the release ran), and Mark draws them as ×
 * (joined by filter id, lib/markedRows useExclusionSplit, which also applies "keep only"). Only the check against the
 * release's own tables (Agreement) keeps all 545, whatever Samples says.
 */
export function useFilterOnly(rows: FilterRow[]): { data: FoData | null; error: string } {
  const { data, error } = useExport<PointsFile>('research/filter_only/points.json', (d) => (Array.isArray(d.points) ? null : 'no points'))
  const mode = useExclMode()
  const marked = useMarkedRows()
  const inView = useMemo(() => {
    if (!data) return NO_POINTS
    const ids = new Set([...rows, ...marked].map((r) => r.id))
    return data.points.filter((p) => ids.has(p.id))
  }, [data, rows, marked])
  const split = useExclusionSplit(inView, (p) => p.id)
  const out = useMemo(() => {
    if (!data) return null
    const points = split.counted, n = data.points.length - points.length
    // under Mark the charts say it themselves (FoExcludedNote), with the × drawn
    const note = mode === 'mark' || n <= 0 ? null : `${n} of the ${data.points.length} ${n === 1 ? 'is' : 'are'} out of these Samples or excluded (Filters → excluded) and left out of every refit.`
    return { points, marked: split.marked, all: data.points, dropped: n, omitDefault: data.omitted_default, note }
  }, [data, split, mode])
  return { data: out, error }
}

/** the excluded release filters under Mark (FoData.marked), for every chart's note without threading them through */
const MarkedCtx = createContext<FPoint[]>(NO_POINTS)
export const FoMarkedProvider = MarkedCtx.Provider
export const useFoMarked = () => useContext(MarkedCtx)

/**
 * A chart's note under Mark (lib/markedRows ExcludedNote): how many excluded filters of its population (every
 * population with `any`) and site it leaves out; `drawn` where it draws them as ×.
 */
export function FoExcludedNote({ s, site, any = false, drawn = false }: { s: Settings; site?: string; any?: boolean; drawn?: boolean }) {
  const marked = useFoMarked()
  return <ExcludedNote count={marked.filter((p) => (any || inPopulation(p, s.population)) && (!site || p.site === site)).length} drawn={drawn} />
}

export const useRelease = () => useExport<ReleaseFile>('research/filter_only/release.json', (d) => (d.blocks ? null : 'no blocks'))

export const siteColor = (meta: MetaFile, site: string) => metaSiteColor(meta, site, FALLBACK[site] ?? INK.accent)

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

/**
 * the page-wide settings in the dock: population and minimum training n on the Page tab, HIPS units on the Plot tab
 * under the MAC that "BC" divides by (lib/pageSidebar)
 */
export function SettingsControls() {
  const [population, setPopulation] = useSearchParam<Population>('fo.pop', 'diagnostic', POP_KEYS)
  const [min, setMin] = useSearchParam<string>('fo.min', String(MIN_TRAIN), MIN_OPTIONS)
  const [unit, setUnit] = useSearchParam<Unit>('fo.unit', 'fabs', ['fabs', 'bc'])
  return <>
    <SidebarControls title="Filter-only transfer limits">
      <Select label="Population" value={population} options={POP_KEYS} onChange={(v) => setPopulation(v as Population)} optionLabel={(o) => POPULATIONS.find((p) => p.key === o)!.label}
        title="Which filters every chart on the page refits on" />
      <Segmented label="Min training n" value={min} options={MIN_OPTIONS} onChange={setMin} optionLabel={(o) => (o === String(MIN_TRAIN) ? `${o} (release)` : o)}
        title="The fewest training filters a quarter needs before it is evaluated" />
    </SidebarControls>
    <PlotControls>
      <Row label="HIPS as">
        <Segmented value={unit} options={['fabs', 'bc'] as const} onChange={setUnit} optionLabel={(o) => (o === 'fabs' ? 'Fabs (Mm⁻¹)' : 'BC (µg/m³)')}
          title="Filter-only charts: Fabs needs no MAC; BC divides every HIPS value, prediction and error by the MAC above, which leaves percentages unchanged" />
      </Row>
    </PlotControls>
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
