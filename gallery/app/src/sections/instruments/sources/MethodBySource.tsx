import { useMemo } from 'react'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Crossplot, sharedCrossDomains, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { f2 } from '@/lib/format'
import { INK, withUnit } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { ExcludedNote, useMarkedRows } from '@/lib/markedRows'
import type { FilterRow, MetaFile, PmfFile } from '@/lib/types'
import { CLASSES, CLASS_COLOR, classify, type AaePair, type AaeRegions } from './aae'
import { rowAae } from './AaeValues'
import { NOTEBOOK, provenance } from './common'
import { usePmfByRow } from './data'
import { pmfDominant } from './PmfAgreement'

/** the notebooks' METHOD_PAIRS: (x, y); HIPS as Fabs / MAC, the gallery's "HIPS BC" */
export const METHOD_PAIRS = {
  'HIPS vs FTIR EC': { x: 'EC (FTIR)', y: 'HIPS BC' },
  'Aeth IR vs FTIR EC': { x: 'EC (FTIR)', y: 'IR BCc (aethalometer)' },
  'Aeth IR vs HIPS': { x: 'HIPS BC', y: 'IR BCc (aethalometer)' },
} as const
export type MethodPair = keyof typeof METHOD_PAIRS
export const METHOD_PAIR_KEYS = Object.keys(METHOD_PAIRS) as MethodPair[]

export const SPLITS = ['PMF dominant (GF)', 'PMF dominant (K_F)', 'AAE class'] as const
export type Split = (typeof SPLITS)[number]

export interface SplitGroup { name: string; color: string }
/** each row's group under a split (null: not in any), and the groups in order */
export function useSplit(pmf: PmfFile | null | undefined, split: Split, pair: AaePair, regions: AaeRegions) {
  const pmfOf = usePmfByRow(pmf)
  return useMemo(() => {
    if (split === 'AAE class') {
      return { groups: CLASSES.map((c) => ({ name: c, color: CLASS_COLOR[c] })), of: (r: FilterRow) => classify(rowAae(r, pair, 'absorption'), regions) as string | null,
        share: (r: FilterRow) => rowAae(r, pair, 'absorption') }
    }
    const rule = split === 'PMF dominant (GF)' ? 'GF' : 'K_F'
    const dom = (r: FilterRow) => { const p = pmfOf(r); return p && pmf ? pmfDominant(p, pmf, rule) : null }
    return { groups: (pmf?.sources ?? []).map((s) => ({ name: s.label, color: s.color })), of: (r: FilterRow) => dom(r)?.source ?? null,
      share: (r: FilterRow) => dom(r)?.share ?? null }
  }, [split, pmf, pmfOf, pair, regions])
}

/**
 * Source-separated regressions (source_apportionment_regression Tasks 1, 2, 4; addis_ababa_source_analysis;
 * dominant_source_comparison): one method pair, every filter in view, then one panel per dominant PMF factor
 * (Addis, by either rule) or per AAE class (every site with an aethalometer). Fits follow the sidebar.
 */
export function MethodBySource({ rows, meta, pmf, pair, regions }: { rows: FilterRow[]; meta: MetaFile; pmf: PmfFile | null; pair: AaePair; regions: AaeRegions }) {
  const [mp, setMp] = useSearchParam<MethodPair>('aes.methodPair', 'Aeth IR vs FTIR EC', METHOD_PAIR_KEYS)
  const [split, setSplit] = useSearchParam<Split>('aes.methodSplit', 'PMF dominant (GF)', SPLITS)
  const { x: xf, y: yf } = METHOD_PAIRS[mp]
  const s = useSplit(pmf, split, pair, regions)
  const isPmf = split !== 'AAE class'
  // Filters → excluded: Mark (lib/markedRows): the excluded filters in view, split the same way, × in their panels, in no fit
  const markedRows = useMarkedRows()
  const panels = useMemo(() => {
    const point = (r: FilterRow, color: string, g: string | null): CrossPoint => ({ x: r[xf] as number, y: r[yf] as number, color, id: r.id,
      tip: [r.id, `${r.site} · ${r.date}`, `${xf}: ${f2(r[xf])}`, `${yf}: ${f2(r[yf])}`, `${split}: ${g ?? '—'}`],
      rec: { id: r.id, source: `Aethalometer sources · ${mp} by ${split}`, site: r.site, date: r.date, fields: [[xf, r[xf] as number], [yf, r[yf] as number], [split, g]] } })
    // with a PMF split, only filters with a PMF day, as the notebooks merged them (inner join)
    const usableOf = (rs: FilterRow[]) => rs.filter((r) => typeof r[xf] === 'number' && typeof r[yf] === 'number' && s.of(r) !== null && (!isPmf || r.code === pmf?.site))
    const usable = usableOf(rows), markedUsable = usableOf(markedRows)
    const allOf = (rs: FilterRow[]) => rs.map((r) => { const g = s.of(r); return point(r, s.groups.find((x) => x.name === g)?.color ?? INK.neutral, g) })
    const by = s.groups.map((g) => ({ name: g.name, color: g.color, points: usable.filter((r) => s.of(r) === g.name).map((r) => point(r, g.color, g.name)),
      marked: markedUsable.filter((r) => s.of(r) === g.name).map((r) => point(r, g.color, g.name)) }))
    return [{ name: isPmf ? `All ${pmf?.site_name ?? ''} filters with a PMF day` : 'All filters with an AAE', color: INK.text, points: allOf(usable), marked: allOf(markedUsable) }, ...by]
  }, [rows, markedRows, xf, yf, s, isPmf, pmf, split, mp])
  const dom = sharedCrossDomains(panels.map((p) => p.points))
  const xl = withUnit(xf, meta.field_units), yl = withUnit(yf, meta.field_units)
  return (
    <ChartFrame id="aeth-sources-method-by-source"
      title={`Method agreement by source — ${mp}, all filters and one panel per ${isPmf ? 'dominant PMF factor' : 'AAE class'}`}
      subtitle={`${isPmf ? `${pmf?.site_name ?? 'Addis'} filters joined to their PMF day; the dominant factor is the largest ${split.includes('GF') ? 'normalised GF fraction' : 'K_F contribution (µg/m³)'}.` : `Every filter in view with an aethalometer AAE, classed fossil ≤ ${regions.fossil_max}, biomass ≥ ${regions.biomass_min} (AAE ${pair} from absorption).`} HIPS is Fabs ÷ MAC (HIPS BC); the aethalometer is IR BCc over the filter's sampling window. Panels share one range; a panel with fewer than three filters draws its points without a fit. The notebooks reported OLS; the sidebar adds the errors-in-variables fit a 1:1 comparison needs (AGENTS.md).`}
      provenance={provenance([NOTEBOOK.regression, NOTEBOOK.addisSource, NOTEBOOK.dominantComparison], 'Rows as filtered in the Filters panel; PMF by filter id, else sampling date.')}
      controls={<>
        <Select label="Pair" value={mp} options={METHOD_PAIR_KEYS} onChange={(v) => setMp(v as MethodPair)} />
        <Select label="Split by" value={split} options={[...SPLITS]} onChange={(v) => setSplit(v as Split)} />
      </>}>
      {!panels[0].points.length ? <Empty>{isPmf ? `No ${pmf?.site_name ?? 'PMF'} filter in view has both ${xf} and ${yf}. The PMF solution covers ${pmf?.site_name ?? 'Addis Ababa'} only.` : `No filter in view has both ${xf} and ${yf}.`}</Empty>
        : <FacetGrid items={panels} minCell={260} maxCols={3} render={(p, w) => (
          <div key={p.name}>
            <Crossplot points={p.points} marked={p.marked} title={`${p.name} · n=${p.points.length}`} xLabel={xl} yLabel={yl} maxSide={Math.max(180, w - 80)} domain={dom.domain} grouping="pooled" />
          </div>
        )} />}
      {panels[0].points.length > 0 && <ExcludedNote count={panels[0].marked.length} drawn />}
    </ChartFrame>
  )
}
