import { useCallback, useMemo } from 'react'
import { CompositionTreemap } from '@/charts/partOfWhole/CompositionTreemap'
import { PAPER_CALENDAR, PAPER_ORDER } from '@/lib/pmfPaper'
import type { FilterRow, MetaFile, PmfFile, PmfRow } from '@/lib/types'
import { ExcludedNote, useCountedFilter, useExclusionSplit } from '@/lib/markedRows'

const PAPER = 'Paper (four seasons)'

/**
 * Stacked area / streamgraph — PMF source contributions over time.
 *
 * The composition chart (charts/partOfWhole/CompositionTreemap) drawn on the PMF factors, so it
 * has every option that chart has: the per-filter time view this chart opens on (stacked area,
 * one bar per sampling day as in the PMF paper's Figure 7, or a streamgraph), and the per-site
 * and per-season summaries (mean or median stacked bars, treemap, circle pack, donut). The bars count the
 * filters one source dominates above a chosen share; that, and the paper's seasons as the grouping, is
 * what the former 'Source apportionment by season' chart drew (its Dashboard tile opens this one by season).
 *
 * Concentration stacks the K_Fn contributions in µg/m³, so a stack's height is the apportioned
 * mass. Share stacks the normalised relative contributions (normalize_gf_fractions), not the
 * µg/m³ divided by their sum (the two differ by up to 25 points); the raw GF1-GF5 columns are
 * PM2.5 mass fractions summing to 0.03-0.46 and would be wrong to stack directly. Sources stack
 * in the paper's order, Fossil Fuel at the bottom up to Charcoal.
 */
export function SourceStack({ pmf, rows: inView, meta, id = 'sourcestack', initialGrouping = 'over time', title = 'Stacked area · Streamgraph — PMF source contributions over time' }: {
  pmf: PmfFile; rows: PmfRow[]; meta: MetaFile
  /** a Dashboard tile's own chart: 'sourceseason' opens on the bars by season (the former Source apportionment by season) */
  id?: string; initialGrouping?: 'per site' | 'by season' | 'over time'; title?: string
}) {
  // each PMF day is a filter: Filters → excluded's one rule (lib/markedRows useCountedFilter) takes it out of the
  // stacks, means, medians and counts; the PMF model's contributions for the other days are as it fitted them
  const isCounted = useCountedFilter()
  const rows = useMemo(() => inView.filter((r) => isCounted(r.id)), [inView, isCounted])
  const nMarked = useExclusionSplit(inView, (r) => r.id).marked.length
  const labels = useMemo(() => [...pmf.sources].sort((a, b) => PAPER_ORDER.indexOf(a.label) - PAPER_ORDER.indexOf(b.label)).map((s) => s.label), [pmf.sources])
  const byId = useMemo(() => new Map(rows.map((r) => [r.id, r])), [rows])
  // each PMF filter as a composition row: its sources' µg/m³ as fields, at the PMF site
  const asRows = useMemo(() => rows.map((r): FilterRow => ({
    id: r.id, site: pmf.site_name, code: pmf.site, color: '', date: r.date, year: r.year, month: r.month, season: r.season,
    lat: null, lon: null, excluded: false, exclusion_reason: null,
    ...Object.fromEntries(labels.map((l) => [l, r.ugm3[l] ?? null])),
  })), [rows, labels, pmf.site, pmf.site_name])
  const share = useCallback((row: FilterRow, f: string) => byId.get(row.id)?.fraction[f] ?? null, [byId])
  const labelOfKey = (k: string | null | undefined) => pmf.sources.find((s) => s.key === k)?.label ?? k ?? '—'
  // the export keys the dominant source (polluted_marine); readers see its name
  const dominant = (row: FilterRow): [string, string | null] => {
    const r = byId.get(row.id)
    return ['dominant source', r?.dominant_source ? `${labelOfKey(r.dominant_source)} (${((r.dominant_fraction ?? 0) * 100).toFixed(0)} %)` : null]
  }

  // dominant_source is meaningful only on the normalised fractions: on the raw GF columns the largest tops out near 0.24
  const dominantOf = useCallback((row: FilterRow) => {
    const r = byId.get(row.id)
    return r?.dominant_source ? { label: labelOfKey(r.dominant_source), share: r.dominant_fraction ?? 0 } : null
  }, [byId]) // eslint-disable-line react-hooks/exhaustive-deps

  return <><CompositionTreemap rows={asRows} meta={meta} id={id} species={labels} palette="PMF paper"
    initialGrouping={initialGrouping} dominantOf={dominantOf} initialUnits="composition (%)" share={share} massField={null}
    calendars={[{ value: PAPER, seasons: PAPER_CALENDAR, title: "the PMF paper's four seasons (its 2023 dates)" }]} initialCalendar={PAPER}
    itemNoun="sources" totalNoun="apportioned" recordSource="the PMF source stack"
    extraTip={(row) => { const [k, v] = dominant(row); return v ? [`${k}: ${v}`] : [] }}
    extraFields={(row) => [dominant(row)]}
    title={title}
    subtitle={`${pmf.n} ETAD filters with a PMF solution. Share stacks each filter's normalised source mix to 100 % (the paper's fractions); concentration stacks the K_Fn contributions in µg/m³, so a stack's height is the apportioned mass. Per site and by season draw the mean or median over the filters instead (seasons from the Seasons choice) and count the filters one source dominates above a chosen share. Click a source in the legend to drop it from the stack; shares are then of the sources shown.`}
    provenance="the estate has exactly 1 stacked-area figure in 827 · react-graph-gallery.com/stacked-area-plot · /streamchart · same chart as Part of whole › Stacked bars" />
    <ExcludedNote count={nMarked} /></>
}
