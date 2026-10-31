import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { f2, pct } from '@/lib/format'
import { useSearchParam } from '@/lib/url'
import { ExcludedNote, FixedSetNote, useMarkedRows } from '@/lib/markedRows'
import type { FilterRow, MetaFile, PmfFile, PmfRow } from '@/lib/types'
import { AAE_PAIRS, CLASSES, CLASS_COLOR, DOMINANCE_LABEL, cell, classify, crosstab, dominantOf, type AaeClass, type AaePair, type AaeRegions, type Dominance } from './aae'
import { rowAae } from './AaeValues'
import { CountMatrix, NOTEBOOK, provenance } from './common'
import { usePmfByRow } from './data'

const DOMINANCES = ['GF', 'K_F'] as const
/** the PMF factors whose fuel the AAE classes name: burning → biomass, fossil fuel → fossil */
const BURNING = new Set(['wood', 'charcoal'])
const expected = (key: string): AaeClass | null => (BURNING.has(key) ? 'Biomass' : key === 'fossil_fuel' ? 'Fossil fuel' : null)

/**
 * The factors as the meteorology notebooks grouped them (meteorology_source_interaction, meteorology_revisions,
 * friday_summary_consolidated: GROUPED_SOURCE_CATEGORIES): biomass burning = charcoal + wood, fossil fuel combustion =
 * fossil fuel + polluted marine, sea salt alone. A day's group is the largest summed share (or µg/m³).
 */
const SOURCE_SETS = ['Factors', 'Grouped'] as const
type SourceSet = (typeof SOURCE_SETS)[number]
interface SourceGroup { label: string; color: string; keys: string[]; aae: AaeClass | null }
const NOTEBOOK_GROUPS: SourceGroup[] = [
  { label: 'Biomass burning (wood + charcoal)', color: '#556B2F', keys: ['wood', 'charcoal'], aae: 'Biomass' },
  { label: 'Fossil fuel comb. (+ polluted marine)', color: '#8B0000', keys: ['fossil_fuel', 'polluted_marine'], aae: 'Fossil fuel' },
  { label: 'Sea Salt Mixed', color: '#1ABC9C', keys: ['sea_salt'], aae: null },
]
const groupsOf = (pmf: PmfFile, set: SourceSet): SourceGroup[] => set === 'Grouped' ? NOTEBOOK_GROUPS
  : pmf.sources.map((s) => ({ label: s.label, color: s.color, keys: [s.key], aae: expected(s.key) }))

/** a PMF day's dominant group: the largest sum of its factors' GF fractions or K_F µg/m³ (a factor alone when ungrouped) */
function dominantGroup(p: PmfRow, pmf: PmfFile, rule: Dominance, groups: SourceGroup[]) {
  const values = rule === 'GF' ? p.fraction : p.ugm3
  const labelOf = new Map(pmf.sources.map((s) => [s.key, s.label]))
  const sums = Object.fromEntries(groups.map((g) => {
    const vs = g.keys.map((k) => values[labelOf.get(k) ?? '']).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
    return [g.label, vs.length ? vs.reduce((a, b) => a + b, 0) : null]
  }))
  return dominantOf(sums, groups.map((g) => g.label))
}

/** a PMF day's dominant source under either rule, with its share */
export function pmfDominant(p: PmfRow, pmf: PmfFile, rule: Dominance) {
  const labels = pmf.sources.map((s) => s.label)
  const d = dominantOf(rule === 'GF' ? p.fraction : p.ugm3, labels)
  return d ? { ...d, key: pmf.sources.find((s) => s.label === d.source)!.key } : null
}

interface Joined { row: FilterRow; p: PmfRow; aae: number | null }
function useJoined(rows: FilterRow[], pmf: PmfFile, pair: AaePair) {
  const pmfOf = usePmfByRow(pmf)
  return useMemo<Joined[]>(() => rows.flatMap((r) => {
    const p = pmfOf(r)
    return p ? [{ row: r, p, aae: rowAae(r, pair, 'absorption') }] : []
  }), [rows, pmfOf, pair])
}

/**
 * The dominant source by AAE against the PMF's dominant factor at Addis (the PMF solution is ETAD only):
 * an agreement matrix of filters, and each factor's AAE as median and interquartile range.
 */
export function PmfVsAae({ rows, meta, pmf, pair, regions }: { rows: FilterRow[]; meta: MetaFile; pmf: PmfFile; pair: AaePair; regions: AaeRegions }) {
  const [rule, setRule] = useSearchParam<Dominance>('aes.pmfRule', 'GF', DOMINANCES)
  const [set, setSet] = useSearchParam<SourceSet>('aes.pmfSet', 'Factors', SOURCE_SETS)
  const groups = groupsOf(pmf, set)
  const joined = useJoined(rows, pmf, pair)
  const withAae = joined.filter((j) => j.aae !== null)
  // Filters → excluded: Mark (lib/markedRows): the excluded filters with a PMF day and an AAE, in no cell or strip
  const nMarked = useJoined(useMarkedRows(), pmf, pair).filter((j) => j.aae !== null).length
  const pairs = withAae.map((j) => ({ j, d: dominantGroup(j.p, pmf, rule, groups), cls: classify(j.aae, regions) }))
  const tab = crosstab(pairs.map((x) => ({ a: x.d?.source ?? null, b: x.cls })))
  const labels = groups.map((g) => g.label)
  const color = new Map(groups.map((g) => [g.label, g.color]))
  const aaeOf = new Map(groups.map((g) => [g.label, g.aae]))
  const matched = pairs.filter((x) => x.d && aaeOf.get(x.d.source))
  const agree = matched.filter((x) => aaeOf.get(x.d!.source) === x.cls).length
  const grouped = set === 'Grouped'
  const strip: IntervalRow[] = labels.map((l) => {
    const v = pairs.filter((x) => x.d?.source === l).map((x) => x.j.aae as number).sort(d3.ascending)
    return { key: l, label: l, labelColor: color.get(l), note: `n=${v.length}`,
      marks: [[{ mid: d3.quantile(v, 0.5), lo: d3.quantile(v, 0.25), hi: d3.quantile(v, 0.75), color: color.get(l) ?? '#888' }]],
      tip: [l, `AAE ${pair} median ${f2(d3.quantile(v, 0.5))}`, `IQR ${f2(d3.quantile(v, 0.25))} to ${f2(d3.quantile(v, 0.75))}`, `${v.length} filters`] }
  })
  const site = meta.sites.find((s) => s.code === pmf.site)?.name ?? pmf.site_name
  return (
    <ChartFrame id="aeth-sources-pmf-vs-aae"
      title={`AAE class against the PMF dominant ${grouped ? 'source group' : 'source'} — ${site} filters with a PMF day and an aethalometer AAE`}
      subtitle={grouped
        ? `Rows: the notebooks' three source groups (meteorology_source_interaction: biomass burning = wood + charcoal, fossil fuel combustion = fossil fuel + polluted marine, sea salt alone), the group with the largest summed ${DOMINANCE_LABEL[rule]} on the filter's day; columns: the filter's AAE ${pair} class from absorption (fossil ≤ ${regions.fossil_max}, biomass ≥ ${regions.biomass_min}). Outlined: where the two name the same fuel. Each cell: filters and their share of the row.`
        : `Rows: the PMF factor with the largest ${DOMINANCE_LABEL[rule]} on the filter's day; columns: the filter's AAE ${pair} class from absorption (fossil ≤ ${regions.fossil_max}, biomass ≥ ${regions.biomass_min}). Outlined: where the two name the same fuel (wood or charcoal → biomass, fossil fuel → fossil). Marine and sea-salt days have no AAE counterpart. Each cell: filters and their share of the row.`}
      provenance={provenance([NOTEBOOK.addis01, NOTEBOOK.dominantComparison, NOTEBOOK.regression], `PMF from pmf.json (normalised GF fractions and K_F µg/m³), joined by filter id, else sampling date (lib/pmfGroups); AAE from "${AAE_PAIRS[pair].field}".`)}
      controls={<>
        <Segmented label="Sources" value={set} options={SOURCE_SETS} onChange={setSet} optionLabel={(o) => (o === 'Factors' ? 'Five factors' : 'Grouped (notebooks)')}
          title="The five PMF factors, or the three groups the meteorology notebooks used: biomass burning (wood + charcoal), fossil fuel combustion (fossil fuel + polluted marine) and sea salt" />
        <Segmented label="Dominant by" value={rule} options={DOMINANCES} onChange={setRule} optionLabel={(o) => (o === 'GF' ? 'GF fraction' : 'K_F µg/m³')} />
      </>}>
      {!withAae.length ? <Empty>No {site} filter in view has both a PMF day and an aethalometer AAE (the PMF covers {pmf.n} days in {pmf.rows[0]?.year}).</Empty> : <>
        <CountMatrix id="aeth-sources-pmf-vs-aae" rows={labels} cols={[...CLASSES]} count={(r, c) => cell(tab, r, c)} rowLabel={grouped ? 'PMF dominant group' : 'PMF dominant source'} colLabel={`AAE ${pair} class`}
          rowColor={(r) => color.get(r) ?? ''} colColor={(c) => d3.color(CLASS_COLOR[c as AaeClass])!.darker(0.7).formatHex()} outline={(r, c) => aaeOf.get(r) === c} />
        <p className="chart-note">{withAae.length} of {joined.length} PMF filters in view have an AAE. Of the {matched.length} whose dominant {grouped ? 'group is biomass burning or fossil fuel combustion' : 'factor is a fuel (wood, charcoal or fossil)'}, {agree} ({pct(matched.length ? agree / matched.length : null)}) fall in the matching AAE class.</p>
        <IntervalStrip blocks={[{ key: 'pmf', rows: strip }]} labelWidth={grouped ? 230 : 150} noteWidth={44} minPanelWidth={150}
          panels={[{ label: `AAE ${pair} (median and interquartile range)`, marker: { value: regions.biomass_min, label: `biomass ≥ ${regions.biomass_min}` } }]} />
        <ExcludedNote count={nMarked} />
      </>}
    </ChartFrame>
  )
}

/** dominant_source_comparison: which factor dominates a PMF day by GF fraction against by K_F µg/m³ (every PMF day). */
export function DominanceRules({ pmf }: { pmf: PmfFile }) {
  const labels = pmf.sources.map((s) => s.label)
  const color = new Map(pmf.sources.map((s) => [s.label, s.color]))
  const pairs = pmf.rows.map((p) => ({ a: pmfDominant(p, pmf, 'GF')?.source ?? null, b: pmfDominant(p, pmf, 'K_F')?.source ?? null }))
  const tab = crosstab(pairs)
  const agree = pairs.filter((x) => x.a && x.a === x.b).length
  const burningLabels = new Set(pmf.sources.filter((s) => BURNING.has(s.key)).map((s) => s.label))
  const toBurning = pairs.filter((x) => x.a !== x.b && burningLabels.has(x.b ?? '') && !burningLabels.has(x.a ?? '')).length
  return (
    <ChartFrame id="aeth-sources-dominance-rules"
      title="Two rules for the dominant source — by normalised GF fraction (rows) and by K_F contribution in µg/m³ (columns), every PMF day"
      subtitle={`The notebooks split the regressions by either rule. They agree on ${agree} of ${pmf.rows.length} days (${pct(agree / Math.max(1, pmf.rows.length))}); of the ${pmf.rows.length - agree} that differ, ${toBurning} are days a burning factor (wood or charcoal) leads by µg/m³ but not by fraction. So "charcoal-dominated" filters are not the same set under the two rules, and a split regression changes with the rule.`}
      provenance={provenance([NOTEBOOK.dominantComparison, NOTEBOOK.dominantThreshold], 'pmf.json: GF normalised with etad_factors.normalize_gf_fractions (raw GF are PM2.5 mass fractions); K_F as exported.')}>
      <CountMatrix id="aeth-sources-dominance-rules" rows={labels} cols={labels} count={(r, c) => cell(tab, r, c)} rowLabel="by GF fraction" colLabel="by K_F µg/m³"
        rowColor={(r) => color.get(r) ?? ''} colColor={(c) => color.get(c) ?? ''} outline={(r, c) => r === c} cellWidth={96} />
      <FixedSetNote what="every day of the PMF solution (pmf.json), not the filters in view" />
    </ChartFrame>
  )
}

/** AAE against the PMF's burning share (wood + charcoal): does a filter's AAE rise with the factors that burn biomass? */
export function AaeVsBurning({ rows, meta, pmf, pair, regions }: { rows: FilterRow[]; meta: MetaFile; pmf: PmfFile; pair: AaePair; regions: AaeRegions }) {
  const joined = useJoined(rows, pmf, pair)
  // Filters → excluded: Mark (lib/markedRows): the excluded filters joined the same way, × only, in no fit
  const markedJoined = useJoined(useMarkedRows(), pmf, pair)
  const burning = pmf.sources.filter((s) => BURNING.has(s.key)).map((s) => s.label)
  const pointsOf = (js: Joined[]): CrossPoint[] => js.flatMap((j) => {
    if (j.aae === null) return []
    const share = burning.reduce((a, l) => a + (j.p.fraction[l] ?? 0), 0)
    const d = pmfDominant(j.p, pmf, 'GF')
    const src = pmf.sources.find((s) => s.label === d?.source)
    return [{ x: 100 * share, y: j.aae, color: src?.color ?? '#888', id: j.row.id,
      tip: [j.row.id, j.row.date, `wood + charcoal ${f2(100 * share)} %`, `AAE ${pair} ${f2(j.aae)}`, `dominant: ${d?.source ?? '—'}`],
      rec: { id: j.row.id, source: 'Aethalometer sources · AAE vs PMF burning share', site: j.row.site, date: j.row.date, fields: [['wood + charcoal share (%)', +(100 * share).toFixed(1)], [`AAE ${pair}`, j.aae], ['PMF dominant (GF)', d?.source ?? null]] } }]
  })
  const pts = pointsOf(joined), marked = pointsOf(markedJoined)
  const site = meta.sites.find((s) => s.code === pmf.site)?.name ?? pmf.site_name
  return (
    <ChartFrame id="aeth-sources-aae-burning"
      title={`AAE against the PMF burning share — ${site}: wood + charcoal share of the day's PMF (x) and the filter's AAE ${pair} (y)`}
      subtitle={`Points coloured by the day's dominant factor (GF). Shaded: the AAE classes. A rising fit would say the absorption AAE tracks the PMF's biomass-burning factors; the axes are different quantities, so read the fit's sign and R² (no 1:1 line).`}
      provenance={provenance([NOTEBOOK.addis01, NOTEBOOK.regression, NOTEBOOK.addisSource], 'Normalised GF fractions from pmf.json, AAE from the aethalometer export.')}>
      {!pts.length ? <Empty>No {site} filter in view has both a PMF day and an aethalometer AAE.</Empty> : <>
        <Crossplot points={pts} marked={marked} title={`AAE ${pair} vs wood + charcoal share`} xLabel="Wood + charcoal share of the PMF (%)" yLabel={`AAE ${pair}`} sameScale={false} grouping="pooled" maxSide={340}
          background={(x, y) => <g>{CLASSES.map((c) => {
            const [a, b] = c === 'Fossil fuel' ? [y.domain()[0], regions.fossil_max] : c === 'Mixed' ? [regions.fossil_max, regions.biomass_min] : [regions.biomass_min, y.domain()[1]]
            const top = y(Math.min(Math.max(b, y.domain()[0]), y.domain()[1])), bot = y(Math.max(Math.min(a, y.domain()[1]), y.domain()[0]))
            return bot > top ? <rect key={c} x={0} y={top} width={x.range()[1]} height={bot - top} fill={CLASS_COLOR[c]} fillOpacity={0.08} /> : null
          })}</g>} />
        <ExcludedNote count={marked.length} drawn />
      </>}
    </ChartFrame>
  )
}
