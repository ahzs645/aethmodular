import { useMemo } from 'react'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { colorsFor } from '@/lib/groupings'
import { coef, f2 } from '@/lib/format'
import { FIT_STYLE, useRegressions } from '@/lib/regressionPrefs'
import { INK, withUnit } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile, PmfFile } from '@/lib/types'
import { thresholdSweep, type AaePair, type AaeRegions } from './aae'
import { NOTEBOOK, provenance } from './common'
import { METHOD_PAIRS, METHOD_PAIR_KEYS, SPLITS, useSplit, type MethodPair, type Split } from './MethodBySource'

const [LOW, HIGH] = colorsFor(2)
/** the notebooks' DOMINANCE_THRESHOLDS */
const NB_THRESHOLDS = [30, 40, 50, 60, 70]
const ALL = 'All'

/**
 * dominant_source_threshold_analysis and threshold_by_source_analysis: split the filters at a threshold on
 * the dominant PMF share (or, new here, on the AAE) and refit each side, for every threshold at once. The
 * notebooks tried five thresholds; the sweep runs through them all, with those five ringed.
 */
export function ThresholdSweep({ rows, pmf, pair, regions }: { rows: FilterRow[]; meta: MetaFile; pmf: PmfFile | null; pair: AaePair; regions: AaeRegions }) {
  const [mp, setMp] = useSearchParam<MethodPair>('aes.sweepPair', 'Aeth IR vs HIPS', METHOD_PAIR_KEYS)
  const [on, setOn] = useSearchParam<Split>('aes.sweepOn', 'PMF dominant (GF)', SPLITS)
  const isPmf = on !== 'AAE class'
  const within = useSplit(pmf, on, pair, regions)
  const groups = isPmf ? [ALL, ...within.groups.map((g) => g.name)] : [ALL]
  // the PMF's sources arrive with pmf.json, so the key is checked here rather than registered with a list
  const [onlyParam, setOnly] = useSearchParam<string>('aes.sweepWithin', ALL)
  const only = groups.includes(onlyParam) ? onlyParam : ALL
  const { methods } = useRegressions()
  const { x: xf, y: yf } = METHOD_PAIRS[mp]

  const data = useMemo(() => {
    const scale = isPmf ? 100 : 1
    const pairs = rows.flatMap((r) => {
      const x = r[xf], y = r[yf], t = within.share(r)
      if (typeof x !== 'number' || typeof y !== 'number' || t == null) return []
      if (isPmf && only !== ALL && within.of(r) !== only) return []
      return [{ x, y, t: t * scale, r }]
    })
    const thresholds = isPmf ? Array.from({ length: 25 }, (_, k) => 26 + 2 * k) : Array.from({ length: 33 }, (_, k) => +(0.6 + 0.05 * k).toFixed(2))
    return { pairs, steps: thresholdSweep(pairs, thresholds) }
  }, [rows, xf, yf, within, isPmf, only])

  const tLabel = isPmf ? `Dominant share threshold (%, ${on.includes('GF') ? 'GF' : 'K_F'})` : `AAE ${pair} threshold`
  const side = (k: 'low' | 'high') => (k === 'low' ? `≤ threshold` : `> threshold`)
  const series = (metric: 'r2' | 'slope' | 'n'): SweepSeries[] => (['low', 'high'] as const).flatMap((k): SweepSeries[] => {
    const color = k === 'low' ? LOW : HIGH
    const pt = (x: number, y: number | null | undefined, n: number) => ({ x, y, tip: [`${side(k)} (${k === 'low' ? 'lower' : 'higher'} ${isPmf ? 'dominance' : 'AAE'})`, `${tLabel}: ${x}`, `${metric === 'n' ? 'filters' : metric === 'r2' ? 'R²' : 'slope'}: ${y == null ? '—' : metric === 'n' ? y : coef(y)}`, `n = ${n}`] })
    if (metric === 'n') return [{ id: k, label: side(k), color, marks: false, curve: 'step' as const, points: data.steps.map((s) => pt(s.threshold, k === 'low' ? s.nLow : s.nHigh, k === 'low' ? s.nLow : s.nHigh)) }]
    if (metric === 'r2') return [{ id: k, label: side(k), color, marks: false, points: data.steps.map((s) => pt(s.threshold, s[k]?.r2, k === 'low' ? s.nLow : s.nHigh)) }]
    return methods.map((m) => ({ id: `${k}-${m}`, label: `${side(k)} · ${FIT_STYLE[m].label}`, color, dash: FIT_STYLE[m].dash, marks: false,
      points: data.steps.map((s) => pt(s.threshold, m === 'ols' ? s[k]?.slope : m === 'deming' ? s[k]?.demingSlope : s[k]?.yorkSlope, k === 'low' ? s.nLow : s.nHigh)) }))
  })
  const chosen = isPmf ? NB_THRESHOLDS.flatMap((t) => ['low', 'high'].map((k) => ({ series: k, x: t }))) : []
  const markers = isPmf ? [] : [{ at: regions.fossil_max, label: `fossil ≤ ${regions.fossil_max}` }, { at: regions.biomass_min, label: `biomass ≥ ${regions.biomass_min}`, labelSide: 'left' as const }]
  const panels = [
    { key: 'r2', yLabel: 'R²', reference: undefined },
    { key: 'slope', yLabel: `Slope (${mp.split(' vs ')[0]} per ${mp.split(' vs ')[1]})`, reference: 1 },
    { key: 'n', yLabel: 'Filters on each side', reference: undefined },
  ] as const
  const site = pmf?.site_name ?? 'Addis Ababa'
  return (
    <ChartFrame id="aeth-sources-threshold-sweep"
      title={`How the fit changes with the threshold — ${mp}, refitted on each side of a threshold on ${isPmf ? 'the dominant PMF share' : `AAE ${pair}`}`}
      subtitle={`${isPmf ? `${site} filters with a PMF day${only !== ALL ? ` whose dominant factor is ${only}` : ''}, split at each threshold into "≤" (more mixed days) and ">" (one factor dominates). The notebooks' five thresholds (30–70 %) are ringed; the rest fill in between.` : `Every filter in view with an aethalometer AAE, split at each AAE: "≤" leans fossil, ">" leans biomass; the region cut points are marked.`} A side needs three filters for a fit; ΔR² in the notebooks is R²(≤) − R²(>). Slopes follow the sidebar's estimators (dashed: Deming / York).`}
      provenance={provenance([NOTEBOOK.dominantThreshold, NOTEBOOK.thresholdBySource, NOTEBOOK.regression], `x: ${xf}, y: ${yf}; ${isPmf ? 'dominant share from pmf.json (GF normalised, or K_F over the day\'s K_F total).' : 'AAE from the aethalometer export.'}`)}
      controls={<>
        <Select label="Pair" value={mp} options={METHOD_PAIR_KEYS} onChange={(v) => setMp(v as MethodPair)} />
        <Select label="Threshold on" value={on} options={[...SPLITS]} onChange={(v) => setOn(v as Split)} optionLabel={(o) => (o === 'AAE class' ? `AAE ${pair}` : o.replace('PMF dominant', 'PMF dominant share'))} />
        {isPmf && <Select label="Within" value={only} options={groups} onChange={(v) => setOnly(v)} optionLabel={(o) => (o === ALL ? 'All dominant sources' : `${o}-dominated days`)} />}
      </>}>
      {data.pairs.length < 6 ? <Empty>Too few filters with {xf}, {yf} and {isPmf ? 'a PMF day' : 'an AAE'} in view ({data.pairs.length}).</Empty> : <>
        <FacetGrid items={[...panels]} minCell={260} maxCols={3} render={(p, w) => (
          <SweepPlot key={p.key} width={w} height={230} series={series(p.key)} xLabel={tLabel} yLabel={p.yLabel} reference={p.reference}
            include={p.key === 'r2' ? [0, 1] : p.key === 'n' ? [0] : []} chosen={p.key === 'slope' ? [] : chosen} xMarkers={markers} />
        )} />
        <Key items={[{ label: `≤ threshold`, color: LOW, kind: 'line' }, { label: `> threshold`, color: HIGH, kind: 'line' }, ...(isPmf ? [{ label: "the notebooks' thresholds", color: INK.text, kind: 'hollow' as const }] : [])]} />
      </>}
    </ChartFrame>
  )
}

/**
 * dominant_source_threshold_analysis's third panel: each filter's difference between the two methods
 * (y − x, in the pair's units) against its dominant share (or AAE). A trend says the methods part ways
 * as one source takes over.
 */
export function ResidualVsDominance({ rows, meta, pmf, pair, regions }: { rows: FilterRow[]; meta: MetaFile; pmf: PmfFile | null; pair: AaePair; regions: AaeRegions }) {
  const [mp, setMp] = useSearchParam<MethodPair>('aes.residPair', 'Aeth IR vs HIPS', METHOD_PAIR_KEYS)
  const [on, setOn] = useSearchParam<Split>('aes.residOn', 'PMF dominant (GF)', SPLITS)
  const isPmf = on !== 'AAE class'
  const s = useSplit(pmf, on, pair, regions)
  const { x: xf, y: yf } = METHOD_PAIRS[mp]
  const pts: CrossPoint[] = useMemo(() => rows.flatMap((r) => {
    const x = r[xf], y = r[yf], t = s.share(r), g = s.of(r)
    if (typeof x !== 'number' || typeof y !== 'number' || t == null) return []
    const tx = isPmf ? 100 * t : t
    return [{ x: tx, y: y - x, color: s.groups.find((q) => q.name === g)?.color ?? INK.neutral, id: r.id,
      tip: [r.id, `${r.site} · ${r.date}`, `${isPmf ? 'dominant share' : `AAE ${pair}`}: ${f2(tx)}`, `${yf} − ${xf}: ${f2(y - x)}`, `${g ?? '—'}`],
      rec: { id: r.id, source: `Aethalometer sources · residual vs ${on}`, site: r.site, date: r.date, fields: [[`${yf} − ${xf}`, +(y - x).toFixed(3)], [isPmf ? 'dominant share (%)' : `AAE ${pair}`, +tx.toFixed(2)], [on, g]] } }]
  }), [rows, xf, yf, s, isPmf, pair, on])
  return (
    <ChartFrame id="aeth-sources-residual"
      title={`Method difference against dominance — ${yf} − ${xf} per filter, against ${isPmf ? 'the dominant PMF share' : `AAE ${pair}`}`}
      subtitle={`Points coloured by ${isPmf ? 'dominant factor' : 'AAE class'}. The fit's r is the notebooks' "residual vs dominance" Pearson r (here as R² with the line's sign). The difference is in the pair's own units (${withUnit(yf, meta.field_units).replace(yf, '').trim() || 'as exported'}); the methods differ in level everywhere (IR BCc runs at about three times HIPS BC with the MAC of 10), so read the trend, not the level.`}
      provenance={provenance([NOTEBOOK.dominantThreshold], 'Same rows and dominance as the threshold sweep.')}
      controls={<>
        <Select label="Pair" value={mp} options={METHOD_PAIR_KEYS} onChange={(v) => setMp(v as MethodPair)} />
        <Select label="Against" value={on} options={[...SPLITS]} onChange={(v) => setOn(v as Split)} optionLabel={(o) => (o === 'AAE class' ? `AAE ${pair}` : o.replace('PMF dominant', 'PMF dominant share'))} />
      </>}>
      {pts.length < 3 ? <Empty>Too few filters in view with {xf}, {yf} and {isPmf ? 'a PMF day' : 'an AAE'}.</Empty>
        : <Crossplot points={pts} title={`${yf} − ${xf}`} xLabel={isPmf ? 'Dominant share (%)' : `AAE ${pair}`} yLabel={`${mp.replace(' vs ', ' − ')} (${withUnit(yf, meta.field_units).slice(yf.length).replace(/[()]/g, '').trim() || 'units'})`} sameScale={false} grouping="pooled" maxSide={360} />}
    </ChartFrame>
  )
}
