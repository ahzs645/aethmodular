import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { ExcludedNote } from '@/lib/markedRows'
import { networkColor } from '@/lib/networks'
import { useSearchParam } from '@/lib/url'
import { INK } from '@/lib/theme'
import { type Channel, type Pair, type PairSite } from './data'
import { macMethods, type MacMethods } from './mac'

const METHODS: { key: string; label: string; slope: (m: MacMethods) => number; intercept: (m: MacMethods) => number; dash?: string }[] = [
  { key: 'm1', label: 'M1 mean of ratios', slope: (m) => m.meanOfRatios, intercept: () => 0 },
  { key: 'm2', label: 'M2 ratio of means', slope: (m) => m.ratioOfMeans, intercept: () => 0 },
  { key: 'm3', label: 'M3 regression with intercept', slope: (m) => m.ols.slope, intercept: (m) => m.ols.intercept, dash: '6 4' },
  { key: 'm4', label: 'M4 regression through the origin', slope: (m) => m.origin, intercept: () => 0 },
]
const colourOf = (i: number) => networkColor(i)
const ABS = ['HIPS', 'aeth'] as const

/**
 * AAARpos.ipynb's "physical constraint check": four ways to turn absorption against EC into one MAC,
 * and whether each says there is no absorption without EC. M3 (least squares with an intercept) does
 * not, and its slope is far from the others wherever the intercept is large (Addis: ~28 Mm⁻¹). The
 * notebook drew EC against Fabs; here absorption is on y, so each MAC is a slope.
 */
export function ConstraintCheck({ pairs, marked = [], sites, ch, sigma }: {
  pairs: Pair[]; sites: PairSite[]; ch: Channel; sigma: number | null
  /** Filters → excluded: Mark, the excluded filters paired the same way (lib/markedRows): × on the crossplots, in no method */
  marked?: Pair[]
}) {
  const [abs, setAbs] = useSearchParam<(typeof ABS)[number]>('af.abs', 'HIPS', ABS)
  const yOf = (p: Pair) => (abs === 'HIPS' ? p.fabs : p.batn)
  const yLabel = abs === 'HIPS' ? 'HIPS Fabs (Mm⁻¹)' : `b_ATN ${ch} (Mm⁻¹)`
  const reference = abs === 'HIPS' ? 10 : sigma
  // the MAC set under Filters → units, beside SPARTAN's 10 when it differs
  const { mac } = useMac()
  const readerMac = abs === 'HIPS' && mac !== 10 ? mac : null
  const per = useMemo(() => sites.map((s) => {
    const own = pairs.filter((p) => p.site === s.name && p.ec !== null && yOf(p) !== null)
    const m = macMethods(own.map((p) => p.ec!), own.map((p) => yOf(p)!))
    return { s, own, m, markedOwn: marked.filter((p) => p.site === s.name && p.ec !== null && yOf(p) !== null) }
  }).filter((x): x is typeof x & { m: MacMethods } => x.m !== null), [pairs, marked, sites, abs, ch]) // eslint-disable-line react-hooks/exhaustive-deps

  const lines = (m: MacMethods): CrossLine[] => [
    ...METHODS.map((d, i) => ({ slope: d.slope(m), intercept: d.intercept(m), color: colourOf(i), label: d.label, dash: d.dash, method: 'ols' as const, stats: null })),
    ...(reference ? [{ slope: reference, intercept: 0, color: INK.muted, label: 'reference', dash: '2 3', method: 'ols' as const, stats: null }] : []),
    ...(readerMac ? [{ slope: readerMac, intercept: 0, color: INK.accent, label: `MAC ${readerMac} (Filters)`, dash: '6 3', method: 'ols' as const, stats: null }] : []),
  ]
  const points = (own: Pair[], color: string): CrossPoint[] => own.map((p) => ({
    x: p.ec!, y: yOf(p)!, color, id: p.id,
    tip: [`${p.id} · ${p.site} · ${p.date}`, `FTIR EC ${fmt(p.ec, 2)} µg/m³`, `${yLabel}: ${fmt(yOf(p), 1)}`, `ratio ${fmt(yOf(p)! / p.ec!, 2)} m²/g`],
    rec: { id: p.id, source: 'Aethalometer · MAC methods', site: p.site, date: p.date, fields: [['EC (FTIR)', p.ec], [yLabel, yOf(p)]] },
  }))
  const blocks: IntervalBlock[] = [{ key: 'sites', rows: per.map(({ s, m }) => ({
    key: s.code, label: s.name, labelColor: s.color, note: `n=${m.n}`,
    marks: [METHODS.map((d, i) => ({ mid: d.slope(m), color: colourOf(i), shape: d.key === 'm3' ? 'hollow' as const : 'dot' as const }))],
    tip: [s.name, ...METHODS.map((d) => `${d.label}: ${fmt(d.slope(m), 2)} m²/g`),
      `M3 intercept ${fmt(m.ols.intercept, 1)} Mm⁻¹ (R² ${fmt(m.ols.r2, 3)}): zero absorption at EC = ${fmt(m.xAtZero, 2)} µg/m³`],
  })) }]

  return (
    <ChartFrame id="aeth-filters-constraint" title="Four MAC methods — and whether absorption is zero when EC is"
      subtitle={`${yLabel} against FTIR EC per filter, with the MAC four ways (AAARpos): M1 the mean of the per-filter ratios, M2 the ratio of the means, M3 least squares with an intercept (dashed), M4 least squares through the origin. Only M3 lets absorption be non-zero at zero EC; the strip gives each method's MAC by site and the hover gives M3's intercept and where its line reaches zero absorption. Dotted grey: ${abs === 'HIPS' ? '10 m²/g, SPARTAN\'s MAC at 633 nm' : `σ_ATN at ${ch}, the MAC at which the aethalometer's BC would equal EC`}. OLS lines, independent of the sidebar's estimators.`}
      provenance="AAARpos.ipynb (physical constraint check, four MAC methods) · ETAD_comprehensive_absorption_analysis.ipynb §12–12.1 (the same four methods on IR Babs / FTIR EC)"
      controls={<Segmented label="Absorption" value={abs} options={ABS} onChange={setAbs} optionLabel={(k) => (k === 'HIPS' ? 'HIPS Fabs' : `b_ATN ${ch}`)} />}>
      {per.length === 0 ? <Empty>No site in view has three filters with FTIR EC and {yLabel}.</Empty> : <>
        <IntervalStrip blocks={blocks} panels={[{ label: 'MAC (m²/g)', reference: reference ?? undefined, marker: readerMac ? { value: readerMac, label: `MAC ${readerMac}` } : undefined, format: (v) => fmt(v, 1) }]} labelWidth={110} noteWidth={44} minPanelWidth={170} />
        <FacetGrid items={per} minCell={280} maxCols={per.length > 1 ? 2 : 1} render={({ s, own, m, markedOwn }, w) => (
          <Crossplot points={points(own, INK.muted)} marked={points(markedOwn, INK.muted)} title={`${s.name} · M3 intercept ${fmt(m.ols.intercept, 1)} Mm⁻¹`} xLabel="FTIR EC (µg/m³)" yLabel={yLabel}
            maxSide={Math.min(420, Math.max(180, w - 80))} fit={false} sameScale={false} extraLines={lines(m)} />
        )} />
        <Key items={[...METHODS.map((d, i) => ({ label: d.label, color: colourOf(i), kind: d.dash ? 'dash' as const : 'line' as const })),
          ...(reference ? [{ label: abs === 'HIPS' ? 'MAC 10 m²/g' : `σ_ATN ${ch} = ${fmt(reference, 2)} m²/g`, color: INK.muted, kind: 'dash' as const }] : []),
          ...(readerMac ? [{ label: `MAC ${readerMac} m²/g (Filters → units)`, color: INK.accent, kind: 'dash' as const }] : [])]} />
        <ExcludedNote count={per.reduce((a, x) => a + x.markedOwn.length, 0)} drawn />
        <Note>The notebook's verdict is about the intercept, not the fit: a positive M3 intercept is absorption that does not scale with EC
          (dust, brown carbon, a filter or loading artefact). M1 is pulled up by low-EC filters, whose ratios are the noisiest.</Note>
      </>}
    </ChartFrame>
  )
}
