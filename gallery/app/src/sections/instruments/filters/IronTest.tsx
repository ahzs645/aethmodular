import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { Crossplot, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { SweepPlot } from '@/components/SweepPlot'
import { fitLine, fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { ExcludedNote } from '@/lib/markedRows'
import { spearman } from '@/lib/weather'
import { seasonOnly, shortSeason } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { seasonColor, seasonsIn, type Channel, type Pair, type PairSite } from './data'
import { closestTo, ironSweep, linspace, ols, ols2 } from './mac'

const ABS = ['HIPS', 'aeth'] as const
const MIN_N = 10, MIN_SEASON = 5
type Row = { p: Pair; ec: number; y: number; fe: number }

/**
 * iron_correction_test.ipynb at any site with iron: is absorption = MAC_BC × EC + MAC_Fe × Iron + c?
 * The two-predictor fit, the sweep of k in absorption − k × Iron against EC (slope, intercept, R²),
 * before and after subtracting the fitted iron term, the season fits before and after, and the
 * residuals of the two-predictor fit against iron and EC. For HIPS Fabs (the notebook) or the
 * aethalometer's b_ATN at the page's channel.
 */
export function IronTest({ pairs, marked = [], meta, sites, ch, sigma }: {
  pairs: Pair[]; meta: MetaFile; sites: PairSite[]; ch: Channel; sigma: number | null
  /** Filters → excluded: Mark, the excluded filters paired the same way (lib/markedRows): × on the crossplots, in no fit, sweep or strip */
  marked?: Pair[]
}) {
  const [abs, setAbs] = useSearchParam<(typeof ABS)[number]>('af.fe.abs', 'HIPS', ABS)
  const yOf = (p: Pair) => (abs === 'HIPS' ? p.fabs : p.batn)
  const yName = abs === 'HIPS' ? 'HIPS Fabs' : `b_ATN ${ch}`
  // the slope the iron-corrected fit should reach: the MAC set under Filters → units (the notebook's 10), or σ_ATN
  const { mac } = useMac()
  const target = abs === 'HIPS' ? mac : sigma ?? NaN
  const eligible = useMemo(() => sites.filter((s) => pairs.filter((p) => p.site === s.name && p.iron !== null && p.ec !== null && yOf(p) !== null).length >= MIN_N),
    [pairs, sites, abs]) // eslint-disable-line react-hooks/exhaustive-deps
  const preferred = eligible.find((s) => s.code === 'ETAD') ?? eligible[0]
  // no option list: the eligible sites change with the channel and the data loading; an absent site falls back below
  const [siteParam, setSite] = useSearchParam<string>('af.fe.site', '')
  const site = eligible.find((s) => s.code === siteParam) ?? preferred

  const t = useMemo(() => {
    if (!site) return null
    const rows: Row[] = pairs.filter((p) => p.site === site.name && p.iron !== null && p.ec !== null && yOf(p) !== null)
      .map((p) => ({ p, ec: p.ec!, y: yOf(p)!, fe: p.iron! }))
    const ec = rows.map((r) => r.ec), y = rows.map((r) => r.y), fe = rows.map((r) => r.fe)
    const base = ols(ec, y), multi = ols2(y, ec, fe)
    if (!base || !multi) return null
    const soilRows = rows.filter((r) => r.p.fineSoil !== null)
    const soil = soilRows.length >= MIN_N ? ols2(soilRows.map((r) => r.y), soilRows.map((r) => r.ec), soilRows.map((r) => r.p.fineSoil!)) : null
    const hiK = Math.abs(multi.b) > 1e-9 ? 3 * multi.b : 1
    const sweep = ironSweep(ec, y, fe, linspace(0, hiK, 200))
    // the closest k on the edge of the sweep (usually k = 0: the slope only moves away from the target) is no correction
    const closest = Number.isFinite(target) ? closestTo(sweep, target) : null
    const atEdge = !!closest && (closest.k === sweep[0]?.k || closest.k === sweep[sweep.length - 1]?.k)
    const best = atEdge ? null : closest
    const corr = (k: number) => y.map((v, i) => v - k * fe[i])
    const fixed = ols(ec, corr(multi.b)), bestFit = best ? ols(ec, corr(best.k)) : null
    const resid = new Map(multi.keep.map((i, k) => [rows[i].p.id, multi.residuals[k]]))
    const seasons = seasonsIn(rows.map((r) => r.p), meta).map((se) => {
      const ix = rows.map((_, i) => i).filter((i) => rows[i].p.season === se)
      const xs = ix.map((i) => ec[i])
      const res = ix.map((i) => resid.get(rows[i].p.id)).filter((v): v is number => v !== undefined).sort(d3.ascending)
      return {
        season: se, n: ix.length, color: seasonColor(meta, se, INK.muted),
        raw: ix.length >= MIN_SEASON ? ols(xs, ix.map((i) => y[i])) : null,
        fixed: ix.length >= MIN_SEASON ? ols(xs, ix.map((i) => y[i] - multi.b * fe[i])) : null,
        best: ix.length >= MIN_SEASON && best ? ols(xs, ix.map((i) => y[i] - best.k * fe[i])) : null,
        resid: res.length >= MIN_SEASON ? { mid: d3.quantileSorted(res, 0.5)!, lo: d3.quantileSorted(res, 0.25)!, hi: d3.quantileSorted(res, 0.75)! } : null,
      }
    })
    // least-squares residuals are uncorrelated with the predictors by construction (Pearson r = 0), so the
    // notebook's r says nothing; a rank correlation still shows a curved or one-sided pattern
    const rFe = spearman(multi.keep.map((i) => fe[i]), multi.residuals), rEc = spearman(multi.keep.map((i) => ec[i]), multi.residuals)
    // the site's excluded filters with all three, drawn as × on the fitted panels (their residual from the same fit)
    const markedRows: Row[] = marked.filter((p) => p.site === site.name && p.iron !== null && p.ec !== null && yOf(p) !== null)
      .map((p) => ({ p, ec: p.ec!, y: yOf(p)!, fe: p.iron! }))
    return { rows, ec, y, fe, base, multi, soil, sweep, best, closest, atEdge, corr, fixed, bestFit, resid, seasons, rFe, rEc, markedRows }
  }, [pairs, marked, meta, site, abs, ch, target]) // eslint-disable-line react-hooks/exhaustive-deps

  const controls = <>
    <Segmented label="Absorption" value={abs} options={ABS} onChange={setAbs} optionLabel={(k) => (k === 'HIPS' ? 'HIPS Fabs' : `b_ATN ${ch}`)} />
    {eligible.length > 1 && <Select label="Site" value={site?.code ?? ''} options={eligible.map((s) => s.code)} onChange={setSite} optionLabel={(c) => eligible.find((s) => s.code === c)?.name ?? c} />}
  </>
  const frame = (body: React.ReactNode) => (
    <ChartFrame id="aeth-filters-iron" title="Iron test — does subtracting iron absorption recover the expected MAC?"
      subtitle={`${yName} = MAC_BC × FTIR EC + MAC_Fe × Iron + c, fitted by least squares over one site's filters with all three (iron_correction_test.ipynb). Then ${yName} − k × Iron is refitted against EC for k from 0 to three times the fitted MAC_Fe, and the k whose slope comes closest to ${abs === 'HIPS' ? `${mac} m²/g (the MAC set under Filters → units; SPARTAN uses 10 at 633 nm)` : `σ_ATN at ${ch} (where the aethalometer's BC would equal EC)`} is marked. Iron in µg/m³, so k is in Mm⁻¹ per µg/m³ (the notebook's per ng/m³ × 1000). Seasons follow the gallery's calendar. OLS throughout, as in the notebook.`}
      provenance="notebooks/analysis/iron_correction_test.ipynb (Tests 1–7) · ETAD_Aug13.ipynb §6–8.5 (BCc, HIPS and EC against iron)"
      controls={controls}>{body}</ChartFrame>
  )
  if (!site) return frame(<Empty>No site in view has {MIN_N} filters with iron, FTIR EC and {yName}.</Empty>)
  if (!t) return frame(<Empty>{site.name}: the two-predictor fit is undefined (too few filters, or iron and EC collinear).</Empty>)

  const { rows, base, multi, soil, sweep, best, closest, atEdge, fixed, bestFit, seasons, markedRows } = t
  const pt = (r: Row, y: number, label: string): CrossPoint => ({
    x: r.ec, y, color: seasonColor(meta, r.p.season, INK.muted), id: r.p.id,
    tip: [`${r.p.id} · ${r.p.date}`, `FTIR EC ${fmt(r.ec, 2)} µg/m³ · Iron ${fmt(r.fe, 3)} µg/m³`, `${label}: ${fmt(y, 1)} Mm⁻¹`, seasonOnly(r.p.season)],
    rec: { id: r.p.id, source: 'Aethalometer · iron test', site: r.p.site, date: r.p.date, fields: [['EC (FTIR)', r.ec], ['Iron', r.fe], [yName, r.y]] },
  })
  const fitLines = (overall: ReturnType<typeof ols>, which: 'raw' | 'fixed' | 'best'): CrossLine[] => [
    ...(overall ? [{ slope: overall.slope, intercept: overall.intercept, color: INK.fit, label: 'all filters', dash: '6 4', method: 'ols' as const, stats: null }] : []),
    ...seasons.flatMap((s) => { const f = s[which]; return f ? [{ slope: f.slope, intercept: f.intercept, color: s.color, label: s.season, method: 'ols' as const, stats: null }] : [] }),
    ...(Number.isFinite(target) ? [{ slope: target, intercept: 0, color: INK.muted, label: 'target', dash: '2 3', method: 'ols' as const, stats: null }] : []),
  ]
  const panelsBA = [
    { key: 'raw', title: `Uncorrected: ${fitLine(base.slope, base.intercept)}, R² ${fmt(base.r2, 2)}`, ys: t.y, k: 0, fit: base, which: 'raw' as const, label: yName },
    { key: 'fixed', title: `− ${fmt(multi.b, 1)} × Iron: ${fitLine(fixed?.slope, fixed?.intercept)}, R² ${fmt(fixed?.r2, 2)}`, ys: t.corr(multi.b), k: multi.b, fit: fixed, which: 'fixed' as const, label: `${yName} − ${fmt(multi.b, 1)} × Iron` },
    ...(best && bestFit ? [{ key: 'best', title: `− ${fmt(best.k, 1)} × Iron (slope ${fmt(target, 1)}): ${fitLine(bestFit.slope, bestFit.intercept)}`, ys: t.corr(best.k), k: best.k, fit: bestFit, which: 'best' as const, label: `${yName} − ${fmt(best.k, 1)} × Iron` }] : []),
  ]
  const allY = panelsBA.flatMap((p) => p.ys), allX = t.ec
  const xDom = d3.scaleLinear().domain([Math.min(0, d3.min(allX)!), d3.max(allX)! * 1.05]).nice().domain() as [number, number]
  const yDom = d3.scaleLinear().domain([Math.min(0, d3.min(allY)!), d3.max(allY)! * 1.05]).nice().domain() as [number, number]
  const kMarks = [{ at: multi.b, label: 'two-predictor MAC_Fe', color: INK.accent }, ...(best ? [{ at: best.k, label: `slope ${fmt(target, 1)}`, color: INK.warn }] : [])]
  const sweepSeries = (f: (s: (typeof sweep)[number]) => number) => [{ id: 'k', label: 'k', color: INK.text, marks: false, points: sweep.map((s) => ({ x: s.k, y: f(s) })) }]
  const seasonBlock: IntervalBlock[] = [{ key: 'seasons', rows: [
    { key: 'all', label: 'All seasons', strong: true, note: `n=${rows.length}`,
      marks: [[{ mid: base.slope, color: INK.text, shape: 'hollow' as const }, { mid: fixed?.slope, color: INK.text }], [{ mid: base.intercept, color: INK.text, shape: 'hollow' as const }, { mid: fixed?.intercept, color: INK.text }], []],
      tip: ['All seasons', `uncorrected: slope ${fmt(base.slope, 2)}, intercept ${fmt(base.intercept, 1)}`, `after − MAC_Fe × Iron: slope ${fmt(fixed?.slope, 2)}, intercept ${fmt(fixed?.intercept, 1)}`] },
    ...seasons.map((s) => ({
      key: s.season, label: shortSeason(seasonOnly(s.season)), labelColor: s.color, note: `n=${s.n}`,
      marks: [[{ mid: s.raw?.slope, color: s.color, shape: 'hollow' as const }, { mid: s.fixed?.slope, color: s.color }],
        [{ mid: s.raw?.intercept, color: s.color, shape: 'hollow' as const }, { mid: s.fixed?.intercept, color: s.color }],
        s.resid ? [{ ...s.resid, color: s.color }] : []],
      tip: [seasonOnly(s.season), s.raw ? `uncorrected: slope ${fmt(s.raw.slope, 2)}, intercept ${fmt(s.raw.intercept, 1)}, R² ${fmt(s.raw.r2, 3)}` : `fewer than ${MIN_SEASON} filters`,
        s.fixed ? `after − MAC_Fe × Iron: slope ${fmt(s.fixed.slope, 2)}, intercept ${fmt(s.fixed.intercept, 1)}, R² ${fmt(s.fixed.r2, 3)}` : '',
        s.resid ? `two-predictor residual: median ${fmt(s.resid.mid, 1)} (IQR ${fmt(s.resid.lo, 1)} to ${fmt(s.resid.hi, 1)}) Mm⁻¹` : ''].filter(Boolean),
    })),
  ] }]
  const spread = (k: 'raw' | 'fixed') => { const v = seasons.map((s) => s[k]?.slope).filter((x): x is number => x != null); return v.length > 1 ? d3.max(v)! - d3.min(v)! : null }
  // the counted filters' residuals from the fit; an excluded filter's (drawn as × only) from the same coefficients
  const markedResid = (r: Row) => r.y - (multi.c + multi.a * r.ec + multi.b * r.fe)
  const residPts = (x: (r: Row) => number, xl: string, src: Row[] = rows, resid: (r: Row) => number | undefined = (r) => t.resid.get(r.p.id)): CrossPoint[] => src.flatMap((r) => {
    const e = resid(r)
    return e === undefined ? [] : [{ x: x(r), y: e, color: seasonColor(meta, r.p.season, INK.muted), id: r.p.id,
      tip: [`${r.p.id} · ${r.p.date}`, `${xl} ${fmt(x(r), 3)}`, `residual ${fmt(e, 1)} Mm⁻¹`, seasonOnly(r.p.season)] }]
  })
  const zero: CrossLine[] = [{ slope: 0, intercept: 0, color: INK.muted, label: 'zero', dash: '4 4', method: 'ols', stats: null }]

  return frame(<>
    <div style={{ overflowX: 'auto' }}>
      <table className="census-table">
        <thead><tr>{[`${site.name}, n=${rows.length}`, 'MAC_BC (m²/g)', 'MAC_Fe (Mm⁻¹ per µg/m³)', 'constant (Mm⁻¹)', 'R²'].map((h) => <th key={h} style={{ textTransform: 'none' }}>{h}</th>)}</tr></thead>
        <tbody>
          <tr><td>{yName} on EC (Test 1)</td><td className="mono">{fmt(base.slope, 2)}</td><td className="mono">—</td><td className="mono">{fmt(base.intercept, 1)}</td><td className="mono">{fmt(base.r2, 3)}</td></tr>
          <tr><td>on EC and iron (Test 2)</td><td className="mono">{fmt(multi.a, 2)}</td><td className="mono">{fmt(multi.b, 1)}</td><td className="mono">{fmt(multi.c, 1)}</td><td className="mono">{fmt(multi.r2, 3)}</td></tr>
          {soil && <tr><td>on EC and fine soil (n={soil.n})</td><td className="mono">{fmt(soil.a, 2)}</td><td className="mono">{fmt(soil.b, 1)} per µg/m³ soil</td><td className="mono">{fmt(soil.c, 1)}</td><td className="mono">{fmt(soil.r2, 3)}</td></tr>}
          <tr><td>− MAC_Fe × Iron, on EC (Test 3)</td><td className="mono">{fmt(fixed?.slope, 2)}</td><td className="mono">{fmt(multi.b, 1)} subtracted</td><td className="mono">{fmt(fixed?.intercept, 1)}</td><td className="mono">{fmt(fixed?.r2, 3)}</td></tr>
          {best && bestFit && <tr><td>− k × Iron for slope {fmt(target, 1)}</td><td className="mono">{fmt(bestFit.slope, 2)}</td><td className="mono">k = {fmt(best.k, 1)}</td><td className="mono">{fmt(bestFit.intercept, 1)}</td><td className="mono">{fmt(bestFit.r2, 3)}</td></tr>}
        </tbody>
      </table>
    </div>
    <Note>
      Season slopes spread {fmt(spread('raw'), 2)} m²/g before the correction and {fmt(spread('fixed'), 2)} after
      ({(spread('fixed') ?? 0) < (spread('raw') ?? 0) ? 'closer' : 'not closer'}). EC and iron together explain {fmt(100 * multi.r2, 0)} % of the variance
      of {yName}, against {fmt(100 * base.r2, 0)} % for EC alone. {multi.a < 0.7 * target ? `MAC_BC stays well below ${fmt(target, 1)} m²/g, so iron alone does not recover the expected MAC here.` : ''}
      {atEdge && closest ? ` No k in the sweep brings the slope to ${fmt(target, 1)}: it is closest at k = ${fmt(closest.k, 1)} (${fmt(closest.slope, 2)} m²/g), the edge of the range, so there is no best-k panel.` : ''}
    </Note>
    <FacetGrid items={[
      { key: 'slope', y: 'Recovered slope (m²/g)', f: (s: (typeof sweep)[number]) => s.slope, ref: Number.isFinite(target) ? target : undefined },
      { key: 'int', y: 'Recovered intercept (Mm⁻¹)', f: (s: (typeof sweep)[number]) => s.intercept, ref: 0 },
      { key: 'r2', y: 'R²', f: (s: (typeof sweep)[number]) => s.r2, ref: undefined },
    ]} minCell={260} maxCols={3} render={(it, w) => (
      <SweepPlot series={sweepSeries(it.f)} xLabel="k (Mm⁻¹ per µg/m³ iron)" yLabel={it.y} width={w} height={220} reference={it.ref} xMarkers={kMarks} />
    )} />
    <FacetGrid items={panelsBA} minCell={260} maxCols={3} render={(p, w) => (
      <Crossplot points={rows.map((r, i) => pt(r, p.ys[i], p.label))} marked={markedRows.map((r) => pt(r, r.y - p.k * r.fe, p.label))} title={p.title} xLabel="FTIR EC (µg/m³)" yLabel={`${p.label} (Mm⁻¹)`}
        maxSide={Math.max(170, w - 80)} fit={false} sameScale={false} domain={xDom} yDomain={yDom} extraLines={fitLines(p.fit, p.which)} />
    )} />
    <IntervalStrip blocks={seasonBlock} labelWidth={110} noteWidth={44} minPanelWidth={170} panels={[
      { heading: 'slope against EC', label: 'm²/g', reference: Number.isFinite(target) ? target : undefined, format: (v) => fmt(v, 1) },
      { heading: 'intercept', label: 'Mm⁻¹', reference: 0, format: (v) => fmt(v, 0) },
      { heading: 'two-predictor residual', label: 'Mm⁻¹ (median, IQR)', reference: 0, format: (v) => fmt(v, 0) },
    ]} />
    <FacetGrid items={[
      { key: 'fe', x: (r: Row) => r.fe, xl: 'Iron (µg/m³)', r: t.rFe },
      { key: 'ec', x: (r: Row) => r.ec, xl: 'FTIR EC (µg/m³)', r: t.rEc },
    ]} minCell={260} maxCols={2} render={(it, w) => (
      <Crossplot points={residPts(it.x, it.xl)} marked={residPts(it.x, it.xl, markedRows, markedResid)} title={`Residual against ${it.xl.split(' (')[0]} · Spearman ρ = ${fmt(it.r, 2)}`} xLabel={it.xl} yLabel="Two-predictor residual (Mm⁻¹)"
        maxSide={Math.max(170, Math.min(360, w - 80))} fit={false} sameScale={false} extraLines={zero} />
    )} />
    <Key items={[...seasons.map((s) => ({ label: seasonOnly(s.season), color: s.color })), { label: 'fit to all filters', color: INK.fit, kind: 'dash' as const },
      { label: Number.isFinite(target) ? `slope ${fmt(target, 1)} through the origin` : 'target', color: INK.muted, kind: 'dash' as const },
      { label: 'uncorrected (strip)', color: INK.muted, kind: 'hollow' as const }, { label: 'after − MAC_Fe × Iron (strip)', color: INK.muted, kind: 'dot' as const }]} />
    <ExcludedNote count={markedRows.length} drawn />
    <Note>The two-predictor residual should be flat against iron (iron fully modelled) and against EC (a single MAC fits), and centred on zero in every season.
      The notebook quoted Pearson r for the first two, but a least-squares fit leaves residuals uncorrelated with its own predictors, so that r is
      zero by construction; the panels give Spearman's rank correlation, which a curved or one-sided pattern still moves. The notebook's
      Kruskal–Wallis test across seasons is left out; the strip shows each season's median and spread instead.</Note>
  </>)
}
