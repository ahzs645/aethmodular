import type { ReactNode } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { Key } from '@/components/Key'
import { SweepPlot } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { ExcludedNote } from '@/lib/markedRows'
import type { FilterRow } from '@/lib/types'
import { INK } from '@/lib/theme'

import { lineKey } from '../../common/blankLines'
import type { Optics } from '../../common/optics'
import { ols, quantile } from '../../common/stats'
import { DataTable, Grid, line, point } from '../../common/ui'
import { refOf, type LinePalette } from './lines'
import { useMarkedPairs } from './marked'

const NARRATIVE_NB = 'research/ftir_hips_chem/hips_offset_narrative.ipynb'

export interface Paired { r: FilterRow; o: Optics }
const keyOf = (o: Optics) => { const ref = refOf(o); return ref ? lineKey(ref) : null }
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/**
 * hips_offset_narrative [10]: tau through time coloured by the calibration line each filter was read against, and
 * tau against FTIR EC with one OLS line per coefficient family. If the tau floor were a line artefact, the families'
 * intercepts would differ; the notebook found them within 0.53–0.58 at Addis.
 */
export function TauByLine({ pairs, site, palette, controls }: { pairs: Paired[]; site: string; palette: LinePalette; controls: ReactNode }) {
  const here = pairs.filter((p) => p.r.site === site && p.o.tau !== null)
  // Filters → excluded: Mark: the site's excluded filters on the tau–EC crossplot, as ×
  const markedHere = useMarkedPairs().filter((p) => p.r.site === site && p.o.tau !== null)
  const keys = [...new Set(here.map((p) => keyOf(p.o)))].sort((a, b) => palette.keys.indexOf(a ?? '') - palette.keys.indexOf(b ?? ''))
  const fam = keys.map((k) => {
    const g = here.filter((p) => keyOf(p.o) === k), withEc = g.filter((p) => num(p.r['EC (FTIR)']) !== null)
    const dates = g.map((p) => p.r.date).sort()
    return { k, n: g.length, from: dates[0], to: dates[dates.length - 1], lot: g[0]?.o.lot ?? '—',
      fit: withEc.length >= 5 ? ols(withEc.map((p) => num(p.r['EC (FTIR)'])!), withEc.map((p) => p.o.tau!)) : null, g, withEc }
  })
  return (
    <ChartFrame id="he-tau-lines" title={`HIPS tau through time, by calibration line — ${site}`}
      subtitle="Left: each filter's HIPS optical depth on its sampling date, coloured by the deployed calibration line (lot · Intercept + Slope·R1) it was read against. Right: tau against FTIR EC with one OLS line per line. A jump in tau where the line changes, or a different intercept per line, would put the Addis floor in the calibration; parallel lines with one floor put it in the samples."
      provenance={`${NARRATIVE_NB} [10] (Addis tau floor; coefficient-family timeline and tau regressions)`} controls={controls}>
      {here.length < 3 ? <Empty>Too few {site} filters with a HIPS tau in view.</Empty> : <>
        <Grid>
          <SweepPlot xLabel="Sampling date" yLabel="HIPS tau" height={300} include={[0]}
            series={fam.map((f) => ({ id: f.k ?? 'none', label: palette.label(f.k), color: palette.color(f.k), width: 0, marks: true, interval: 'none' as const,
              points: f.g.map((p) => ({ x: new Date(p.r.date), y: p.o.tau, id: p.r.id, tip: [p.r.id, `${p.r.site} · ${p.r.date}`, `line ${palette.label(f.k)}`, `tau ${fmt(p.o.tau, 3)} · Fabs ${fmt(p.o.fabs, 1)} Mm⁻¹`] })) }))} />
          <Crossplot sameScale={false} fit={false} maxSide={340} title="tau vs FTIR EC, OLS per line" xLabel="FTIR EC (µg/m³)" yLabel="HIPS tau"
            extraLines={fam.flatMap((f) => line(f.fit, palette.color(f.k), `tau ${f.k}`, '6 3'))}
            points={fam.flatMap((f) => f.withEc.map((p) => point(p.r, num(p.r['EC (FTIR)'])!, p.o.tau!, palette.color(f.k), 'FTIR EC', 'tau', 'Research · HIPS envelope', [`line ${palette.label(f.k)}`])))}
            marked={markedHere.flatMap((p) => { const ec = num(p.r['EC (FTIR)']); return ec !== null ? [point(p.r, ec, p.o.tau!, palette.color(keyOf(p.o)), 'FTIR EC', 'tau', 'Research · HIPS envelope', [`line ${palette.label(keyOf(p.o))}`])] : [] })} />
        </Grid>
        <DataTable rows={fam} rowKey={(f) => f.k ?? 'none'} columns={[
          { key: 'l', label: 'Calibration line', value: (f) => <span style={{ color: palette.color(f.k), fontWeight: 600 }}>{palette.label(f.k)}</span> },
          { key: 'n', label: 'n', num: true, value: (f) => f.n },
          { key: 'd', label: 'Samples', value: (f) => `${f.from} – ${f.to}` },
          { key: 'm', label: 'Median tau', num: true, value: (f) => fmt(quantile(f.g.map((p) => p.o.tau!), 0.5), 3) },
          { key: 's', label: 'tau-on-EC slope', num: true, value: (f) => fmt(f.fit?.slope, 4) },
          { key: 'i', label: 'tau intercept', num: true, value: (f) => fmt(f.fit?.intercept, 4) },
          { key: 'r', label: 'R²', num: true, value: (f) => fmt(f.fit?.r2, 3) },
        ]} />
        <Key items={fam.map((f) => ({ label: palette.label(f.k), color: palette.color(f.k) }))} />
        <ExcludedNote count={markedHere.length} drawn />
      </>}
    </ChartFrame>
  )
}

/**
 * hips_offset_narrative [23]: tau against PM2.5 mass, every site, and the loading-regime table beside it (median
 * tau, sample T1 over the site's mean field-blank T1, the shares past T1 < 50 % of blank and tau > 1).
 */
export function TauPm({ pairs, blanks, sites, colorOfSite }: { pairs: Paired[]; blanks: Optics[]; sites: string[]; colorOfSite: (s: string) => string }) {
  const rowsBySite = sites.map((s) => {
    const g = pairs.filter((p) => p.r.site === s && p.o.tau !== null)
    const pm = g.filter((p) => num(p.r['PM2.5 mass']) !== null)
    const code = g[0]?.r.code
    const bl = blanks.filter((b) => b.code === code && b.T1 !== null).map((b) => b.T1!)
    const blankT1 = bl.length ? bl.reduce((a, b) => a + b, 0) / bl.length : NaN
    const ratio = g.filter((p) => p.o.T1 !== null).map((p) => p.o.T1! / blankT1)
    return { s, g, pm, fit: pm.length >= 5 ? ols(pm.map((p) => num(p.r['PM2.5 mass'])!), pm.map((p) => p.o.tau!)) : null, nBlank: bl.length,
      tau: quantile(g.map((p) => p.o.tau!), 0.5), ratio: Number.isFinite(blankT1) ? quantile(ratio, 0.5) : NaN,
      below: ratio.length && Number.isFinite(blankT1) ? (100 * ratio.filter((v) => v < 0.5).length) / ratio.length : NaN,
      over1: g.length ? (100 * g.filter((p) => p.o.tau! > 1).length) / g.length : NaN }
  })
  const all = rowsBySite.flatMap((x) => x.pm)
  // Filters → excluded: Mark: the excluded filters of these sites on the crossplot, as ×; in no fit and no row of the table
  const marked = useMarkedPairs().filter((p) => sites.includes(p.r.site) && p.o.tau !== null)
  return (
    <ChartFrame id="he-tau-pm" title="HIPS tau against PM2.5 mass, and the loading regime by site"
      subtitle="Each filter's HIPS optical depth against the gallery's date-matched PM2.5 mass, coloured by site, with an OLS line per site; the intercept is the tau floor at zero PM2.5. The table is the notebook's loading-regime readout: sample T1 relative to the site's mean field-blank T1, and how often a site's filters sit far from the blank regime the calibration was fitted in."
      provenance={`${NARRATIVE_NB} [23] (ChemSpec loading regime; tau vs PM2.5 mass)`}>
      {all.length < 5 ? <Empty>Too few filters with both HIPS tau and PM2.5 mass in view.</Empty> : <>
        <Crossplot sameScale={false} fit={false} maxSide={380} title="tau vs PM2.5 mass, OLS per site" xLabel="PM2.5 mass (µg/m³)" yLabel="HIPS tau"
          extraLines={rowsBySite.flatMap((x) => line(x.fit, colorOfSite(x.s), `tau-pm ${x.s}`))}
          points={rowsBySite.flatMap((x) => x.pm.map((p) => point(p.r, num(p.r['PM2.5 mass'])!, p.o.tau!, colorOfSite(x.s), 'PM2.5', 'tau', 'Research · HIPS envelope')))}
          marked={marked.flatMap((p) => { const pm = num(p.r['PM2.5 mass']); return pm !== null ? [point(p.r, pm, p.o.tau!, colorOfSite(p.r.site), 'PM2.5', 'tau', 'Research · HIPS envelope')] : [] })} />
        <DataTable rows={rowsBySite} rowKey={(x) => x.s} columns={[
          { key: 's', label: 'Site', value: (x) => <span style={{ color: colorOfSite(x.s), fontWeight: 600 }}>{x.s}</span> },
          { key: 'n', label: 'n', num: true, value: (x) => x.g.length },
          { key: 'f', label: 'tau = a + b·PM2.5', num: true, value: (x) => (x.fit ? `${fmt(x.fit.intercept, 3)} + ${fmt(x.fit.slope, 4)}·PM` : '—') },
          { key: 'r', label: 'R²', num: true, value: (x) => fmt(x.fit?.r2, 3) },
          { key: 't', label: 'Median tau', num: true, value: (x) => fmt(x.tau, 3) },
          { key: 'b', label: 'Median T1 / blank', num: true, title: 'sample T1 over the mean T1 of the site\'s field blanks', value: (x) => fmt(x.ratio, 2) },
          { key: 'h', label: 'T1 < 50 % blank (%)', num: true, value: (x) => fmt(x.below, 1) },
          { key: 'o', label: 'tau > 1 (%)', num: true, value: (x) => fmt(x.over1, 1) },
        ]} />
      </>}
      <Key items={[...sites.map((s) => ({ label: s, color: colorOfSite(s) })), { label: 'OLS per site', color: INK.text, kind: 'line' }]} />
      <ExcludedNote count={marked.length} drawn />
    </ChartFrame>
  )
}
