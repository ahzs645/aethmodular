import { ChartFrame } from '@/components/ChartFrame'
import { Crossplot, sharedCrossDomains, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { ExcludedNote } from '@/lib/markedRows'
import { INK } from '@/lib/theme'

import { lineKey, rSuppression } from '../../common/blankLines'
import type { Optics } from '../../common/optics'
import { quantile } from '../../common/stats'
import { line, point } from '../../common/ui'
import { refOf, type LinePalette } from './lines'
import { useMarkedPairs } from './marked'
import type { Paired } from './TauViews'

const WARREN_NB = 'research/ftir_hips_chem/warren_meeting_slides.ipynb'
const keyOf = (o: Optics) => { const ref = refOf(o); return ref ? lineKey(ref) : null }

/**
 * warren_meeting_slides [30] (slide 10c): per site, raw R1 against T1 with every filter on the calibration line it
 * was reduced with (T1 = Intercept + Slope·R1). The red marker is the site's median sample; the bar runs from it to
 * the R1 its line expects at that T1, and the shortfall is the median R suppression.
 */
export function RowLines({ pairs, blanks, sites, palette, colorOfSite }: { pairs: Paired[]; blanks: Optics[]; sites: string[]; palette: LinePalette; colorOfSite: (s: string) => string }) {
  // Filters → excluded: Mark: each site's excluded samples on their line, as ×; in no median, count or axis range
  const markedAll = useMarkedPairs()
  const sample = (p: Paired) => point(p.r, p.o.R1!, p.o.T1!, palette.color(keyOf(p.o)), 'R1', 'T1', 'Research · HIPS optics', [`line ${palette.label(keyOf(p.o))}`, `tau ${fmt(p.o.tau, 3)}`])
  const facets = sites.map((s) => {
    const smp = pairs.filter((p) => p.r.site === s && p.o.R1 !== null && p.o.T1 !== null && keyOf(p.o))
    const code = smp[0]?.r.code ?? pairs.find((p) => p.r.site === s)?.r.code
    const bl = blanks.filter((b) => b.code === code && b.R1 !== null && b.T1 !== null)
    const sup = smp.flatMap((p) => { const v = rSuppression({ R1: p.o.R1!, T1: p.o.T1!, intercept: p.o.cal_intercept!, slope: p.o.cal_slope! }); return v ? [{ p, ...v }] : [] })
    const keys = [...new Set([...smp.map((p) => keyOf(p.o)!), ...bl.flatMap((b) => keyOf(b) ?? [])])].sort((a, b) => palette.keys.indexOf(a) - palette.keys.indexOf(b))
    const marked = markedAll.filter((p) => p.r.site === s && p.o.R1 !== null && p.o.T1 !== null && keyOf(p.o)).map(sample)
    const points: CrossPoint[] = [
      ...smp.map(sample),
      ...bl.map((b) => ({ x: b.R1!, y: b.T1!, color: palette.color(keyOf(b)), hollow: true, tip: [`${b.filter_id} (field blank)`, `line ${palette.label(keyOf(b))}`, `R1 ${fmt(b.R1, 0)} · T1 ${fmt(b.T1, 0)}`] })),
    ]
    return { s, points, marked, keys, nS: smp.length, nB: bl.length,
      med: sup.length ? { R1: quantile(sup.map((v) => v.p.o.R1!), 0.5), T1: quantile(sup.map((v) => v.p.o.T1!), 0.5), rExp: quantile(sup.map((v) => v.rExpected), 0.5),
        pct: quantile(sup.map((v) => v.pct), 0.5), tau: quantile(sup.map((v) => v.p.o.tau ?? NaN).filter(Number.isFinite), 0.5) } : null }
  })
  // the axes take in where each median's line expects it, so the suppression bar is drawn whole
  const dom = sharedCrossDomains([...facets.map((f) => f.points), facets.flatMap((f) => (f.med ? [{ x: f.med.rExp, y: f.med.T1, color: '', tip: [] }] : []))], false)
  const used = [...new Set(facets.flatMap((f) => f.keys))].sort((a, b) => palette.keys.indexOf(a) - palette.keys.indexOf(b))
  return (
    <ChartFrame id="he-row-lines" title="Each filter against its own calibration line — raw R1 / T1 by site"
      subtitle="Samples filled and field blanks hollow, both coloured by the deployed line they were reduced with; each line present at the site is drawn as T1 = Intercept + Slope·R1. The red diamond is the site's median sample and the bar runs to the R1 its line expects at that T1: the median R suppression, (1 − R1 / R1 expected). Shared axes, as the notebook's 2 × 2."
      provenance={`${WARREN_NB} [30] (slide 10c, per-site R1/T1 against each row's line)`}>
      <FacetGrid items={facets} minCell={280} title={(f) => f.s} note={(f) => (f.med ? `R suppression ${fmt(f.med.pct, 0)} % · tau ${fmt(f.med.tau, 2)}` : '')} render={(f, w) => (
        <div style={{ width: w }}>
          <Crossplot sameScale={false} fit={false} maxSide={w - 72} domain={dom.domain} yDomain={dom.yDomain}
            title={`${f.nS} samples, ${f.nB} field blanks`}
            xLabel="Raw HIPS sphere signal R1" yLabel="Raw HIPS plate signal T1"
            extraLines={f.keys.flatMap((k) => { const ref = palette.ref(k); return ref ? line(ref, palette.color(k), `row line ${k}`, '6 3') : [] })}
            background={(x, y) => f.med && <g>
              <line x1={x(f.med.rExp)} x2={x(f.med.R1)} y1={y(f.med.T1)} y2={y(f.med.T1)} stroke={INK.negative} strokeWidth={2} />
              <path d={`M${x(f.med.R1)},${y(f.med.T1) - 9}l9,9l-9,9l-9,-9z`} fill={INK.negative} stroke={INK.panel} strokeWidth={1.5} />
              <line x1={x(f.med.rExp)} x2={x(f.med.rExp)} y1={y(f.med.T1) - 6} y2={y(f.med.T1) + 6} stroke={INK.negative} strokeWidth={2} />
            </g>}
            points={f.points} marked={f.marked} />
        </div>
      )} />
      <Key items={[...used.map((k) => ({ label: palette.label(k), color: palette.color(k), kind: 'dash' as const })), { label: 'field blank', color: INK.muted, kind: 'hollow' as const },
        { label: 'median sample → R1 its line expects', color: INK.negative, kind: 'diamond' as const }]} />
      <ExcludedNote count={facets.reduce((a, f) => a + f.marked.length, 0)} drawn />
      <p className="chart-note">Here colour is the calibration line, not the site. {facets.map((f, i) => <span key={f.s}>{i ? ' · ' : ''}<span style={{ color: colorOfSite(f.s), fontWeight: 600 }}>{f.s}</span>: {f.keys.length} line{f.keys.length === 1 ? '' : 's'}{f.med ? `, median R1 ${fmt(f.med.R1, 0)} where its line expects ${fmt(f.med.rExp, 0)}` : ''}</span>)}.</p>
    </ChartFrame>
  )
}
