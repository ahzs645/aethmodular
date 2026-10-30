/**
 * The filter-level charts: held-out predictions against reported HIPS (stability 05), training against test EC
 * ranges (stability 07) and held-out errors by the EC source rows' CalibrationSetId (stability 08).
 */
import { ChartFrame, Note, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { quantile } from '../../common/stats'
import { Grid } from '../../common/ui'
import { quarterIndex, type Scheme } from '../../common/filterOnly'
import { NB_STAB, SCHEMES, SOURCE, blocksOf, popLabel, schemeLabel, siteColor, type Settings, type SiteRun } from './data'

/** stability 05: each filter's prediction with its own quarter withheld, against its reported HIPS */
export function HeldOutScatter({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  const [scheme, setScheme] = useSearchParam<Scheme>('fo.scat.split', 'leave_quarter_out', SCHEMES)
  const label = s.unit === 'bc' ? `HIPS BC at MAC ${fmt(s.mac, 1)} (µg/m³)` : 'HIPS Fabs (Mm⁻¹)'
  return (
    <ChartFrame id="fo-scatter" title="Held-out predictions — block-specific errors remain"
      subtitle={`${popLabel(s.population)} population · ${schemeLabel(scheme).toLowerCase()} · OLS prediction with the filter's quarter withheld · 1:1 reference, no fitted line`}
      provenance={<>{NB_STAB} figure 05</>} source={`${SOURCE}/Predictions.tsx › HeldOutScatter`}
      tip={<>Each prediction comes from a fit that never saw the filter's quarter. The 1:1 line compares predicted and reported HIPS in the same units; no regression is fitted to this comparison. FTIR-predicted EC itself is not established as an independent reference.</>}
      controls={<Segmented label="Split" value={scheme} options={SCHEMES} onChange={setScheme} optionLabel={schemeLabel} />}>
      <Grid min={300}>
        {runs.map((r) => {
          const color = siteColor(meta, r.site)
          const preds = blocksOf(r, scheme).flatMap((b) => b.preds).filter((q) => Number.isFinite(q.pred.ols))
          const mae = preds.reduce((a, q) => a + Math.abs(q.err.ols), 0) / Math.max(1, preds.length)
          const pts: CrossPoint[] = preds.map((q) => ({ x: q.p.fabs * s.k, y: q.pred.ols * s.k, color, id: q.p.id, site: r.site,
            tip: [q.p.id, `${r.site} · ${q.p.date} · ${q.block}`, `reported ${fmt(q.p.fabs * s.k, 2)} · predicted ${fmt(q.pred.ols * s.k, 2)} ${s.u}`, `FTIR EC ${fmt(q.p.ec, 2)} µg/m³${q.outside ? ' (outside training range)' : ''}`],
            rec: { id: q.p.id, source: 'Research · filter-only held-out prediction', site: r.site, date: q.p.date, fields: [[`Reported ${label}`, q.p.fabs * s.k], [`Held-out prediction (${s.u})`, q.pred.ols * s.k], ['FTIR EC (µg/m³)', q.p.ec]] } }))
          return <Crossplot key={r.site} points={pts} title={`${r.site} · n ${preds.length} · MAE ${fmt(mae * s.k, 2)}`} xLabel={`Reported ${label}`} yLabel={`Held-out prediction (${s.u})`} fit={false} grouping="none" maxSide={340} />
        })}
      </Grid>
    </ChartFrame>
  )
}

/** stability 07: each withheld quarter's EC range against its training range, with the count outside */
export function ConcentrationRanges({ runs, meta, s }: { runs: SiteRun[]; meta: MetaFile; s: Settings }) {
  const [scheme, setScheme] = useSearchParam<Scheme>('fo.range.split', 'leave_quarter_out', SCHEMES)
  const blocks: IntervalBlock[] = runs.map((r) => {
    const color = siteColor(meta, r.site)
    return { key: r.site, title: r.site, color, rows: blocksOf(r, scheme).filter((b) => b.testN > 0).map((b) => {
      const train = b.trainEc, test = b.testEc!
      const med = (xs: number[]) => quantile(xs, 0.5)
      return { key: `${r.site}${b.block}`, label: b.block, note: b.outsideN === null ? 'no training' : `${b.outsideN}/${b.testN} outside`,
        tip: [`${r.site} · ${b.block}`, train ? `training EC ${fmt(train[0], 2)}–${fmt(train[1], 2)} µg/m³ (n ${b.trainN})` : 'no training filters', `test EC ${fmt(test[0], 2)}–${fmt(test[1], 2)} µg/m³ (n ${b.testN})`, b.outsideN === null ? '' : `${b.outsideN} test filters outside the training range`],
        marks: [[
          ...(train ? [{ mid: med(r.cohort.filter((p) => (scheme === 'leave_quarter_out' ? quarterIndex(p.date) !== b.q : quarterIndex(p.date) < b.q)).map((p) => p.ec)), lo: train[0], hi: train[1], color: '#9a9a9a', dy: -4, r: 3 }] : []),
          { mid: med(b.preds.map((q) => q.p.ec)), lo: test[0], hi: test[1], color, dy: 4, r: 3 },
        ]] }
    }) }
  })
  return (
    <ChartFrame id="fo-ranges" title="Training and test EC ranges — how far each withheld quarter reaches beyond its training"
      subtitle={`${popLabel(s.population)} population · ${schemeLabel(scheme).toLowerCase()} · FTIR-predicted EC range of each quarter's training (grey) and test (site colour) filters, dot at the median`}
      provenance={<>{NB_STAB} figure 07</>} source={`${SOURCE}/Predictions.tsx › ConcentrationRanges`}
      tip={<>"Outside" counts test filters below the lowest or above the highest training EC; they are kept. Different ranges can change how hard prediction is, and a zero count does not make the distributions comparable or the EC precise.</>}
      controls={<Segmented label="Split" value={scheme} options={SCHEMES} onChange={setScheme} optionLabel={schemeLabel} />}>
      <Key items={[{ label: 'Training EC range', color: '#9a9a9a', kind: 'line' }, { label: 'Test EC range', color: INK.text, kind: 'line' }]} />
      <IntervalStrip blocks={blocks} labelWidth={84} noteWidth={92} rowHeight={22} panels={[{ label: 'FTIR-predicted EC (µg/m³)' }]} />
    </ChartFrame>
  )
}

/** stability 08: held-out OLS errors grouped by the CalibrationSetId reported on each filter's EC_ftir rows */
export function MetadataErrors({ runs, meta, s }: { runs: SiteRun[]; meta: MetaFile; s: Settings }) {
  const [scheme, setScheme] = useSearchParam<Scheme>('fo.meta.split', 'leave_quarter_out', SCHEMES)
  const blocks: IntervalBlock[] = runs.map((r) => {
    const color = siteColor(meta, r.site)
    const preds = blocksOf(r, scheme).flatMap((b) => b.preds).filter((q) => Number.isFinite(q.err.ols))
    const ids = [...new Set(preds.map((q) => q.p.cal))].sort()
    return { key: r.site, title: r.site, color, rows: ids.map((id) => {
      const e = preds.filter((q) => q.p.cal === id).map((q) => q.err.ols * s.k)
      const q1 = quantile(e, 0.25), q3 = quantile(e, 0.75), iqr = q3 - q1
      // box-plot whiskers: the furthest errors within 1.5 × IQR of the box
      const wlo = Math.min(...e.filter((v) => v >= q1 - 1.5 * iqr)), whi = Math.max(...e.filter((v) => v <= q3 + 1.5 * iqr))
      const dates = preds.filter((q) => q.p.cal === id).map((q) => q.p.date).sort()
      return { key: `${r.site}${id}`, label: `ID ${id}`, note: `n=${e.length}`,
        tip: [`${r.site} · CalibrationSetId ${id}`, `${e.length} filters, ${dates[0]} to ${dates[dates.length - 1]}`, `median error ${signed(quantile(e, 0.5), 2)} ${s.u}`, `IQR ${fmt(q1, 2)} to ${fmt(q3, 2)}`],
        marks: [[{ mid: quantile(e, 0.5), lo: wlo, hi: whi, color, faint: true, r: 0 }, { mid: quantile(e, 0.5), lo: q1, hi: q3, color, shape: 'square' as const, r: 4 }]] }
    }) }
  })
  return (
    <ChartFrame id="fo-metadata" title="Errors by reported calibration-set identifier — the identifiers track different filter groups"
      subtitle={`${popLabel(s.population)} population · ${schemeLabel(scheme).toLowerCase()} · held-out OLS prediction − reported HIPS; square at the median, bar the interquartile range, thin line the 1.5 × IQR whiskers`}
      provenance={<>{NB_STAB} figure 08</>} source={`${SOURCE}/Predictions.tsx › MetadataErrors`}
      tip={<>CalibrationSetId values come from the filters' own EC_ftir source rows. At Addis, ID 17 covers 34 early filters through 22 March 2023 and ID 11 the 156 later ones from 29 March, so date and identifier cannot be separated. The IDs are not documented FTIR model versions, LotId is not a confirmed batch, and ChemSpec method codes 217/218 are not used in their place.</>}
      controls={<Segmented label="Split" value={scheme} options={SCHEMES} onChange={setScheme} optionLabel={schemeLabel} />}>
      <IntervalStrip blocks={blocks} labelWidth={84} noteWidth={56} rowHeight={24} panels={[{ label: `Prediction − HIPS (${s.u})`, reference: 0 }]} />
      <Note>The notebook draws these as box plots; here the box is the bar and the whiskers the thin line, with the same 1.5 × IQR rule.</Note>
    </ChartFrame>
  )
}
