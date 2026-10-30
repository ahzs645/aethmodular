/**
 * Training on earlier ID-11 filters only (proportionality 06 and 07): the later-period folds refitted on every
 * earlier filter or only on those whose EC_ftir rows report CalibrationSetId 11, both scored on the same later
 * ID-11 filters, and which folds both choices can fit. The release ran this at Addis; the other sites carry the
 * same identifiers and can be read the same way.
 */
import { useMemo } from 'react'
import type { ScalePoint } from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { id11Sensitivity, type Id11Row } from '../../common/filterOnly'
import { Grid, SitePick } from '../../common/ui'
import { FoExcludedNote, NB_PROP, SITES, SOURCE, barLayer, popLabel, shortQ, siteColor, type Settings, type SiteRun } from './data'

const GREY = '#8a8a8a'

function useId11(runs: SiteRun[], s: Settings): [string, (v: string) => void, Id11Row[]] {
  const [site, setSite] = useSearchParam<string>('fo.id11.site', SITES[0], SITES)
  const rows = useMemo(() => { const r = runs.find((x) => x.site === site)!; return id11Sensitivity(r.later, r.cohort, s.minTrain) }, [runs, site, s.minTrain])
  return [site, setSite, rows]
}

/** proportionality 06: OLS MAE and signed error on the common later ID-11 filters under both training choices */
export function Id11Errors({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  const [site, setSite, rows] = useId11(runs, s)
  const color = siteColor(meta, site)
  const common = rows.filter((r) => r.common)
  // the pooled comparison over every common filter (filter-weighted), as the notebook's note quotes it
  const pool = (k: 'all' | 'id11', m: 'mae' | 'mse') => common.reduce((a, r) => a + r[k][m].ols * r.testN, 0) / Math.max(1, common.reduce((a, r) => a + r.testN, 0))
  const nCommon = common.reduce((a, r) => a + r.testN, 0)
  const series = (m: 'mae' | 'mse'): SweepSeries[] => [
    { id: 'all', label: 'All earlier training', color: GREY, dash: '6 3', points: common.map((r) => ({ x: r.block, y: r.all[m].ols * s.k, tip: tip(r) })) },
    { id: 'id11', label: 'Earlier ID 11 only', color, width: 2, points: common.map((r) => ({ x: r.block, y: r.id11[m].ols * s.k, tip: tip(r) })) },
  ]
  const tip = (r: Id11Row) => [`${r.block} · ${r.testN} ID-11 test filters`, `all earlier (n ${r.allEarlierN}): MAE ${fmt(r.all.mae.ols * s.k, 2)}, bias ${signed(r.all.mse.ols * s.k, 2)}`, `ID 11 only (n ${r.id11EarlierN}): MAE ${fmt(r.id11.mae.ols * s.k, 2)}, bias ${signed(r.id11.mse.ols * s.k, 2)}`]
  return (
    <ChartFrame id="fo-id11" title="Training on earlier ID-11 filters — the same later ID-11 filters under two training choices"
      subtitle={`${site} · ${popLabel(s.population)} population · later period · OLS with intercept · identifier and date stay confounded`}
      provenance={<>{NB_PROP} figure 06 (Addis)</>} source={`${SOURCE}/Id11.tsx › Id11Errors`}
      tip={<>At the release's settings the same 127 later Addis ID-11 filters are compared under both choices: OLS MAE falls from 4.009 to 3.884 Mm⁻¹ and mean error from +1.621 to +0.619 Mm⁻¹. It is a bounded predictive sensitivity; because ID 11 begins on a date, the change cannot be attributed to a model change. No residual means are subtracted.</>}
      controls={<SitePick value={site} options={SITES} onChange={setSite} />}>
      {common.length ? <>
        <Key items={[{ label: 'All earlier training', color: GREY, kind: 'dash' }, { label: 'Earlier ID 11 only', color, kind: 'line' }]} />
        <Grid min={320}>
          <SweepPlot series={series('mae')} xScale="point" height={240} xFormat={shortQ} yLabel={`MAE (${s.u})`} include={[0]} />
          <SweepPlot series={series('mse')} xScale="point" height={240} xFormat={shortQ} yLabel={`Mean signed error (${s.u})`} reference={0} />
        </Grid>
        <Note>{nCommon} common filters in {common.length} quarters: MAE {fmt(pool('all', 'mae') * s.k, 3)} → {fmt(pool('id11', 'mae') * s.k, 3)} {s.u}, mean error {signed(pool('all', 'mse') * s.k, 3)} → {signed(pool('id11', 'mse') * s.k, 3)}.</Note>
      </> : <Empty>No later quarter at {site} has ID-11 test filters with both training choices fitted.</Empty>}
      <FoExcludedNote s={s} site={site} />
    </ChartFrame>
  )
}

/** proportionality 07: training filters under each choice, and which ID-11 test quarters both can fit */
export function Id11Support({ runs, s, meta }: { runs: SiteRun[]; s: Settings; meta: MetaFile }) {
  const [site, setSite, rows] = useId11(runs, s)
  const color = siteColor(meta, site)
  const first = rows.find((r) => r.common)
  const train: SweepSeries[] = [
    { id: 'all', label: 'All earlier', color: GREY, dash: '6 3', points: rows.map((r) => ({ x: r.block, y: r.allEarlierN, tip: [r.block, `all earlier ${r.allEarlierN}`, `earlier ID 11 ${r.id11EarlierN}`] })) },
    { id: 'id11', label: 'Earlier ID 11', color, width: 2, points: rows.map((r) => ({ x: r.block, y: r.id11EarlierN, tip: [r.block, `all earlier ${r.allEarlierN}`, `earlier ID 11 ${r.id11EarlierN}`] })) },
  ]
  const test: SweepSeries[] = [{ id: 'test', label: 'ID-11 test filters', color, width: 0, r: 0, points: rows.map((r) => ({ x: r.block, y: r.testN, tip: [r.block, `${r.testN} ID-11 test filters`, r.common ? 'compared' : r.status.replace(/_/g, ' ')] })) }]
  return (
    <ChartFrame id="fo-id11-support" title="Which ID-11 folds are supported — restricted training changes which early folds can be compared"
      subtitle={`${site} · ${popLabel(s.population)} population · later period · a quarter is compared only when both training choices meet the same minimum (${s.minTrain})`}
      provenance={<>{NB_PROP} figure 07 (Addis)</>} source={`${SOURCE}/Id11.tsx › Id11Support`}
      tip={<>At the release's settings the first two Addis ID-11 test quarters cannot be compared; common support begins in 2023Q3 and covers five quarters. Compare those same filters, not the earlier 155-filter later-period aggregate.</>}
      controls={<SitePick value={site} options={SITES} onChange={setSite} />}>
      <Key items={[{ label: 'All earlier', color: GREY, kind: 'dash' }, { label: 'Earlier ID 11', color, kind: 'line' }, { label: 'Test quarter compared', color, kind: 'bar' }, { label: 'Not compared', color, kind: 'faded' }]} />
      <Grid min={320}>
        <SweepPlot series={train} xScale="point" height={240} xFormat={shortQ} yLabel="Training filters" include={[0]} yMarkers={[{ at: s.minTrain, label: `minimum ${s.minTrain}`, color: INK.muted }]} />
        <SweepPlot series={test} xScale="point" height={240} xFormat={shortQ} yLabel="ID-11 test filters" include={[0, Math.max(0, ...rows.map((r) => r.testN)) * 1.1]}
          background={barLayer(rows.map((r) => ({ x: r.block, y: r.testN, color, faint: !r.common })))}
          children={({ x, ih }) => <g>{rows.filter((r) => r.testN && !r.common).map((r) => <text key={r.block} x={(x as ScalePoint<string>)(r.block) ?? 0} y={ih - 4} textAnchor="middle" fontSize={9} fill={INK.muted}>NA</text>)}</g>} />
      </Grid>
      <Note>{first ? `Common support begins in ${first.block} and covers ${rows.filter((r) => r.common).length} quarters.` : 'No quarter is supported under both choices.'}</Note>
      <FoExcludedNote s={s} site={site} />
    </ChartFrame>
  )
}
