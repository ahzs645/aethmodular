import { useMemo } from 'react'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { IntervalStrip } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { SweepPlot } from '@/components/SweepPlot'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { SectionPageProps } from '../../types'
import { quantile } from '../common/stats'
import { DataTable, Grid, SERIES, num, pText, useConcRows } from '../common/ui'
import { ols } from '../common/stats'
import type { QuantileReference } from '@/lib/referencePopulation'
import type { FilterRow } from '@/lib/types'

const FTIR26 = 'research/ftir_ec_phase3/ftir_26_improve_hips_origin.ipynb'
const FTIR37 = 'research/ftir_ec_phase3/ftir_37_hips_loading_and_curvature.ipynb'
const COMMAND = 'uv run python gallery/data/export_research_improve_origin.py'

type Row = Record<string, number | string | null>
interface Origin {
  generated: string; bridge_n: number; bridge_sites: number
  intercepts: { range: string; n: number; sites: number; slope: number | null; intercept: number | null; ci_lo: number | null; ci_hi: number | null; share_of_addis_c: number | null }[]
  lambda: { lambda: number; slope: number; intercept: number }[]
  zero_ec: { n: number; median: number; mean: number; p25: number; p75: number; negative_share: number }
  scale: { n: number; median_fabs: number; addis_c: number; n_at_or_above_c: number; addis_ols_intercept?: number; n_at_or_above_addis_ols?: number }
  curvature: { ledger: Row[]; cv: Row[]; curves: { model: string; EC_ugm3: number; predicted_Fabs: number }[]; sites: Row[]; tests: Row[] }
  blank_domain: { bins: Row[]; quality: Row[] }
}
const n = (v: unknown) => (typeof v === 'number' ? v : null)
const MODEL_COLOR: Record<string, string> = { linear: INK.fit, quadratic: SERIES[1], spline: SERIES[0] }

/**
 * The IMPROVE side of the Addis intercept: does IMPROVE HIPS read a nonzero Fabs at EC = 0 (ftir_26), and is
 * its Fabs–TOR response curved enough to lift an apparent intercept at high loading (ftir_37)? From
 * gallery/data/export_research_improve_origin.py, which runs the notebooks' own join and analysis code on the
 * Davis data folder.
 */
export function ImproveOrigin({ rows: viewRows, meta }: SectionPageProps) {
  const { data, error } = useExport<Origin>('research/improve_origin/origin.json')
  const rows = useConcRows(viewRows, meta)
  if (error || !data) return (
    <ChartFrame id="io-missing" title="IMPROVE HIPS at EC = 0, and its curvature" provenance={FTIR26}>
      <Empty>{error ? <>Not exported here ({error}). It needs the Davis data folder (FTIR/local_db tables, the SPARTAN HIPS batch); where it is available run <code>{COMMAND}</code>.</> : 'Loading…'}</Empty>
    </ChartFrame>
  )
  const c = data.scale.addis_c
  const iv = data.intercepts
  const curves = Object.entries(data.curvature.curves.reduce<Record<string, { x: number; y: number }[]>>((acc, p) => ({ ...acc, [p.model]: [...(acc[p.model] ?? []), { x: p.EC_ugm3, y: p.predicted_Fabs }] }), {}))
  const sites = data.curvature.sites
  const shift = sites.map((s) => n(s.high_minus_low_intercept_Mm1)).filter((v): v is number => v !== null)
  const quad = sites.map((s) => n(s.quadratic_coefficient_per_IQR2)).filter((v): v is number => v !== null)
  const q = data.blank_domain.quality[0] ?? {}
  return (
    <>
      <ChartFrame id="io-intercepts" title="Does IMPROVE HIPS read a nonzero Fabs at EC = 0? — the intercept over five fit ranges"
        subtitle={`OLS of HIPS Fabs on quartz TOR EC over ${data.bridge_n.toLocaleString()} IMPROVE filters at ${data.bridge_sites} sites (run_ftir_26.py's join), with a 95 % site-cluster bootstrap interval. The dashed line is the Addis intercept C = ${c} Mm⁻¹ the notebook measures against; every IMPROVE intercept is a few per cent of it at most.`}
        provenance={`${FTIR26} (Part 1)`}>
        <IntervalStrip blocks={[{ key: 'i', rows: iv.map((r) => ({ key: r.range, label: r.range, note: `n=${r.n.toLocaleString()}`,
          marks: [[{ mid: r.intercept, lo: r.ci_lo, hi: r.ci_hi, color: INK.accent }]], tip: [r.range, `intercept ${fmt(r.intercept, 3)} Mm⁻¹ [${fmt(r.ci_lo, 3)}, ${fmt(r.ci_hi, 3)}]`, `slope ${fmt(r.slope, 2)} m²/g`, `${fmt(100 * (r.share_of_addis_c ?? NaN), 1)} % of the Addis C`, `${r.sites} sites`] })) }]}
          panels={[{ label: 'Fabs at EC = 0 (Mm⁻¹)', reference: 0, marker: { value: c, label: `Addis C ${c}` } }]} labelWidth={260} noteWidth={80} />
        <DataTable rows={iv} rowKey={(r) => r.range} columns={[
          { key: 'r', label: 'Fit range', value: (r) => r.range }, { key: 'n', label: 'n', num: true, value: (r) => r.n.toLocaleString() }, { key: 's', label: 'Sites', num: true, value: (r) => r.sites },
          { key: 'sl', label: 'Slope (m²/g)', num: true, value: (r) => fmt(r.slope, 2) }, { key: 'i', label: 'Intercept (Mm⁻¹)', num: true, value: (r) => fmt(r.intercept, 3) },
          { key: 'ci', label: '95 % site bootstrap', num: true, value: (r) => (r.ci_lo === null ? '—' : `${fmt(r.ci_lo, 3)} to ${fmt(r.ci_hi, 3)}`) },
          { key: 'c', label: '% of Addis C', num: true, value: (r) => fmt(100 * (r.share_of_addis_c ?? NaN), 1) }]} />
        <Note>No fit at all: on the {data.zero_ec.n.toLocaleString()} filters whose TOR EC is ≤ 0, median HIPS Fabs is {fmt(data.zero_ec.median, 2)} Mm⁻¹ (IQR {fmt(data.zero_ec.p25, 2)}–{fmt(data.zero_ec.p75, 2)}) and {fmt(100 * data.zero_ec.negative_share, 1)} % read negative. Only {data.scale.n_at_or_above_c.toLocaleString()} of {data.scale.n.toLocaleString()} IMPROVE filters carry a total Fabs of {c} Mm⁻¹ or more (median {fmt(data.scale.median_fabs, 2)}){data.scale.addis_ols_intercept != null ? `, and ${data.scale.n_at_or_above_addis_ols?.toLocaleString()} reach the gallery's own Addis OLS intercept of ${fmt(data.scale.addis_ols_intercept, 1)} Mm⁻¹` : ''}: an offset that size could not hide in IMPROVE. The per-site median here uses sites with at least 100 filters, so it differs from the notebook's.</Note>
      </ChartFrame>

      <FabsSurvival origin={data} rows={rows} />

      <ChartFrame id="io-lambda" title="Whatever the error ratio, OLS is the upper bound — the pooled Deming intercept against λ"
        subtitle="Deming regression of Fabs on TOR EC (EC > 0) with λ = Var(Fabs error) / Var(EC error) from 0.1 to 10⁶. The intercept rises monotonically toward the OLS value, so the OLS column above bounds every errors-in-variables weighting."
        provenance={`${FTIR26} (the λ sweep)`}>
        <Grid>
          <SweepPlot series={[{ id: 'i', label: 'Deming intercept', color: INK.accent, points: data.lambda.map((l) => ({ x: l.lambda, y: l.intercept })) }]} xScale="log" xLabel="λ" yLabel="Intercept (Mm⁻¹)" include={[0]} title="Intercept" />
          <SweepPlot series={[{ id: 's', label: 'Deming slope', color: INK.good, points: data.lambda.map((l) => ({ x: l.lambda, y: l.slope })) }]} xScale="log" xLabel="λ" yLabel="Slope (m²/g)" include={[0]} title="Slope" />
        </Grid>
      </ChartFrame>

      <ChartFrame id="io-curvature" title="Is IMPROVE's Fabs–TOR response curved? — held-out-site models and the per-site intercept shift"
        subtitle={`Linear, quadratic and spline models of Fabs on TOR EC (with log OC/EC and an interaction), fitted on four fifths of the sites and scored on the held-out fifth, sites weighted equally. Left: the fitted curves at the median OC/EC. Right: per site (≥ 200 filters), how much the linear intercept rises when only the upper half of the loading range is fitted.`}
        provenance={`${FTIR37} (curvature_analysis)`}>
        <Grid>
          <SweepPlot series={curves.map(([m, pts]) => ({ id: m, label: m, color: MODEL_COLOR[m] ?? INK.muted, marks: false, points: pts }))} xLabel="TOR EC (µg/m³)" yLabel="Predicted Fabs (Mm⁻¹)" include={[0]} title="Fitted curves" />
          <IntervalStrip blocks={[{ key: 's', rows: [
            { key: 'shift', label: 'High − low half intercept (Mm⁻¹)', note: `n=${shift.length}`, marks: [[{ mid: quantile(shift, 0.5), lo: quantile(shift, 0.25), hi: quantile(shift, 0.75), color: INK.accent }]] },
          ] }]} panels={[{ label: 'Per-site shift, median and IQR', reference: 0 }]} labelWidth={200} noteWidth={50} minPanelWidth={180} />
        </Grid>
        <Key items={curves.map(([m]) => ({ label: m, color: MODEL_COLOR[m] ?? INK.muted, kind: 'line' as const }))} />
        <DataTable rows={data.curvature.cv} rowKey={(r) => String(r.model)} columns={[
          { key: 'm', label: 'Model', value: (r) => String(r.model) }, { key: 'n', label: 'n', num: true, value: (r) => (n(r.n) ?? 0).toLocaleString() },
          { key: 'p', label: 'Pooled RMSE (Mm⁻¹)', num: true, value: (r) => fmt(r.pooled_RMSE_Mm1, 4) }, { key: 's', label: 'Site-balanced RMSE', num: true, value: (r) => fmt(r.site_balanced_RMSE_Mm1, 4) },
          { key: 'c', label: 'vs linear', num: true, value: (r) => `${fmt(r.site_balanced_RMSE_change_pct_vs_linear, 1)} %` }, { key: 'r2', label: 'Pooled R²', num: true, value: (r) => fmt(r.pooled_R2, 3) }]} />
        <Note>{data.curvature.tests.map((t) => `${t.test}: median ${fmt(t.median_effect, 3)} over ${t.n_sites} sites, Wilcoxon p ${pText(n(t.wilcoxon_p) ?? NaN)}`).join('. ')}. Within sites the curvature is mostly concave (median quadratic coefficient {fmt(quantile(quad, 0.5), 3)} per IQR²): fitting only heavier filters lifts the apparent intercept. That supports a loading-driven intercept, but on IMPROVE's scale it is well under 1 Mm⁻¹, not Addis's ~{c}.</Note>
      </ChartFrame>

      <ChartFrame id="io-blank" title="SPARTAN samples outside their blanks' R1 range — the whole shipped HIPS batch, by Fabs quintile"
        subtitle={`For every SPARTAN PM2.5 sample in SPARTAN_HIPS_Batch1-51, the field blanks read on the same lot and calibration coefficients give an R1 range. ${n(q.sample_filters)?.toLocaleString() ?? '—'} samples, ${n(q.blank_rows) ?? '—'} blank rows, ${n(q.calibration_sets_with_blanks) ?? '—'} calibration sets with blanks; ${fmt(q.sample_blank_domain_coverage_pct, 1)} % of samples have one. The HIPS operating envelope page repeats this for the focus sites.`}
        provenance={`${FTIR37} (hips_blank_domain)`}>
        <SweepPlot series={[{ id: 'b', label: 'below the blank R1 range', color: INK.negative, points: data.blank_domain.bins.map((b) => ({ x: n(b.Fabs_median) ?? 0, y: n(b.below_blank_pct), tip: [`Fabs ${fmt(b.Fabs_min, 1)}–${fmt(b.Fabs_max, 1)} Mm⁻¹`, `${fmt(b.below_blank_pct, 1)} % below, ${fmt(b.outside_blank_pct, 1)} % outside`, `median R1 position ${fmt(b.r1_position_median, 2)}`, `n ${b.n}`] })) },
          { id: 'o', label: 'outside it', color: INK.muted, dash: '5 4', points: data.blank_domain.bins.map((b) => ({ x: n(b.Fabs_median) ?? 0, y: n(b.outside_blank_pct) })) }]}
          xLabel="Median HIPS Fabs of the quintile (Mm⁻¹)" yLabel="Samples (%)" include={[0]} />
        <Key items={[{ label: 'below the blank R1 range', color: INK.negative, kind: 'line' }, { label: 'outside it, either side', color: INK.muted, kind: 'dash' }]} />
      </ChartFrame>
    </>
  )
}

const ADDIS_COLOR = '#F39C12'

/** the share (%) of a quantile table's values at or above v, read between the two quantiles around it; null above the top step */
function shareAtOrAbove(q: number[], v: number): number | null {
  const steps = q.length - 1
  if (v <= q[0]) return 100
  if (v > q[steps - 1]) return null
  const i = q.findIndex((x) => x >= v) - 1
  const frac = q[i + 1] > q[i] ? (v - q[i]) / (q[i + 1] - q[i]) : 0
  return 100 * (1 - (i + frac) / steps)
}
/** the x axis starts here: below it sits the IMPROVE noise floor (Fabs near zero), not the tail this chart is about */
const X_MIN = 0.05
/** label the decades only on the log axes */
const decade = (v: number) => (Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-6 ? String(+v.toPrecision(1)) : '')
const shareText = (v: number | null) => (v === null ? '< 0.1 %' : `${fmt(v, v < 1 ? 2 : 1)} %`)

/**
 * ftir_26 [29B]: how rare a total Fabs of the Addis intercept's size is in IMPROVE. The notebook's curve is the
 * survival of Fabs over its 160,023 HIPS × TOR bridge filters; the gallery ships the whole IMPROVE network's Fabs as
 * 1,001 quantiles (improve_reference.json, 388,159 filters), so the curve here is that population down to its 0.1 %
 * step, with the bridge's own counts (origin.json) as points. The Addis curve, its median and its OLS intercept are
 * computed from the gallery rows in view.
 */
function FabsSurvival({ origin, rows }: { origin: Origin; rows: FilterRow[] }) {
  const { data: ref, error } = useExport<QuantileReference>('improve_reference.json')
  const addis = useMemo(() => rows.filter((r) => r.code === 'ETAD' && num(r, 'HIPS Fabs') !== null), [rows])
  const fabs = useMemo(() => addis.map((r) => num(r, 'HIPS Fabs')!).sort((a, b) => a - b), [addis])
  const fit = useMemo(() => {
    const pairs = addis.flatMap((r) => { const ec = num(r, 'EC (FTIR)'), f = num(r, 'HIPS Fabs'); return ec !== null && f !== null ? [[ec, f] as const] : [] })
    return ols(pairs.map((p) => p[0]), pairs.map((p) => p[1]))
  }, [addis])
  const q = ref?.quantiles['HIPS Fabs']?.q
  if (!q) return (
    <ChartFrame id="io-survival" title="How rare is the Addis intercept's Fabs in IMPROVE?" provenance={`${FTIR26} (Figure 2B)`}>
      <Empty>{error ? <>The IMPROVE quantiles (improve_reference.json) are not here: {error}.</> : 'Loading the IMPROVE quantiles…'}</Empty>
    </ChartFrame>
  )
  const c = origin.scale.addis_c, sc = origin.scale
  const steps = q.length - 1
  // at or above each quantile; the top entry (the maximum) has nothing above it on a log axis
  const network = q.slice(0, steps).flatMap((x, i) => (x >= X_MIN ? [{ x, y: 100 * (1 - i / steps), tip: [`Fabs ${fmt(x, 2)} Mm⁻¹`, `${fmt(100 * (1 - i / steps), 1)} % of IMPROVE filters at or above`] }] : []))
  const site = fabs.flatMap((x, i) => (x > 0 ? [{ x, y: (100 * (fabs.length - i)) / fabs.length, tip: [`Addis Fabs ${fmt(x, 1)} Mm⁻¹`, `${fmt((100 * (fabs.length - i)) / fabs.length, 1)} % of the Addis filters in view at or above`] }] : []))
  const median = fabs.length ? fabs[Math.floor((fabs.length - 1) / 2)] / 2 + fabs[Math.ceil((fabs.length - 1) / 2)] / 2 : null
  const bridge = [
    { x: sc.median_fabs, y: 50, tip: ['ftir_26 bridge median', `Fabs ${fmt(sc.median_fabs, 2)} Mm⁻¹`] },
    { x: c, y: (100 * sc.n_at_or_above_c) / sc.n, tip: [`ftir_26 bridge: ${sc.n_at_or_above_c} of ${sc.n.toLocaleString()} at or above ${c}`] },
    ...(sc.addis_ols_intercept != null && sc.n_at_or_above_addis_ols != null ? [{ x: sc.addis_ols_intercept, y: (100 * sc.n_at_or_above_addis_ols) / sc.n,
      tip: [`ftir_26 bridge: ${sc.n_at_or_above_addis_ols} at or above ${fmt(sc.addis_ols_intercept, 1)}`] }] : []),
  ]
  const markers = [{ at: c, label: `Addis C ${c}`, color: ADDIS_COLOR },
    ...(median !== null ? [{ at: median, label: `Addis median ${fmt(median, 1)}`, color: ADDIS_COLOR }] : []),
    ...(fit && fit.intercept > 0 ? [{ at: fit.intercept, label: `Addis OLS intercept ${fmt(fit.intercept, 1)}`, color: INK.negative }] : [])]
  return (
    <ChartFrame id="io-survival" title={`${c} Mm⁻¹ of total absorption is rare in IMPROVE — the Fabs survival curve against Addis`}
      subtitle={`Share of filters whose total HIPS Fabs is at or above each value, log–log (x from ${X_MIN} Mm⁻¹; ${fmt(100 * (1 - (q.findIndex((x) => x >= X_MIN)) / steps), 0)} % of IMPROVE filters read more). Grey: every IMPROVE filter in the gallery (${ref!.quantiles['HIPS Fabs'].n.toLocaleString()} at ${ref!.n_sites} sites, ${steps + 1} quantiles, so the curve stops at its 0.1 % step; the maximum is ${fmt(q[steps], 1)} Mm⁻¹). The notebook drew the ${sc.n.toLocaleString()} filters of its HIPS × TOR bridge instead: its counts are the black points (median ${fmt(sc.median_fabs, 2)}; ${sc.n_at_or_above_c} at or above C = ${c}, ${fmt((100 * sc.n_at_or_above_c) / sc.n, 3)} %, 1 in ${Math.round(sc.n / sc.n_at_or_above_c).toLocaleString()}). On the whole network ${shareText(shareAtOrAbove(q, c))} reach C (median ${fmt(q[Math.round(steps / 2)], 2)}): the bridge filters are a little cleaner, the tail the same order. Orange: the ${fabs.length} Addis filters in view, median ${fmt(median, 1)} Mm⁻¹ (ftir_26: 47.11 on every ETAD HIPS row)${fit ? `; their OLS of Fabs on FTIR EC has intercept ${fmt(fit.intercept, 1)} Mm⁻¹ (n = ${fit.n}), which ${shareText(shareAtOrAbove(q, fit.intercept))} of IMPROVE filters reach in total` : ''}. Fabs carries no MAC.`}
      provenance={<>{FTIR26} (Figure 2B, cell 29). IMPROVE quantiles: <code>improve_reference.json</code>; bridge counts: <code>research/improve_origin/origin.json</code> scale; Addis: the gallery rows in view.</>}>
      <SweepPlot series={[
        { id: 'improve', label: 'IMPROVE (all filters, quantiles)', color: INK.muted, points: network, marks: false, width: 2 },
        { id: 'addis', label: 'Addis filters in view', color: ADDIS_COLOR, points: site, marks: false, curve: 'step' },
        { id: 'bridge', label: 'ftir_26 bridge counts', color: INK.text, points: bridge, marks: true, width: 0 },
      ]} xScale="log" yScale="log" xDomain={[X_MIN, Math.max(q[steps - 1], fabs[fabs.length - 1] ?? 0) * 1.15]} yDomain={[0.01, 100]} yFormat={decade}
        xLabel="Total HIPS Fabs on the filter (Mm⁻¹, log)" yLabel="% of filters at or above (log)" height={320} xMarkers={markers} />
      <Key items={[{ label: 'IMPROVE, every filter (quantiles)', color: INK.muted, kind: 'line' }, { label: 'Addis filters in view', color: ADDIS_COLOR, kind: 'line' }, { label: 'ftir_26 bridge (160,023 filters): counts', color: INK.text, kind: 'dot' }]} />
    </ChartFrame>
  )
}
