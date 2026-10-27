import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { SweepPlot } from '@/components/SweepPlot'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { IMPROVE_GREY, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { DensityPlot, type DensityPoint } from '../common/density'
import { useOptics } from '../common/optics'
import { Bars, DataTable, Grid, SERIES } from '../common/ui'

const COMMAND = 'uv run python gallery/data/export_research_improve_envelope.py'
const NB = {
  first: 'research/improve_hips_offset/improve_first_order_loading_range_analysis.ipynb',
  may: 'research/improve_hips_offset/improve_spartan_may_full_analysis.ipynb',
  warren: 'research/improve_hips_offset/warren_cena_improve_prep_analysis.ipynb',
  anne: 'research/improve_hips_offset/anne_spartan_improve_ec_mass_fabs.ipynb',
  fed: 'research/improve_hips_offset/next_tests_2026_04/00_fed_addis_loading_envelope.ipynb',
  matched: 'research/improve_hips_offset/etad_vs_improve_ec_matched_comparisons.ipynb',
  narrative: 'research/improve_hips_offset/improve_hips_offset_narrative.ipynb',
  smoke: 'research/improve_hips_offset/improve_smoke_event_qc.ipynb',
  rt: 'research/improve_hips_offset/improve_fed_rt_proxy_figure2.ipynb',
  white: 'research/improve_hips_offset/improve_white_style_rt_calibration_space.ipynb',
  mass: 'research/spartan/spartan_vs_improve_mass_comparison.ipynb',
}

interface Q { n: number; p05: number | null; p25: number | null; p50: number | null; p75: number | null; p95: number | null }
interface Fit { n: number; slope: number | null; intercept: number | null; r2: number | null }
interface Density { x_max: number; y_max: number; bins: number; cells: number[][]; n: number; beyond: number }
interface Envelope {
  generated: string
  improve: { n: number; n_post: number; sites: number; date_min: string; date_max: string; area_cm2: number }
  bounds: Record<'fabs' | 'ec_ug' | 'ec_ug_cm2' | 'ec_conc', { n: number; p05: number; p95: number; min: number; max: number }>
  screens: { step: string; kind: string; n: number; sites: number }[]
  density: { mass: Density; surface: Density }
  matched: { group: string; fabs: Q; fabs_ec: Q; ec_conc: Q; ec_ug: Q; fit: Fit }[]
  sweep: ({ threshold: number; sites: number; median_fabs_ec: number | null } & Fit)[]
  intercepts: ({ range: string } & Fit)[]
  smoke: { group: string; n: number; sites: number; smoke_like: number; smoke_share: number | null; median_oc_ec: number | null }[]
  rt: { pairs: Record<string, { ref: string; trans: string; n: number; x: [number, number]; y: [number, number]; bins: number; cells: number[][]; comparable: number[][] }>
    initial_line?: Fit; deficit?: ({ fabs_lo: number; fabs_hi: number | null } & Q)[] } | null
  mass_space: ({ group: string; quantity: string } & Q)[]
}

const qMark = (q: Q, color: string, wide = false) => ({ mid: q.p50, lo: wide ? q.p05 : q.p25, hi: wide ? q.p95 : q.p75, color })

/**
 * Addis against the whole IMPROVE network in loading space: the Addis windows, how few IMPROVE filters survive
 * them, where the network sits in (EC mass, fAbs), EC-matched subsets, the fAbs threshold sweep, what a
 * selection on fAbs does to the line, the smoke screen, the FED 635 nm R/T proxies, and the networks side by
 * side in mass on the filter. Summaries from gallery/data/export_research_improve_envelope.py; the IMPROVE
 * exports are not in the repo.
 */
export function ImproveEnvelope({ rows, meta }: SectionPageProps) {
  const { data, error } = useExport<Envelope>('research/improve_envelope/envelope.json')
  const { rows: optics } = useOptics()
  const [space, setSpace] = useSearchParam<'mass' | 'surface'>('ie.space', 'mass', ['mass', 'surface'])
  const [pair, setPair] = useSearchParam<string>('ie.rt', 'minimum', ['initial', 'final', 'minimum'])
  const byId = useMemo(() => new Map((optics ?? []).filter((o) => o.type !== 'FB').map((o) => [o.id, o])), [optics])
  const sites = useMemo(() => [...new Set(rows.map((r) => r.site))], [rows])
  const colorOf = (s: string) => rows.find((r) => r.site === s)?.color ?? INK.text
  // the focus-site filters in view, in the same coordinates (FTIR's EC mass on the filter, HIPS Fabs)
  const spartan: DensityPoint[] = useMemo(() => rows.flatMap((r) => {
    const o = byId.get(r.id)
    if (!o || o.fabs === null || o.ec_ug === null || r.excluded) return []
    const x = space === 'mass' ? o.ec_ug : o.area_cm2 ? o.ec_ug / o.area_cm2 : null
    return x === null ? [] : [{ x, y: o.fabs, color: r.color, tip: [r.id, `${r.site} · ${r.date}`, `EC ${fmt(o.ec_ug, 1)} µg on the filter (${fmt(o.area_cm2 ? o.ec_ug / o.area_cm2 : null, 2)} µg/cm²)`, `fAbs ${fmt(o.fabs, 1)} Mm⁻¹`] }]
  }), [rows, byId, space])

  if (error || !data) return (
    <ChartFrame id="ie-missing" title="Addis against IMPROVE in loading space" provenance={NB.first}>
      <Empty>{error ? <>The IMPROVE summaries are not here ({error}). They need the IMPROVE FED exports, which the repo does not hold: where they are available (AETHMODULAR_IMPROVE_DIR, or the Davis Data Drive folder), run <code>{COMMAND}</code> from the repo root.</> : 'Loading the IMPROVE summaries…'}</Empty>
    </ChartFrame>
  )
  const b = data.bounds
  const d = data.density[space]
  const xLabel = space === 'mass' ? 'EC on the filter (µg)' : 'EC surface loading (µg/cm²)'
  const box = space === 'mass' ? { x: [b.ec_ug.p05, b.ec_ug.p95] as [number, number], y: [b.fabs.p05, b.fabs.p95] as [number, number], label: 'Addis p05–p95' }
    : { x: [b.ec_ug_cm2.p05, b.ec_ug_cm2.p95] as [number, number], y: [b.fabs.p05, b.fabs.p95] as [number, number], label: 'Addis p05–p95' }
  const rt = data.rt?.pairs[pair]
  const matchedColors = [rows.find((r) => r.code === 'ETAD')?.color ?? '#F39C12', IMPROVE_GREY, '#6a51a3']
  const matchedRows: IntervalRow[] = data.matched.map((m, i) => ({ key: m.group, label: m.group, labelColor: matchedColors[i], note: `n=${m.fabs.n.toLocaleString()}`,
    marks: [[qMark(m.fabs, matchedColors[i])], [qMark(m.fabs_ec, matchedColors[i])], [qMark(m.ec_conc, matchedColors[i])], [qMark(m.ec_ug, matchedColors[i])]],
    tip: [m.group, `fAbs median ${fmt(m.fabs.p50, 1)} (IQR ${fmt(m.fabs.p25, 1)}–${fmt(m.fabs.p75, 1)})`, `fAbs / EC median ${fmt(m.fabs_ec.p50, 2)}`, `n ${m.fabs.n}`] }))
  const quantities = [...new Set(data.mass_space.map((m) => m.quantity))]
  const groups = [...new Set(data.mass_space.map((m) => m.group))]
  const nameOf = (code: string) => rows.find((r) => r.code === code)?.site ?? code
  const massRows: IntervalRow[] = groups.map((g, i) => {
    const color = g === 'IMPROVE' ? IMPROVE_GREY : rows.find((r) => r.code === g)?.color ?? SERIES[i % SERIES.length]
    return { key: g, label: g === 'IMPROVE' ? 'IMPROVE (post-2003)' : nameOf(g), labelColor: color,
      marks: quantities.map((q) => { const m = data.mass_space.find((x) => x.group === g && x.quantity === q); return m ? [qMark(m, color, true)] : [] }) }
  })

  return (
    <>
      <ChartFrame id="ie-screens" title="How many IMPROVE filters look like Addis? — the Addis windows, alone and in sequence"
        subtitle={`Addis bounds (p05–p95 of the focus-site export, exclusions applied): fAbs ${fmt(b.fabs.p05, 1)}–${fmt(b.fabs.p95, 1)} Mm⁻¹, EC on the filter ${fmt(b.ec_ug.p05, 1)}–${fmt(b.ec_ug.p95, 1)} µg, EC surface loading ${fmt(b.ec_ug_cm2.p05, 2)}–${fmt(b.ec_ug_cm2.p95, 2)} µg/cm² (IMPROVE at ${data.improve.area_cm2} cm²). IMPROVE: ${data.improve.n.toLocaleString()} cleaned filters at ${data.improve.sites} sites, ${data.improve.date_min} to ${data.improve.date_max}.`}
        provenance={<>{NB.first} · {NB.may} §4 · {NB.warren}</>} exportable={false}>
        <Bars label="IMPROVE filters (log scale)" format={(v) => (10 ** v >= 1000 ? `${fmt(10 ** v / 1000, 0)}k` : fmt(10 ** v, 0))}
          bars={data.screens.map((s) => ({ key: s.step, label: s.step, value: Math.log10(Math.max(1, s.n)), note: `${s.n.toLocaleString()} · ${s.sites} sites`, tip: [s.step, `${s.n.toLocaleString()} filters at ${s.sites} sites`, s.kind === 'alone' ? 'this bound on its own' : s.kind === 'sequence' ? 'bounds combined' : 'baseline'] }))} />
        <Note>Matching on EC alone finds thousands of IMPROVE filters; adding Addis's fAbs leaves {data.screens[data.screens.length - 1]?.n ?? '—'}. EC-loading-matched IMPROVE filters do not reach Addis's absorption.</Note>
      </ChartFrame>

      <ChartFrame id="ie-density" title="The IMPROVE cloud and the four SPARTAN sites — fAbs against EC on the filter"
        subtitle={`Every post-2003 IMPROVE filter with a volume, binned (${d.n.toLocaleString()} filters; ${d.beyond.toLocaleString()} beyond the axes), with the focus-site filters in view on top and the Addis p05–p95 window dashed. Linear axes from zero, as the loading-envelope notebook pins them.`}
        provenance={<>{NB.anne} · {NB.fed} · {NB.may} §3</>}
        controls={<Segmented label="x" value={space} options={['mass', 'surface'] as const} onChange={setSpace} optionLabel={(o) => (o === 'mass' ? 'EC mass (µg)' : 'EC per area (µg/cm²)')} />}>
        <DensityPlot cells={d.cells} bins={d.bins} x={[0, d.x_max]} y={[0, d.y_max]} xLabel={xLabel} yLabel="fAbs (Mm⁻¹)" points={spartan} box={box} maxSide={520} />
        <Key items={sites.map((s) => ({ label: s, color: colorOf(s) }))} />
      </ChartFrame>

      <ChartFrame id="ie-matched" title="Matched on EC, do IMPROVE filters absorb like Addis? — EC-mass and EC-concentration matches"
        subtitle="IMPROVE filters inside the Addis p05–p95 of EC mass on the filter, or of EC concentration, against Addis itself: median and interquartile range of fAbs, fAbs / EC, EC concentration and EC mass. The same EC can carry much less absorption."
        provenance={NB.matched}>
        <IntervalStrip blocks={[{ key: 'm', rows: matchedRows }]} panels={[{ label: 'fAbs (Mm⁻¹)' }, { label: 'fAbs / EC (m²/g)', reference: meta.mac_value }, { label: 'EC (µg/m³)' }, { label: 'EC mass (µg)' }]}
          labelWidth={220} noteWidth={60} minPanelWidth={170} />
        <DataTable rows={data.matched} rowKey={(r) => r.group} columns={[
          { key: 'g', label: 'Group', value: (r) => r.group }, { key: 'n', label: 'n', num: true, value: (r) => r.fit.n.toLocaleString() },
          { key: 's', label: 'OLS slope, fAbs ~ EC', num: true, value: (r) => fmt(r.fit.slope, 2) }, { key: 'i', label: 'Intercept (Mm⁻¹)', num: true, value: (r) => fmt(r.fit.intercept, 2) },
          { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.fit.r2, 3) }]} />
        <Note>These subsets are selected on EC, so their slopes are descriptions of the window, not calibrations.</Note>
      </ChartFrame>

      <ChartFrame id="ie-sweep" title="Selecting on fAbs manufactures an intercept — OLS of fAbs on EC above each fAbs threshold"
        subtitle="Post-2003 IMPROVE filters with fAbs at or above a threshold, fitted fAbs = slope × EC + intercept. Truncating the response flattens the slope and lifts the intercept by itself, which is why the loading-matched comparison comes first."
        provenance={`${NB.narrative} (§5 fAbs threshold sweep)`}>
        <Grid>
          <SweepPlot series={[{ id: 's', label: 'slope', color: INK.accent, points: data.sweep.map((s) => ({ x: s.threshold, y: s.slope, tip: [`fAbs ≥ ${s.threshold}`, `n ${s.n.toLocaleString()} at ${s.sites} sites`, `slope ${fmt(s.slope, 2)} · R² ${fmt(s.r2, 2)}`] })) }]}
            xLabel="fAbs threshold (Mm⁻¹)" yLabel="OLS slope (m²/g)" include={[0]} title="Slope" />
          <SweepPlot series={[{ id: 'i', label: 'intercept', color: INK.negative, points: data.sweep.map((s) => ({ x: s.threshold, y: s.intercept, tip: [`fAbs ≥ ${s.threshold}`, `intercept ${fmt(s.intercept, 2)} Mm⁻¹`, `median fAbs / EC ${fmt(s.median_fabs_ec, 2)}`] })) }]}
            xLabel="fAbs threshold (Mm⁻¹)" yLabel="Intercept (Mm⁻¹)" reference={0} title="Intercept" />
        </Grid>
      </ChartFrame>

      <ChartFrame id="ie-intercepts" title="What IMPROVE HIPS reads at EC = 0 in the FED pull — OLS intercepts over fit ranges"
        subtitle="fAbs on FED TOR EC. The FED analogue of ftir_26's Part 1 (which uses the Davis HIPS × TOR join; see the IMPROVE origin page); the cleaned FED cache keeps only EC > 0."
        provenance={`${NB.first} · research/ftir_ec_phase3/ftir_26_improve_hips_origin.ipynb (Part 1 analogue)`} exportable={false}>
        <DataTable rows={data.intercepts} rowKey={(r) => r.range} columns={[
          { key: 'r', label: 'Fit range', value: (r) => r.range }, { key: 'n', label: 'n', num: true, value: (r) => r.n.toLocaleString() },
          { key: 's', label: 'Slope', num: true, value: (r) => fmt(r.slope, 2) }, { key: 'i', label: 'Intercept (Mm⁻¹)', num: true, value: (r) => fmt(r.intercept, 3) },
          { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.r2, 3) }]} />
      </ChartFrame>

      <ChartFrame id="ie-smoke" title="Are the Addis-like IMPROVE filters smoke events? — OC/EC ≥ 5"
        subtitle="The share of filters with TOR OC/EC ≥ 5 (improve_smoke_event_qc's OC_EC_SMOKE_MIN) among all post-2003 filters with a loading, the Addis-comparable ones (all three windows) and the fAbs ≥ 70 Mm⁻¹ tail. Addis's own FTIR OC/EC is about 2."
        provenance={NB.smoke} exportable={false}>
        <DataTable rows={data.smoke} rowKey={(r) => r.group} columns={[
          { key: 'g', label: 'Group', value: (r) => r.group }, { key: 'n', label: 'n', num: true, value: (r) => r.n.toLocaleString() }, { key: 's', label: 'Sites', num: true, value: (r) => r.sites },
          { key: 'k', label: 'OC/EC ≥ 5', num: true, value: (r) => r.smoke_like.toLocaleString() }, { key: 'p', label: 'Share', num: true, value: (r) => (r.smoke_share === null ? '—' : `${fmt(100 * r.smoke_share, 0)} %`) },
          { key: 'm', label: 'Median OC/EC', num: true, value: (r) => fmt(r.median_oc_ec, 2) }]} />
      </ChartFrame>

      {data.rt && <ChartFrame id="ie-rt" title="FED 635 nm reflectance and transmittance — the Figure 2 proxy and the White-style calibration space"
        subtitle={`IMPROVE's exported laser R/T ratios (the TOR analyser's, not HIPS's registered detector record), binned with each cell coloured by its median fAbs; the Addis-comparable filters on top. Initial readings stand in for the clean-filter line (FED exports no field blanks): Trans_I = ${fmt(data.rt.initial_line?.slope, 3)} × Ref_I + ${fmt(data.rt.initial_line?.intercept, 1)} (R² ${fmt(data.rt.initial_line?.r2, 2)}). Right: how far the minimum readings fall below that line, by fAbs.`}
        provenance={<>{NB.rt} · {NB.white}</>}
        controls={<Segmented label="Readings" value={pair} options={Object.keys(data.rt.pairs)} onChange={setPair} />}>
        <Grid>
          {rt ? <DensityPlot cells={rt.cells} bins={rt.bins} x={rt.x} y={rt.y} xLabel={`${rt.ref.replace('_Val', '')} (ratio)`} yLabel={`${rt.trans.replace('_Val', '')} (ratio)`} mode="value" valueLabel="median fAbs (Mm⁻¹)" maxSide={420}
            points={rt.comparable.map(([x, y, f]) => ({ x, y, color: INK.warn, hollow: true, tip: ['Addis-comparable IMPROVE filter', `fAbs ${fmt(f, 1)} Mm⁻¹`] }))}>
            {pair === 'initial' && data.rt.initial_line?.slope != null ? ({ x, y }) => <line x1={x(rt.x[0])} x2={x(rt.x[1])} y1={y(data.rt!.initial_line!.intercept! + data.rt!.initial_line!.slope! * rt.x[0])} y2={y(data.rt!.initial_line!.intercept! + data.rt!.initial_line!.slope! * rt.x[1])} stroke={INK.fit} strokeWidth={1.6} strokeDasharray="6 3" /> : undefined}
          </DensityPlot> : <Empty>No {pair} readings in the export.</Empty>}
          {data.rt.deficit?.length ? <SweepPlot series={[{ id: 'd', label: 'deficit', color: INK.accent, interval: 'whisker', points: data.rt.deficit.map((q) => ({ x: `${fmt(q.fabs_lo, 0)}${q.fabs_hi === null ? '+' : `–${fmt(q.fabs_hi, 0)}`}`, y: q.p50, lo: q.p25, hi: q.p75, tip: [`fAbs ${fmt(q.fabs_lo, 0)}${q.fabs_hi === null ? '+' : `–${fmt(q.fabs_hi, 0)}`} Mm⁻¹`, `median ${fmt(q.p50, 1)} (IQR ${fmt(q.p25, 1)}–${fmt(q.p75, 1)})`, `n ${q.n.toLocaleString()}`] })) }]}
            xScale="point" xLabel="fAbs bin (Mm⁻¹)" yLabel="Trans_M − initial line" reference={0} /> : null}
        </Grid>
        <Key items={[{ label: 'Addis-comparable IMPROVE filters', color: INK.warn, kind: 'hollow' }, ...(pair === 'initial' ? [{ label: 'initial-reading line', color: INK.fit, kind: 'dash' as const }] : [])]} />
      </ChartFrame>}

      <ChartFrame id="ie-mass" title="The networks in mass on the filter — IMPROVE against each focus site"
        subtitle={`Median and p05–p95 of EC, OC, iron and PM2.5 mass on the filter (µg) and of fAbs × volume / area, a volume-free optical depth (IMPROVE ≈ 33 m³ a filter, SPARTAN ≈ 7). SPARTAN EC and OC are FTIR's own mass on the filter.`}
        provenance={NB.mass}>
        <IntervalStrip blocks={[{ key: 'mass', rows: massRows }]} panels={quantities.map((q) => ({ label: q }))} labelWidth={150} minPanelWidth={160} />
      </ChartFrame>
    </>
  )
}
