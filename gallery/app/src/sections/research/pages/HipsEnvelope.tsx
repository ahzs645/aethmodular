import { useMemo } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { Crossplot } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { SweepPlot } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'

import type { SectionPageProps } from '../../types'
import { opticsJoin } from '../common/envelope'
import { envelopeOf, useOptics, windowOf, OPTICS_COMMAND, OPTICS_SITES, type Envelope, type Optics } from '../common/optics'
import { ols, originFit, quantile, spearman, theilSen } from '@/lib/stats'
import { BASIS_NOTE, Bars, DataTable, Grid, SitePick, SitesNote, line, ofSite, point, seasonsOf, shortSeason, sitesOf, useConcRows, useSite } from '../common/ui'
import { linePalette } from './hips/lines'
import { ResidualCrossplots, ResidualSsa } from './hips/ResidualViews'
import { RowLines } from './hips/RowLines'
import { TauByLine, TauPm } from './hips/TauViews'
import { useMarkedPairs } from './hips/marked'
import { ExcludedNote, useMarkedRows } from '@/lib/markedRows'
import { namedColor } from '@/lib/siteColours'

const ENVELOPE_NB = 'research/ftir_hips_chem/addis_06_hips_ptfe_operating_envelope.ipynb'
const RT_NB = 'research/improve_hips_offset/improve_fed_rt_proxy_figure2.ipynb (SPARTAN R1/T1 analog)'
const FTIR37_NB = 'research/ftir_ec_phase3/ftir_37_hips_loading_and_curvature.ipynb (blank domain)'

const SCREEN: { key: keyof Envelope | 'tau' | 't' | 'r'; label: string }[] = [
  { key: 'ec', label: 'FTIR EC' }, { key: 'hipsBc', label: 'HIPS BC' }, { key: 'tau', label: 'HIPS tau' }, { key: 't', label: 'HIPS t' }, { key: 'r', label: 'HIPS r' },
  { key: 'ecLoad', label: 'EC loading (µg/cm²)' }, { key: 'ocLoad', label: 'OC loading (µg/cm²)' }, { key: 'omLoad', label: 'OM loading (µg/cm²)' },
  { key: 'pmLoad', label: 'PM2.5 loading (µg/cm²)' }, { key: 'ocEc', label: 'OC / EC' }, { key: 'omEc', label: 'OM / EC' }, { key: 'pmEc', label: 'PM2.5 / EC' },
  { key: 'ironLoad', label: 'Iron loading (µg/cm²)' }, { key: 'ironEc', label: 'Iron / EC' },
]
const val = (e: Envelope, k: (typeof SCREEN)[number]['key']): number | null => (k === 'tau' || k === 't' || k === 'r' ? e.o[k] : (e[k as keyof Envelope] as number | null))

/** quantile marks for a strip row: median dot, IQR whisker (or p05–p95 when `wide`) */
const qrow = (key: string, label: string, values: (number | null)[], color: string, hollow = false, wide = false): IntervalRow => {
  const v = values.filter((x): x is number => x !== null && Number.isFinite(x))
  return { key, label, note: `n=${v.length}`, labelColor: color, marks: [[{ mid: quantile(v, 0.5), lo: quantile(v, wide ? 0.05 : 0.25), hi: quantile(v, wide ? 0.95 : 0.75), color, shape: hollow ? 'hollow' : 'dot' }]],
    tip: [label, `median ${fmt(quantile(v, 0.5), 3)}`, `p05 ${fmt(quantile(v, 0.05), 3)} · p25 ${fmt(quantile(v, 0.25), 3)} · p75 ${fmt(quantile(v, 0.75), 3)} · p95 ${fmt(quantile(v, 0.95), 3)}`, `n ${v.length}`] }
}

/** ftir_37: where a sample's R1 sits against the R1 range of the field blanks read on the same lot and calibration line */
function blankDomain(samples: Optics[], blanks: Optics[]) {
  const key = (o: Optics) => `${o.lot}|${o.cal_intercept}|${o.cal_slope}`
  const range = new Map<string, { n: number; lo: number; hi: number }>()
  for (const b of blanks) if (b.R1 !== null) {
    const k = key(b), cur = range.get(k)
    range.set(k, cur ? { n: cur.n + 1, lo: Math.min(cur.lo, b.R1), hi: Math.max(cur.hi, b.R1) } : { n: 1, lo: b.R1, hi: b.R1 })
  }
  const rows = samples.filter((s) => s.fabs !== null && s.R1 !== null).map((s) => {
    const d = range.get(key(s))
    return { s, covered: !!d, below: !!d && s.R1! < d.lo, outside: !!d && (s.R1! < d.lo || s.R1! > d.hi), pos: d && d.hi > d.lo ? (s.R1! - d.lo) / (d.hi - d.lo) : null }
  })
  const fabs = rows.map((r) => r.s.fabs!)
  const cuts = [0.2, 0.4, 0.6, 0.8].map((q) => quantile(fabs, q))
  const bins = Array.from({ length: 5 }, (_, i) => rows.filter((r) => (i === 0 || r.s.fabs! > cuts[i - 1]) && (i === 4 || r.s.fabs! <= cuts[i])))
  return { rows, keys: range.size, bins: bins.map((b, i) => ({ q: i + 1, n: b.length, fabs: quantile(b.map((r) => r.s.fabs!), 0.5), min: Math.min(...b.map((r) => r.s.fabs!)), max: Math.max(...b.map((r) => r.s.fabs!)),
    below: b.length ? (100 * b.filter((r) => r.below).length) / b.length : NaN, outside: b.length ? (100 * b.filter((r) => r.outside).length) / b.length : NaN,
    pos: quantile(b.map((r) => r.pos).filter((v): v is number => v !== null), 0.5), covered: b.filter((r) => r.covered).length })) }
}

/**
 * Is Addis pushing HIPS on PTFE outside its operating envelope? Raw optical coordinates, mass per area, tau against
 * PM2.5, three fit modes, the residual by loading bin, the residual screen against loading and composition with its
 * crossplots and the AERONET SSA panel, the lot screen and tau by calibration line, the SPARTAN R1/T1 "Figure 2"
 * analog with its blank line, each filter against its own line, where loaded samples sit against the blank R1
 * domain, and the Addis windows the IMPROVE comparisons screen with. Every chart follows Samples through the rows. The
 * export holds only the four focus sites, so a filter of another site is read from its gallery row where its row can say
 * (common/envelope opticsJoin: loadings, fits, the residual charts, tau against PM2.5). What needs a raw reading or a
 * deployed line (t, r, R1, T1, the lot and calibration screens, tau by line, each filter's own line, the blank domain)
 * stays on the focus sites and names the others in a note; field blanks are undated, so they follow the site only.
 */
export function HipsEnvelope({ rows: viewRows, meta }: SectionPageProps) {
  // envelopeOf works out loadings from concentrations itself, so it takes µg/m³ rows whatever the basis
  const rows = useConcRows(viewRows, meta)
  const { mac } = useMac()
  const { rows: optics, error } = useOptics()
  const [site, setSite, sites] = useSite(rows, 'he')
  const [resid, setResid] = useSearchParam<'relative' | 'absolute'>('he.resid', 'relative', ['relative', 'absolute'])
  const byId = useMemo(() => new Map((optics ?? []).filter((o) => o.type !== 'FB').map((o) => [o.id, o])), [optics])
  // the export's row where it has the filter, at any other site a stand-in from the gallery row (common/envelope)
  const opticsOf = useMemo(() => opticsJoin(optics ?? []), [optics])
  const heldCodes = useMemo(() => new Set((optics ?? []).map((o) => o.code)), [optics])
  const codes = useMemo(() => new Set(rows.map((r) => r.code)), [rows])
  const blanks = useMemo(() => (optics ?? []).filter((o) => o.type === 'FB' && codes.has(o.code)), [optics, codes])
  const env = useMemo(() => rows.flatMap((r) => { const o = opticsOf(r); const e = o ? envelopeOf(r, o, mac) : null; return e ? [e] : [] }), [rows, opticsOf, mac])
  const siteNames = useMemo(() => sitesOf(rows), [rows])
  // the sites in view by what they can draw: those the export holds, the rest (read from their rows), and those with a filter that has HIPS Fabs and FTIR EC
  const heldSites = useMemo(() => siteNames.filter((s) => rows.some((r) => r.site === s && heldCodes.has(r.code))), [siteNames, rows, heldCodes])
  const rowSites = siteNames.filter((s) => !heldSites.includes(s))
  const envSites = siteNames.filter((s) => env.some((e) => e.row.site === s))
  const noEnv = siteNames.filter((s) => !envSites.includes(s))
  const colorOfSite = (s: string) => rows.find((r) => r.site === s)?.color ?? INK.text
  const nameOfCode = (c: string) => rows.find((r) => r.code === c)?.site ?? c
  const siteEnv = env.filter((e) => e.row.site === site)
  // the export's own filters in view: what a raw reading or a deployed line needs
  const pairs = useMemo(() => rows.flatMap((r) => { const o = byId.get(r.id); return o ? [{ r, o }] : [] }), [rows, byId])
  // the same with the stand-ins, for what tau alone can draw
  const tauPairs = useMemo(() => rows.flatMap((r) => { const o = opticsOf(r); return o ? [{ r, o }] : [] }), [rows, opticsOf])
  const palette = useMemo(() => linePalette(optics ?? []), [optics])
  const seasons = useMemo(() => seasonsOf(ofSite(rows, site), meta), [rows, site, meta])
  // Filters → excluded: Mark (lib/markedRows): the excluded filters, drawn as × on the crossplots below, in no fit or median
  const markedRows = useConcRows(useMarkedRows(), meta)
  const markedPairs = useMarkedPairs()

  if (error || !optics) return (
    <ChartFrame id="he-missing" title="HIPS operating envelope" provenance={ENVELOPE_NB}>
      <LoadState what="the HIPS optics" error={error} failed={<>The raw HIPS optics export is missing ({error}). Run <code>{OPTICS_COMMAND}</code> from the repo root; it reads research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl.</>} />
    </ChartFrame>
  )

  // 1. optical regime
  const opt = rows.flatMap((r) => { const o = byId.get(r.id); return o && o.t !== null && o.r !== null ? [{ r, o }] : [] })
  const blankPts = blanks.filter((b) => b.t !== null && b.r !== null)
  const markedOpt = markedRows.flatMap((r) => { const o = byId.get(r.id); return o && o.t !== null && o.r !== null && codes.has(r.code) ? [point(r, o.t, o.r, r.color, 't', 'r', 'Research · HIPS optics', [`tau ${fmt(o.tau, 3)}`])] : [] })
  const markedEnv = markedRows.flatMap((r) => { const o = r.site === site ? opticsOf(r) : null; const e = o ? envelopeOf(r, o, mac) : null; return e ? [e] : [] })
  const tauRows = heldSites.flatMap((s) => {
    const code = rows.find((r) => r.site === s)?.code
    return [qrow(`${s}-s`, `${s} · samples`, opt.filter((p) => p.r.site === s).map((p) => p.o.tau), colorOfSite(s), false, true),
      qrow(`${s}-b`, `${s} · blanks`, blanks.filter((b) => b.code === code).map((b) => b.tau), colorOfSite(s), true, true)]
  })
  // 2. mass per area
  const loadPanels = [{ k: 'ecLoad', label: 'EC (µg/cm²)' }, { k: 'ocLoad', label: 'OC (µg/cm²)' }, { k: 'pmLoad', label: 'PM2.5 (µg/cm²)' }, { k: 'bcLoad', label: 'HIPS BC (µg/cm²)' }] as const
  const loadRows: IntervalRow[] = envSites.map((s) => {
    const es = env.filter((e) => e.row.site === s)
    return { key: s, label: s, labelColor: colorOfSite(s), note: `n=${es.length}`, marks: loadPanels.map((p) => qrow('', '', es.map((e) => e[p.k]), colorOfSite(s), false, true).marks[0]) }
  })
  // 3. fit modes
  const fits = envSites.map((s) => {
    const es = env.filter((e) => e.row.site === s), xs = es.map((e) => e.ec), ys = es.map((e) => e.hipsBc)
    return { site: s, o: ols(xs, ys), z: originFit(xs, ys), t: theilSen(xs, ys) }
  })
  // 4. residual by EC-loading sextile at the chosen site
  const withLoad = siteEnv.filter((e) => e.ecLoad !== null)
  const sext = [1, 2, 3, 4, 5].map((k) => quantile(withLoad.map((e) => e.ecLoad!), k / 6))
  const bins = Array.from({ length: 6 }, (_, i) => withLoad.filter((e) => (i === 0 || e.ecLoad! > sext[i - 1]) && (i === 5 || e.ecLoad! <= sext[i]))).filter((b) => b.length)
  const residOf = (e: Envelope) => (resid === 'relative' ? e.relResid : e.resid)
  const residLabel = resid === 'relative' ? '(HIPS BC − FTIR EC) / FTIR EC' : 'HIPS BC − FTIR EC (µg/m³)'
  // 5. residual screen
  const screen = SCREEN.map((s) => {
    const ok = siteEnv.flatMap((e) => { const v = val(e, s.key); return v !== null && Number.isFinite(v) ? [{ v, e: residOf(e) }] : [] })
    return { ...s, n: ok.length, rho: ok.length >= 5 ? spearman(ok.map((o) => o.v), ok.map((o) => o.e)) : null }
  }).filter((s) => s.rho !== null).sort((a, b) => b.rho! ** 2 - a.rho! ** 2)
  // 6. lot / calibration screen: all three come from the export, which has no row for another site
  const lotEnv = siteEnv.filter((e) => heldCodes.has(e.row.code))
  const groupBy = (keyOf: (e: Envelope) => string | null) => {
    const m = new Map<string, Envelope[]>()
    for (const e of lotEnv) { const k = keyOf(e) ?? '(none)'; m.set(k, [...(m.get(k) ?? []), e]) }
    return [...m.entries()].map(([k, es]) => ({ k, n: es.length, from: es.map((e) => e.row.date).sort()[0], to: es.map((e) => e.row.date).sort().reverse()[0],
      tau: quantile(es.map((e) => e.o.tau ?? NaN), 0.5), bc: quantile(es.map((e) => e.hipsBc), 0.5), ec: quantile(es.map((e) => e.ec), 0.5),
      rel: quantile(es.map((e) => e.relResid), 0.5), iqr: quantile(es.map((e) => e.relResid), 0.75) - quantile(es.map((e) => e.relResid), 0.25) })).sort((a, b) => b.n - a.n)
  }
  const lots = [
    { key: 'Lot', rows: groupBy((e) => e.o.lot) },
    { key: 'Calibration line (intercept / slope)', rows: groupBy((e) => (e.o.cal_intercept !== null ? `${e.o.cal_intercept} / ${e.o.cal_slope}` : null)) },
    { key: 'HIPS analysis month', rows: groupBy((e) => e.o.analysis_date?.slice(0, 7) ?? null) },
  ]
  // 7–8. R1 / T1 and the blank domain: the filters in view with a raw reading; blanks follow the site only (undated)
  const rawPairs = pairs.filter((p) => p.o.R1 !== null && p.o.T1 !== null)
  const rawMarked = markedPairs.filter((p) => p.o.R1 !== null && p.o.T1 !== null)
  const rtSites = heldSites.filter((s) => rawPairs.some((p) => p.r.site === s) || blanks.some((b) => b.code === rows.find((r) => r.site === s)?.code))
  const domain = blankDomain(pairs.map((p) => p.o), blanks)
  const domainSites = heldSites.filter((s) => domain.rows.some((d) => d.s.code === rows.find((r) => r.site === s)?.code))
  // 9. the Addis windows
  const addis = env.filter((e) => e.row.site === 'Addis Ababa')
  const windows = ([['HIPS Fabs (Mm⁻¹)', addis.map((e) => e.o.fabs)], ['HIPS tau', addis.map((e) => e.o.tau)], ['HIPS t', addis.map((e) => e.o.t)], ['HIPS r', addis.map((e) => e.o.r)],
    ['EC mass on the filter (µg)', addis.map((e) => e.o.ec_ug)], ['EC loading (µg/cm²)', addis.map((e) => e.ecLoad)], ['OC loading (µg/cm²)', addis.map((e) => e.ocLoad)],
    ['OM loading (µg/cm²)', addis.map((e) => e.omLoad)], ['PM2.5 loading (µg/cm²)', addis.map((e) => e.pmLoad)]] as const).map(([q, v]) => ({ q, ...windowOf([...v]) }))

  const siteLabel = site || 'no site in view'
  const residControls = <><SitePick value={site} options={sites} onChange={setSite} /><Segmented label="Residual" value={resid} options={['relative', 'absolute'] as const} onChange={setResid} /></>

  const lotCols = [
    { key: 'k', label: 'Group', value: (r: ReturnType<typeof groupBy>[number]) => r.k },
    { key: 'n', label: 'n', num: true, value: (r: ReturnType<typeof groupBy>[number]) => r.n },
    { key: 'd', label: 'Samples', value: (r: ReturnType<typeof groupBy>[number]) => `${r.from} – ${r.to}` },
    { key: 't', label: 'Median tau', num: true, value: (r: ReturnType<typeof groupBy>[number]) => fmt(r.tau, 3) },
    { key: 'b', label: 'Median HIPS BC', num: true, value: (r: ReturnType<typeof groupBy>[number]) => fmt(r.bc, 2) },
    { key: 'e', label: 'Median FTIR EC', num: true, value: (r: ReturnType<typeof groupBy>[number]) => fmt(r.ec, 2) },
    { key: 'r', label: 'Median relative residual', num: true, value: (r: ReturnType<typeof groupBy>[number]) => fmt(r.rel, 2) },
    { key: 'i', label: 'IQR', num: true, value: (r: ReturnType<typeof groupBy>[number]) => fmt(r.iqr, 2) },
  ]

  return (
    <>
      <ChartFrame id="he-optics" title="Raw HIPS optical coordinates — samples against field blanks"
        subtitle={`Left: the HIPS transmittance (t) and reflectance (r) proxies of every sample (filled) and field blank (hollow), coloured by site. Right: HIPS optical depth tau, samples and blanks, median with p05–p95. If Addis is a boundary case it should separate here, not only in Fabs / MAC.${meta.basis === 'mass' ? ` ${BASIS_NOTE}` : ''}`}
        provenance={`${ENVELOPE_NB} (§1 optical regime)`}>
        {!opt.length && !blankPts.length ? <Empty>No filter in view has raw HIPS readings: pick {OPTICS_SITES} under Samples.</Empty> : <Grid>
          <Crossplot points={[...opt.map((p) => point(p.r, p.o.t!, p.o.r!, p.r.color, 't', 'r', 'Research · HIPS optics', [`tau ${fmt(p.o.tau, 3)}`])),
            ...blankPts.map((b) => ({ x: b.t!, y: b.r!, color: colorOfSite(nameOfCode(b.code)), hollow: true, tip: [`${b.filter_id} (field blank)`, `lot ${b.lot}`, `t ${fmt(b.t, 3)} · r ${fmt(b.r, 3)} · tau ${fmt(b.tau, 3)}`] }))]}
            marked={markedOpt} sameScale={false} title="HIPS r vs t" xLabel="HIPS transmittance proxy, t" yLabel="HIPS reflectance proxy, r" fit={false} />
          <IntervalStrip blocks={[{ key: 'tau', rows: tauRows }]} panels={[{ label: 'HIPS tau (median, p05–p95)' }]} labelWidth={150} noteWidth={50} minPanelWidth={200} />
        </Grid>}
        {heldSites.length > 0 && <Key items={[...heldSites.map((s) => ({ label: s, color: colorOfSite(s) })), { label: 'field blank', color: INK.muted, kind: 'hollow' }]} />}
        <SitesNote sites={rowSites} why="the raw t, r and field blanks are in the HIPS optics export only, which holds the four focus sites" />
      </ChartFrame>

      <ChartFrame id="he-loading" title="Mass per area — what is on the filter, not in the air"
        subtitle="Surface loading = mass on the filter / deposit area: FTIR's own mass for EC and OC, concentration × sampled volume for PM2.5 (the gallery's date-matched value) and for HIPS BC (Fabs / MAC). Median with p05–p95 by site."
        provenance={`${ENVELOPE_NB} (§2 mass-per-area envelope)`}>
        {loadRows.length ? <IntervalStrip blocks={[{ key: 'load', rows: loadRows }]} panels={loadPanels.map((p) => ({ label: p.label }))} labelWidth={110} noteWidth={50} minPanelWidth={180} />
          : <Empty>No filter in view has HIPS Fabs and FTIR EC together: pick a site with both under Samples.</Empty>}
        <SitesNote sites={rowSites.filter((s) => envSites.includes(s))} why="read from the gallery row, not the HIPS optics export: EC and OC are concentration × sampled volume, on a 3.53 cm² deposit" />
        <SitesNote sites={noEnv} why="no filter in view has HIPS Fabs and FTIR EC together, so it is not drawn here" />
      </ChartFrame>

      <TauPm pairs={tauPairs} blanks={blanks} sites={siteNames} others={rowSites} colorOfSite={colorOfSite} />

      <ChartFrame id="he-fits" title="Slope and intercept separated — OLS, through the origin and Theil–Sen by site"
        subtitle={`HIPS BC (Fabs / ${mac}) on FTIR EC. A different effective MAC mostly changes the slope; a positive intercept that survives the robust fit points to a floor (blank or lot mismatch, optical nonlinearity, a loaded-filter response). Filters with both values positive.`}
        provenance={`${ENVELOPE_NB} (§4 slope vs intercept separation)`} exportable={false}>
        {!fits.length ? <Empty>No filter in view has HIPS Fabs and FTIR EC together: pick a site with both under Samples.</Empty> : <DataTable rows={fits} rowKey={(r) => r.site} columns={[
          { key: 's', label: 'Site', value: (r) => <span style={{ color: colorOfSite(r.site), fontWeight: 600 }}>{r.site}</span> },
          { key: 'n', label: 'n', num: true, value: (r) => r.o?.n ?? '—' },
          { key: 'os', label: 'OLS slope', num: true, value: (r) => fmt(r.o?.slope, 3) },
          { key: 'oi', label: 'OLS intercept', num: true, value: (r) => fmt(r.o?.intercept, 3) },
          { key: 'r2', label: 'OLS R²', num: true, value: (r) => fmt(r.o?.r2, 3) },
          { key: 'z', label: 'Through-origin slope', num: true, value: (r) => fmt(r.z?.slope, 3) },
          { key: 'ts', label: 'Theil–Sen slope', num: true, value: (r) => fmt(r.t?.slope, 3) },
          { key: 'ti', label: 'Theil–Sen intercept', num: true, value: (r) => fmt(r.t?.intercept, 3) },
        ]} />}
        <SitesNote sites={noEnv} why="no filter in view has HIPS Fabs and FTIR EC together, so it has no fit here" />
      </ChartFrame>

      <ChartFrame id="he-bins" title={`Does the residual change with deposited EC? — ${siteLabel}, by EC-loading sextile`}
        subtitle={`Filters in six equal-count bins of EC surface loading; the median ${residLabel} per bin, with its interquartile range. Compression at high loading would bend the residual down as loading rises.`}
        provenance={`${ENVELOPE_NB} (§4 Addis binned median test)`}
        controls={<><SitePick value={site} options={sites} onChange={setSite} /><Segmented label="Residual" value={resid} options={['relative', 'absolute'] as const} onChange={setResid} /></>}>
        {bins.length < 3 ? <Empty>Too few {site} filters in view with an EC loading: pick a site with HIPS Fabs, FTIR EC and a sampled volume.</Empty> : <Grid>
          <SweepPlot series={[{ id: 'b', label: 'median residual', color: colorOfSite(site), interval: 'whisker', points: bins.map((b) => {
            const v = b.map(residOf)
            return { x: quantile(b.map((e) => e.ecLoad!), 0.5), y: quantile(v, 0.5), lo: quantile(v, 0.25), hi: quantile(v, 0.75), tip: [`EC loading ${fmt(Math.min(...b.map((e) => e.ecLoad!)), 2)}–${fmt(Math.max(...b.map((e) => e.ecLoad!)), 2)} µg/cm²`, `median residual ${fmt(quantile(v, 0.5), 3)}`, `median tau ${fmt(quantile(b.map((e) => e.o.tau ?? NaN), 0.5), 3)}`, `n ${b.length}`] }
          }) }]} xLabel="Median EC loading (µg/cm²)" yLabel={resid === 'relative' ? 'Relative residual' : 'Residual (µg/m³)'} reference={0} include={[0]} />
          <Crossplot points={siteEnv.map((e) => point(e.row, e.ec, e.hipsBc, namedColor(seasons, e.row.season), 'FTIR EC', 'HIPS BC', 'Research · HIPS envelope', [`EC loading ${fmt(e.ecLoad, 2)} µg/cm²`]))}
            marked={markedEnv.map((e) => point(e.row, e.ec, e.hipsBc, namedColor(seasons, e.row.season), 'FTIR EC', 'HIPS BC', 'Research · HIPS envelope', [`EC loading ${fmt(e.ecLoad, 2)} µg/cm²`]))}
            title="HIPS BC vs FTIR EC, binned medians" xLabel="FTIR EC (µg/m³)" yLabel="HIPS BC (µg/m³)" fit={false}
            foreground={(x, y) => {
              // over the dots: the bin medians run through the densest part of the cloud, so under them the line disappears
              const at = bins.map((b) => [x(quantile(b.map((e) => e.ec), 0.5)), y(quantile(b.map((e) => e.hipsBc), 0.5))] as const)
              const d = at.map(([px, py], i) => `${i ? 'L' : 'M'}${px},${py}`).join('')
              return <g>
                <path d={d} fill="none" stroke={INK.panel} strokeWidth={5} strokeLinejoin="round" strokeOpacity={0.9} />
                <path d={d} fill="none" stroke={INK.fit} strokeWidth={2.2} strokeLinejoin="round" />
                {at.map(([px, py], i) => <circle key={i} cx={px} cy={py} r={4} fill={INK.fit} stroke={INK.panel} strokeWidth={1.5} />)}
              </g>
            }} />
        </Grid>}
        <Key items={[...seasons.map((s) => ({ label: shortSeason(s.name), color: s.color })), { label: 'binned medians', color: INK.fit, kind: 'line' }]} />
      </ChartFrame>

      <ChartFrame id="he-screen" title={`Residual mechanism screen — ${siteLabel}`}
        subtitle={`Spearman ρ² of the ${resid} HIPS residual with optical loading, deposited mass, composition and iron; blue where the residual rises with the variable, red where it falls. A ranking of what deserves a closer test, not a causal claim.`}
        provenance={`${ENVELOPE_NB} (§3 residual mechanism screen)`}>
        {!screen.length ? <Empty>Too few {site} filters in view with HIPS Fabs and FTIR EC: pick a site with both.</Empty> : <Bars label="Spearman ρ²" max={Math.max(0.1, ...screen.map((s) => s.rho! ** 2))}
          bars={screen.map((s) => ({ key: s.label, label: s.label, value: s.rho! ** 2, signed: s.rho, note: `ρ ${fmt(s.rho, 2)}`, tip: [s.label, `Spearman ρ ${fmt(s.rho, 3)}`, `n ${s.n}`] }))} />}
      </ChartFrame>

      <ResidualCrossplots env={siteEnv} site={site} relative={resid === 'relative'} mac={mac} seasons={seasons} controls={residControls} />

      <ResidualSsa env={siteEnv} site={site} relative={resid === 'relative'} mac={mac} seasons={seasons} controls={residControls} />

      <ChartFrame id="he-lots" title={`Lot and calibration screen — ${siteLabel}`}
        subtitle="The relative residual and tau by filter lot, by the HIPS calibration line a filter was read against, and by HIPS analysis month. A lot or calibration mechanism would show as a jump between groups. A quick screen, not the blank audit of hips_offset_narrative.ipynb."
        provenance={`${ENVELOPE_NB} (§5 lot and calibration screen)`} exportable={false}>
        {!lotEnv.length ? <Empty>No {site} filter in view has a lot, calibration line and analysis month on record: pick {OPTICS_SITES}.</Empty> : lots.map((l) => <div key={l.key} style={{ marginBottom: 12 }}>
          <p className="subtitle" style={{ margin: '0 0 4px' }}>{l.key}</p>
          <DataTable rows={l.rows.slice(0, 12)} rowKey={(r) => r.k} columns={lotCols} />
        </div>)}
        <SitesNote sites={rowSites} why="the lot, calibration line and analysis month are in the HIPS optics export only, so this screen stays on the four focus sites" />
      </ChartFrame>

      <TauByLine pairs={pairs} site={site} others={rowSites} palette={palette} controls={<SitePick value={site} options={sites} onChange={setSite} />} />

      <ChartFrame id="he-rt" title="SPARTAN R1 / T1 — the Warren Figure 2 analog, samples against field blanks"
        subtitle="Raw HIPS reflectance R1 against transmittance T1 per site: samples hollow grey, field blanks in the site colour, with the blanks' OLS line where there are at least three. Closer in spirit to Warren et al. Figure 2 than the IMPROVE FED ratios, but still not its registered detector record."
        provenance={RT_NB}>
        {/* one R1 / T1 range for every site under Panel axes: Shared (Own: each site its own) */}
        {!rtSites.length ? <Empty>No filter in view has a raw HIPS R1 and T1: pick {OPTICS_SITES} under Samples.</Empty> : <FacetGrid items={rtSites} minCell={280} site={(s) => s} render={(s, w) => {
          const code = rows.find((r) => r.site === s)?.code
          const smp = rawPairs.filter((p) => p.r.site === s).map((p) => p.o)
          const bl = blanks.filter((b) => b.code === code && b.R1 !== null && b.T1 !== null)
          const fit = bl.length >= 3 ? ols(bl.map((b) => b.R1!), bl.map((b) => b.T1!)) : null
          return <Crossplot sameScale={false} fit={false} maxSide={w} title={`${s}${fit ? ` · blank line T1 = ${fmt(fit.slope, 2)} R1 ${fit.intercept >= 0 ? '+' : '−'} ${fmt(Math.abs(fit.intercept), 0)}` : ''}`}
            xLabel="HIPS R1" yLabel="HIPS T1" extraLines={line(fit, colorOfSite(s), 'blank OLS', '6 3')}
            points={[...smp.map((o) => ({ x: o.R1!, y: o.T1!, color: INK.muted, hollow: true, tip: [o.filter_id, `lot ${o.lot}`, `R1 ${fmt(o.R1, 0)} · T1 ${fmt(o.T1, 0)}`, `Fabs ${fmt(o.fabs, 1)}`] })),
              ...bl.map((o) => ({ x: o.R1!, y: o.T1!, color: colorOfSite(s), tip: [`${o.filter_id} (field blank)`, `lot ${o.lot}`, `R1 ${fmt(o.R1, 0)} · T1 ${fmt(o.T1, 0)}`] }))]}
            marked={rawMarked.filter((p) => p.r.site === s).map((p) => point(p.r, p.o.R1!, p.o.T1!, INK.muted, 'R1', 'T1', 'Research · HIPS optics', [`lot ${p.o.lot}`]))} />
        }} />}
        <ExcludedNote count={rawMarked.length} drawn />
        <SitesNote sites={rowSites} why="raw R1 and T1 are in the HIPS optics export only, which holds the four focus sites" />
      </ChartFrame>

      <RowLines pairs={pairs} blanks={blanks} sites={heldSites} others={rowSites} palette={palette} colorOfSite={colorOfSite} />

      <ChartFrame id="he-domain" title="Loaded filters outside the blank domain — share of samples below the blanks' R1 range, by Fabs quintile"
        subtitle={`For each sample, the field blanks read on the same lot and calibration line (${domain.keys} such groups among the sites in view) give an R1 range; a sample below it sits where the calibration has no blank. ftir_37 found this share rising from under 1 % to about 32 % in the highest-loading fifth across the shipped SPARTAN batch; here, ${domainSites.length === heldCodes.size ? 'the four focus sites' : domainSites.join(', ') || 'no site in view'}.`}
        provenance={FTIR37_NB}>
        {!domain.rows.length ? <Empty>No filter in view has a raw HIPS R1: pick {OPTICS_SITES} under Samples.</Empty> : <><Grid>
          <SweepPlot series={[{ id: 'below', label: 'below the blank R1 range', color: INK.negative, points: domain.bins.map((b) => ({ x: b.fabs, y: b.below, tip: [`Fabs quintile ${b.q}: ${fmt(b.min, 1)}–${fmt(b.max, 1)} Mm⁻¹`, `${fmt(b.below, 1)} % below, ${fmt(b.outside, 1)} % outside`, `${b.covered} of ${b.n} have blanks on their lot/line`] })) },
            { id: 'outside', label: 'outside it (either side)', color: INK.muted, dash: '5 4', points: domain.bins.map((b) => ({ x: b.fabs, y: b.outside })) }]}
            xLabel="Median HIPS Fabs of the quintile (Mm⁻¹)" yLabel="Samples (%)" include={[0]} />
          <DataTable rows={domain.bins} rowKey={(r) => String(r.q)} columns={[
            { key: 'q', label: 'Fabs quintile', value: (r) => `Q${r.q}` },
            { key: 'r', label: 'Fabs range (Mm⁻¹)', num: true, value: (r) => `${fmt(r.min, 1)}–${fmt(r.max, 1)}` },
            { key: 'n', label: 'n', num: true, value: (r) => r.n },
            { key: 'c', label: 'With blanks', num: true, value: (r) => r.covered },
            { key: 'b', label: 'Below (%)', num: true, value: (r) => fmt(r.below, 1) },
            { key: 'o', label: 'Outside (%)', num: true, value: (r) => fmt(r.outside, 1) },
            { key: 'p', label: 'Median R1 position', num: true, value: (r) => fmt(r.pos, 2) },
          ]} />
        </Grid>
        <Key items={[{ label: 'below the blank R1 range', color: INK.negative, kind: 'line' }, { label: 'outside it, either side', color: INK.muted, kind: 'dash' }]} /></>}
        <ExcludedNote count={rawMarked.length} />
        <SitesNote sites={rowSites} why="raw R1 is in the HIPS optics export only, which holds the four focus sites" />
      </ChartFrame>

      <ChartFrame id="he-windows" title="The Addis windows — what an Addis-like IMPROVE filter would have to match"
        subtitle="Addis samples in view: p05, p25, median, p75 and p95 of the variables the IMPROVE comparisons screen on. Matching IMPROVE by fAbs alone selects on the response and can manufacture a compressed slope; match on loading first."
        provenance={`${ENVELOPE_NB} (§7 IMPROVE handoff) · research/improve_hips_offset (ETAD p05–p95 bounds)`} exportable={false}>
        {!addis.length ? <Empty>No Addis Ababa filter in view has HIPS Fabs and FTIR EC: this chart is Addis's own, so pick Addis Ababa under Samples.</Empty> : <DataTable rows={windows} rowKey={(r) => r.q} columns={[
          { key: 'q', label: 'Variable', value: (r) => r.q }, { key: 'n', label: 'n', num: true, value: (r) => r.n },
          ...(['min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'] as const).map((k) => ({ key: k, label: k, num: true, value: (r: (typeof windows)[number]) => fmt(r[k], 3) })),
        ]} />}
      </ChartFrame>
    </>
  )
}


