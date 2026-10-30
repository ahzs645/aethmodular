import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented, Toggle } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'

import type { SectionPageProps } from '../../types'
import {
  CONSTRUCTION_LABEL, exposeSamples, fitBlankLines, lineBreakdown, lineKey, lineLabel, networkLines, polyval, poolLots, recomputeCheck, siteExposure,
  type BlankLine, type BlankPoint, type Construction, type NetworkBlankLine, type SamplePoint,
} from '../common/blankLines'
import { useOptics, OPTICS_COMMAND } from '../common/optics'
import { DataTable, SitePick, useSite } from '../common/ui'
import { curvePath, linePalette, refOf } from './hips/lines'
import { useCalibRuns } from './spectralCommon'

const NB47 = 'research/ftir_ec_phase3/ftir_47_blank_lines_per_deployed_line.ipynb'
const NB53 = 'research/ftir_ec_phase3/ftir_53_committee_backup_figures.ipynb'

/** ftir_47 §2's displayed site table for the four focus sites (whole SPARTAN batch, FB + LB blanks of every site) */
const FTIR47_SITES: Record<string, { n: number; below: number; above: number; lineQuad: number; lineQuadP90: number; lotQuad: number }> = {
  ETAD: { n: 239, below: 0.439, above: 0.0, lineQuad: 0.206, lineQuadP90: 0.494, lotQuad: 0.535 },
  INDH: { n: 152, below: 0.178, above: 0.007, lineQuad: 0.606, lineQuadP90: 1.4, lotQuad: 0.718 },
  USPA: { n: 158, below: 0.0, above: 0.076, lineQuad: 0.241, lineQuadP90: 0.555, lotQuad: 0.937 },
  CHTS: { n: 192, below: 0.01, above: 0.036, lineQuad: 0.313, lineQuadP90: 0.727, lotQuad: 0.686 },
}

type BlankSet = 'network' | 'focus'
const SET_LABEL: Record<BlankSet, string> = { network: 'network fits', focus: 'focus-site blanks' }
const pct = (v: number) => fmt(100 * v, 1)

/**
 * ftir_47 (and what it means for ftir_53 01): HIPS blank lines belong to deployed calibration lines, keyed on
 * (lot, Intercept, Slope), never on lot. Per deployed line: its field blanks' T1 against R1 with the shipped line,
 * the per-line linear and quadratic refits and the lot-pooled curve the notebook retires; the line ledger; which
 * samples sit outside their own line's blank R1 range and how far each construction moves their Fabs; and, per
 * line, the Fabs each construction implies. Fabs is in Mm⁻¹ and does not depend on the MAC.
 */
export function BlankLines({ rows }: SectionPageProps) {
  const { rows: optics, error } = useOptics()
  const { runs } = useCalibRuns()
  const [set, setSet] = useSearchParam<BlankSet>('bl.set', 'network', ['network', 'focus'])
  const [withSamples, setWithSamples] = useSearchParam<'0' | '1'>('bl.samples', '0', ['0', '1'])
  const [cons, setCons] = useSearchParam<Exclude<Construction, 'deployed'>>('bl.cons', 'line_quad', ['line_lin', 'line_quad', 'lot_quad'])
  const [site, setSite, sites] = useSite(rows, 'bl')

  const ids = useMemo(() => new Set(rows.map((r) => r.id)), [rows])
  const siteOfCode = useMemo(() => new Map(rows.map((r) => [r.code, { site: r.site, color: r.color }])), [rows])
  // a line's blanks do not depend on which samples are in view: every focus-site field blank on a deployed line
  const blanks: BlankPoint[] = useMemo(() => (optics ?? []).flatMap((o) => {
    const ref = refOf(o)
    return o.type === 'FB' && ref && o.R1 !== null && o.T1 !== null && o.T1 > 0 ? [{ ...ref, R1: o.R1, T1: o.T1, date: o.analysis_date }] : []
  }), [optics])
  // the samples in view with a shipped Fabs, a line and a volume (dv = 100·A/V)
  const samples: SamplePoint[] = useMemo(() => (optics ?? []).flatMap((o) => {
    const ref = refOf(o)
    return o.type !== 'FB' && ids.has(o.id) && ref && o.R1 !== null && o.T1 !== null && o.fabs !== null && o.volume_m3 && o.area_cm2
      ? [{ ...ref, id: o.id, code: o.code, R1: o.R1, T1: o.T1, fabs: o.fabs, dv: (100 * o.area_cm2) / o.volume_m3 }] : []
  }), [optics, ids])

  const network = useMemo(() => networkLines(runs?.hips?.blanks as Record<string, NetworkBlankLine> | null | undefined), [runs])
  const focus = useMemo(() => fitBlankLines(blanks), [blanks])
  const useNet = set === 'network' && network.length > 0
  const lines = useNet ? network : focus
  const lots = useMemo(() => poolLots(blanks, focus), [blanks, focus])
  const exps = useMemo(() => exposeSamples(samples, lines, lots), [samples, lines, lots])
  const palette = useMemo(() => linePalette(optics ?? [], network), [optics, network])

  if (error || !optics) return (
    <ChartFrame id="bl-missing" title="Blank lines per deployed line" provenance={NB47}>
      <Empty>{error ? <>The raw HIPS optics export is missing ({error}). Run <code>{OPTICS_COMMAND}</code> from the repo root.</> : 'Loading the HIPS optics…'}</Empty>
    </ChartFrame>
  )

  const byKey = new Map(lines.map((l) => [l.key, l])), focusByKey = new Map(focus.map((l) => [l.key, l])), netByKey = new Map(network.map((l) => [l.key, l]))
  const setControl = <Segmented label="Blank lines" value={set} options={['network', 'focus'] as const} onChange={setSet} optionLabel={(o) => SET_LABEL[o]} />
  const setNote = useNet
    ? `Per-line refits are the explorer's (calibration_runs.json hips.blanks: hips_lab.blank_lines over every SPARTAN FB and LB blank, ftir_47's LINE_FIT); the lot-pooled curves can only be refitted from the ${blanks.length} focus-site field blanks exported here.`
    : `Every refit uses the ${blanks.length} field blanks of the four focus sites (hips_optics/filters.json), not ftir_47's network FB + LB set: lines with fewer than five such blanks get no refit.`

  // 1. the blanks per lot
  const lotNames = [...new Set(blanks.map((b) => b.lot))].sort()
  const facets = lotNames.map((lot) => {
    const bl = blanks.filter((b) => b.lot === lot), keys = [...new Set(bl.map(lineKey))].sort((a, b) => palette.keys.indexOf(a) - palette.keys.indexOf(b))
    const sm = withSamples === '1' ? samples.filter((s) => s.lot === lot) : []
    const pts: CrossPoint[] = [
      ...sm.map((s) => ({ x: s.R1, y: s.T1, color: palette.color(lineKey(s)), hollow: true, opacity: 0.5, id: s.id, tip: [s.id, `sample · line ${palette.label(lineKey(s))}`, `R1 ${fmt(s.R1, 0)} · T1 ${fmt(s.T1, 0)} · Fabs ${fmt(s.fabs, 1)} Mm⁻¹`] })),
      ...bl.map((b) => ({ x: b.R1, y: b.T1, color: palette.color(lineKey(b)), tip: [`field blank · ${b.date ?? 'no analysis date'}`, `line ${palette.label(lineKey(b))}`, `R1 ${fmt(b.R1, 0)} · T1 ${fmt(b.T1, 0)}`] })),
    ]
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
    const fits = keys.map((k) => byKey.get(k)).filter((l): l is BlankLine => !!l)
    const lo = Math.min(...xs, ...fits.map((l) => l.r1Min)), hi = Math.max(...xs, ...fits.map((l) => l.r1Max))
    const pool = lots.find((p) => p.lot === lot)
    return { lot, bl, keys, pts, fits, pool, x: [Math.floor((lo - 8) / 10) * 10, Math.ceil((hi + 8) / 10) * 10] as [number, number],
      y: [Math.floor((Math.min(...ys) - 30) / 50) * 50, Math.ceil((Math.max(...ys) + 30) / 50) * 50] as [number, number] }
  })

  // 2. the ledger, one row per deployed line that has a blank here or in the network
  const ledgerKeys = [...new Set([...focus.map((l) => l.key), ...network.filter((l) => lotNames.includes(l.lot)).map((l) => l.key)])].sort((a, b) => palette.keys.indexOf(a) - palette.keys.indexOf(b))
  const ledger = ledgerKeys.map((k) => ({ k, f: focusByKey.get(k), net: netByKey.get(k), nS: samples.filter((s) => lineKey(s) === k).length }))

  // 3. exposure by site
  const sitesHere = siteExposure(exps).sort((a, b) => b.outside - a.outside)
  const nameOf = (code: string) => siteOfCode.get(code)?.site ?? code
  const MARKS: { c: Exclude<Construction, 'deployed'>; color: string; shape: 'dot' | 'hollow' | 'square' }[] = [
    { c: 'line_lin', color: '#2c7fb8', shape: 'hollow' }, { c: 'line_quad', color: '#2c7fb8', shape: 'dot' }, { c: 'lot_quad', color: INK.negative, shape: 'square' }]
  const expRows: IntervalRow[] = sitesHere.map((s, i) => ({
    key: s.code, label: nameOf(s.code), labelColor: siteOfCode.get(s.code)?.color, note: `n=${s.n}`,
    tip: [nameOf(s.code), `${s.n} samples on a fitted line (${s.nUnfit} on an unfitted one)`, `${pct(s.below)} % below, ${pct(s.above)} % above its own line's blank R1 range`,
      ...MARKS.map((m) => `${CONSTRUCTION_LABEL[m.c]}: median |ΔFabs| ${fmt(s.med[m.c], 2)}, p90 ${fmt(s.p90[m.c], 2)} Mm⁻¹`)],
    marks: [MARKS.map((m, j) => ({ mid: s.med[m.c], color: m.color, shape: m.shape, dy: (j - 1) * 6 })),
      [{ mid: 100 * s.below, color: '#6a51a3', dy: -4 }, { mid: 100 * s.above, color: '#d08c2c', shape: 'square' as const, dy: 4 }]],
    strong: i === 0,
  }))
  const agree = sitesHere.filter((s) => FTIR47_SITES[s.code])

  // 4. the Fabs each line implies at the chosen site
  const siteCode = rows.find((r) => r.site === site)?.code
  const siteExps = exps.filter((e) => e.s.code === siteCode)
  const breakdown = lineBreakdown(siteExps)
  const check = recomputeCheck(exps)
  // ΔFabs against the sample's R1: where it leaves its line's blank R1 range the refit extrapolates
  const fabsPts: CrossPoint[] = siteExps.flatMap((e) => {
    const d = e.dfabs[cons], k = lineKey(e.s)
    return d !== null ? [{ x: e.s.R1, y: d, color: palette.color(k), id: e.s.id,
      tip: [e.s.id, `line ${palette.label(k)}`, `Fabs deployed ${fmt(e.fabs.deployed, 2)} · ${CONSTRUCTION_LABEL[cons]} ${fmt(e.fabs[cons], 2)} Mm⁻¹ (Δ ${fmt(d, 2)})`, `R1 ${fmt(e.s.R1, 0)}${e.line ? ` (blank range ${fmt(e.line.r1Min, 0)}–${fmt(e.line.r1Max, 0)})` : ''}`] }] : []
  })
  const bands = breakdown.filter((b) => b.line)
  const r1Lo = Math.min(...fabsPts.map((p) => p.x), ...bands.map((b) => b.line!.r1Min)), r1Hi = Math.max(...fabsPts.map((p) => p.x), ...bands.map((b) => b.line!.r1Max))
  const dMax = Math.max(0.5, ...fabsPts.map((p) => Math.abs(p.y))) * 1.1

  return (
    <>
      <ChartFrame id="bl-lines" title="Field blanks' T1 against R1, by deployed calibration line"
        subtitle={`One panel per PTFE lot. Field blanks coloured by the deployed line they were reduced with; dashed: the shipped line over its blanks' R1 range; solid: the per-line quadratic refit; dotted black: one quadratic through all the lot's blanks; dash-dot grey: the lot-pooled linear. In a lot with several lines the blanks of successive calibrations sit in different R1 bands, so a pooled curve threads between them and bends where no blank exists. ${setNote}`}
        provenance={`${NB47} [8] (blank_lines_per_lot.png)`}
        controls={<>{setControl}<Toggle label="Samples" checked={withSamples === '1'} onChange={(v) => setWithSamples(v ? '1' : '0')} title="draw the lot's samples in view, hollow" /></>}>
        {!blanks.length ? <Empty>No field blanks with a deployed line in the optics export.</Empty> : <FacetGrid items={facets} minCell={320} render={(f, w) => (
          <div style={{ width: w }}>
            <Crossplot sameScale={false} fit={false} maxSide={w - 72} domain={f.x} yDomain={f.y} xLabel="Blank R1 (counts)" yLabel="Blank T1 (counts)"
              title={`lot ${f.lot}: ${f.keys.length} deployed line${f.keys.length === 1 ? '' : 's'}, ${f.bl.length} field blanks${f.pool ? ` · pooled rms ${fmt(f.pool.rmsLin, 0)} / ${fmt(f.pool.rmsQuad, 0)}` : ''}`}
              background={(x, y) => <g fill="none">
                {f.fits.map((l) => <g key={l.key} stroke={palette.color(l.key)}>
                  <path d={curvePath((r) => l.intercept + l.slope * r, l.r1Min, l.r1Max, x, y)} strokeWidth={1.3} strokeDasharray="6 4" />
                  {l.quad && <path d={curvePath((r) => polyval(l.quad!, r), l.r1Min, l.r1Max, x, y)} strokeWidth={2.2} />}
                </g>)}
                {f.pool?.quad && <path d={curvePath((r) => polyval(f.pool!.quad!, r), f.pool.r1Min, f.pool.r1Max, x, y)} stroke={INK.text} strokeWidth={2} strokeDasharray="2 3" />}
                {f.pool?.lin && <path d={curvePath((r) => polyval(f.pool!.lin!, r), f.pool.r1Min, f.pool.r1Max, x, y)} stroke={INK.muted} strokeWidth={1.2} strokeDasharray="8 3 2 3" />}
              </g>}
              points={f.pts} />
          </div>
        )} />}
        <Key items={[...[...new Set(facets.flatMap((f) => f.keys))].map((k) => ({ label: palette.label(k), color: palette.color(k) })),
          { label: 'shipped line', color: INK.muted, kind: 'dash' as const }, { label: 'per-line quadratic', color: INK.muted, kind: 'line' as const },
          { label: 'lot-pooled quadratic (dotted)', color: INK.text, kind: 'dash' as const }, { label: 'lot-pooled linear (dash-dot)', color: INK.muted, kind: 'dash' as const }]} />
      </ChartFrame>

      <ChartFrame id="bl-ledger" title="The blank ledger — per deployed line, then per lot"
        subtitle="Scatter of the blanks about the shipped line and about the refits (counts rms, numpy's std). Left block: the focus sites' field blanks, refitted here. Right block: the explorer's network fits as calibration_runs.json carries them, which are ftir_47 §1's table to the count. A shipped line within a few counts of its own blanks' OLS is the same object as the per-line linear refit."
        provenance={`${NB47} §1 (blank_ledger_per_line.csv, blank_ledger_per_lot_pooled.csv)`} exportable={false}>
        <DataTable rows={ledger} rowKey={(r) => r.k} columns={[
          { key: 'l', label: 'Deployed line', value: (r) => <span style={{ color: palette.color(r.k), fontWeight: 600 }}>{palette.label(r.k)}</span> },
          { key: 'd', label: 'Blanks read', value: (r) => (r.f?.from ? `${r.f.from} – ${r.f.to}` : '—') },
          { key: 'n', label: 'Focus n', num: true, value: (r) => r.f?.n ?? 0 },
          { key: 'r', label: 'Focus R1', num: true, value: (r) => (r.f ? `${fmt(r.f.r1Min, 0)}–${fmt(r.f.r1Max, 0)}` : '—') },
          { key: 'rd', label: 'rms shipped', num: true, value: (r) => fmt(r.f?.rmsDeployed, 1) },
          { key: 'rl', label: 'rms lin', num: true, value: (r) => fmt(r.f?.rmsLin, 1) },
          { key: 'rq', label: 'rms quad', num: true, value: (r) => fmt(r.f?.rmsQuad, 1) },
          { key: 'c', label: 'Curvature', value: (r) => (r.f?.quad ? (r.f.quad[2] > 0 ? 'convex (+)' : 'concave (−)') : r.f ? 'unfit (< 5)' : '—') },
          { key: 'nn', label: 'Network n', num: true, value: (r) => r.net?.n ?? '—' },
          { key: 'nr', label: 'Network R1', num: true, value: (r) => (r.net ? `${fmt(r.net.r1Min, 0)}–${fmt(r.net.r1Max, 0)}` : '—') },
          { key: 'nl', label: 'Network rms lin / quad', num: true, value: (r) => (r.net ? `${fmt(r.net.rmsLin, 1)} / ${fmt(r.net.rmsQuad, 1)}` : '—') },
          { key: 's', label: 'Samples in view', num: true, value: (r) => r.nS },
        ]} />
        <p className="subtitle" style={{ margin: '12px 0 4px' }}>Every focus-site field blank of a lot pooled into one curve</p>
        <DataTable rows={lots} rowKey={(r) => r.lot} columns={[
          { key: 'l', label: 'Lot', value: (r) => r.lot },
          { key: 'n', label: 'Blanks', num: true, value: (r) => r.n },
          { key: 'k', label: 'Deployed lines', num: true, value: (r) => `${r.nLines} (${r.nFitted} refitted)` },
          { key: 'r', label: 'R1', num: true, value: (r) => `${fmt(r.r1Min, 0)}–${fmt(r.r1Max, 0)}` },
          { key: 'rl', label: 'Pooled rms lin', num: true, value: (r) => fmt(r.rmsLin, 1) },
          { key: 'rq', label: 'Pooled rms quad', num: true, value: (r) => fmt(r.rmsQuad, 1) },
          { key: 'o', label: 'Fitted lines share an R1 interval', value: (r) => (r.shareInterval === null ? '—' : r.shareInterval ? 'yes' : 'no') },
        ]} />
        <Note>ftir_47 on the whole batch: lot 251's 373 blanks pooled scatter 43 / 32 counts (linear / quadratic) against 5–10 per line, and lot 248's 40 / 40 against 4–8; its three fitted lines share no R1 interval. Here only the field blanks of Beijing, Delhi, JPL and Addis are exported (lot 251's 2423.6 line has 4 of its 63, lot 248's 2262.0 line 4 of 16), so those two lines get no focus-site refit and the pooled scatter is smaller.</Note>
      </ChartFrame>

      <ChartFrame id="bl-exposure" title="Which sites the construction moves — |ΔFabs| against the deployed line, and samples outside their own line's blanks"
        subtitle={`Samples in view on a refitted line. Left: median |ΔFabs| from the deployed line under the per-line linear (hollow), per-line quadratic (filled) and lot-pooled quadratic (red square) refits; the tooltip has the p90. Right: the share of samples whose R1 is below (purple: heavier than any blank of their line) or above (orange) their own line's blank R1 range. ${setNote}`}
        provenance={`${NB47} [11] (network_site_exposure.png; network_site_table.csv)`} controls={setControl}>
        {!expRows.length ? <Empty>No samples in view on a refitted line.</Empty> : <IntervalStrip blocks={[{ key: 'sites', rows: expRows }]} labelWidth={110} noteWidth={50} minPanelWidth={240}
          panels={[{ label: 'Median |ΔFabs| vs deployed line (Mm⁻¹)', domain: [0, Math.max(0.5, ...sitesHere.flatMap((s) => MARKS.map((m) => s.med[m.c])).filter(Number.isFinite)) * 1.1] },
            { label: 'Samples outside own line\'s blank R1 range (%)', domain: [0, Math.max(10, ...sitesHere.map((s) => 100 * Math.max(s.below, s.above))) * 1.1] }]} />}
        <Key items={[{ label: 'per-line linear', color: '#2c7fb8', kind: 'hollow' }, { label: 'per-line quadratic', color: '#2c7fb8' }, { label: 'lot-pooled quadratic', color: INK.negative, kind: 'square' },
          { label: 'R1 below blank range', color: '#6a51a3' }, { label: 'R1 above blank range', color: '#d08c2c', kind: 'square' }]} />
        {agree.length > 0 && <>
          <p className="subtitle" style={{ margin: '12px 0 4px' }}>Against ftir_47's displayed site table (whole SPARTAN batch, blank lines fitted on every site's FB and LB blanks)</p>
          <DataTable rows={agree} rowKey={(s) => s.code} columns={[
            { key: 's', label: 'Site', value: (s) => nameOf(s.code) },
            { key: 'n', label: 'n here / ftir_47', num: true, value: (s) => `${s.n} / ${FTIR47_SITES[s.code].n}` },
            { key: 'b', label: 'Below (%)', num: true, value: (s) => `${pct(s.below)} / ${pct(FTIR47_SITES[s.code].below)}` },
            { key: 'a', label: 'Above (%)', num: true, value: (s) => `${pct(s.above)} / ${pct(FTIR47_SITES[s.code].above)}` },
            { key: 'q', label: 'Per-line quad median |ΔFabs|', num: true, value: (s) => `${fmt(s.med.line_quad, 2)} / ${fmt(FTIR47_SITES[s.code].lineQuad, 2)}` },
            { key: 'p', label: 'p90', num: true, value: (s) => `${fmt(s.p90.line_quad, 2)} / ${fmt(FTIR47_SITES[s.code].lineQuadP90, 2)}` },
            { key: 'l', label: 'Lot-pooled median |ΔFabs|', num: true, title: 'here pooled over the focus-site field blanks only; ftir_47 pooled every blank of the lot', value: (s) => `${fmt(s.med.lot_quad, 2)} / ${fmt(FTIR47_SITES[s.code].lotQuad, 2)}` },
          ]} />
          <Note>With the network fits and every filter in view, the per-line columns are ftir_47's computation on the gallery's subset of its samples: the shares outside the blank range agree to about a percentage point and the median |ΔFabs| to a few hundredths of a Mm⁻¹ at Addis, Beijing and JPL. Delhi differs most because the export has a sampled volume for fewer Delhi filters than the batch. The lot-pooled column cannot match: ftir_47 pooled each lot's blanks across the whole batch, and only the focus sites' field blanks are exported.</Note>
        </>}
      </ChartFrame>

      <ChartFrame id="bl-fabs" title={`The Fabs each line implies — ${site}`}
        subtitle={`Each sample's Fabs recomputed as 100·ln(top / T1)·A / V with top the chosen curve at its R1 (its own line's refit, or its lot's pooled quadratic), minus the shipped deployed-line Fabs, against R1 and coloured by line; the bars along the top are each line's blank R1 range, so a sample left of its bar is heavier than any blank its line was fitted on. The table gives the median Fabs per line under every construction (ftir_47's Pasadena breakdown, for any site). Fabs in Mm⁻¹, before any MAC. ${setNote}`}
        provenance={`${NB47} [16] (pasadena_per_line_breakdown.csv) and §2`}
        controls={<><SitePick value={site} options={sites} onChange={setSite} /><Segmented label="Recomputed with" value={cons} options={['line_lin', 'line_quad', 'lot_quad'] as const} onChange={setCons} optionLabel={(o) => CONSTRUCTION_LABEL[o]} />{setControl}</>}>
        {!siteExps.length ? <Empty>No {site} samples in view with a deployed line, a shipped Fabs and a sampled volume.</Empty> : <>
          {!fabsPts.length ? <Empty>No {site} sample is on a line with a {CONSTRUCTION_LABEL[cons]} refit under the {SET_LABEL[set]}.</Empty> :
            <Crossplot sameScale={false} fit={false} maxSide={520} domain={[Math.floor((r1Lo - 5) / 10) * 10, Math.ceil((r1Hi + 5) / 10) * 10]} yDomain={[-dMax, dMax]}
              title={`${fabsPts.length} of ${siteExps.length} samples recomputed`} xLabel="Sample R1 (counts)" yLabel={`ΔFabs, ${CONSTRUCTION_LABEL[cons]} − deployed (Mm⁻¹)`}
              background={(x, y) => <g>
                <line x1={x.range()[0]} x2={x.range()[1]} y1={y(0)} y2={y(0)} stroke={INK.axis} />
                {bands.map((b, i) => <rect key={b.key} x={x(b.line!.r1Min)} width={Math.max(1, x(b.line!.r1Max) - x(b.line!.r1Min))} y={4 + 7 * i} height={5} fill={palette.color(b.key)} opacity={0.75} />)}
              </g>}
              points={fabsPts} />}
          <DataTable rows={breakdown} rowKey={(b) => b.key} columns={[
            { key: 'l', label: 'Deployed line', value: (b) => <span style={{ color: palette.color(b.key), fontWeight: 600 }}>{lineLabel(b.ref)}</span> },
            { key: 'n', label: 'n', num: true, value: (b) => b.n },
            { key: 'r', label: 'Sample R1', num: true, value: (b) => `${fmt(b.r1Min, 0)}–${fmt(b.r1Max, 0)}` },
            { key: 'br', label: 'Blank R1', num: true, value: (b) => (b.line ? `${fmt(b.line.r1Min, 0)}–${fmt(b.line.r1Max, 0)}` : 'not refitted') },
            { key: 'd', label: 'Median Fabs, deployed', num: true, value: (b) => fmt(b.median.deployed, 2) },
            { key: 'li', label: 'per-line lin', num: true, value: (b) => fmt(b.median.line_lin, 2) },
            { key: 'q', label: 'per-line quad', num: true, value: (b) => fmt(b.median.line_quad, 2) },
            { key: 'p', label: 'lot-pooled quad', num: true, value: (b) => fmt(b.median.lot_quad, 2) },
            { key: 'x', label: 'Pooled / deployed', num: true, value: (b) => fmt(b.median.lot_quad / b.median.deployed, 2) },
          ]} />
        </>}
        <Key items={breakdown.map((b) => ({ label: lineLabel(b.ref), color: palette.color(b.key) }))} />
        <Note>The deployed line reproduces the shipped Fabs to {fmt(check.maxDiff, 3)} Mm⁻¹ at most over {check.n} samples in view (ftir_47: 0.0000 over 3,047; the export rounds the deposit area). For JPL on the network fits, ftir_47's own table has the 2423.6 line's 56 filters at a median 4.84 Mm⁻¹ deployed, 5.23 per-line quadratic and 8.72 lot-pooled; the export carries 28 of them with a HIPS reading (28 more JPL rows from late 2022 have none).</Note>
      </ChartFrame>

      <ChartFrame id="bl-york" title="The York ladder and ftir_53's five-site table — not redrawn here" provenance={`${NB47} [14] (five_target_york_ladder.png) · ${NB53} (01)`} exportable={false}>
        <Note>ftir_47 [14] refits the five explorer targets with York (prediction against Fabs / 10) under four blank-line constructions. Its inputs are not all in the browser: the per-site HIPS uncertainty models that set σ<sub>x</sub> (hips_lab.sigma_fabs), the lot-pooled quadratics of the whole batch, and Bishoftu's raw T1 / R1, which the optics export does not carry (Bishoftu is not one of its four focus sites), so the ladder is not drawn and ftir_53 01 stays static. The explorer's own York fits under the deployed line and the per-line linear and quadratic refits (adjudication configuration, ocec-450 × AIRSpec, k = 9) are on Calibration → HIPS diagnostics, from calibration_runs.json; they are ftir_47 [13]'s adjudication rows: Addis −1.51 / −1.48 / −1.48 µg/m³, Bishoftu −0.39 / −0.34 / −0.33, Beijing +0.63 / +0.67 / +0.68, Delhi −1.32 / −1.24 / −1.25, Pasadena slope 3.15 / 3.06 / 3.04.</Note>
      </ChartFrame>
    </>
  )
}
