import { useMemo } from 'react'
import { ChartFrame, ExplainOnly, Note } from '@/components/ChartFrame'
import { Select } from '@/components/Controls'
import { ExportState, LoadState } from '@/components/ExportState'
import { BundleContext, Figure, type FigureSpec, type MeetingBundle } from '@/components/Figure'
import { Key } from '@/components/Key'
import { PageToc } from '@/components/PageToc'
import { Marker } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { correction, linearise, noiseGain, saturate } from '@/lib/hipsLinearise'
import { Markdown } from '@/lib/markdown'
import { median, quantile } from '@/lib/stats'
import { rowsOf } from '@/lib/table'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { SPEC_MEETINGS } from '@/meetings'
import { AgainstHips, Bend, COMMAND, CorrectionCurve, CorrectionPlot, Limits, LINEAR, MEASURED, PROVENANCE, Readouts, SeasonLot, SIDE, WORKFLOW, Worked, pathOf, range, useCeiling, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'

/** the meeting pages this one borrows figures from (src/meetings/<id>.meeting.ts, data/meetings/<id>.json) */
const CLOSURE = 'closure-20261110'
const NETWORK = 'hips-network-20261111'
const PROCEDURE = 'hips-procedure-20261111'
const SEASONS = 'hips-seasons-20261111'
const ADDIS = 'Addis Ababa, Ethiopia'

/** a figure declared on another meeting page: its id there, with an optional patch and the bundle text shown under "?" */
type Borrowed = string | { id: string; patch?: Partial<FigureSpec>; about?: string }

/**
 * Figures declared on another meeting page, drawn here from that page's own bundle, so the two cannot differ: the
 * spec (title, columns, notes) comes from src/meetings/<from>.meeting.ts and the rows from data/meetings/<from>.json.
 * A figure keeps its id, so a choice made here (a site, a variant) is the same choice on its own page.
 */
function Borrow({ from, figures, columns }: { from: string; figures: Borrowed[]; columns?: 2 }) {
  const spec = SPEC_MEETINGS.find((m) => m.id === from)
  const { data, error } = useExport<MeetingBundle>(`meetings/${from}.json`)
  const declared = useMemo(() => spec?.sections.flatMap((s) => s.figures ?? []) ?? [], [spec])
  if (!data) return <ExportState what={`the ${spec?.short ?? from} data`} error={error} command={spec?.command} />
  return (
    <BundleContext.Provider value={data}>
      <div className={columns === 2 ? 'figure-pair' : undefined}>
        {figures.map((b) => {
          const pick = typeof b === 'string' ? { id: b } : b
          const figure = declared.find((f) => f.id === pick.id)
          if (!figure) return <ExportState key={pick.id} title={pick.id} error={`no figure "${pick.id}" in src/meetings/${from}.meeting.ts`} />
          return <Figure key={pick.id} {...({ ...figure, ...pick.patch } as FigureSpec)}
            about={pick.about && data.text[pick.about] ? <Markdown text={data.text[pick.about]} /> : undefined} />
        })}
      </div>
    </BundleContext.Provider>
  )
}

const Part = ({ children }: { children: string }) => <h2 className="meeting-section-title">{children}</h2>

/**
 * HIPS on dark filters, part by part: the walk-through of the linearisation in the order it is explained. Where the
 * bend is seen (IMPROVE thermal EC), what linearising does to one filter, how it applies to Addis, how it applies to
 * every SPARTAN site (and how a site's correction is made), what it changes in the earlier results, what the
 * correction costs and what would avoid it (less air through the filter), and what is open.
 *
 * Nothing is computed for this page alone. The interactive charts are the "Linearising HIPS at Addis" page's
 * (pages/HipsLinearisePage), on one ceiling control; every other figure is borrowed from the 10 and 11 Nov meeting
 * pages by id (Borrow). The one new chart, a site's filters on the correction curve, reads the 11 Nov procedure's
 * per-filter table.
 */
export function HipsPartsPage() {
  const { data, error } = useExport<Linearise>('hips_linearise/linearise.json')
  const ceil = useCeiling(data)
  if (error || !data) return (
    <ChartFrame id="hw-missing" title="HIPS on dark filters, part by part" provenance={WORKFLOW}>
      <LoadState what="the linearisation export" error={error} command={COMMAND} />
    </ChartFrame>
  )
  const primary = data.improve_ceilings[0]
  const band = `${fmt(primary.tau_inf, 2)} [${fmt(primary.tau_inf_lo, 2)}, ${fmt(primary.tau_inf_hi, 2)}]`
  return (
    <PageToc>
      <Figure kind="table" id="hw-parts" title="HIPS reads low on dark filters, part by part — where it is seen, what linearising does, Addis, and every SPARTAN site"
        subtitle={`One curve, measured on IMPROVE against thermal EC (ceiling ${band}), read backwards on each filter. The ceiling buttons on the charts below set the ceiling for every interactive chart on this page.`}
        columns={[{ key: 'part', label: 'Part' }, { key: 'question', label: 'Question' }, { key: 'shows', label: 'What it shows' }]}
        data={[
          { part: '1. IMPROVE', question: 'Where is the bend seen?', shows: 'HIPS optical depth against thermal EC on the same filters, and the same bend in every subset of sites' },
          { part: '2. Linearising', question: 'What does it do to one filter?', shows: 'One filter through every step, and the correction factor against how dark the filter is' },
          { part: '3. Addis', question: 'How does it apply to Addis?', shows: 'Where the Addis filters sit, the fit of FTIR EC on HIPS before and after, the ceiling, the seasons' },
          { part: '4. SPARTAN', question: 'How is it made for each site?', shows: 'The same curve on each site\'s own filters: optical depth, factor, status, and the check on the site\'s own data' },
          { part: '5. Earlier results', question: 'What does it change?', shows: 'The saved calibrations, the FTIR level and the MAC, re-read against linearised HIPS' },
          { part: '6. The cost', question: 'What are the downsides, and the fix?', shows: 'What the correction does to noise and uncertainty, and how much less air through the filter would avoid it' },
          { part: '7. Open', question: 'What is not settled?', shows: 'The limits of a borrowed ceiling, and the pathways that are done, open or blocked' },
        ]} />

      <Part>1. Where it is seen: on IMPROVE filters, HIPS falls behind thermal EC as the filter darkens</Part>
      <Bend data={data} ceil={ceil} />
      <Borrow from={CLOSURE} figures={[{ id: 'c1110-improve-ratio', about: 'improve' }, 'c1110-ceilings']} />

      <Part>2. What linearising does: each filter is read back through the curve</Part>
      <Worked data={data} ceil={ceil} />
      <CorrectionCurve data={data} ceil={ceil} />

      <Part>3. Addis: the filters sit on the bend, and linearising takes the intercept to zero</Part>
      <Borrow from={NETWORK} figures={['h1111-tau']} />
      <AgainstHips data={data} ceil={ceil} />
      <Limits data={data} />
      <Borrow from={PROCEDURE} figures={[{ id: 'hp-traced', about: 'trace' }]} />
      <Borrow from={SEASONS} figures={[{ id: 'hs-within', about: 'within' }, 'hs-site-seasons']} />
      <SeasonLot data={data} ceil={ceil} />

      <Part>4. Every SPARTAN site: one curve, applied to each filter; a site's numbers are rebuilt from its filters</Part>
      <PerSiteRecipe data={data} />
      <SiteOnCurve data={data} ceil={ceil} />
      <Borrow from={PROCEDURE} figures={[{ id: 'hp-points', about: 'correct' }]} />
      <Borrow from={SEASONS} figures={[{ id: 'hs-levels', about: 'levels' }]} />
      <Borrow from={NETWORK} figures={[{ id: 'h1111-sites', about: 'sites' }, { id: 'h1111-replay', about: 'volume' }]} />
      <Borrow from={PROCEDURE} figures={[{ id: 'hp-status', about: 'identify' }, { id: 'hp-after', about: 'check' }, 'hp-checked']} />

      <Part>5. What it changes: every result scored against HIPS ÷ 10 is re-read</Part>
      <Borrow from={CLOSURE} figures={[{ id: 'c1110-cand-season', about: 'candidates' }, 'c1110-cand-overall', { id: 'c1110-ftir-level', about: 'source' }, { id: 'c1110-mac-bars', about: 'level' }, 'c1110-rereads']} />

      <Part>6. What linearising costs, and what would fix it: less air through the filter</Part>
      <Downsides data={data} ceil={ceil} />
      <VolumeReplay data={data} ceil={ceil} />
      <Borrow from={NETWORK} figures={[{ id: 'h1111-options', about: 'decision' }]} />

      <Part>7. What is not settled</Part>
      <Borrow from={CLOSURE} figures={[{ id: 'c1110-pathways', about: 'next' }]} />
      <Borrow from={PROCEDURE} figures={['hp-revisit']} />
    </PageToc>
  )
}

// ------------------------------------------------------------------ how a site's correction is made
/** The recipe as a table: what is shared by every site, what is the filter's own, and what the site's own data are for. */
function PerSiteRecipe({ data }: { data: Linearise }) {
  const primary = data.improve_ceilings[0]
  const c = fmt(primary.tau_inf, 2), area = fmt(100 * data.deposit_area_cm2, 0)
  return (
    <>
      <Figure kind="table" id="hw-recipe" title="How a site's linearised HIPS is made — no site has a curve of its own; each filter is corrected from its own optical depth"
        subtitle="SPARTAN filters have no thermal EC, so the curve cannot be fitted at a site. The ceiling is IMPROVE's for every site; what differs from site to site is how dark its filters are."
        columns={[{ key: 'step', label: 'Step' }, { key: 'from', label: 'Comes from' }, { key: 'how', label: 'How' }, { key: 'level', label: 'Made per' }]}
        data={[
          { step: '1. The curve', from: `IMPROVE, ${primary.samples.toLocaleString()} samples at ${primary.sites} sites, against thermal EC`, how: `One ceiling, ${c} [${fmt(primary.tau_inf_lo, 2)}, ${fmt(primary.tau_inf_hi, 2)}]`, level: 'network (the same at every site)' },
          { step: '2. Optical depth', from: 'The filter\'s HIPS Fabs and its sampled volume', how: `Fabs × volume ÷ ${area}`, level: 'filter' },
          { step: '3. Expected under-read', from: 'Step 2 and the ceiling', how: `optical depth ÷ ${c}`, level: 'filter' },
          { step: '4. Linearised value', from: 'Step 3', how: 'Fabs ÷ (1 − under-read), with a low and a high from the ends of the ceiling\'s interval', level: 'filter' },
          { step: '5. Status', from: 'Step 3', how: 'use (up to a fifth), caution (a fifth to three tenths), shape only (beyond)', level: 'filter' },
          { step: '6. The site\'s numbers', from: 'Its corrected filters', how: 'Median BC, seasonal contrast, slope and intercept against FTIR EC are recomputed from the corrected filters; no site factor is applied', level: 'site' },
          { step: '7. The check', from: 'The site\'s own HIPS, FTIR EC and weighed PM2.5', how: 'After correcting, HIPS should rise in proportion to FTIR EC (exponent 1). This tests the correction; it does not set it', level: 'site (and season)' },
        ]} />
      <ExplainOnly>
        <section className="panel">
          <header className="frame-head"><div><h2>Why per filter and not per site</h2></div></header>
          <p>The factor depends only on the filter's own optical depth, and the optical depth is the air's absorption times the volume of air drawn through the filter. Two sites with the same air get different corrections if they sample different volumes, and one site gets different corrections on a clean day and a dark one. A single factor for a site would over-correct its light filters and under-correct its dark ones; the bars under "one factor for a group" below show by how much.</p>
          <p>A site's "under-read" on the ranking chart is the under-read of its median filter. It describes the site; it is not what is applied to its filters.</p>
        </section>
      </ExplainOnly>
    </>
  )
}

interface PointRow { id: string; site: string; date: string; fabs: number; volume: number; tau: number; status: string }

/** the 11 Nov procedure's per-filter rows, and its sites darkest first (as on the network pages) */
function usePoints() {
  const { data: bundle, error } = useExport<MeetingBundle>(`meetings/${PROCEDURE}.json`)
  const all = useMemo(() => rowsOf<PointRow>(bundle?.tables.points).filter((r) => Number.isFinite(r.tau) && r.volume > 0), [bundle])
  const sites = useMemo(() => {
    const by = new Map<string, number[]>()
    for (const r of all) by.set(r.site, [...(by.get(r.site) ?? []), r.tau])
    return [...by].map(([site, taus]) => ({ site, tau: median(taus) })).sort((a, b) => b.tau - a.tau)
  }, [all])
  const [chosen, setSite] = useSearchParam<string>('hw.site', ADDIS)
  const site = sites.some((s) => s.site === chosen) ? chosen : sites[0]?.site ?? ADDIS
  const rows = useMemo(() => all.filter((r) => r.site === site), [all, site])
  const control = <Select label="Site" value={site} options={sites.map((s) => s.site)} onChange={setSite}
    optionLabel={(o) => `${o} · τ ${fmt(sites.find((s) => s.site === o)?.tau, 2)}`} />
  return { bundle, error, all, rows, site, control }
}

/** the under-read a filter would show had `volume` m³ been drawn through it: its loading back through the curve, rescaled, read forward */
const underReadAt = (r: PointRow, volume: number, ceiling: number): number => saturate((linearise(r.tau, ceiling) * volume) / r.volume, ceiling) / ceiling

// ------------------------------------------------------------------ what the correction costs
/** The downsides of correcting after the fact, each with its size on the Addis filters and what would remove it. */
function Downsides({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const c = ceil.ceiling
  const lo = data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi = data.ceilings.find((k) => k.key === 'hi')!.tau_inf
  const taus = data.filters.map((f) => f.tau)
  const med = median(taus), dark = Math.max(...taus)
  // how dark IMPROVE gets: the darkest group of any IMPROVE population (hips_loading.IMPROVE_CEILING.reach), not Adama's five filters
  const reach = Math.max(...data.improve_ceilings.filter((p) => p.sites > 1).map((p) => p.max_stratum_tau))
  const beyond = taus.filter((t) => t > reach).length / taus.length
  const eras = data.improve_ceilings.filter((p) => /^\d{4}/.test(p.population))
  return (
    <Figure kind="table" id="hw-downsides" title="What linearising costs — it corrects after the fact, so noise and the ceiling's uncertainty grow with the correction"
      subtitle={`Sizes are for the ${taus.length} Addis filters at the chosen ceiling (${fmt(c, 2)}). The first four shrink if the filter is lighter to begin with; the last two need thermal EC on dark filters.`}
      columns={[{ key: 'cost', label: 'Downside' }, { key: 'size', label: 'How large at Addis' }, { key: 'fix', label: 'What would fix it' }]}
      data={[
        { cost: 'Noise is multiplied', size: `An error in the measured optical depth grows × ${fmt(noiseGain(med, c), 1)} on the median filter and × ${fmt(noiseGain(dark, c), 1)} on the darkest`, fix: 'A lighter filter: less air drawn through it' },
        { cost: 'The factor is only as good as the ceiling', size: `× ${fmt(correction(med, c), 2)} on the median filter, anywhere from × ${fmt(correction(med, hi), 2)} to × ${fmt(correction(med, lo), 2)} across the ceiling's interval`, fix: 'A lighter filter (the range narrows fast), or a ceiling measured at Addis' },
        { cost: 'It grows without bound', size: `No finite value at or past the ceiling; the darkest Addis filter is at ${fmt(100 * dark / c, 0)} % of it`, fix: 'A lighter filter' },
        { cost: 'It cannot set a level on dark filters', size: 'Every Addis filter on file is "shape only": usable for the shape of a comparison, not its level', fix: 'New filters at a lower volume; the filters on file stay as they are' },
        { cost: 'The ceiling is borrowed', size: `The darkest group of IMPROVE filters, in any subset, is at optical depth ${fmt(reach, 2)}; ${fmt(100 * beyond, 0)} % of Addis filters are darker${eras.length === 2 ? `, and the ceiling differs by period (${eras.map((p) => `${fmt(p.tau_inf, 2)} for ${p.population}`).join(', ')})` : ''}`, fix: 'HIPS and thermal EC on the same Addis filters, some loaded to optical depth 0.5–1.5' },
        { cost: 'It is still optical', size: 'Linearised HIPS is absorption, not EC; turning it into EC still needs a MAC that Addis does not have', fix: 'Thermal EC at Addis' },
      ]} />
  )
}

/**
 * The same air at another sampled volume: each filter's loading is taken back through the curve, rescaled to the
 * volume and read forward again (hips_loading.at_volume), and the site's median filter is drawn against volume. The
 * marks are SPARTAN's three volumes; the readouts give the correction that would be left at each.
 */
function VolumeReplay({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const { bundle, error, rows, site, control } = usePoints()
  const c = ceil.ceiling
  const lo = data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi = data.ceilings.find((k) => k.key === 'hi')!.tau_inf
  const usable = useMemo(() => rows.filter((r) => r.tau < c), [rows, c])
  const at = useMemo(() => (v: number, q = 0.5) => quantile(usable.map((r) => underReadAt(r, v, c)), q), [usable, c])
  const VOLUMES = [1.8, 3.6, 7.2]
  const points = useMemo<ScatterPoint[]>(() => (usable.length ? VOLUMES.map((v) => ({ key: `v${v}`, x: v, y: at(v), color: LINEAR, r: 6, opacity: 1,
    tip: [`${fmt(v, 1)} m³`, `median filter under-reads ${fmt(100 * at(v), 0)} %`, `correction × ${fmt(1 / (1 - at(v)), 2)}`] })) : []), [usable, at]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!bundle) return <ExportState title="The same air at another sampled volume" what="the 11 Nov procedure data" error={error} command={SPEC_MEETINGS.find((m) => m.id === PROCEDURE)?.command} />
  // the volume at which the median filter loses a fifth (the edge of "use, linearised"): the curve rises with volume, so bisect
  let a = 0.05, b = 12
  for (let i = 0; i < 40; i++) { const m = (a + b) / 2; if (at(m) > 0.2) b = m; else a = m }
  const fifth = at(12) > 0.2 ? (a + b) / 2 : NaN
  const now = median(usable.map((r) => r.volume)), asSampled = median(usable.map((r) => r.tau / c))
  const vs = range(0.2, 8, 0.2)
  const under = ({ x, y }: ScatterScales) => <g pointerEvents="none">
    <path d={pathOf([...vs.map((v): [number, number] => [x(v), y(at(v, 0.75))]), ...[...vs].reverse().map((v): [number, number] => [x(v), y(at(v, 0.25))])]) + 'Z'} fill={LINEAR} opacity={0.12} />
    <path d={pathOf(vs.map((v) => [x(v), y(at(v))]))} fill="none" stroke={LINEAR} strokeWidth={2} />
    {Number.isFinite(fifth) && fifth < 8 && <Marker orient="v" at={x(fifth)} from={y(0)} to={y(0.2)} color={MEASURED} label={`a fifth at ${fmt(fifth, 1)} m³`} />}
  </g>
  // what is left of the correction at a volume: the factor on the median filter, its range across the ceiling, and the noise gain
  const left = (v: number) => {
    const tau = at(v) * c, L = linearise(tau, c)
    return { factor: correction(tau, c), lo: correction(saturate(L, hi), hi), hi: correction(saturate(L, lo), lo), noise: noiseGain(tau, c) }
  }
  return (
    <ChartFrame id="hw-volume" title="Less air, lighter filter — the same air at a lower sampled volume needs a smaller, better-known correction"
      subtitle="Expected under-read of the site's median filter against the volume of air drawn through it. Line: the median filter; band: the middle half of its filters. Dots: SPARTAN's three volumes, 1.8, 3.6 and 7.2 m³. Lines at a fifth (use, linearised) and three tenths (shape only beyond)."
      provenance={<>{PROVENANCE} Per-filter rows: <code>meetings/{PROCEDURE}.json</code>, table <code>points</code>; the replay is <code>hips_loading.at_volume</code>.</>}
      controls={<>{control}{ceil.control}</>}>
      <ScatterPlot points={points} xLabel="Sampled volume (m³)" yLabel="Expected under-read, median filter" xDomain={[0, 8]} yDomain={[0, 0.6]} nice={false}
        yFormat={(v: number) => `${Math.round(100 * v)} %`} maxSide={SIDE} outside="pin" note={false} refY={[0.2, 0.3]} under={under} />
      <Readouts items={[
        { label: `As sampled, ${fmt(now, 1)} m³`, value: `${fmt(100 * asSampled, 0)} %`, sub: `${usable.length.toLocaleString()} filters at ${site.split(',')[0]}` },
        ...[3.6, 1.8].map((v) => { const k = left(v); return { label: `At ${fmt(v, 1)} m³`, value: `${fmt(100 * at(v), 0)} %`, sub: `× ${fmt(k.factor, 2)} (${fmt(k.lo, 2)}–${fmt(k.hi, 2)}), noise × ${fmt(k.noise, 1)}` } }),
        { label: 'A fifth or less needs', value: Number.isFinite(fifth) ? `${fmt(fifth, 1)} m³` : 'any volume', color: MEASURED, sub: 'on the median filter' },
      ]} />
      <Note>The cost of sampling less air: less mass on the filter for every other analysis (weighing, XRF, ions, FTIR), and a step in the reported record where the volume changes, which the linearised series should not show. It only helps new filters. Delhi's volume was cut from about 7 to 3.6 to 1.8 m³; its filters at two volumes are consistent with the curve (the 11 Nov network page), on filters from different years.</Note>
    </ChartFrame>
  )
}

/**
 * One site's filters on the correction curve. The curve and its band are the same for every site (CorrectionPlot, as
 * on the Addis chart of part 2); choosing a site only changes where the dots fall. Rows: the 11 Nov procedure's
 * per-filter table (hips-procedure-20261111, table "points"); the factor is recomputed here at the chosen ceiling.
 */
function SiteOnCurve({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const { bundle, error, all, rows, control } = usePoints()
  const c = ceil.ceiling
  const lo = data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi = data.ceilings.find((k) => k.key === 'hi')!.tau_inf
  const colors = bundle?.colors ?? {}
  const points = useMemo<ScatterPoint[]>(() => rows.flatMap((r) => {
    const k = correction(r.tau, c)
    return Number.isFinite(k) ? [{ key: r.id, x: r.tau, y: k, color: colors[r.status] ?? INK.muted,
      tip: [`${r.id} · ${r.date}`, `Fabs ${fmt(r.fabs, 1)} Mm⁻¹ × ${fmt(r.volume, 2)} m³ → τ ${fmt(r.tau, 2)}`, `× ${fmt(k, 2)} → ${fmt(r.fabs * k, 1)} Mm⁻¹`, r.status] }] : []
  }), [rows, c, colors])
  if (!bundle) return <ExportState title="One site's filters on the curve" what="the 11 Nov procedure data" error={error} command={SPEC_MEETINGS.find((m) => m.id === PROCEDURE)?.command} />
  const taus = rows.map((r) => r.tau), med = median(taus)
  const factors = (k: number) => median(taus.map((t) => correction(t, k)).filter(Number.isFinite))
  const statuses = [...new Set(all.map((r) => r.status))].sort((a, b) => Object.keys(colors).indexOf(a) - Object.keys(colors).indexOf(b))
  const share = (s: string) => (rows.length ? rows.filter((r) => r.status === s).length / rows.length : NaN)
  const past = rows.filter((r) => !(r.tau < c)).length
  return (
    <ChartFrame id="hw-site-curve" title="One site's filters on the curve — the curve is shared; a site differs only in where its filters fall on it"
      subtitle="Correction factor 1 / (1 − τ/τ∞) against each filter's measured optical depth (Fabs × its sampled volume ÷ 353). Choose a site, darkest first. The line and its band are the same for every site."
      provenance={<>{PROVENANCE} Per-filter rows: <code>meetings/{PROCEDURE}.json</code>, table <code>points</code>.</>}
      controls={<>
        {control}
        {ceil.control}
      </>}>
      <CorrectionPlot points={points} ceiling={c} lo={lo} hi={hi} median={med} medianLabel={`median τ ${fmt(med, 2)} → × ${fmt(correction(med, c), 2)}`} />
      <Key items={[{ label: `chosen ceiling ${fmt(c, 2)}`, color: LINEAR, kind: 'line' }, { label: `ceiling ${fmt(lo, 2)} to ${fmt(hi, 2)} (90 %)`, color: LINEAR, kind: 'faded' },
        ...statuses.map((s) => ({ label: s, color: colors[s] ?? INK.muted, kind: 'dot' as const }))]} />
      <Readouts items={[
        { label: 'Filters', value: rows.length.toLocaleString(), sub: `median volume ${fmt(median(rows.map((r) => r.volume)), 1)} m³` },
        { label: 'Median optical depth', value: fmt(med, 2) },
        { label: 'Median factor ×', value: fmt(factors(c), 2), sub: `× ${fmt(factors(hi), 2)} to × ${fmt(factors(lo), 2)} across the ceiling` },
        ...statuses.map((s) => ({ label: s, value: `${fmt(100 * share(s), 0)} %`, color: colors[s] })),
      ]} />
      <Note>Statuses are the 11 Nov procedure's, set at the central ceiling: use up to a fifth under-read, caution to three tenths, shape only beyond.{past ? ` ${past} of this site's filters are at or past the chosen ceiling and have no finite value with it.` : ''}</Note>
    </ChartFrame>
  )
}
