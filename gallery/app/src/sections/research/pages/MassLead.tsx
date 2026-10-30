import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { Crossplot, type CrossLine, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalMark, type IntervalRow } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt, signed } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { SidebarControls } from '@/lib/pageSidebar'
import { num as cell, type Row } from '@/lib/table'
import { seasonOnly } from '@/lib/seasons'
import { INK } from '@/lib/theme'
import type { CalibRunsFile, FilterRow, MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import {
  CONDITIONING, addedVariable, framed, lstsq, massPerArea, partialSlopes, seasonCols, seedOf, siteFit,
  type Estimate, type Framing, type MassFilter, type PartialRow, type SiteFit,
} from '../common/massLead'
import { OPTICS_COMMAND, useOptics } from '../common/optics'
import { quantile } from '../common/stats'
import { BASIS_NOTE, DataTable, SERIES, num, point, useConcRows } from '../common/ui'
import { presetName, rows as tableRows, useCalibRuns, useSpectral, type SpectralExport } from './spectralCommon'

const FTIR46 = 'research/ftir_ec_phase3/ftir_46_paired_increment_and_mass.ipynb'
/** ftir_46 used the locked lowest-OC/EC 800 + AIRSpec calibration (and a k = 8 440 run the export does not hold) */
const LOCKED = 'ocec800_airspec'
const NB_CONFIG = 'locked 800 + AIRSpec (rule k)'
/** the explorer's five targets in ftir_46's order */
const TARGETS = [['addis', 'ETAD'], ['etbi', 'ETBI'], ['chts', 'CHTS'], ['indh', 'INDH'], ['uspa', 'USPA']] as const
const POOLED = 'pooled: 4 non-Addis sites'
/** the explorer's groups outside Ethiopia are calendar seasons */
const GROUP_COLOR: Record<string, string> = { DJF: SERIES[0], MAM: SERIES[2], JJA: SERIES[1], SON: SERIES[3] }
const FRAMING_LABEL: Record<Framing, string> = { volume: 'per volume (µg/m³)', deposit: 'per deposit (µg/cm²)' }
const NB_FRAMING: Record<Framing, string> = { volume: 'per volume', deposit: 'per deposit' }

interface SiteData { target: string; code: string; site: string; color: string; nEval: number; nView: number; nMass: number; filters: MassFilter[] }
interface SiteResult extends SiteData { fit: Record<Framing, SiteFit | null> }

const ci = (e: Estimate | null | undefined, d = 3) => (e ? `${fmt(e.est, d)} [${fmt(e.lo, d)}, ${fmt(e.hi, d)}]` : '—')
/** a notebook row's value and interval, × k (a negative k swaps the ends); `lo`/`hi` name the interval columns */
const nbCi = (r: Row | undefined, key: string, k = 1, d = 3, lo = `${key}_lo`, hi = `${key}_hi`) => {
  const v = cell(r?.[key]), a = (cell(r?.[lo]) ?? NaN) * k, b = (cell(r?.[hi]) ?? NaN) * k
  return v == null ? '—' : `${fmt(v * k, d)} [${fmt(Math.min(a, b), d)}, ${fmt(Math.max(a, b), d)}]`
}
/** does a live estimate fall inside the notebook's interval (× k)? */
const inside = (e: Estimate | null | undefined, r: Row | undefined, key: string, k = 1, lo = `${key}_lo`, hi = `${key}_hi`) => {
  const a = cell(r?.[lo]), b = cell(r?.[hi])
  return e && a != null && b != null ? e.est >= Math.min(a * k, b * k) && e.est <= Math.max(a * k, b * k) : null
}

/**
 * ftir_46's mass lead, computed here: the explorer's per-filter predictions for a calibration preset
 * (calibration_runs.json) joined to the gallery rows' PM2.5 mass and the HIPS optics export's volume, deposit
 * area and lot. At fixed prediction, does the calibration residual f(X) − Fabs/MAC fall with mass, at Addis and
 * at the other explorer sites? OLS partial slopes with a per-filter bootstrap (common/massLead), so every number
 * follows the filters in view, the hand exclusions and the MAC.
 */
export function MassLead({ rows: viewRows, meta }: SectionPageProps) {
  const rows = useConcRows(viewRows, meta)
  const { runs, error: runsError } = useCalibRuns()
  const { rows: optics, error: opticsError } = useOptics()
  // the notebook's own tables (the ML section's residual export), for the comparison only; the page draws without them
  const { data: nb } = useSpectral('residual', 'ml')
  const { mac } = useMac()
  const presets = useMemo(() => (runs?.presets ?? []).filter((p) => TARGETS.every(([t]) => runs!.runs.some((r) => r.preset === p.key && r.target === t))), [runs])
  const [presetParam, setPreset] = useSearchParam<string>('mass.preset', LOCKED)
  const preset = presets.some((p) => p.key === presetParam) ? presetParam : LOCKED
  const byId = useMemo(() => new Map(rows.map((r) => [r.id, r])), [rows])

  const sites = useMemo((): SiteData[] | null => {
    if (!runs || !optics) return null
    const opt = new Map(optics.filter((o) => o.type !== 'FB').map((o) => [o.id, o]))
    return TARGETS.map(([target, code]) => {
      const run = runs.runs.find((r) => r.preset === preset && r.target === target)
      const site = meta.sites.find((s) => s.code === code)?.name ?? runs.targets[target]?.site ?? code
      const color = meta.sites.find((s) => s.code === code)?.color ?? runs.targets[target]?.color ?? INK.muted
      const filters: MassFilter[] = []
      let nView = 0, nMass = 0
      run?.eval.id.forEach((id, i) => {
        const r = id ? byId.get(id) : undefined
        if (!id || !r) return
        nView++
        const mass = num(r, 'PM2.5 mass'), fabs = run.eval.ref[i], fx = run.eval.pred[i]
        if (mass === null) return
        nMass++
        if (!Number.isFinite(fabs) || !Number.isFinite(fx)) return
        const o = opt.get(id), date = run.eval.date?.[i] ?? r.date
        filters.push({ id, group: run.eval.group[i], month: date ? Number(date.slice(5, 7)) : null, fabs, fx, mass,
          volume: o?.volume_m3 ?? null, area: o?.area_cm2 ?? null, lot: o?.lot ?? null })
      })
      return { target, code, site, color, nEval: run?.eval.id.length ?? 0, nView, nMass, filters }
    })
  }, [runs, optics, preset, byId, meta.sites])

  // ftir_46's Addis deposit-loading range: the Addis filters' own mass per area, here those in view
  const range = useMemo((): [number, number] | null => {
    const a = (sites?.[0].filters ?? []).map(massPerArea).filter((v): v is number => v !== null)
    return a.length ? [Math.min(...a), Math.max(...a)] : null
  }, [sites])
  const results = useMemo((): SiteResult[] | null => {
    if (!sites) return null
    const pooled: SiteData = { target: 'pooled', code: '', site: POOLED, color: INK.text, nEval: d(sites.slice(1), 'nEval'), nView: d(sites.slice(1), 'nView'), nMass: d(sites.slice(1), 'nMass'),
      filters: sites.slice(1).flatMap((s) => s.filters) }
    return [...sites, pooled].map((s) => ({ ...s, fit: {
      volume: siteFit(s.filters, 'volume', mac, range, { seed: seedOf(`${s.site}|volume`) }),
      deposit: siteFit(s.filters, 'deposit', mac, range, { seed: seedOf(`${s.site}|deposit`) }),
    } }))
  }, [sites, mac, range])

  if (!runs || !optics || !results) {
    const error = runsError || opticsError
    return <ChartFrame id="mass-lead" title="The mass lead" provenance={FTIR46} exportable={false}>
      <Empty>{error ? <>Could not load {runsError ? 'the explorer readouts (calibration_runs.json)' : <>the HIPS optics export (run <code>{OPTICS_COMMAND}</code>)</>}: {error}.</> : 'Loading the explorer readouts and the HIPS optics…'}</Empty>
    </ChartFrame>
  }
  const p = presets.find((x) => x.key === preset)
  const locked = preset === LOCKED
  return <>
    <SidebarControls title="Mass lead page">
      <Select label="Calibration" value={preset} options={presets.map((x) => x.key)} onChange={setPreset}
        optionLabel={(k) => { const x = presets.find((q) => q.key === k); return x ? presetName(x) : k }} />
    </SidebarControls>
    <AddisResidual addis={results[0]} byId={byId} meta={meta} mac={mac} nb={locked ? nb : null} presetLabel={p ? presetName(p) : preset} />
    <SiteCoefficients results={results} mac={mac} nb={locked ? nb : null} range={range} presetLabel={p ? presetName(p) : preset} runs={runs} />
    <ResidualGrid results={results.slice(0, 5)} byId={byId} meta={meta} mac={mac} presetLabel={p ? presetName(p) : preset} />
  </>
}
const d = (s: SiteData[], k: 'nEval' | 'nView' | 'nMass') => s.reduce((a, x) => a + x[k], 0)

/** a filter's colour: the meta's season for the explorer's Ethiopian groups, a fixed calendar-season colour elsewhere */
const groupColor = (meta: MetaFile, g: string) =>
  [...meta.seasons, ...Object.values(meta.season_conventions ?? {}).flat()].find((s) => s.name === g || seasonOnly(s.name) === g)?.color ?? GROUP_COLOR[g] ?? INK.muted
const groupsOf = (filters: MassFilter[]) => [...new Set(filters.map((f) => f.group))]

/**
 * ftir_46 §5 (cell 13): the Addis residual against mass, raw and added-variable, per volume or per deposit, and the
 * partial-slope table under the notebook's three conditionings.
 */
function AddisResidual({ addis, byId, meta, mac, nb, presetLabel }: { addis: SiteResult; byId: Map<string, FilterRow>; meta: MetaFile; mac: number; nb: SpectralExport | null; presetLabel: string }) {
  const [framing, setFraming] = useSearchParam<Framing>('mass-addis.framing', 'volume', ['volume', 'deposit'])
  const table = useMemo(() => [...partialSlopes(addis.filters, 'volume', mac, { seed: 3 }), ...partialSlopes(addis.filters, 'deposit', mac, { seed: 5 })], [addis.filters, mac])
  const view = useMemo(() => {
    const F = framed(addis.filters.filter((f) => f.month !== null), framing, mac)
    if (F.use.length < 10) return null
    const unit = framing === 'volume' ? 'µg/m³' : 'µg/cm²'
    const raw = lstsq([F.m], F.r)
    const av = addedVariable([F.fx, ...seasonCols(F.use.map((f) => f.month!))], F.m, F.r)
    const pts = (xs: number[], ys: number[], xl: string, yl: string): CrossPoint[] => F.use.flatMap((f, i) => {
      const r = byId.get(f.id)
      return r ? [point(r, xs[i], ys[i], groupColor(meta, f.group), xl, yl, 'Research · the mass lead', [`f(X) ${fmt(f.fx, 2)} µg/m³ · Fabs ${fmt(f.fabs, 1)} Mm⁻¹`, `PM2.5 ${fmt(f.mass, 1)} µg/m³`])] : []
    })
    const meanM = F.m.reduce((a, b) => a + b, 0) / F.m.length, meanR = F.r.reduce((a, b) => a + b, 0) / F.r.length
    const rawLine: CrossLine[] = raw ? [{ slope: raw[0], intercept: meanR - raw[0] * meanM, color: INK.muted, label: `raw slope ${signed(raw[0], 4)}`, method: 'ols', stats: null }] : []
    const part = table.find((t) => t.framing === framing && t.conditioning === 'season')?.slope
    const avLine: CrossLine[] = part ? [{ slope: part.est, intercept: 0, color: INK.negative, label: `partial slope ${signed(part.est, 4)}`, method: 'ols', stats: null }] : []
    return { unit, raw, part, panels: [
      { key: 'raw', title: `Raw residual vs mass, ${FRAMING_LABEL[framing]}`, points: pts(F.m, F.r, `mass (${unit})`, `residual (${unit})`), lines: rawLine,
        xLabel: `Gravimetric PM2.5, ${FRAMING_LABEL[framing]}`, yLabel: `f(X) − HIPS EC-equivalent (${unit}, MAC ${mac})` },
      ...(av ? [{ key: 'av', title: `Added-variable: residual ~ mass | f(X), season`, points: pts(av.rx, av.ry, `mass | f(X), season`, `residual | f(X), season`), lines: avLine,
        xLabel: `mass | f(X), season (${unit})`, yLabel: `residual | f(X), season (${unit})` }] : []),
    ] }
  }, [addis.filters, framing, mac, byId, meta, table])

  const nbRows = tableRows(nb, 'addis_residual_vs_mass_partial_slopes')
  const nbOf = (r: PartialRow) => nbRows.find((x) => String(x.framing) === FRAMING_LABEL[r.framing] && String(x.conditioning) === CONDITIONING[r.conditioning])
  // the conditioned slopes are −β/MAC (f(X) in the design), so the notebook's MAC 10 values × 10/MAC; the raw one is not a rescaling
  const kOf = (r: PartialRow) => (r.conditioning === 'none' ? (mac === 10 ? 1 : null) : 10 / mac)
  const groups = groupsOf(addis.filters)
  const nbSeason = nbRows.find((x) => String(x.framing) === FRAMING_LABEL.volume && String(x.conditioning) === CONDITIONING.season)
  const liveSeason = table.find((t) => t.framing === 'volume' && t.conditioning === 'season')
  const rawRow = table.find((t) => t.framing === 'volume' && t.conditioning === 'none')
  const lastAddis = meta.sites.find((x) => x.code === 'ETAD')?.date_max
  const agreeNote = nbSeason && liveSeason?.slope
    ? ` Run on the notebook's 239 filters (with the tracked ETAD mass table) this code reproduces its table to the printed digits (massLead.test.mjs). On the ${liveSeason.n} here that slope is ${ci(liveSeason.slope, 4)} against the notebook's ${nbCi(nbSeason, 'slope', 10 / mac, 4, 'ci_lo', 'ci_hi')}${mac === 10 ? '' : ` (× 10/${mac})`}, ${inside(liveSeason.slope, nbSeason, 'slope', 10 / mac, 'ci_lo', 'ci_hi') ? 'inside' : 'outside'} its interval: the gallery's Addis rows ${lastAddis ? `end ${lastAddis}` : 'end earlier'}, the notebook's 239 run to Jan 2026, and the later filters pull the slope toward zero.`
    : ''
  return (
    <ChartFrame id="mass-lead-addis" title="At fixed prediction the Addis residual falls with mass; the raw scatter hides it"
      subtitle={`${presetLabel}, Addis: residual r = f(X) − Fabs/${mac} against gravimetric PM2.5 (the gallery rows), ${addis.filters.length} filters in view with mass of the explorer's ${addis.nEval}. Mass and f(X) correlate at ${fmt(addis.fit.volume?.corrMassFx, 2)}, so the raw slope (left, ${ci(rawRow?.slope, 4)} per volume) mixes f(X)'s own rise with mass; the slope at fixed f(X) and season (right) is the one to read${liveSeason?.slope && liveSeason.slope.hi < 0 ? ', and its interval is below zero: heavier deposits read more HIPS per unit of FTIR EC' : ', and on these filters its interval reaches zero'}. Per deposit uses each filter's volume and deposit area from the HIPS optics export. With f(X) in the design the slope is −β/MAC, β the mass coefficient of Fabs ~ f(X) + mass, so it scales as 10/MAC; the raw slope does not and is recomputed at MAC ${mac}.${agreeNote}`}
      provenance={<>Reproduces {FTIR46} §5 (cell 13). Predictions: <code>calibration_runs.json</code> (explorer eval); mass: <code>filters.json</code> PM2.5; volume, area, lot: <code>research/hips_optics/filters.json</code>. Fits and the 2,000-draw per-filter bootstrap in <code>sections/research/common/massLead.ts</code>.</>}
      controls={<Segmented label="Framing" value={framing} options={['volume', 'deposit'] as const} onChange={setFraming} optionLabel={(o) => (o === 'volume' ? 'Per volume' : 'Per deposit')} />}>
      {view ? <>
        <FacetGrid items={view.panels} minCell={300} maxCols={2} title={(pn) => pn.title} render={(pn) =>
          <Crossplot title="" points={pn.points} xLabel={pn.xLabel} yLabel={pn.yLabel} fit={false} grouping="none" extraLines={pn.lines} sameScale={false} maxSide={420} />} />
        <Key items={[...groups.map((g) => ({ label: g, color: groupColor(meta, g) })), { label: 'raw OLS', color: INK.muted, kind: 'line' as const }, { label: 'partial slope | f(X), season', color: INK.negative, kind: 'line' as const }]} />
      </> : <Empty>Fewer than 10 Addis filters in view carry PM2.5 mass and a date.</Empty>}
      <DataTable rows={table} rowKey={(r) => `${r.framing}|${r.conditioning}`} columns={[
        { key: 'f', label: 'Framing', value: (r) => FRAMING_LABEL[r.framing] },
        { key: 'c', label: 'Conditioning', value: (r) => CONDITIONING[r.conditioning] },
        { key: 'n', label: 'n', num: true, value: (r) => r.n },
        { key: 's', label: `Slope [95 % bootstrap], MAC ${mac}`, num: true, value: (r) => ci(r.slope, 4) },
        { key: 'i', label: '× mass IQR', num: true, value: (r) => fmt(r.slopeXIqr, 2) },
        ...(nb ? [{ key: 'nb', label: mac === 10 ? 'ftir_46 (239 filters)' : `ftir_46 × 10/${mac}`, num: true, value: (r: PartialRow) => { const k = kOf(r); return k === null ? 'MAC 10 only' : nbCi(nbOf(r), 'slope', k, 4, 'ci_lo', 'ci_hi') } }] : []),
      ]} />
      <Note>Units: per volume, µg/m³ of residual per µg/m³ of mass; per deposit, µg/cm² per µg/cm². The full conditioning adds sample volume and the HIPS lot ({[...new Set(addis.filters.map((f) => f.lot).filter(Boolean))].sort().join(', ')} in view). {BASIS_NOTE}</Note>
    </ChartFrame>
  )
}

/**
 * ftir_46 §7–8 [19a]: the mass coefficient in Fabs ~ f(X) + mass at each explorer site, per volume (Mm⁻¹ per µg/m³,
 * MAC-free) and per deposit (µg of HIPS-EC-equivalent per µg of mass, at the reader's MAC), full range or inside
 * the Addis deposit-loading range, against the notebook's.
 */
function SiteCoefficients({ results, mac, nb, range, presetLabel, runs }: { results: SiteResult[]; mac: number; nb: SpectralExport | null; range: [number, number] | null; presetLabel: string; runs: CalibRunsFile }) {
  const [which, setWhich] = useSearchParam<'full' | 'addis'>('mass-sites.range', 'addis', ['full', 'addis'])
  const nbRows = tableRows(nb, 'cross_site_mass_slopes').filter((r) => String(r.config) === NB_CONFIG)
  const nbOf = (site: string, f: Framing) => nbRows.find((r) => String(r.site) === site && String(r.framing) === NB_FRAMING[f])
  const key = which === 'full' ? 'mass_coef' : 'mass_coef_addis_range'
  const est = (s: SiteResult, f: Framing) => (which === 'full' ? s.fit[f]?.massCoef : s.fit[f]?.inRange) ?? null
  // per deposit the Fabs side is Fabs/MAC, so the notebook's MAC 10 coefficient × 10/MAC
  const kOf = (f: Framing) => (f === 'deposit' ? 10 / mac : 1)
  const mark = (e: Estimate | null, color: string): IntervalMark[] => (e ? [{ mid: e.est, lo: e.lo, hi: e.hi, color }] : [])
  const nbMark = (r: Row | undefined, f: Framing): IntervalMark[] => {
    const v = cell(r?.[key])
    return v == null ? [] : [{ mid: v * kOf(f), lo: (cell(r?.[`${key}_lo`]) ?? NaN) * kOf(f), hi: (cell(r?.[`${key}_hi`]) ?? NaN) * kOf(f), color: INK.muted, shape: 'diamond', dy: 6, r: 3, faint: true }]
  }
  const rowsOut: IntervalRow[] = results.map((s) => {
    const nv = nbOf(s.site, 'volume')
    const nLive = which === 'full' ? s.fit.volume?.n : s.fit.volume?.nInRange
    const nNb = cell(nv?.[which === 'full' ? 'n' : 'n_addis_range'])
    return { key: s.site, label: s.site, labelColor: s.color, strong: s.target === 'addis',
      note: `n=${nLive ?? 0}${nb ? ` · nb ${nNb ?? '—'}` : ''}`,
      marks: (['volume', 'deposit'] as Framing[]).map((f) => [...mark(est(s, f), s.color), ...(nb ? nbMark(nbOf(s.site, f), f) : [])]),
      tip: [s.site, ...(['volume', 'deposit'] as Framing[]).map((f) => `${NB_FRAMING[f]}: ${ci(est(s, f), f === 'volume' ? 2 : 3)}${nb ? ` · ftir_46 ${nbCi(nbOf(s.site, f), key, kOf(f), f === 'volume' ? 2 : 3)}` : ''}`)] }
  })
  // the agreement sentence: which live coefficients sit inside the notebook's interval (per volume)
  const verdict = nb ? results.flatMap((s) => {
    const e = est(s, 'volume'), r = nbOf(s.site, 'volume')
    if (!e || !r) return []
    return [`${s.site} ${fmt(e.est, 2)} vs ${fmt(cell(r[key]), 2)}${inside(e, r, key) ? '' : ' (outside its interval)'}`]
  }) : []
  const gen = tableRows(nb, 'mass_term_ftir53')
  const noMass = results.slice(0, 5).filter((s) => s.filters.length < 6).map((s) => s.nView === 0 ? `${s.site} (no gallery rows in view)` : `${s.site} (${s.nMass} of ${s.nView} rows in view carry PM2.5)`)
  return (
    <ChartFrame id="mass-lead-sites" title="Does mass add absorption at fixed FTIR EC? — at every site, more in Ethiopia"
      subtitle={`${presetLabel}: the coefficient on PM2.5 mass in Fabs ~ f(X) + mass at each explorer site, ${which === 'addis' && range ? `inside the Addis deposit-loading range (${fmt(range[0], 1)}–${fmt(range[1], 1)} µg/cm², the Addis filters in view)` : 'over each site\'s full range'}; dots computed here (per-filter bootstrap 95 %), grey diamonds the notebook's. Per volume it is Mm⁻¹ per µg/m³ and no MAC enters; per deposit it is µg of HIPS-EC-equivalent per µg of mass at MAC ${mac} (× 10/MAC of the notebook's). ${verdict.length ? `Per volume, live vs ftir_46: ${verdict.join('; ')}.` : nb ? '' : 'The notebook ran the locked 800 + AIRSpec calibration only, so its values are drawn for that preset.'} The live fits use only filters whose gallery row carries PM2.5 (counts below): Delhi's mass is mostly missing and Bishoftu has no gallery rows, so neither matches the notebook's n, which read mass from the SPARTAN catalogs.${gen.length ? ` Research → Does it generalise draws ftir_53's copy of the notebook's per-deposit Addis-range column (${gen.map((r) => `${r.site} ${fmt(r.mass_coef_addis_range, 3)}`).join(', ')}, MAC 10): the diamonds here at MAC 10, per deposit, Addis range.` : ''}`}
      provenance={<>Reproduces {FTIR46} §7–8 (cells 17 and 19, the per-site slopes). Notebook values: <code>public/data/ml/residual.json</code> (cross_site_mass_slopes, mass_term_ftir53). Explorer runs: <code>calibration_runs.json</code> ({runs.mode}).</>}
      controls={<Segmented label="Range" value={which} options={['addis', 'full'] as const} onChange={setWhich} optionLabel={(o) => (o === 'addis' ? 'Addis loading range' : 'Full range')} />}>
      <IntervalStrip blocks={[{ key: 's', rows: rowsOut }]} labelWidth={190} noteWidth={nb ? 92 : 50}
        panels={[{ label: 'Per volume: Mm⁻¹ per µg/m³ of mass', reference: 0, heading: 'Per volume (MAC-free)' },
          { label: `Per deposit: µg EC-eq per µg mass (MAC ${mac})`, reference: 0, heading: 'Per deposit' }]} />
      <Key items={[{ label: 'computed here (site colour), bootstrap 95 %', color: INK.text }, ...(nb ? [{ label: 'ftir_46 (locked 800 + AIRSpec), its interval', color: INK.muted, kind: 'diamond' as const }] : [])]} />
      <DataTable rows={results} rowKey={(s) => s.site} columns={[
        { key: 's', label: 'Site', value: (s) => s.site },
        { key: 'e', label: 'Explorer filters', num: true, value: (s) => s.nEval },
        { key: 'v', label: 'In view', num: true, value: (s) => s.nView },
        { key: 'm', label: 'With PM2.5', num: true, value: (s) => s.nMass },
        { key: 'd', label: 'With volume, area', num: true, value: (s) => s.fit.deposit?.n ?? 0 },
        { key: 'r', label: 'In Addis range', num: true, value: (s) => s.fit.deposit?.nInRange ?? 0 },
        { key: 'c', label: 'corr(mass, f(X))', num: true, value: (s) => fmt(s.fit.volume?.corrMassFx, 2) },
        { key: 'p', label: `Residual slope | f(X), MAC ${mac}`, num: true, value: (s) => ci(s.fit.volume?.rPartial, 4) },
        ...(nb ? [{ key: 'nb', label: 'ftir_46 n (with mass)', num: true, value: (s: SiteResult) => cell(nbOf(s.site, 'volume')?.n) ?? '—' }] : []),
      ]} />
      {noMass.length > 0 && <Note>No fit (fewer than 6 filters with mass): {noMass.join('; ')}.</Note>}
    </ChartFrame>
  )
}

/** ftir_46 [19b]: the residual against mass concentration at each site, coloured by group, with its partial slope at fixed f(X) */
function ResidualGrid({ results, byId, meta, mac, presetLabel }: { results: SiteResult[]; byId: Map<string, FilterRow>; meta: MetaFile; mac: number; presetLabel: string }) {
  const shown = results.filter((s) => s.filters.length >= 6 && s.fit.volume)
  const frames = shown.map((s) => framed(s.filters, 'volume', mac))
  const xHi = quantile(frames.flatMap((F) => F.m), 0.99)
  const rs = frames.flatMap((F) => F.r), yLo = quantile(rs, 0.005) - 0.5, yHi = quantile(rs, 0.995) + 0.5
  const panels = shown.map((s, k) => {
    const F = frames[k], fit = s.fit.volume!
    const mm = F.m.reduce((a, b) => a + b, 0) / F.m.length, mr = F.r.reduce((a, b) => a + b, 0) / F.r.length
    const points: CrossPoint[] = F.use.flatMap((f, i) => { const r = byId.get(f.id); return r ? [point(r, F.m[i], F.r[i], groupColor(meta, f.group), 'PM2.5', 'residual', 'Research · the mass lead', [`group ${f.group}`])] : [] })
    const slope = fit.rPartial?.est
    const lines: CrossLine[] = slope != null ? [{ slope, intercept: mr - slope * mm, color: INK.negative, label: `partial slope | f(X) ${signed(slope, 3)}`, method: 'ols', stats: null }] : []
    const beyond = F.m.filter((m) => m > xHi).length + F.r.filter((r) => r < yLo || r > yHi).length
    return { s, points, lines, fit, beyond, groups: groupsOf(s.filters) }
  })
  return (
    <ChartFrame id="mass-lead-grid" title="The residual against mass at each site — the slope at fixed prediction, drawn through the cloud"
      subtitle={`${presetLabel}: r = f(X) − Fabs/${mac} against PM2.5 mass concentration, one panel per explorer site with mass in view, on shared axes (x to the 99th percentile, y to the 0.5–99.5th). The red line is the partial slope at fixed f(X) (−β/MAC, so × 10/MAC of MAC 10's) drawn through the site means; the panel title gives β, the MAC-free mass coefficient of Fabs ~ f(X) + mass in Mm⁻¹ per µg/m³. Colours: Ethiopian seasons at Addis, calendar seasons elsewhere (the explorer's groups).`}
      provenance={<>Reproduces {FTIR46} §8 (cell 19, the residual-vs-mass grid), per volume.</>}>
      {panels.length ? <FacetGrid items={panels} minCell={240} maxCols={5}
        title={(pn) => `${pn.s.site} (n=${pn.fit.n}${pn.beyond ? `, ${pn.beyond} beyond frame` : ''})`}
        note={(pn) => `β ${ci(pn.fit.massCoef, 2)}`}
        render={(pn) => <Crossplot title="" points={pn.points} xLabel="PM2.5 (µg/m³)" yLabel={`f(X) − Fabs/${mac} (µg/m³)`} fit={false} grouping="none" extraLines={pn.lines} sameScale={false}
          domain={[0, xHi * 1.02]} yDomain={[yLo, yHi]} maxSide={300} />} /> : <Empty>No explorer site in view has 6 or more filters with PM2.5 mass.</Empty>}
      <Key items={[...new Set(panels.flatMap((pn) => pn.groups))].map((g) => ({ label: g, color: groupColor(meta, g) }))} />
    </ChartFrame>
  )
}
