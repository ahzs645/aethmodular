/**
 * Research → Python PMF, what the fits are checked against (charts 5–9): held-out total OM, the archived KF solution,
 * the gallery's own measurements, the preparation and cohort sensitivity, and the QC inventory against the gallery's
 * exclusions. Every per-filter statistic here is recomputed in the browser on the filters that count under
 * Filters → excluded (common/pythonPmf.ts, node-tested against the package's saved values); with every filter
 * counted it equals the package's.
 */
import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Figure } from '@/components/Figure'
import { HeatCell } from '@/components/HeatCell'
import { Key } from '@/components/Key'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { ExcludedNote, FixedSetNote, useCountedFilter, useExclusionSplit, useKnownFilter, useMarkedRows } from '@/lib/markedRows'
import { num, type Row } from '@/lib/table'
import { FONT, INK } from '@/lib/theme'
import type { FilterRow, MetaFile, PmfFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { BASIS_NOTE, DataTable, useConcRows } from '../../common/ui'
import { archivedMatch, baseId, disagrees, pearson, qcCell, scores, type Scores } from '../../common/pythonPmf'
import {
  CasePick, FACTOR_COLOR, FACTORS, REPORT, archivedFor, caseFilters, caseLabel, closest, factorName, tableRows,
  type PmfCase, type PythonPmfExport,
} from './data'

const FOLD_COLOR = ['#2c7fb8', '#d08c2c', '#4f9d69', '#b5446e', '#6a51a3']
const ODD = 'ETAD-0037'
const SOURCE = 'Research · Python PMF'

// --------------------------------------------------------------------------------------------- 5. held-out OM

interface OmItem { id: string; base: string; date: string; fold: number; obs: number; pred: number; mean: number }

const scoreCells = (s: Scores | null) => s ? `RMSE ${fmt(s.rmse, 3)} · MAE ${fmt(s.mae, 3)} · R² ${fmt(s.r2, 3)} (n = ${s.n})` : '—'

export function HeldOut({ data }: { data: PythonPmfExport }) {
  const [which, setWhich] = useSearchParam<'heldout' | 'conditional'>('ppmf-om.which', 'heldout', ['heldout', 'conditional'])
  const items = useMemo((): OmItem[] => which === 'heldout'
    ? tableRows(data, 'heldout').map((r) => ({ id: String(r.sample_id), base: baseId(String(r.sample_id)), date: String(r.date), fold: Number(r.fold), obs: Number(r.observed_OM_ug_m3), pred: Number(r.predicted_OM_ug_m3), mean: Number(r.training_mean_baseline_ug_m3) }))
    : tableRows(data, 'conditional').map((r) => ({ id: String(r.sample_id), base: baseId(String(r.sample_id)), date: String(r.date), fold: Number(r.date_block_fold), obs: Number(r.OM_ug_m3), pred: Number(r.conditional_heldout_OM_prediction_ug_m3), mean: Number(r.training_mean_baseline_ug_m3) })),
  [data, which])
  const split = useExclusionSplit(items, (i) => i.base)
  const live = scores(split.counted.map((i) => i.obs), split.counted.map((i) => i.pred))
  const base = scores(split.counted.map((i) => i.obs), split.counted.map((i) => i.mean))
  const odd = split.counted.find((i) => i.base === ODD)
  const oddShare = odd && live ? (odd.pred - odd.obs) ** 2 / live.sse : null
  const saved = which === 'heldout'
    ? { rmse: data.heldout_pooled.RMSE_ug_m3, mae: data.heldout_pooled.MAE_ug_m3, r2: data.heldout_pooled.R_squared }
    : { rmse: data.conditional.heldout.rmse_ug_m3, mae: data.conditional.heldout.mae_ug_m3, r2: data.conditional.heldout.r2 }
  const savedFolds = which === 'heldout'
    ? tableRows(data, 'heldout_folds').map((r) => ({ fold: Number(r.fold), from: String(r.heldout_date_min), to: String(r.heldout_date_max), n: num(r.n), rmse: num(r.RMSE_ug_m3), r2: num(r.R_squared) }))
    : tableRows(data, 'conditional_folds').map((r) => ({ fold: Number(r.fold), from: String(r.test_date_min), to: String(r.test_date_max), n: num(r.n), rmse: num(r.rmse_ug_m3), r2: num(r.r2) }))
  const folds = savedFolds.map((f) => {
    const rs = split.counted.filter((i) => i.fold === f.fold)
    return { ...f, live: scores(rs.map((i) => i.obs), rs.map((i) => i.pred)) }
  })
  const all = split.nOut === 0
  const points: CrossPoint[] = items.map((i) => ({
    x: i.obs, y: i.pred, id: i.base, color: i.base === ODD ? INK.negative : FOLD_COLOR[i.fold - 1], hollow: i.base === ODD,
    tip: [i.id, `${i.date} · fold ${i.fold}`, `lab OM ${fmt(i.obs, 2)} · predicted ${fmt(i.pred, 2)} µg/m³`, ...(i.base === ODD ? ['unresolved spectrum/reference discrepancy: kept in every score'] : [])],
    rec: { id: i.base, source: SOURCE, site: 'Addis Ababa', date: i.date, fields: [['Lab FTIR OM (µg/m³)', i.obs], ['Predicted OM (µg/m³)', i.pred], ['Fold', i.fold]] },
  }))
  return (
    <ChartFrame id="ppmf-om" title={`Held-out total OM — ${scoreCells(live)}`}
      subtitle={`${which === 'heldout'
        ? 'Five contiguous date blocks are withheld in turn: profiles are refitted on the other 126–127 reference filters (two starts), a nonnegative no-intercept regression maps training contributions to lab OM, and the held-out spectra are projected onto the training profiles to predict OM without their lab values.'
        : 'The conditional check: the full 246-filter profiles are kept fixed and only OM is held out by date block (OM = G b, nonnegative, no intercept). It is not end-to-end: the held-out spectra helped fit the profiles.'} Scores are live on the filters that count (${live?.n ?? 0} of ${items.length}); the package's, with all ${items.length}: RMSE ${fmt(saved.rmse, 3)}, MAE ${fmt(saved.mae, 3)}, R² ${fmt(saved.r2, 3)} µg/m³${all ? ' — identical here' : ''}. The training-mean baseline scores ${scoreCells(base)}. ETAD-0037-5 (red ring) is an unresolved spectrum/reference discrepancy: lab OM 2.87 µg/m³, its spectrum nearly flat and negative${oddShare != null ? `; it carries ${fmt(100 * oddShare, 1)} % of the pooled squared error` : ''}. The report keeps it in every score, and so does this chart unless you exclude it by hand. Lab OM is from the same FTIR family, so this is same-method consistency, not source validation.`}
      provenance={<>Reproduces {REPORT} §8 (the held-out table and analysis/figures/heldout_organic_mass_validation.png). Data: python_pmf/results/mass_holdout/predictions.csv, fold_metrics.csv, summary.json; analysis/conditional_mass/etad246_huber/conditional_predictions.csv, fold_metrics.csv, summary.json. Live scores: common/pythonPmf.ts scores.</>}
      controls={<Segmented label="Prediction" value={which} options={['heldout', 'conditional'] as const} onChange={setWhich}
        optionLabel={(o) => (o === 'heldout' ? 'Spectra held out (end-to-end)' : 'Fixed full-data profiles')} />}>
      <Crossplot title="" points={points} xLabel="Lab FTIR OM (µg/m³)" yLabel="Predicted OM (µg/m³)" fit={false} grouping="none" maxSide={440} />
      <Key items={[...FOLD_COLOR.map((c, i) => ({ label: `fold ${i + 1}`, color: c })), { label: 'ETAD-0037-5', color: INK.negative, kind: 'hollow' as const }, { label: '1:1', color: INK.identity, kind: 'dash' as const }]} />
      <DataTable rows={[
        { what: `Live: the ${live?.n ?? 0} filters that count`, n: live?.n, rmse: live?.rmse, mae: live?.mae, r2: live?.r2, r: live?.r },
        { what: `The package's, all ${items.length} filters`, n: items.length, rmse: saved.rmse, mae: saved.mae, r2: saved.r2, r: which === 'heldout' ? data.heldout_pooled.Pearson_r : data.conditional.heldout.pearson_r },
        { what: 'Live training-mean baseline', n: base?.n, rmse: base?.rmse, mae: base?.mae, r2: base?.r2, r: base?.r },
      ]} rowKey={(r) => r.what} highlight={(r) => r.what.startsWith('Live:')} columns={[
        { key: 'w', label: 'Pooled scores', value: (r) => r.what },
        { key: 'n', label: 'n', num: true, value: (r) => r.n ?? '—' },
        { key: 'rmse', label: 'RMSE (µg/m³)', num: true, value: (r) => fmt(r.rmse, 3) },
        { key: 'mae', label: 'MAE (µg/m³)', num: true, value: (r) => fmt(r.mae, 3) },
        { key: 'r2', label: 'R²', num: true, value: (r) => fmt(r.r2, 3) },
        { key: 'r', label: 'Pearson r', num: true, value: (r) => fmt(r.r, 3) },
      ]} />
      <DataTable rows={folds} rowKey={(f) => String(f.fold)} columns={[
        { key: 'f', label: 'Fold (held-out dates)', value: (f) => `${f.fold}: ${f.from} – ${f.to}` },
        { key: 'n', label: 'n counted / saved', num: true, value: (f) => `${f.live?.n ?? 0} / ${f.n ?? '—'}` },
        { key: 'r', label: 'RMSE live', num: true, value: (f) => fmt(f.live?.rmse, 3) },
        { key: 'rs', label: 'RMSE saved', num: true, value: (f) => fmt(f.rmse, 3) },
        { key: 'q', label: 'R² live', num: true, value: (f) => fmt(f.live?.r2, 3) },
        { key: 'qs', label: 'R² saved', num: true, value: (f) => fmt(f.r2, 3) },
      ]} />
      {which === 'conditional' && <Note>{data.conditional.outside_reference_dates} of the {data.conditional.allocated_filters} modelled ETAD filters fall outside the OM reference dates ({data.conditional.reference_dates.join(' to ')}): their total-OM estimates from these fixed profiles would be temporal extrapolations, and a good total does not validate how it splits between factors.</Note>}
    </ChartFrame>
  )
}

// --------------------------------------------------------------------------------------------- 6. new vs archived

export function Archived({ data, c, setCase, archived }: { data: PythonPmfExport; c: PmfCase | null; setCase: (k: string) => void; archived: PmfFile | null }) {
  const [factor, setFactor] = useSearchParam<string>('ppmf-kf.factor', '1', ['1', '2', '3', '4', '5'])
  const counts = useCountedFilter()
  const matches = c ? archivedFor(data, c.key) : new Map()
  const label = (kf: number) => archived?.sources[kf - 1]?.label ?? `KF${kf}`
  const live = useMemo(() => (c && archived ? archivedMatch(data.contributions[c.key], archived.rows, [...matches.entries()].map(([f, m]) => ({ factor: f, kf: m.kf })), label, counts) : []),
    [c, archived, data, counts]) // eslint-disable-line react-hooks/exhaustive-deps
  const k = Number(factor), m = matches.get(k)
  const points = useMemo((): CrossPoint[] => {
    if (!c || !archived || !m) return []
    const byId = new Map(archived.rows.map((r) => [r.id, r]))
    return caseFilters(data, c.key, []).flatMap((f) => {
      const a = byId.get(f.base), y = a?.ugm3?.[label(m.kf)]
      if (!a || y == null || !Number.isFinite(f.g[k - 1])) return []
      return [{ x: f.g[k - 1], y, id: f.base, color: FACTOR_COLOR[k - 1], tip: [f.id, f.date, `${factorName(k)} G ${fmt(f.g[k - 1], 3)}`, `archived ${label(m.kf)} ${fmt(y, 2)} µg/m³`],
        rec: { id: f.base, source: SOURCE, site: 'Addis Ababa', date: f.date, fields: [[`${factorName(k)} G`, f.g[k - 1]], [`archived ${label(m.kf)} (µg/m³)`, y]] } }]
    })
  }, [c, archived, data, k, m]) // eslint-disable-line react-hooks/exhaustive-deps
  const allCount = live.every((l) => l.n === 102)
  const row = live.find((l) => l.factor === k)
  return (
    <ChartFrame id="ppmf-archived" title={c ? `New vs archived — ${factorName(k)} against the archived ${m ? label(m.kf) : '—'} factor, r = ${fmt(row?.r, 3)}` : 'New vs archived'}
      subtitle={`Each new factor's contribution against the archived KF factor (µg/m³, the gallery's pmf.json) the package matched it to, on the ${row?.n ?? 0} common filters that count (the archived solution's 102). The one-to-one assignment was chosen to maximise these same correlations, so they are descriptive best matches, not an independent recovery test, and no archived label or mass coefficient is carried onto a new factor. r is live; the package's (all 102 counted) is beside it${allCount ? ', and equal here' : ''}.`}
      provenance={<>Reproduces {REPORT} §6 (the archived-pattern table). Data: python_pmf/analysis/historical_factor_pattern_matches.csv; the archived contributions from <code>public/data/pmf.json</code>. Live r: common/pythonPmf.ts archivedMatch.</>}
      controls={<>
        <CasePick data={data} value={c} onChange={setCase} etadOnly />
        <Segmented label="Factor" value={factor} options={['1', '2', '3', '4', '5']} onChange={setFactor} optionLabel={(o) => `F${o}`} />
      </>}>
      {!c ? <Empty>The archived solution covers Addis only: pick an ETAD cohort.</Empty> : !archived ? <Empty>Loading the archived solution (pmf.json)…</Empty> : <>
        <Crossplot title="" points={points} xLabel={`${factorName(k)} G (spectral amplitude per m³)`} yLabel={`Archived ${m ? label(m.kf) : ''} (µg/m³)`} fit={false} grouping="none" sameScale={false} maxSide={400} />
        <DataTable rows={live} rowKey={(l) => String(l.factor)} highlight={(l) => l.factor === k} columns={[
          { key: 'f', label: 'New factor', value: (l) => factorName(l.factor) },
          { key: 'a', label: 'Matched archived factor', value: (l) => `KF${l.kf} (${label(l.kf)})` },
          { key: 'n', label: 'n', num: true, value: (l) => l.n },
          { key: 'r', label: 'r live', num: true, value: (l) => fmt(l.r, 3) },
          { key: 's', label: 'r saved (102)', num: true, value: (l) => fmt(matches.get(l.factor)?.r, 3) },
        ]} />
      </>}
    </ChartFrame>
  )
}

// --------------------------------------------------------------------------------------------- 7. vs gallery measurements

interface Variable { key: string; label: string; short: string; value: (r: FilterRow) => number | null; saved?: string }

function Matrix({ rows, cols, cell, picked, onPick }: { rows: string[]; cols: Variable[]; cell: (f: number, v: Variable) => { r: number | null; n: number; saved: number | null }; picked: [number, string]; onPick: (f: number, v: string) => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const labelW = 74, top = 22, ch = 30
  const cw = Math.max(60, Math.min(120, (width - labelW - 8) / Math.max(1, cols.length)))
  const colours = useColourScheme('ppmf-matrix', 'diverging')
  const color = d3.scaleQuantize<string>().domain([-1, 1]).range(colours.stops)
  return (
    <div ref={ref} style={{ position: 'relative', overflowX: 'auto' }}>
      <svg width={labelW + cols.length * cw + 4} height={top + rows.length * ch + 4} fontFamily={FONT.family}>
        {cols.map((c, j) => <text key={c.key} x={labelW + j * cw + cw / 2} y={top - 7} textAnchor="middle" fontSize={11} fill={INK.text}>{c.short}</text>)}
        {rows.map((name, i) => <g key={name}>
          <text x={labelW - 6} y={top + i * ch + ch / 2} dy="0.35em" textAnchor="end" fontSize={11.5} fill={FACTOR_COLOR[i]}>{name}</text>
          {cols.map((c, j) => {
            const v = cell(i + 1, c)
            return <HeatCell key={c.key} x={labelW + j * cw} y={top + i * ch} width={cw - 2} height={ch - 2} fill={v.r === null ? null : color(v.r)} label={v.r === null ? '' : fmt(v.r, 2)}
              outline={picked[0] === i + 1 && picked[1] === c.key} onClick={() => onPick(i + 1, c.key)}
              onEnter={(e) => tip.show(e, [`${name} × ${c.label}`, `r ${fmt(v.r, 3)} (n ${v.n})`, ...(v.saved != null ? [`package (102 filters): ${fmt(v.saved, 3)}`] : []), 'click: draw it below'])} onLeave={tip.hide} />
          })}
        </g>)}
      </svg>
      <ColorLegend scale={color} scheme={colours.scheme} label="Pearson r" format={(v) => fmt(v, 1)} />
      {tip.node}
    </div>
  )
}

export function Measurements({ data, c, setCase, rows: viewRows, meta }: { data: PythonPmfExport; c: PmfCase | null; setCase: (k: string) => void; rows: FilterRow[]; meta: MetaFile }) {
  const mac = useMac()
  const rows = useConcRows(viewRows, meta)
  const markedAll = useConcRows(useMarkedRows(), meta)
  const [set, setSet] = useSearchParam<'view' | 'p102'>('ppmf-meas.set', 'view', ['view', 'p102'])
  const [pf, setPf] = useSearchParam<string>('ppmf-meas.factor', '3', ['1', '2', '3', '4', '5'])
  const vars: Variable[] = useMemo(() => [
    { key: 'bc', label: `HIPS BC at ${mac.label} (Fabs/MAC, µg/m³)`, short: `HIPS BC`, value: (r) => mac.fromFabs(num(r['HIPS Fabs'] as number)), saved: 'hips_Fabs_Mm1' },
    { key: 'ec', label: 'EC (FTIR, µg/m³)', short: 'FTIR EC', value: (r) => num(r['EC (FTIR)'] as number) },
    { key: 'oc', label: 'OC (FTIR, µg/m³)', short: 'FTIR OC', value: (r) => num(r['OC (FTIR)'] as number) },
    { key: 'om', label: 'OM (FTIR, µg/m³)', short: 'FTIR OM', value: (r) => num(r['OM (FTIR)'] as number) },
    { key: 'pm', label: 'PM2.5 mass (µg/m³)', short: 'PM2.5', value: (r) => num(r['PM2.5 mass'] as number), saved: 'pm25_gravimetric_ug_m3' },
  ], [mac])
  const [pv, setPv] = useSearchParam<string>('ppmf-meas.var', 'bc', vars.map((v) => v.key))
  const keep = useMemo(() => new Set(data.comparison_filters.map(baseId)), [data])
  const joined = useMemo(() => {
    if (!c) return { counted: [], marked: [] }
    const byBase = new Map(caseFilters(data, c.key, []).map((f) => [f.base, f]))
    const join = (rs: FilterRow[]) => rs.flatMap((r) => { const f = byBase.get(r.id); return f && (set === 'view' || keep.has(r.id)) ? [{ r, f }] : [] })
    return { counted: join(rows), marked: join(markedAll) }
  }, [c, data, rows, markedAll, set, keep])
  const supporting = tableRows(data, 'supporting')
  const savedOf = (f: number, v: Variable) => (c && v.saved ? num(supporting.find((s) => s.case === c.key && Number(s.factor) === f && s.reference_variable === v.saved)?.pearson_r ?? null) : null)
  const cell = (f: number, v: Variable) => {
    const xs: number[] = [], ys: number[] = []
    for (const { r, f: pf_ } of joined.counted) { const y = v.value(r), x = pf_.g[f - 1]; if (y != null && Number.isFinite(x)) { xs.push(x); ys.push(y) } }
    return { r: pearson(xs, ys), n: xs.length, saved: savedOf(f, v) }
  }
  const k = Number(pf), variable = vars.find((v) => v.key === pv) ?? vars[0]
  const toPoint = ({ r, f }: { r: FilterRow; f: { id: string; g: number[] } }): CrossPoint[] => {
    const y = variable.value(r), x = f.g[k - 1]
    return y == null || !Number.isFinite(x) ? [] : [{ x, y, id: r.id, color: FACTOR_COLOR[k - 1], tip: [f.id, r.date ?? '', `${factorName(k)} G ${fmt(x, 3)}`, `${variable.label} ${fmt(y, 2)}`],
      rec: { id: r.id, source: SOURCE, site: r.site, date: r.date, fields: [[`${factorName(k)} G`, x], [variable.label, y]] } }]
  }
  const pick = cell(k, variable)
  const archived = c ? archivedFor(data, c.key) : new Map()
  return (
    <ChartFrame id="ppmf-measurements" title={c ? `Factors against the gallery's measurements — ${caseLabel(c)}` : 'Factors against the gallery\'s measurements'}
      subtitle={`Pearson r of each factor's contribution with the gallery's own measurements of the same filter (joined by base id), on ${set === 'view' ? 'the Addis rows in view' : 'the package\'s 102 comparison filters among the rows in view'} that count: your dates, seasons and exclusions apply. HIPS BC follows your MAC (r does not: it is scale-free, so BC at any MAC gives Fabs's r). Shared seasonality and correlated sources make strong correlations easy, so these are supporting context, not source identification. Hover a cell for the package's own value (its 102 filters, HIPS Fabs and gravimetric PM2.5); click one to draw it.`}
      provenance={<>Extends {REPORT} §6 (analysis/supporting_variable_correlations.csv) to the gallery's measurements. Live r: common/pythonPmf.ts pearson.</>}
      controls={<>
        <CasePick data={data} value={c} onChange={setCase} etadOnly />
        <Segmented label="Filters" value={set} options={['view', 'p102'] as const} onChange={setSet} optionLabel={(o) => (o === 'view' ? 'Addis rows in view' : 'Package\'s 102')} />
      </>}>
      {!c ? <Empty>Only the Addis filters have gallery rows: pick an ETAD cohort.</Empty> : !joined.counted.length ? <Empty>No Addis rows in view join this case's filters.</Empty> : <>
        <Matrix rows={FACTORS.map(factorName)} cols={vars} cell={cell} picked={[k, variable.key]} onPick={(f, v) => { setPf(String(f)); setPv(v) }} />
        <div className="controls-row" style={{ display: 'flex', gap: 12, flexWrap: 'wrap', margin: '8px 0' }}>
          <Segmented label="Factor" value={pf} options={['1', '2', '3', '4', '5']} onChange={setPf} optionLabel={(o) => `F${o}`} />
          <Select label="Measurement" value={variable.key} options={vars.map((v) => v.key)} onChange={setPv} optionLabel={(o) => vars.find((v) => v.key === o)?.short ?? o} />
        </div>
        <p className="chart-note" style={{ margin: '4px 0' }}><strong>{factorName(k)} × {variable.short}</strong>: r = {fmt(pick.r, 3)} (n = {pick.n}){pick.saved != null ? `; the package's ${fmt(pick.saved, 3)} on its 102` : ''}{archived.get(k) ? ` · ${closest(archived.get(k))}` : ''}</p>
        <Crossplot title=""
          points={joined.counted.flatMap(toPoint)} marked={joined.marked.flatMap(toPoint)} xLabel={`${factorName(k)} G`} yLabel={variable.label} fit={false} grouping="none" sameScale={false} maxSide={400} />
        <DataTable rows={[...FACTORS]} rowKey={(f) => String(f)} columns={[
          { key: 'f', label: `Against the package (${set === 'p102' ? 'its 102 filters' : 'switch Filters to Package\'s 102 to compare like for like'})`, value: (f) => factorName(f) },
          ...vars.filter((v) => v.saved).flatMap((v) => [
            { key: `${v.key}-l`, label: `${v.short} r live (n)`, num: true, value: (f: number) => { const x = cell(f, v); return `${fmt(x.r, 3)} (${x.n})` } },
            { key: `${v.key}-s`, label: `${v.key === 'bc' ? 'HIPS Fabs' : 'gravimetric PM2.5'} r saved`, num: true, value: (f: number) => fmt(savedOf(f, v), 3) },
          ]),
        ]} />
        <Note>On the package's 102, HIPS Fabs (and so BC) gives the package's r exactly. PM2.5 differs slightly: the gallery's PM2.5 mass differs from the package's gravimetric values on eight July–August 2023 filters (ETAD-0094 to ETAD-0104, by up to 1 µg/m³). {BASIS_NOTE}</Note>
      </>}
      <ExcludedNote count={joined.marked.length} drawn />
    </ChartFrame>
  )
}

// --------------------------------------------------------------------------------------------- 8. sensitivity

export function Sensitivity({ data }: { data: PythonPmfExport }) {
  const label = (k: string) => { const c = data.cases.find((x) => x.key === k); return c ? caseLabel(c) : k }
  const rows = useMemo(() => {
    const m = tableRows(data, 'case_matches').filter((r) => r.reference_case !== r.other_case)
    const pairs = [...new Set(m.map((r) => `${r.reference_case}|${r.other_case}`))]
    return pairs.map((p): Row => {
      const [ref, other] = p.split('|'), rs = m.filter((r) => r.reference_case === ref && r.other_case === other)
      const cos = rs.map((r) => Number(r.profile_cosine)), cr = rs.map((r) => Number(r.contribution_r_common_filters))
      const med = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]
      return { reference: `against ${label(ref)}`, other: `${label(other)} (${rs[0]?.n_common_physical_filters} common)`,
        cos: med(cos), cosLo: Math.min(...cos), cosHi: Math.max(...cos), r: med(cr), rLo: Math.min(...cr), rHi: Math.max(...cr) }
    })
  }, [data]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Figure kind="intervals" id="ppmf-sensitivity" data={rows} block="reference" row="other" labelWidth={330}
      panels={[{ value: 'cos', lo: 'cosLo', hi: 'cosHi', label: 'Matched profile cosine: range over the five factors (dot: median)', reference: 1 },
        { value: 'r', lo: 'rLo', hi: 'rHi', label: 'Contribution r on the common filters: range (dot: median)', reference: 1 }]}
      title="Preparation and cohort sensitivity — the factors survive a loss or blank change, less so a baseline change"
      subtitle="Each alternative five-factor case matched one-to-one to a reference case by uncertainty-weighted spectral cosine; contribution r uses only identical physical filters. Against the 102-filter fit, a row mixes cohort extension, loss, preprocessing and profile reallocation; against each site's full primary cohort, it isolates the one change. The 244-filter case drops the two historically omitted filters (ETAD-0091, ETAD-0100) and is indistinguishable from the 246."
      provenance={<>Reproduces {REPORT} §5 (the two preparation-sensitivity tables). Data: python_pmf/analysis/five_factor_case_matches.csv.</>}
      aside={<FixedSetNote what="the package's saved case-to-case matches, each pair compared on all its common filters" />} />
  )
}

// --------------------------------------------------------------------------------------------- 9. QC inventory

const HOLD_LABEL: Record<string, string> = {
  'not_PM2.5': 'not PM2.5 (PM10)', field_blank: 'field blank', missing_date_or_volume: 'missing date or volume',
  only_one_same_lot_site_blank: 'only one same-lot site blank (no SD)', historical_sensitivity_hold_only_no_current_registry_exclusion: 'historical hold only',
  documented_large_hole: 'documented large hole',
}
const STAND = { counted: 'gallery counts it', excluded: 'gallery excludes it', absent: 'no gallery row' } as const
const ORDER = { counted: 0, excluded: 1, absent: 2 } as const

export function Inventory({ data }: { data: PythonPmfExport }) {
  const counts = useCountedFilter()
  const known = useKnownFilter()
  const groups = useMemo(() => {
    const out = new Map<string, { site: string; pmf: string; gallery: keyof typeof STAND; ids: string[]; disagree: boolean }>()
    for (const r of tableRows(data, 'qc')) {
      const id = String(r.sample_id), b = baseId(id), modelled = r.model_eligible === true
      const cellv = qcCell(modelled, known(b), counts(b))
      const pmf = modelled ? 'modelled' : `held: ${HOLD_LABEL[String(r.hold_reason)] ?? String(r.hold_reason)}`
      const key = `${r.site_code}|${pmf}|${cellv.gallery}`
      const g = out.get(key) ?? { site: String(r.site_code), pmf, gallery: cellv.gallery, ids: [], disagree: disagrees(cellv) }
      g.ids.push(id); out.set(key, g)
    }
    return [...out.values()].sort((a, b) => a.site.localeCompare(b.site) || (a.pmf === 'modelled' ? -1 : b.pmf === 'modelled' ? 1 : a.pmf.localeCompare(b.pmf)) || ORDER[a.gallery] - ORDER[b.gallery])
  }, [data, counts, known])
  const off = groups.filter((g) => g.disagree)
  return (
    <ChartFrame id="ppmf-inventory" title={`QC inventory against the gallery — ${off.length ? `${off.reduce((a, g) => a + g.ids.length, 0)} filter${off.reduce((a, g) => a + g.ids.length, 0) === 1 ? '' : 's'} where the two disagree` : 'no disagreement'}`}
      subtitle="Every physical filter in the package's combined inventory (296 ETAD, 72 ETBI records): whether the PMF models it or holds it and why, against what the gallery does with the same filter now (Filters → excluded and your hand exclusions included). Highlighted rows disagree: the PMF holds a filter the gallery counts, or models one the gallery leaves out. The filters with no gallery row (ETAD after the gallery's Addis series ends, every ETBI filter, field blanks) stay in every PMF chart above; they only cannot join gallery measurements."
      provenance={<>Reproduces {REPORT} §1 (data recovered and corrected). Data: python_pmf/analysis/All_Filter_QC_Inventory.csv.</>}>
      <DataTable rows={groups} rowKey={(g) => `${g.site}|${g.pmf}|${g.gallery}`} highlight={(g) => g.disagree} columns={[
        { key: 's', label: 'Site', value: (g) => g.site },
        { key: 'p', label: 'Python PMF', value: (g) => g.pmf },
        { key: 'g', label: 'Gallery', value: (g) => STAND[g.gallery] },
        { key: 'n', label: 'Filters', num: true, value: (g) => g.ids.length },
        { key: 'i', label: 'Which', value: (g) => (g.ids.length <= 6 ? g.ids.join(', ') : `${g.ids[0]} … ${g.ids[g.ids.length - 1]}`) },
      ]} />
      {off.length > 0 && <Note>{off.map((g) => `${g.ids.join(', ')}: ${g.pmf.replace('held: ', 'the PMF holds it (')}${g.pmf.startsWith('held') ? ')' : ''}, ${STAND[g.gallery]}`).join('; ')}.{off.some((g) => g.ids.some((i) => i.startsWith('ETAD-0158'))) ? ' ETAD-0158-6 has a large hole documented in a laboratory memo; the gallery has no such exclusion yet.' : ''}</Note>}
    </ChartFrame>
  )
}
