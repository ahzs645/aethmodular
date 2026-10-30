import { useMemo } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Figure, FORMAT, type FormatName } from '@/components/Figure'
import { Key } from '@/components/Key'
import { useChartOptions } from '@/lib/chartOptions'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { ExcludedNote, useMarkedRows } from '@/lib/markedRows'
import { uniq, type Row, type Value } from '@/lib/table'
import { FONT, INK } from '@/lib/theme'
import type { FilterRow } from '@/lib/types'
import { median } from '../../common/stats'
import { provenance, rows as tableRows, type SpectralExport } from '../spectralCommon'
import { SMOKE_CALS, SMOKE_SETS, smokePoints, smokeScores, type SmokePoint } from './data'

const TOOL = SMOKE_CALS[2].label
const LOT_COLORS: Record<string, string> = { "Lot 251 (spartan_ec 02's set)": '#d62728', 'Lot 248': '#9a9a9a' }

/** one row per point and calibration, for the crossplot's two pickers */
const crossRows = (points: SmokePoint[]) => points.filter((p) => p.set !== 'lot251').flatMap((p) => SMOKE_CALS.map((c): Row => ({
  set: SMOKE_SETS[p.set], calibration: c.label, HIPS: p.hips, EC: p[c.key], id: p.id, date: p.date, scan: p.scan ?? '—',
})))
/** one row per Addis filter: the general EC against the lot-251 tool's, by filter lot */
const lotRows = (points: SmokePoint[], lot: ReadonlySet<string>) => points.filter((p) => p.set === 'filter').map((p): Row => ({
  general: p.reported, tool: p.smoke, id: p.id, date: p.date, lot: lot.has(p.id) ? "Lot 251 (spartan_ec 02's set)" : 'Lot 248',
}))
/** a row as a crossplot point, hovering as the page's Figure crossplots do */
const crossPoint = (r: Row, x: string, y: string, xLabel: string, yLabel: string, tips: string[], color: string): CrossPoint[] => {
  const xv = r[x], yv = r[y]
  if (typeof xv !== 'number' || typeof yv !== 'number' || !Number.isFinite(xv) || !Number.isFinite(yv)) return []
  return [{ x: xv, y: yv, color, id: String(r.id), tip: [String(r.id), ...tips.map((c) => `${c}: ${r[c] ?? '—'}`), `${xLabel}: ${f2(xv)}`, `${yLabel}: ${f2(yv)}`] }]
}

// the saved verdicts' calibration names, as the page names them
const SAVED_NAME: Record<string, string> = {
  'Regular reported EC (pre-CAL-0)': SMOKE_CALS[0].label, 'Deployed general (reported)': SMOKE_CALS[0].label,
  'CAL-0 refit (reference)': SMOKE_CALS[1].label, 'Regular (all-data CAL-0)': SMOKE_CALS[1].label,
  'Smoke-only (biomass)': TOOL,
}

/**
 * July07 (both notebooks) and spartan_ec 02: SPARTAN's lot-251 biomass tool calibration against the deployed
 * general EC and the 906-filter CAL-0 refit, at Addis, against HIPS. The per-filter tables are joined to the
 * gallery's Addis rows (data.ts), so HIPS = Fabs / MAC follows the reader's MAC and the filters in view, and the
 * fits are recomputed live; the notebooks' own verdicts (MAC 10) sit beside them.
 */
export function SmokeFigures({ data, rows }: { data: SpectralExport; rows: FilterRow[] }) {
  const m = useMac()
  const tables = useMemo(() => ({ addis: tableRows(data, 'smoke_addis'), etad: tableRows(data, 'smoke_etad'), lot251: tableRows(data, 'lot251_set') }), [data])
  const points = useMemo(() => smokePoints(rows, tables, m.mac), [rows, tables, m.mac])
  const scores = useMemo(() => smokeScores(points, m.mac), [points, m.mac])

  // Filters → excluded: Mark (lib/markedRows): the excluded Addis rows in view, joined as the rows in view are, drawn as
  // × on the two crossplots and in no fit, scorecard row or median
  const markedRows = useMarkedRows()
  const marked = useMemo(() => smokePoints(markedRows, tables, m.mac), [markedRows, tables, m.mac])
  const cross = useMemo(() => crossRows(points), [points])
  const crossMarked = useMemo(() => crossRows(marked), [marked])
  const lot = new Set([...points, ...marked].filter((p) => p.set === 'lot251').map((p) => p.id))
  const perFilter = lotRows(points, lot), perFilterMarked = lotRows(marked, lot)
  const lot251 = points.filter((p) => p.set === 'lot251')
  const delta = lot251.length ? median(lot251.map((p) => p.smoke - p.reported)) : null
  const ratio = lot251.length ? median(lot251.map((p) => p.smoke / p.reported)) : null
  const filterScores = scores.filter((s) => s.set === SMOKE_SETS.filter)
  const [general, , tool] = filterScores

  const saved: Row[] = [
    ...tableRows(data, 'smoke_verdict_addis').map((r): Row => ({ notebook: '07-07 before-CAL-0 · HIPS on EC', calibration: SAVED_NAME[String(r.calibration)] ?? r.calibration,
      n: r.n, slope: r.slope, intercept: r.intercept, R2: r.R2, implied: null, gap: null })),
    ...tableRows(data, 'smoke_verdict_etad').map((r): Row => ({ notebook: '07-07 regular vs smoke · EC on HIPS', calibration: SAVED_NAME[String(r.calibration)] ?? r.calibration,
      n: r.n, slope: r.slope, intercept: r.intercept, R2: r.R2, implied: r.implied_MAC, gap: r.median_HIPS_minus_EC })),
  ]

  if (!points.length) return <ChartFrame id="ethiopia-smoke-hips" title="SPARTAN's lot-251 biomass (smoke) calibration at Addis" exportable={false}>
    <Empty>No Addis filters with HIPS in view. The July07 comparison joins its tables to the gallery's Addis rows: include Addis Ababa under Filters.</Empty>
  </ChartFrame>
  return <>
    <SmokeHips cross={cross} marked={crossMarked}
      subtitle={`July07 asked whether a smoke-only calibration brings FTIR EC closer to HIPS. "Regular CAL-0" is the 906-filter IMPROVE set refitted at k = 20 (the set the gallery's "Biomass-smoke (906)" preset trains on); "smoke" is a different model, the coefficients SPARTAN's calibration tool exported for lot 251. All three are read on the gallery's Addis rows in view, HIPS = Fabs ÷ ${m.mac} (the notebooks: 10). Per Addis filter at this MAC the deployed EC runs ${f2(general?.slope)}x ${(general?.intercept ?? 0) < 0 ? '−' : '+'} ${f2(Math.abs(general?.intercept ?? 0))} (R² ${f2(general?.r2)}) and the lot-251 tool ${f2(tool?.slope)}x ${(tool?.intercept ?? 0) < 0 ? '−' : '+'} ${f2(Math.abs(tool?.intercept ?? 0))} (R² ${f2(tool?.r2)}), OLS. The line drawn is the gallery's Deming (λ ${f2(m.lambda)}).`}
      provenance={provenance(data, ['smoke_addis', 'smoke_etad'], 'research/July07/regular_before_cal0_vs_smoke_hips.ipynb and regular_vs_smoke_ec_vs_hips.ipynb; HIPS Fabs and the reported EC from the gallery rows (filters.json).')} />

    <ScoreTable id="ethiopia-smoke-scores" data={scores as unknown as Row[]} nMarked={new Set(marked.map((p) => p.id)).size}
      columns={[{ key: 'set', label: 'Filters' }, { key: 'calibration', label: 'Calibration' }, { key: 'n', label: 'n', format: 'int' },
        { key: 'slope', label: `Slope, EC on HIPS (MAC ${m.mac})`, format: 'f3' }, { key: 'intercept', label: 'Intercept (µg/m³)', format: 'f2' }, { key: 'r2', label: 'R²', format: 'f3' },
        { key: 'slopeHipsOnEc', label: 'Slope, HIPS on EC', format: 'f3' }, { key: 'interceptHipsOnEc', label: 'Intercept, HIPS on EC', format: 'f2' },
        { key: 'offOne', label: '|slope − 1|', format: 'f2' }, { key: 'gap', label: 'Median HIPS − EC', format: 'f2' }, { key: 'impliedMac', label: 'Implied MAC', format: 'f2' }]}
      title="July07's scorecard, live — the deployed EC has the shallowest slope and the highest R² of the three"
      subtitle={`OLS on the Addis rows in view at MAC ${m.mac}. Filters: one point per gallery Addis row with the July07 table's EC for its date (the before-CAL-0 notebook's join, the two scans of a filter averaged); one point per ETAD spectrum joined on ExternalFilterId (the regular-vs-smoke notebook's join; its 259 spectra include 49 from ETAD-0249 on, which are not gallery rows); spartan_ec 02's lot-251 filters. A row needs all three ECs, so the calibrations in a block share their filters. The before-CAL-0 notebook regressed HIPS on EC (FTIR EC on x); those columns are its orientation. Implied MAC = MAC ÷ slope, the MAC at which the slope would be 1; it does not depend on the MAC set.`}
      provenance={provenance(data, ['smoke_addis', 'smoke_etad', 'lot251_set'])} />

    <Figure kind="table" id="ethiopia-smoke-saved" data={saved}
      columns={[{ key: 'notebook', label: 'Notebook · orientation' }, { key: 'calibration', label: 'Calibration' }, { key: 'n', label: 'n', format: 'int' },
        { key: 'slope', label: 'Slope (MAC 10)', format: 'f2' }, { key: 'intercept', label: 'Intercept (MAC 10)', format: 'f2' }, { key: 'R2', label: 'R²', format: 'f3' },
        { key: 'implied', label: 'Implied MAC', format: 'f2' }, { key: 'gap', label: 'Median HIPS − EC (MAC 10)', format: 'f2' }]}
      title="The verdicts as July07 saved them, at MAC 10"
      subtitle="The exporter refuses the per-filter tables unless they reproduce these. The live scorecard matches the regular-vs-smoke rows on that notebook's 259 spectra (src/sections/research/pages/ethiopia/data.test.mjs) and the before-CAL-0 rows to two decimals on the gallery's 189 filters under Filters → excluded: Hide (that notebook's 2024-08-07 point averages ETAD-0241 with the PM10 filter ETAD-0248, which the export excludes; Include puts it back as a filter of its own). Its reported-EC row has n = 209: the reported EC it could match."
      provenance={provenance(data, ['smoke_verdict_addis', 'smoke_verdict_etad'])} />

    <ChartFrame id="ethiopia-smoke-lot251"
      title="spartan_ec 02 with the real biomass EC — the lot-251 tool reads higher than the general calibration on most Addis filters"
      subtitle={`Notebook 02 built this comparison before the tool's coefficients existed and ran it on a placeholder (the general EC × 1.18 on the higher-OC half); the exporter leaves those columns out. Here the placeholder is replaced by July07's lot-251 tool EC, per gallery Addis filter, coloured by filter lot. On 02's lot-251 set (n = ${lot251.length}) the median change is ${delta == null ? '—' : `${delta >= 0 ? '+' : '−'}${f2(Math.abs(delta))}`} µg/m³ and the median ratio ${f2(ratio)}: 02's hypothesis (biomass EC > general EC) holds, but against HIPS (above) it widens the gap.`}
      provenance={provenance(data, ['lot251_set', 'smoke_addis'], 'research/spartan_ec_2026_06_16/02_biomass_calibration_comparison.ipynb fig02 (panel 1).')}>
      {!perFilter.length ? <Empty>No rows to draw.</Empty> : <>
        <Crossplot title="" points={perFilter.flatMap((r) => crossPoint(r, 'general', 'tool', LOT_X, LOT_Y, ['date', 'lot'], LOT_COLORS[String(r.lot)]))}
          marked={perFilterMarked.flatMap((r) => crossPoint(r, 'general', 'tool', LOT_X, LOT_Y, ['date', 'lot'], LOT_COLORS[String(r.lot)]))}
          xLabel={LOT_X} yLabel={LOT_Y} fit={false} maxSide={440} />
        <Key items={[...uniq(perFilter, 'lot').map((g) => ({ label: String(g), color: LOT_COLORS[String(g)] })), { label: '1:1', color: INK.identity, kind: 'dash' as const }]} />
      </>}
      <ExcludedNote count={perFilterMarked.length} drawn />
    </ChartFrame>
  </>
}

const LOT_X = 'Deployed general FTIR EC (µg/m³)', LOT_Y = 'Lot-251 biomass tool EC (µg/m³)'

/**
 * July07's crossplot, with its two pickers (kept in the link as the page's Figure crossplots keep theirs): the
 * excluded Addis rows in view are drawn as × under Mark, in no fit.
 */
function SmokeHips({ cross, marked, subtitle, provenance }: { cross: Row[]; marked: Row[]; subtitle: string; provenance: React.ReactNode }) {
  const m = useMac()
  const id = 'ethiopia-smoke-hips'
  const sets = uniq(cross, 'set').map(String), cals = uniq(cross, 'calibration').map(String)
  const o = useChartOptions(id, [
    { key: 'set', label: 'Filters', persist: true, default: SMOKE_SETS.filter, style: sets.length > 4 ? 'select' : 'segmented', choices: sets },
    { key: 'calibration', label: 'Calibration', persist: true, default: TOOL, style: cals.length > 4 ? 'select' : 'segmented', choices: cals },
  ])
  const xLabel = `HIPS Fabs / ${m.mac} (µg/m³)`, yLabel = 'FTIR EC (µg/m³)'
  const chosen = (r: Row) => String(r.set) === o.values.set && String(r.calibration) === o.values.calibration
  const pts = cross.filter(chosen).flatMap((r) => crossPoint(r, 'HIPS', 'EC', xLabel, yLabel, ['date', 'scan'], INK.text))
  const out = marked.filter(chosen).flatMap((r) => crossPoint(r, 'HIPS', 'EC', xLabel, yLabel, ['date', 'scan'], INK.text))
  return (
    <ChartFrame id={id} controls={o.controls()} provenance={provenance} subtitle={subtitle}
      title="SPARTAN's lot-251 biomass (smoke) calibration at Addis — steeper against HIPS and less tight than the deployed EC">
      {!pts.length ? <Empty>No rows to draw.</Empty>
        : <Crossplot title="" points={pts} marked={out} xLabel={xLabel} yLabel={yLabel} maxSide={460} errorRatio={m.lambda} />}
      <ExcludedNote count={out.length} drawn />
    </ChartFrame>
  )
}

/** the live scorecard, laid out as the page's Figure tables are, with the note on what it leaves out under Mark */
function ScoreTable({ id, data, columns, title, subtitle, provenance, nMarked }: {
  id: string; data: Row[]; columns: { key: string; label: string; format?: FormatName }[]; title: string; subtitle: string; provenance: React.ReactNode; nMarked: number
}) {
  const cell = (format: FormatName | undefined, v: Value) => (format ? FORMAT[format](v) : v == null ? '—' : String(v))
  return (
    <ChartFrame id={id} title={title} subtitle={subtitle} provenance={provenance} exportable={false}>
      <div style={{ overflowX: 'auto' }}>
        <table className="census-table">
          <thead><tr>{columns.map((c) => <th key={c.key} style={{ textTransform: 'none', letterSpacing: 0 }}>{c.label}</th>)}</tr></thead>
          <tbody>{data.map((r, i) => <tr key={i}>{columns.map((c) => <td key={c.key} style={typeof r[c.key] === 'number' ? { fontFamily: FONT.mono, textAlign: 'right' } : undefined}>{cell(c.format, r[c.key])}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <ExcludedNote count={nMarked} />
    </ChartFrame>
  )
}
