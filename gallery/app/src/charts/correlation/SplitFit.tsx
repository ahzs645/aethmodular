import { useMemo, useState } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Crossplot, sharedCrossDomains, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { FieldSelect } from '@/components/FieldSelect'
import { Key } from '@/components/Key'
import { Marker, RefLine } from '@/components/RefLines'
import { robustDomain, sameUnits } from '@/lib/axes'
import { coef, f2, fmt } from '@/lib/format'
import { colorsFor } from '@/lib/groupings'
import { FIT_STYLE, useRegressions } from '@/lib/regressionPrefs'
import { medianOf, parseCuts, quantileSorted, residuals, roundCut, siteMedianCuts, splitAt, splitTable, splitValue, type ResidualKind, type SplitItem, type SplitRow } from '@/lib/splitFit'
import { pearson, regression, spearman } from '@/lib/stats'
import { INK, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { ExcludedNote, useMarkedPoints } from '@/lib/markedRows'

const [LOW, HIGH] = colorsFor(2)
const CUT_MODES = ['value', 'median', 'site'] as const
type CutMode = (typeof CUT_MODES)[number]
const CUT_LABEL: Record<CutMode, string> = { value: 'A value', median: 'The median', site: "Each site's median" }
const RESIDUALS = ['auto', 'difference', 'fit'] as const
const PANELS = ['Pooled', 'Per site'] as const
const NONE = 'none'
/** the notebooks' case: iron over FTIR EC, cut at IRON_EC_THRESHOLDS = 50 … 300 ng/µg (iron here in µg/m³, so ÷ 1000) */
const FE = 'Iron', EC = 'EC (FTIR)'
const FE_EC_CUT = 0.1
const FE_EC_TABLE = '0.05, 0.1, 0.15, 0.2, 0.3'

interface Item extends SplitItem { row: FilterRow }

/**
 * Split at a cutoff, fit each side, residual against the splitting variable — the iron/EC threshold analysis that
 * primary_tasks_notebook (Task 1.2: 50 … 300 ng Fe per µg EC), Task_Analysis_Notebook (c34–38, the same by
 * concentration group) and follow_up_analysis (c24, at each site's median) ran repeatedly, made generic: x and y are
 * the tab's field pickers, the splitting variable any field (or one field over another), the fits the sidebar's.
 *
 * Each row of panels is ≤ cutoff, > cutoff, and every filter's residual (y − x, the notebooks') against the variable,
 * with the cutoff marked. The table under it is the notebooks' summary: each side's n, R², slope and ΔR² at a list of
 * cutoffs. Per site draws one row per site, which with "each site's median" is follow_up_analysis's split; the
 * notebooks' concentration groups are the Filters panel's sites (Delhi + Addis, Beijing + JPL).
 */
export function SplitFit({ rows, meta, xField, yField }: { rows: FilterRow[]; meta: MetaFile; xField: string; yField: string }) {
  const has = (f: string) => rows.some((r) => typeof r[f] === 'number')
  const [onParam, setOn] = useSearchParam<string>('splitfit.on', '')
  const [perParam, setPer] = useSearchParam<string>('splitfit.per', '')
  const [mode, setMode] = useSearchParam<CutMode>('splitfit.cut', 'value', CUT_MODES)
  const [atParam, setAt] = useSearchParam<string>('splitfit.at', '')
  const [resParam, setRes] = useSearchParam<(typeof RESIDUALS)[number]>('splitfit.resid', 'auto', RESIDUALS)
  const [panelsParam, setPanels] = useSearchParam<(typeof PANELS)[number]>('splitfit.panels', 'Pooled', PANELS)
  const [tableParam, setTable] = useSearchParam<string>('splitfit.table', '')
  const { methods } = useRegressions()

  // the variable: the reader's, else iron (over FTIR EC) as the notebooks had it, else the first field in view
  const on = onParam || (has(FE) ? FE : meta.fields.find(has) ?? meta.fields[0] ?? '')
  const per = perParam === NONE ? null : perParam || (on === FE && has(EC) ? EC : null)
  const feEc = on === FE && per === EC
  const tName = per ? `${on} / ${per}` : on
  const tUnit = per ? (sameUnits(meta.field_units?.[on], meta.field_units?.[per]) ? '' : `${meta.field_units?.[on] ?? '?'} per ${meta.field_units?.[per] ?? '?'}`) : meta.field_units?.[on] ?? ''
  const tLabel = tUnit ? `${tName} (${tUnit.replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹')})` : tName
  const units = sameUnits(meta.field_units?.[xField], meta.field_units?.[yField])
  const resKind: ResidualKind = resParam === 'auto' ? (units ? 'difference' : 'fit') : resParam

  // every filter with x, y and the variable (the notebooks' valid_mask)
  const items = useMemo<Item[]>(() => rows.flatMap((r) => {
    const x = r[xField], y = r[yField], t = splitValue(r, on, per)
    return typeof x === 'number' && Number.isFinite(x) && typeof y === 'number' && Number.isFinite(y) && t !== null ? [{ x, y, t, site: r.site, row: r }] : []
  }), [rows, xField, yField, on, per])
  // Filters → excluded: Mark (lib/markedRows): the excluded filters as items too, drawn as × on the ≤ / > panels only
  const markedItems = useMarkedPoints<Item>((rs) => rs.flatMap((r) => {
    const x = r[xField], y = r[yField], t = splitValue(r, on, per)
    return typeof x === 'number' && Number.isFinite(x) && typeof y === 'number' && Number.isFinite(y) && t !== null ? [{ x, y, t, site: r.site, row: r }] : []
  }), [xField, yField, on, per])
  const sites = useMemo(() => meta.sites.map((s) => s.name).filter((s) => items.some((i) => i.site === s)), [meta.sites, items])
  const perSite = panelsParam === 'Per site' && sites.length > 1

  // the cutoff: typed (else the notebooks' 0.1 for Fe/EC, else the median rounded), the pooled median, or each site's
  const autoAt = feEc ? FE_EC_CUT : roundCut(medianOf(items.map((i) => i.t)))
  const typed = atParam.trim() === '' ? NaN : Number(atParam)
  const at = Number.isFinite(typed) ? typed : autoAt
  // a cutoff being typed (null: show the one in use)
  const [atDraft, setAtDraft] = useState<string | null>(null)
  const siteCuts = useMemo(() => siteMedianCuts(items), [items])
  const groups = useMemo(() => (perSite ? sites.map((s) => ({ name: s, items: items.filter((i) => i.site === s) })) : [{ name: 'All sites in view', items }]), [perSite, sites, items])
  const cutFor = (g: { items: Item[] }) => (it: Item) => mode === 'site' ? siteCuts.get(it.site) ?? NaN : mode === 'median' ? medianOf(g.items.map((i) => i.t)) : at
  // the cutoff a group's residual panel marks: one value, or none when its sites are cut at their own medians
  const markOf = (g: { name: string; items: Item[] }) => (mode === 'value' ? at : mode === 'median' ? medianOf(g.items.map((i) => i.t)) : perSite ? siteCuts.get(g.name) ?? null : sites.length === 1 ? siteCuts.get(sites[0]) ?? null : null)

  // the summary table's cutoffs: typed, else the notebooks' for Fe/EC, else the quartiles of the variable
  const tableCuts = useMemo(() => {
    const own = parseCuts(tableParam)
    if (own.length) return own
    if (feEc) return parseCuts(FE_EC_TABLE)
    const s = items.map((i) => i.t).sort((a, b) => a - b)
    return [...new Set([0.25, 0.5, 0.75].map((p) => roundCut(quantileSorted(s, p))))].filter(Number.isFinite)
  }, [tableParam, feEc, items])
  const [tableDraft, setTableDraft] = useState<string | null>(null)
  const tables = useMemo(() => groups.map((g) => ({ name: g.name, rows: splitTable(g.items, tableCuts, { errorsInVariables: true }) })), [groups, tableCuts])
  const nonOls = methods.find((m) => m !== 'ols')

  const tipOf = (it: Item, res: number | null): string[] => [it.row.id, `${it.site} · ${it.row.date}`, `${xField}: ${f2(it.x)}`, `${yField}: ${f2(it.y)}`,
    `${tName}: ${fmt(it.t, 3)}`, ...(res === null ? [] : [`residual: ${f2(res)}`])]
  const pointOf = (it: Item, color: string, x: number, y: number, res: number | null): CrossPoint => ({
    x, y, color, id: it.row.id, site: it.site, season: it.row.season, tip: tipOf(it, res),
    rec: { id: it.row.id, source: `Correlation · split at a cutoff on ${tName}`, site: it.site, date: it.row.date,
      fields: [[xField, it.x], [yField, it.y], [tName, +it.t.toPrecision(4)], ...(res === null ? [] : [['residual', +res.toFixed(3)] as [string, number]])] },
  })

  const blocks = useMemo(() => groups.map((g) => {
    const cut = cutFor(g)
    const { low, high } = splitAt(g.items, cut)
    const res = residuals(g.items, resKind)
    const resOf = new Map(g.items.map((it, i) => [it, res?.[i] ?? null]))
    const side = (it: Item) => (it.t <= cut(it) ? LOW : HIGH)
    const lowPts = low.map((it) => pointOf(it, LOW, it.x, it.y, resOf.get(it) ?? null))
    const highPts = high.map((it) => pointOf(it, HIGH, it.x, it.y, resOf.get(it) ?? null))
    const resPts = res ? g.items.flatMap((it, i) => (Number.isFinite(cut(it)) ? [pointOf(it, side(it), it.t, res[i], res[i])] : [])) : []
    const r = res ? pearson(g.items.map((i) => i.t), res) : null
    // Spearman beside it: a ratio with a near-zero denominator (Beijing's Fe/EC reaches ±10⁵) swamps Pearson's r
    const rho = res ? spearman(g.items.map((i) => i.t), res) : null
    // the residual's trend is an OLS of residual on the variable (the notebooks' Pearson r), whatever the sidebar fits
    const trend = res ? regression(g.items.map((i) => i.t), res) : null
    // the group's excluded filters, each side of the same cutoff (a cutoff never moves for them)
    const own = markedItems.filter((it) => !perSite || it.site === g.name)
    const markedLow = own.filter((it) => Number.isFinite(cut(it)) && it.t <= cut(it)).map((it) => pointOf(it, LOW, it.x, it.y, null))
    const markedHigh = own.filter((it) => Number.isFinite(cut(it)) && it.t > cut(it)).map((it) => pointOf(it, HIGH, it.x, it.y, null))
    return { name: g.name, n: g.items.length, low: lowPts, high: highPts, res: resPts, r, rho, trend, mark: markOf(g), markedLow, markedHigh }
  }), [groups, mode, at, siteCuts, resKind, xField, yField, tName, perSite, sites, markedItems]) // eslint-disable-line react-hooks/exhaustive-deps

  // one range for every ≤ / > panel, so the slopes compare by eye; the residual panels clip the variable's 1–99 %
  const shared = sharedCrossDomains(blocks.flatMap((b) => [b.low, b.high]), units)
  const tDom = robustDomain(items.map((i) => i.t), 0.01, 0.99).domain
  const resDom = robustDomain(blocks.flatMap((b) => b.res.map((p) => p.y)), 0.01, 0.99, { include: [0] }).domain
  const pad = (d: [number, number]): [number, number] => { const w = (d[1] - d[0]) || 1; return [d[0] - 0.04 * w, d[1] + 0.04 * w] }
  const outside = items.filter((i) => i.t < tDom[0] || i.t > tDom[1]).length
  const xl = withUnit(xField, meta.field_units), yl = withUnit(yField, meta.field_units)
  const resLabel = resKind === 'difference' ? `${yField} − ${xField}` : `${yField} − pooled OLS fit`
  const cutText = (b: { mark: number | null }) => (b.mark === null ? "each site's median" : fmt(b.mark, 3))
  const cutWord = mode === 'value' ? fmt(at, 3) : mode === 'median' ? 'the median' : "each site's median"

  const panelsOf = (b: (typeof blocks)[number]) => [
    { key: 'low', title: `${tName} ≤ ${cutText(b)}`, points: b.low, marked: b.markedLow },
    { key: 'high', title: `${tName} > ${cutText(b)}`, points: b.high, marked: b.markedHigh },
    { key: 'res', title: `Residual · r = ${fmt(b.r, 3)}, ρ = ${fmt(b.rho, 3)} · n = ${b.res.length}`, points: b.res, marked: undefined },
  ]

  return (
    <ChartFrame id="split-fit"
      title={`Split at a cutoff — ${yField} against ${xField}, fitted either side of ${tName} = ${cutWord}`}
      subtitle={`x and y are the tab's fields; the filters are split on ${tName} (any field, or one field over another), low ≤ the cutoff, high above it, and each side is fitted with the sidebar's estimators (a side needs three filters). The third panel is every filter's ${resKind === 'difference' ? 'difference y − x, as the notebooks plotted it' : 'distance from the pooled OLS line (y − x means little when x and y differ in units)'} against the variable, coloured by side, with its OLS trend, Pearson r (the notebooks') and Spearman ρ (which one extreme ratio cannot swamp): a trend says the two measurements part ways as the variable grows. The table is the notebooks' summary at several cutoffs (OLS R² and slope; ΔR² = R²(low) − R²(high)).${feEc ? ' Iron is exported in µg/m³, so Fe/EC here is the notebooks\' ng/µg ÷ 1000: their thresholds 50, 100, 150, 200, 300 are 0.05 … 0.3.' : ''} HIPS BC follows the MAC; the notebooks used MAC 10.`}
      provenance={'research/ftir_hips_chem/primary_tasks_notebook.ipynb c14–21 (analyze_iron_threshold) · Task_Analysis_Notebook.ipynb c32–39 (by concentration group: pick Delhi + Addis Ababa, or Beijing + JPL, in the Filters panel) · follow_up_analysis.ipynb c24 (split at each site\'s median, Per site) · dominant_source_threshold_analysis follows the same recipe on the PMF share (Aethalometer › Sources)'}
      controls={<>
        <FieldSelect label="Split on" value={on} meta={meta} onChange={setOn} title="The variable the filters are split on" />
        <FieldSelect label="÷" value={per ?? NONE} meta={meta} onChange={setPer} extraOptions={[NONE]} ratios={false}
          title="Divide the variable by another field (iron ÷ FTIR EC is the notebooks' Fe/EC); none: the field itself" />
        <Segmented label="Cut at" value={mode} options={CUT_MODES} optionLabel={(m) => CUT_LABEL[m]} onChange={setMode}
          title="A typed value; the median of the filters in each row of panels; or each site at its own median (follow_up_analysis)" />
        {mode === 'value' && <label className="control" title="The cutoff on the variable; a filter at the cutoff is on the low side">
          Cutoff
          <input className="select" type="number" step="any" style={{ width: 90 }} value={atDraft ?? String(at)} aria-label="Cutoff"
            onChange={(e) => setAtDraft(e.target.value)} onBlur={() => { if (atDraft !== null) setAt(atDraft.trim()); setAtDraft(null) }}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} />
        </label>}
        <Segmented label="Residual" value={resParam} options={RESIDUALS} onChange={setRes}
          optionLabel={(o) => (o === 'auto' ? 'Auto' : o === 'difference' ? 'y − x' : 'From the pooled fit')}
          title="y − x (the notebooks'), or the distance from the OLS line through every filter; Auto picks y − x when x and y share a unit" />
        {sites.length > 1 && <Segmented label="Panels" value={panelsParam} options={PANELS} onChange={setPanels} title="Every site in view pooled, or one row of panels per site" />}
        <label className="control" title="The cutoffs the summary table lists: commas or spaces between them">
          Table at
          <input className="select" style={{ width: 150 }} value={tableDraft ?? tableCuts.join(', ')} aria-label="Table cutoffs"
            onChange={(e) => setTableDraft(e.target.value)} onBlur={() => { if (tableDraft !== null) setTable(parseCuts(tableDraft).join(', ')); setTableDraft(null) }}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} />
        </label>
      </>}>
      {items.length < 6 ? <Empty>Too few filters in view with {xField}, {yField} and {tName} ({items.length}).</Empty> : <>
        {blocks.map((b) => (
          <div key={b.name} style={{ marginBottom: 12 }}>
            {(perSite || sites.length > 1) && <p className="facet-title">{b.name}<span style={{ color: 'var(--ink-muted)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 11 }}> n={b.n}</span></p>}
            <FacetGrid items={panelsOf(b)} minCell={250} maxCols={3} maxCell={380} render={(p, w) => p.key === 'res'
              ? (p.points.length < 3 ? <Empty>Too few filters for the residual.</Empty>
                : <Crossplot points={p.points} title={p.title} xLabel={tLabel} yLabel={resLabel} sameScale={false} grouping="none" maxSide={w - 70}
                  domain={pad(tDom)} yDomain={pad(resDom)}
                  extraLines={b.trend ? [{ slope: b.trend.slope, intercept: b.trend.intercept, color: INK.text, label: `OLS trend, R² ${coef(b.trend.r2)}`, method: 'ols', stats: b.trend }] : []}
                  background={(x, y) => <>
                    <RefLine orient="h" at={y(0)} from={0} to={x.range()[1]} />
                    {b.mark !== null && <Marker orient="v" at={x(b.mark)} from={0} to={y.range()[0]} label={`cutoff ${fmt(b.mark, 3)}`} color={INK.text} />}
                  </>} />)
              : (p.points.length === 0 ? <Empty>No filters on this side.</Empty>
                : <Crossplot points={p.points} marked={p.marked} title={`${p.title} · n = ${p.points.length}`} xLabel={xl} yLabel={yl} sameScale={units} grouping="pooled" maxSide={w - 70}
                  domain={shared.domain} yDomain={shared.yDomain} />)} />
          </div>
        ))}
        <Key items={[{ label: `low: ${tName} ≤ cutoff`, color: LOW }, { label: `high: ${tName} > cutoff`, color: HIGH }]} />
        <ExcludedNote fields={[xField, yField]} drawn />
        {outside > 0 && <p className="chart-note">{outside} of {items.length} filters have {tName} outside its 1–99 % range and fall outside the residual panels (they are in the fits and the table).</p>}
        <SplitTables tables={tables} tName={tName} nonOls={nonOls} />
      </>}
    </ChartFrame>
  )
}

/** the notebooks' threshold summary, one table per row of panels */
function SplitTables({ tables, tName, nonOls }: { tables: { name: string; rows: SplitRow[] }[]; tName: string; nonOls?: 'deming' | 'york' }) {
  const slopeOf = (s: SplitRow['low']) => (nonOls === 'deming' ? s?.demingSlope : nonOls === 'york' ? s?.yorkSlope ?? s?.demingSlope : null)
  const extra = nonOls ? FIT_STYLE[nonOls].label : null
  return <div style={{ overflowX: 'auto', marginTop: 8 }}>
    {tables.map((t) => (
      <table key={t.name} className="census-table" style={{ marginBottom: 10 }}>
        <thead>
          {tables.length > 1 && <tr><th colSpan={extra ? 10 : 8} style={{ textAlign: 'left' }}>{t.name}</th></tr>}
          <tr><th>{tName} cutoff</th><th>low n</th><th>low R²</th><th>low slope</th>{extra && <th>low {extra}</th>}<th>high n</th><th>high R²</th><th>high slope</th>{extra && <th>high {extra}</th>}<th>ΔR²</th></tr>
        </thead>
        <tbody>
          {t.rows.map((r) => (
            <tr key={r.cut}>
              <td>{fmt(r.cut, 3)}</td>
              <td>{r.nLow}</td><td>{r.low ? coef(r.low.r2) : '—'}</td><td>{r.low ? coef(r.low.slope) : '—'}</td>{extra && <td>{r.low ? coef(slopeOf(r.low)) : '—'}</td>}
              <td>{r.nHigh}</td><td>{r.high ? coef(r.high.r2) : '—'}</td><td>{r.high ? coef(r.high.slope) : '—'}</td>{extra && <td>{r.high ? coef(slopeOf(r.high)) : '—'}</td>}
              <td>{r.dR2 === null ? '—' : `${r.dR2 >= 0 ? '+' : '−'}${coef(Math.abs(r.dR2))}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    ))}
    <p className="chart-note">R² and slope are OLS (scipy.stats.linregress, as the notebooks printed){extra ? `; the ${extra} column follows the sidebar's estimator` : ''}. A side with fewer than three filters has no fit.</p>
  </div>
}
