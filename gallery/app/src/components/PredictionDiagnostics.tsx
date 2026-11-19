import { useMemo } from 'react'
import { SavedFigure, type ExportedTables } from '@/components/SavedFigure'
import { useChartOptions, type OptionDef } from '@/lib/chartOptions'
import { SidebarControls } from '@/lib/pageSidebar'
import { residualRows } from '@/lib/predictionDiagnostics'
import { rowsOf, uniq } from '@/lib/table'
import { ANALYSIS_SEASON_COLOR } from '@/lib/seasons'

/**
 * Shared by saved protocol diagnostics and any other per-filter prediction export: one table of predictions drawn
 * three ways (agreement, residual against loading, residual against domain distance). The `choose` columns pick
 * which rows of the table all three charts draw, so they are one choice for the group, in Page options under
 * `group` (URL `${id}.${column}`), not one per chart; each chart names the group and the choice in its options row.
 */
export function PredictionDiagnostics({ bundle, table, id, mac, group, filterIdColumn = 'ExternalFilterId', choose = [], domain = true }: {
  bundle: ExportedTables; table: string; id: string; mac: number
  /** what this table of predictions is, as the reader should see it named in Page options and on each chart */
  group: string
  filterIdColumn?: string
  choose?: { column: string; label?: string; default?: string }[]
  domain?: boolean
}) {
  const all = useMemo(() => (table in bundle.tables ? rowsOf(bundle.tables[table]) : []), [bundle, table])
  // the choice was each chart's own until 19 Nov 2026 (`${id}-agreement.${column}` …): a link that set one still opens on it
  const o = useChartOptions(id, choose.map((c): OptionDef => {
    const options = uniq(all, c.column).map(String)
    return { key: c.column, label: c.label ?? c.column, persist: true, default: c.default ?? options[0] ?? '', style: options.length > 4 ? 'select' : 'segmented', choices: options,
      legacy: ['agreement', 'loading', 'distance'].map((chart) => `${id}-${chart}.${c.column}`) }
  }))
  const picked = choose.map((c) => String(o.values[c.column])).join('|')
  const prepared = useMemo(() => ({ ...bundle, tables: { ...bundle.tables,
    ...(table in bundle.tables ? { [table]: residualRows(all.filter((r) => choose.every((c) => String(r[c.column]) === String(o.values[c.column]))), mac) } : {}),
  } }), [bundle, table, all, picked, mac]) // eslint-disable-line react-hooks/exhaustive-deps
  const controls = o.controls()
  // only the columns that are a choice (more than one value in the table) are worth naming under the title
  const chosen = choose.filter((c) => uniq(all, c.column).length > 1).map((c) => `${c.label ?? c.column}: ${o.values[c.column]}`)
  // where each chart's own copy of the control used to be: which predictions it is drawing, always in view
  const lead = <span className="control hint" title={controls ? 'One choice for the three charts of this group, in Page options' : undefined}>{[group, ...chosen].join(' · ')}</span>
  const common = { bundle: prepared, table, filterIdColumn, lead }
  return <>
    {controls && <SidebarControls title={group}>{controls}</SidebarControls>}
    <SavedFigure {...common} kind="crossplot" id={`${id}-agreement`} x="observed" y="predicted" color="season" colors={ANALYSIS_SEASON_COLOR}
      xLabel={`Observed HIPS Fabs/${mac} (µg/m³)`} yLabel="Predicted EC (µg/m³)" title="Per-filter optical agreement" />
    <SavedFigure {...common} kind="scatter" id={`${id}-loading`} x="observed" y="residual" color="season" refY={[0]} colors={ANALYSIS_SEASON_COLOR}
      xLabel={`HIPS Fabs/${mac} (µg/m³)`} yLabel="Prediction − optical BC (µg/m³)" title="Prediction residual versus loading" />
    {domain && <SavedFigure {...common} kind="scatter" id={`${id}-distance`} x="D2" y="residual" color="season" refY={[0]} colors={ANALYSIS_SEASON_COLOR}
      xLabel="Squared score-space distance D²" yLabel="Prediction − optical BC (µg/m³)" title="Prediction residual versus domain distance"
      variants={[{ key: 'distance', label: 'Domain diagnostic', choices: [{ value: 'D2', label: 'D²' }, { value: 'Q', patch: { x: 'Q', xLabel: 'Spectral reconstruction residual Q' } }] }]} />}
  </>
}
