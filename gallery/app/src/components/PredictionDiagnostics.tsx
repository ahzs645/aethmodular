import { useMemo } from 'react'
import { SavedFigure, type ExportedTables } from '@/components/SavedFigure'
import { residualRows } from '@/lib/predictionDiagnostics'
import { rowsOf } from '@/lib/table'
import { ANALYSIS_SEASON_COLOR } from '@/lib/seasons'

/** Shared by saved protocol diagnostics and any other per-filter prediction export. */
export function PredictionDiagnostics({ bundle, table, id, mac, filterIdColumn = 'ExternalFilterId', choose = [], domain = true }: {
  bundle: ExportedTables; table: string; id: string; mac: number; filterIdColumn?: string
  choose?: { column: string; label?: string; default?: string }[]
  domain?: boolean
}) {
  const prepared = useMemo(() => ({ ...bundle, tables: { ...bundle.tables,
    ...(table in bundle.tables ? { [table]: residualRows(rowsOf(bundle.tables[table]), mac) } : {}),
  } }), [bundle, table, mac])
  const common = { bundle: prepared, table, filterIdColumn, choose }
  return <>
    <SavedFigure {...common} kind="crossplot" id={`${id}-agreement`} x="observed" y="predicted" color="season" colors={ANALYSIS_SEASON_COLOR}
      xLabel={`Observed HIPS Fabs/${mac} (µg/m³)`} yLabel="Predicted EC (µg/m³)" title="Per-filter optical agreement" />
    <SavedFigure {...common} kind="scatter" id={`${id}-loading`} x="observed" y="residual" color="season" refY={[0]} colors={ANALYSIS_SEASON_COLOR}
      xLabel={`HIPS Fabs/${mac} (µg/m³)`} yLabel="Prediction − optical BC (µg/m³)" title="Prediction residual versus loading" />
    {domain && <SavedFigure {...common} kind="scatter" id={`${id}-distance`} x="D2" y="residual" color="season" refY={[0]} colors={ANALYSIS_SEASON_COLOR}
      xLabel="Squared score-space distance D²" yLabel="Prediction − optical BC (µg/m³)" title="Prediction residual versus domain distance"
      variants={[{ key: 'distance', label: 'Domain diagnostic', choices: [{ value: 'D2', label: 'D²' }, { value: 'Q', patch: { x: 'Q', xLabel: 'Spectral reconstruction residual Q' } }] }]} />}
  </>
}
