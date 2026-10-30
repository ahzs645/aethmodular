import { useCallback, useMemo, useRef, useState } from 'react'
import type { IChart, IMutField, IRow } from '@kanaries/graphic-walker'
import type { FilterRow, MetaFile } from '@/lib/types'
import { builderDraft, builderFields, builderRows, builderScales, builderSchema, readBuilderDraft } from '@/lib/chartBuilder'
import { downloadBlob } from '@/lib/export'
import { useTheme } from '@/lib/colorTheme'
import { Empty } from '@/components/ChartFrame'
import { GraphicWalkerAdapter, type ChartBuilderEngine } from './GraphicWalkerAdapter'

// Chart definitions survive tab switches, but measurements are never stored here. Separate
// schemas (including units) get separate drafts so a mass view cannot relabel a concentration chart.
const sessionDrafts = new Map<string, IChart[]>()

export default function GraphicWalkerWorkspace({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const fields = useMemo(() => builderFields(meta), [meta])
  const data = useMemo(() => builderRows(rows, meta), [rows, meta])
  const schema = builderSchema(fields)
  return <Editor key={schema} rows={rows} meta={meta} fields={fields} data={data} schema={schema} />
}

function Editor({ rows, meta, fields, data, schema }: { rows: FilterRow[]; meta: MetaFile; fields: IMutField[]; data: IRow[]; schema: string }) {
  const theme = useTheme()
  const scales = useMemo(() => builderScales(meta), [meta])
  const engine = useRef<ChartBuilderEngine | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const [initialCharts] = useState(() => sessionDrafts.get(schema))
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const saveSession = useCallback((charts: IChart[]) => {
    sessionDrafts.delete(schema)
    sessionDrafts.set(schema, charts)
    if (sessionDrafts.size > 8) sessionDrafts.delete(sessionDrafts.keys().next().value!)
  }, [schema])
  const reportError = useCallback((cause: Error) => setError(cause.message), [])
  const download = () => {
    setError('')
    try {
      if (!engine.current) throw new Error('Wait for the chart builder to finish opening, then retry.')
      const draft = builderDraft(engine.current.exportCharts(), fields, rows, meta, window.location.hash)
      downloadBlob(new Blob([JSON.stringify(draft, null, 2)], { type: 'application/json' }), 'air-gallery-chart-draft.json')
      setMessage('Draft downloaded. Includes chart definitions and filter IDs; measurement rows stay in the gallery.')
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not download this draft.') }
  }
  const importDraft = async (file: File) => {
    setError(''); setMessage('')
    try {
      if (file.size > 5_000_000) throw new Error('Choose a chart draft smaller than 5 MB.')
      const draft = readBuilderDraft(JSON.parse(await file.text()), fields)
      if (!engine.current) throw new Error('Wait for the chart builder to finish opening, then retry.')
      engine.current.importCharts(draft.charts)
      setMessage(`Loaded ${draft.charts.length} chart definition(s) on the current ${rows.length} filters. The draft originally used ${draft.context.filterIds.length} filters. Gallery filters and measurements were not changed.`)
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not read this draft.') }
  }
  return <>
    <div className="chart-builder-actions">
      <button type="button" className="btn" onClick={download}>Download draft</button>
      <button type="button" className="btn" onClick={() => input.current?.click()}>Import draft</button>
      <input ref={input} type="file" accept=".json,application/json" hidden onChange={event => {
        const file = event.target.files?.[0]
        if (file) void importDraft(file)
        event.target.value = ''
      }} />
      <span className="chart-builder-note">Download to keep your draft after reloading.</span>
    </div>
    {message && <p className="chart-builder-message" role="status">{message}</p>}
    {error && <p className="chart-builder-message" role="alert">{error}</p>}
    {!rows.length && <Empty>No filters match this selection. Adjust the Filters panel to explore data.</Empty>}
    <div className="chart-builder-scroll" style={!rows.length ? { display: 'none' } : undefined}>
      <GraphicWalkerAdapter ref={engine} data={data} fields={fields} charts={initialCharts}
        theme={theme} scales={scales} onError={reportError} onDispose={saveSession} />
    </div>
  </>
}
