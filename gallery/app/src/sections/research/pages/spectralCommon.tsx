/**
 * What the spectral and calibration research pages share: their export
 * (public/data/research/spectral/<key>.json, gallery/data/export_research_spectral.py),
 * the calibration explorer's per-filter readouts the gallery already has
 * (calibration_runs.json), and the intro panel that says what each page reproduces,
 * where every table was read from, and what a Drive re-run would add.
 */
import { type ReactNode } from 'react'
import { ChartFrame } from '@/components/ChartFrame'
import { ExportState } from '@/components/ExportState'
import { useCalibrationRuns } from '@/lib/calibrationFiles'
import { useExport } from '@/lib/data'
import { Markdown } from '@/lib/markdown'
import { usePrefs } from '@/lib/prefs'
import { rowsOf, type Row, type Table } from '@/lib/table'

export interface SpectralSource { table: string; rows: number; notebook: string; read: string; path: string; cell?: string; source_export_generated?: string }
export interface SpectralMissing { table: string; what: string; needs: string; run: string; after?: string | null }
export interface SpectralExport {
  schema_version: number
  id: string
  title: string
  generated: string
  generated_from: string
  /** where the export sits under public/data (older exports: research/spectral/<id>.json) */
  file?: string
  text: Record<string, string>
  sources: SpectralSource[]
  missing: SpectralMissing[]
  tables: Record<string, Table>
}

export const EXPORTER = 'gallery/data/export_research_spectral.py'

export function useSpectral(key: string, folder = 'research/spectral') {
  return useExport<SpectralExport>(`${folder}/${key}.json`, (d) => (d.schema_version === 1 ? null : 'unsupported data version'))
}

/** A table of the export as rows (empty when the export lacks it). */
export const rows = (data: SpectralExport | null, name: string): Row[] => rowsOf(data?.tables[name])

/** The explorer readouts, through the same loader as the Calibration tab (lib/calibrationFiles). */
export function useCalibRuns() {
  const { data, error } = useCalibrationRuns()
  return { runs: data, error }
}

/** The provenance line: the notebooks a chart stands in for, and where its numbers were read. */
export function provenance(data: SpectralExport | null, tables: string[], extra?: string): ReactNode {
  const used = (data?.sources ?? []).filter((s) => tables.includes(s.table))
  const reads = [...new Set(used.map((s) => `${s.path}${s.read === 'csv' || s.read === 'tracked csv' ? '' : ` (${s.read}${s.source_export_generated ? `; saved ${s.source_export_generated.slice(0, 10)}` : ''})`}`))]
  return <>
    {reads.length > 0 && <>Read from {reads.join('; ')}. </>}
    {extra && <>{extra} </>}
    {data && <>Export: <code>{data.generated_from || EXPORTER}</code> → <code>{data.file ?? `research/spectral/${data.id}.json`}</code>.</>}
  </>
}

/** Loading and error states for a page that needs its export (and optionally the explorer readouts): components/ExportState with the exporter's command. */
export function Waiting({ error, what = 'the page data', exporter = EXPORTER }: { error?: string; what?: string; exporter?: string }) {
  return <ExportState what={what} error={error} command={`uv run --locked --no-sync python ${exporter}`} />
}

/**
 * The first panel of each page: what it brings in, from which notebooks, the headline read off
 * the same tables the charts draw, and the per-filter tables that only a re-run on the machine
 * with the Drive (or the notebooks' output folders) can add. Shown only with Explain on (lib/prefs).
 */
export function PageIntro({ id, title, notebooks, data, children }: {
  id: string; title: string; notebooks: string; data: SpectralExport; children?: ReactNode
}) {
  // a page's intro (headline numbers, method, what is not exported yet) is reading guidance: only with Explain on
  const { explain } = usePrefs()
  const reads = [...new Set(data.sources.map((s) => s.read))]
  if (!explain) return null
  return (
    <ChartFrame id={id} title={title} exportable={false}
      provenance={<>Notebooks: {notebooks}. Tables read as: {reads.join(', ')}. Exported {data.generated.slice(0, 10)} by <code>{data.generated_from || EXPORTER}</code>.</>}>
      <div className="subtitle">
        {Object.values(data.text).map((t, i) => <Markdown key={i} text={t} />)}
        {children}
        {data.missing.length > 0 && <>
          <p style={{ marginBottom: 4 }}><strong>Not on this page yet</strong> — these need the notebooks' gitignored outputs (spectra on the Drive). Run, then re-export:</p>
          <ul style={{ marginTop: 0 }}>
            {data.missing.map((m) => <li key={m.table}>{m.what}: <code>{m.run}</code> (writes <code>{m.needs}</code>), then <code>uv run --locked --no-sync python {data.generated_from || EXPORTER}</code>.{m.after && <> First it needs {m.after}.</>}</li>)}
          </ul>
        </>}
      </div>
    </ChartFrame>
  )
}

/** A short label for a calibration_runs preset (its exported label, preprocessing in reader-facing words). */
export { presetName } from '@/charts/calibration/common'
