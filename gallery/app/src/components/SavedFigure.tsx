/** Saved-table figures share availability, exclusions and provenance; Figure owns all rendering. */
import { useMemo } from 'react'
import { Figure, type FigureSpec } from '@/components/Figure'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { baseFilterId } from '@/lib/highlight'
import { mediaKey, useEtadByMedia } from '@/lib/etadMedia'
import { ExcludedNote, useExclusionSplit } from '@/lib/markedRows'
import { rowsOf, type Table } from '@/lib/table'

export interface ExportedTables {
  tables: Record<string, Table>
  sources: { table: string; path: string; read: string; source_export_generated?: string }[]
}
type SavedSpec<T> = T extends FigureSpec ? Omit<T, 'data' | 'markedData' | 'table'> & {
  bundle: ExportedTables; table: string
  /** Physical filter IDs or MediaIds; omit for fitted summaries/reference spectra. */
  filterIdColumn?: string
} : never

export function SavedFigure({ bundle, table, filterIdColumn, ...spec }: SavedSpec<FigureSpec>) {
  const byMedia = useEtadByMedia(filterIdColumn === 'MediaId')
  const all = useMemo(() => rowsOf(bundle.tables[table]).map((r) => {
    const raw = filterIdColumn ? r[filterIdColumn] : null
    const id = raw == null || raw === '' ? null : filterIdColumn === 'MediaId' ? byMedia?.get(mediaKey(raw)) ?? null : baseFilterId(String(raw))
    return { ...r, __filterId: id }
  }), [bundle.tables[table], filterIdColumn, byMedia])
  const split = useExclusionSplit(all, (r) => r.__filterId)
  const sources = bundle.sources.filter((s) => s.table === table)
  const provenance = sources.map((s) => `${s.path} (${s.read}${s.source_export_generated ? `; saved ${s.source_export_generated.slice(0, 10)}` : ''})`).join('; ')
  if (!(table in bundle.tables)) return <ChartFrame id={spec.id} title={spec.title} exportable={false} provenance={provenance}>
    <Empty>The saved results for this comparison are not available in this data export.</Empty>
  </ChartFrame>
  const identity = spec.kind === 'scatter' || spec.kind === 'crossplot' ? { idColumn: '__filterId' } : {}
  return <Figure {...spec} {...identity} data={split.counted} markedData={split.marked} provenance={provenance}
    aside={filterIdColumn ? <ExcludedNote count={split.nOut} drawn={spec.kind === 'scatter' || spec.kind === 'crossplot'} /> : undefined} />
}
