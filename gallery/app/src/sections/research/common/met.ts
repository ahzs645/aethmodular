/**
 * The Addis station's day on each filter's sample date (gallery/data/export_research_addis_met.py), the way the
 * meteorology notebooks joined it, added to the rows and the field list of the research pages that offer it,
 * beside the gallery's own window means. Absent the export, rows and meta pass through unchanged.
 */
import { useMemo } from 'react'
import { useExport } from '@/lib/data'
import type { FilterRow, MetaFile } from '@/lib/types'

interface MetFile { fields: string[]; field_units: Record<string, string>; rows: Record<string, (number | null)[]> }
export const SAMPLE_DAY_GROUP = 'Addis station, sample day (notebooks)'

export function useAddisMet(rows: FilterRow[], meta: MetaFile): { rows: FilterRow[]; meta: MetaFile; fields: string[] } {
  const data = useExport<MetFile>('research/addis_met/filters.json').data
  return useMemo(() => {
    if (!data) return { rows, meta, fields: [] }
    const out = rows.map((r) => { const v = data.rows[r.id]; return v ? { ...r, ...Object.fromEntries(data.fields.map((f, i) => [f, v[i]])) } : r })
    return { rows: out, fields: data.fields, meta: { ...meta, fields: [...meta.fields, ...data.fields], field_units: { ...meta.field_units, ...data.field_units },
      field_groups: [...meta.field_groups, { label: SAMPLE_DAY_GROUP, fields: data.fields }] } }
  }, [rows, meta, data])
}
