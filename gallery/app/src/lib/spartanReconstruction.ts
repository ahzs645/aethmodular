/**
 * SPARTAN's own PM2.5 mass reconstruction, filter by filter (spartan_composition.json, written by
 * gallery/data/export_spartan_composition.py), as stackable fields on the filter rows: the twelve
 * components of SPARTAN's network composition figure, which sum to each filter's PM2.5 mass.
 *
 * The Stacked bars (charts/partOfWhole/CompositionTreemap) open on these. They are a different basis
 * from the measured species: Dust already holds the silicon, aluminium, iron, titanium and calcium,
 * Sodium and Chloride are SPARTAN's sea salt split (not the sodium ion), Water and Residual are by
 * difference. So a stack takes its components from one basis or the other, never both.
 */
import { useEffect, useMemo, useState } from 'react'
import { loadJson } from './data'
import type { FilterRow, MetaFile } from './types'

interface CompositionFile {
  columns: string[]
  slices: { key: string; label: string }[]
  filters: (string | number)[][]
}

/** the components, in the figure's legend order (the foot of each bar first), keyed as the export keys them */
export const RECON_PARTS: readonly { key: string; label: string }[] = [
  { key: 'bc', label: 'Black Carbon' }, { key: 'teo', label: 'TEO' }, { key: 'dust', label: 'Dust' },
  { key: 'na', label: 'Sodium' }, { key: 'cl', label: 'Chloride' }, { key: 'k', label: 'Potassium' },
  { key: 'no3', label: 'Nitrate' }, { key: 'nh4', label: 'Ammonium' }, { key: 'so4', label: 'Sulfate' },
  { key: 'water', label: 'Water' }, { key: 'oc', label: 'Org. Carbon' }, { key: 'rm', label: 'Residual' },
]
export const RECON_FIELDS = RECON_PARTS.map((p) => p.label)

/** the component families (the set editor names them): what each component is, not the stacking order */
const RECON_GROUPS = [
  { label: 'Carbon', fields: ['Black Carbon', 'Org. Carbon'] },
  { label: 'Ions', fields: ['Sulfate', 'Nitrate', 'Ammonium', 'Potassium'] },
  { label: 'Sea salt', fields: ['Sodium', 'Chloride'] },
  { label: 'Mineral', fields: ['Dust', 'TEO'] },
  { label: 'By difference', fields: ['Water', 'Residual'] },
]

/** the meta that names each component's family and unit */
export const reconMeta = (meta: MetaFile): MetaFile => ({
  ...meta,
  field_groups: RECON_GROUPS,
  field_units: { ...meta.field_units, ...Object.fromEntries(RECON_FIELDS.map((f) => [f, 'ug/m3'])) },
})

/**
 * The filter rows with a SPARTAN mass balance, each carrying the twelve components under the figure's
 * labels and the balance's own PM2.5 (so the stack and the mass read the same filters). The labels
 * shared with measured ions (Sodium, Nitrate, …) take the reconstruction's values on these copies only.
 * null while the file loads; rows: [] where no filter has a balance (IMPROVE).
 */
export function useReconstructedRows(rows: FilterRow[], enabled: boolean): { rows: FilterRow[]; missing: number } | null {
  const [file, setFile] = useState<CompositionFile | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    if (!enabled || file) return
    let live = true
    loadJson<CompositionFile>('spartan_composition.json').then((d) => { if (live) setFile(d) }).catch(() => { if (live) setFailed(true) })
    return () => { live = false }
  }, [enabled, file])
  return useMemo(() => {
    if (!enabled) return null
    if (failed) return { rows: [], missing: rows.length }
    if (!file) return null
    const col = Object.fromEntries(file.columns.map((c, i) => [c, i]))
    // an export from before the filter ID column cannot be joined exactly
    if (col.id === undefined) return { rows: [], missing: rows.length }
    const byId = new Map(file.filters.map((f) => [f[col.id] as string, f]))
    const out: FilterRow[] = []
    for (const r of rows) {
      const f = byId.get(r.id)
      if (!f) continue
      const parts = Object.fromEntries(RECON_PARTS.map((p) => [p.label, f[col[p.key]] as number]))
      out.push({ ...r, ...parts, 'PM2.5 mass': f[col.pm] as number })
    }
    return { rows: out, missing: rows.length - out.length }
  }, [enabled, failed, file, rows])
}
