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

/**
 * What each component is made from (gallery/data/export_spartan_composition.py, SPARTAN's ReconstrPM25 parameters),
 * and the measured species in the gallery that sit underneath it, shown beside it on the same filters
 */
export const RECON_SOURCE: Record<string, { from: string; measured: string[] }> = {
  'Black Carbon': { from: "SPARTAN's BC PM2.5: HIPS absorption, or estimated from the HIPS-SSR curve (methods 219-221)", measured: ['BC (public HIPS)', 'HIPS BC', 'EC (FTIR)'] },
  TEO: { from: "SPARTAN's Trace Element Oxides: the trace elements (ICP-MS) as their oxides", measured: ['Zinc', 'Copper', 'Lead', 'Manganese', 'Chromium', 'Vanadium', 'Nickel', 'Arsenic', 'Cadmium', 'Antimony', 'Barium', 'Strontium', 'Tin'] },
  Dust: { from: "SPARTAN's Fine Soil: Al, Si, Ca, Fe and Ti (XRF/ICP-MS) with the regional mineral-to-aluminium ratio and correction factor", measured: ['Aluminum', 'Silicon', 'Calcium (element)', 'Iron', 'Titanium'] },
  Sodium: { from: "SPARTAN's Sea Salt ÷ 2.54: the sea-salt sodium (Sea Salt = 2.54 [Na+]ss), not the sodium ion", measured: ['Sodium', 'Sodium (element)'] },
  Chloride: { from: "SPARTAN's Sea Salt × 1.54 / 2.54: the rest of the sea salt (Malm et al. 1994)", measured: ['Chlorine'] },
  Potassium: { from: 'Potassium Ion PM2.5 (ion chromatography, ChemSpecPM25)', measured: ['Potassium', 'Potassium (element)'] },
  Nitrate: { from: 'Nitrate Ion PM2.5 (ion chromatography)', measured: ['Nitrate'] },
  Ammonium: { from: 'Ammonium Ion PM2.5 (ion chromatography; newer filters: Teflon NH4 plus 0.29 × nylon NO3, method 621)', measured: ['Ammonium'] },
  Sulfate: { from: 'Sulfate Ion PM2.5 (ion chromatography)', measured: ['Sulfate'] },
  Water: { from: "particle-bound water at the weighing RH (35%): by difference, what is left of SPARTAN's 'Residual Matter with OC' balance once the other components and OC are taken out", measured: [] },
  'Org. Carbon': { from: 'OC PM2.5 (FTIR)', measured: ['OC (FTIR)'] },
  Residual: { from: "SPARTAN's Residual Matter with OC: by difference, mainly the non-carbon mass of the organics (compare FTIR OM less OC)", measured: ['OM (FTIR)'] },
}

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
