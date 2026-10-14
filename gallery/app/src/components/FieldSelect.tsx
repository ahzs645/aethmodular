import type { MetaFile } from '@/lib/types'
import { sentence } from '@/components/ChartFrame'
import { Combobox } from '@/components/Combobox'

/** "ug/m3" -> "µg/m³" for labels. */
export const prettyUnit = (u: string | undefined) =>
  u ? u.replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹') : ''

/**
 * Measurement picker grouped by family (carbon / optical / FTIR groups /
 * mass / ions / metals). Groups come from meta.json so the web app and the
 * exporter agree on what belongs where — a flat list of 28 fields hides that
 * these are four instruments answering different questions.
 */
export function FieldSelect({
  label,
  value,
  meta,
  onChange,
  extraOptions = [],
  title,
}: {
  label: string
  value: string
  meta: MetaFile
  onChange: (v: string) => void
  /** non-measurement choices (e.g. "Sample count") pinned above the groups */
  extraOptions?: string[]
  title?: string
}) {
  const options = [
    ...extraOptions.map((o) => ({ key: o, label: sentence(o) })),
    ...meta.field_groups.flatMap((g) => g.fields.map((f) => {
      const u = prettyUnit(meta.field_units?.[f])
      const cov = meta.coverage?.[f]
      return { key: f, label: u ? `${f} — ${u}` : f, group: g.label, hint: cov !== undefined ? String(cov) : undefined }
    })),
  ]
  return <Combobox label={label} value={value} options={options} onChange={onChange} title={title} />
}
