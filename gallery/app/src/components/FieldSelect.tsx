import type { MetaFile } from '@/lib/types'
import { sentence } from '@/components/ChartFrame'
import { Combobox } from '@/components/Combobox'
import { DEFINE_RATIO, useRatios } from '@/components/RatioDialog'
import { RATIO_GROUP } from '@/lib/viewFields'
import { describeField } from '@/lib/glossary'

/** "ug/m3" -> "µg/m³" for labels. */
export const prettyUnit = (u: string | undefined) =>
  u ? u.replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹') : ''

/**
 * Measurement picker grouped by family (carbon / optical / FTIR groups /
 * mass / ions / metals). Groups come from meta.json so the web app and the
 * exporter agree on what belongs where — a flat list of 28 fields hides that
 * these are four instruments answering different questions. Where App provides ratios
 * (components/RatioDialog), the list ends with "Define a ratio…", which opens the dialog and
 * picks the ratio it makes; `ratios={false}` leaves it out (a picker whose data has no ratios).
 */
export function FieldSelect({
  label,
  value,
  meta,
  onChange,
  extraOptions = [],
  title,
  ratios = true,
}: {
  label: string
  value: string
  meta: MetaFile
  onChange: (v: string) => void
  /** non-measurement choices (e.g. "Sample count") pinned above the groups */
  extraOptions?: string[]
  title?: string
  ratios?: boolean
}) {
  const ratio = useRatios()
  const define = ratios && ratio
  const options = [
    ...extraOptions.map((o) => ({ key: o, label: sentence(o) })),
    ...meta.field_groups.flatMap((g) => g.fields.map((f) => {
      const u = prettyUnit(meta.field_units?.[f])
      const cov = meta.coverage?.[f]
      return { key: f, label: u ? `${f} — ${u}` : f, group: g.label, hint: cov !== undefined ? String(cov) : undefined }
    })).filter((o) => define || o.group !== RATIO_GROUP),
    ...(define ? [{ key: DEFINE_RATIO, label: 'Define a ratio…', group: RATIO_GROUP }] : []),
  ]
  const pick = (v: string) => (v === DEFINE_RATIO && define ? define.open(onChange) : onChange(v))
  return <Combobox label={label} value={value} options={options} onChange={pick} title={title ?? describeField(value)} />
}
