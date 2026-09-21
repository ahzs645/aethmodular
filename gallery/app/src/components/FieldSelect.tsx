import type { MetaFile } from '@/lib/types'

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
  return (
    <label className="control" title={title}>
      {label}
      <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
        {extraOptions.length > 0 && (
          <optgroup label="—">
            {extraOptions.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </optgroup>
        )}
        {meta.field_groups.map((g) => (
          <optgroup key={g.label} label={g.label}>
            {g.fields.map((f) => {
              const u = prettyUnit(meta.field_units?.[f])
              const cov = meta.coverage?.[f]
              return (
                <option key={f} value={f}>
                  {u ? `${f} — ${u}` : f}
                  {cov !== undefined ? ` (${cov})` : ''}
                </option>
              )
            })}
          </optgroup>
        ))}
      </select>
    </label>
  )
}
