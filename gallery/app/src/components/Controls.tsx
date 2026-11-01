/**
 * The small option controls used in chart frames: Select (a Combobox), Segmented and Toggle, plus the
 * `sentence` helper for option text. ChartFrame re-exports them, so imports from '@/components/ChartFrame'
 * keep working.
 */
import { Combobox } from '@/components/Combobox'
import { estimatorText, useEstimator } from '@/lib/regressionPrefs'

export function Select({
  label,
  value,
  options,
  onChange,
  title,
  optionLabel,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
  title?: string
  /** display text per option when the value is a code (e.g. a site code shown by name) */
  optionLabel?: (o: string) => string
}) {
  const method = useEstimator()
  return <Combobox label={label} value={value} title={title} onChange={onChange}
    options={options.map((o) => ({ key: o, label: estimatorText(optionLabel ? optionLabel(o) : o, method) }))} />
}

/**
 * Sentence case for option text: capitalise the first word unless it is a single
 * letter (x, y, n) or already starts with a capital, digit or symbol. Display
 * only; the option values (often URL params) stay as written.
 */
export const sentence = (s: string) => /^[a-z]{2}/.test(s) ? s[0].toUpperCase() + s.slice(1) : s

/**
 * Segmented control: for 2–5 mutually exclusive options that deserve to be visible at once (more
 * than five, or long labels: Select). Labels follow the page's estimator ('Deming' reads 'OLS'
 * under OLS), as Select's do.
 */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  title,
  optionLabel,
}: {
  label?: string
  value: T
  options: readonly T[]
  onChange: (v: T) => void
  title?: string
  /** display text per option; defaults to the option in sentence case */
  optionLabel?: (o: T) => string
}) {
  const method = useEstimator()
  const text = (o: T) => estimatorText(optionLabel ? optionLabel(o) : sentence(o), method)
  // too long for one row in a column beside the chart: stacked there (styles.css .seg.long); four or more
  // options wrap into a broken grid well before 28 characters
  const chars = options.reduce((n, o) => n + text(o).length, 0)
  const long = chars > 28 || (options.length >= 4 && chars > 20)
  return (
    <span className="control" title={title}>
      {label}
      <span className={`seg${long ? ' long' : ''}`} role="radiogroup">
        {options.map((o) => (
          <button key={o} type="button" role="radio" aria-checked={o === value} className={o === value ? 'active' : ''} onClick={() => onChange(o)}>
            {text(o)}
          </button>
        ))}
      </span>
    </span>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
  title,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  title?: string
}) {
  return (
    <label className="toggle" title={title}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}
