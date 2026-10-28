import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { Segmented, Select } from '@/components/ChartFrame'
import { baselineName } from '@/lib/labels'
import { shortSeason } from '@/lib/seasons'

/**
 * The pickers every research page repeats: baseline, season group and blank
 * bank. One widget per dimension, one name per option (lib/labels, lib/seasons),
 * whatever page it sits on. Each chart still owns its URL key, so links keep
 * working; the options come from the page's data, not from module state.
 */

/** Baselines a page offers (set once at the page root from its export). */
const BaselineOptions = createContext<readonly string[]>(['AIRSpec'])
export function BaselineOptionsProvider({ methods, children }: { methods: readonly string[]; children: ReactNode }) {
  return <BaselineOptions.Provider value={methods}>{children}</BaselineOptions.Provider>
}
export const useBaselineOptions = () => useContext(BaselineOptions)

export function BaselinePicker({ value, onChange, options, label = 'Baseline', short = false, as = 'segmented', backgroundRank, title }: {
  value: string
  onChange: (method: string) => void
  /** defaults to the page's baselines (BaselineOptionsProvider) */
  options?: readonly string[]
  label?: string
  short?: boolean
  as?: 'segmented' | 'select'
  /** for exports that call every VIBES run "VIBES" */
  backgroundRank?: number | null
  /** hover text: what the chosen baseline is (a run's rank, its channels) */
  title?: string
}) {
  const pageOptions = useBaselineOptions()
  const list = options ?? pageOptions
  const name = (m: string) => baselineName(m, { short, backgroundRank })
  return as === 'select'
    ? <Select label={label} value={value} options={[...list]} optionLabel={name} onChange={onChange} title={title} />
    : <Segmented label={label} value={value} options={list} optionLabel={name} onChange={onChange} title={title} />
}

/** A season group (e.g. "All Addis", "Kiremt (Jun-Sep)"), full or short names. */
export function SeasonPicker({ value, onChange, groups, label = 'Season', short = false, as = 'segmented' }: {
  value: string
  onChange: (group: string) => void
  groups: readonly string[]
  label?: string
  short?: boolean
  as?: 'segmented' | 'select'
}) {
  const name = (g: string) => (short ? shortSeason(g) : g)
  return as === 'select'
    ? <Select label={label} value={value} options={[...groups]} optionLabel={name} onChange={onChange} />
    : <Segmented label={label} value={value} options={groups} optionLabel={name} onChange={onChange} />
}

/** A VIBES blank bank, named from the export's own labels, optionally with its chosen rank. */
export function BankPicker({ value, onChange, banks, labels, ranks, label = 'Background' }: {
  value: string
  onChange: (bank: string) => void
  banks: readonly string[]
  labels: Record<string, string>
  ranks?: Record<string, number>
  label?: string
}) {
  return <Select label={label} value={value} options={[...banks]}
    optionLabel={(b) => `${labels[b] ?? b}${ranks?.[b] != null ? ` (rank ${ranks[b]})` : ''}`} onChange={onChange} />
}

/** A count from a URL value, kept to 1..max (anything unreadable falls back). */
export const clampCount = (v: string, max: number, fallback: number) => {
  const n = Math.round(Number(v))
  return String(Number.isFinite(n) && n >= 1 ? Math.min(max, n) : fallback)
}

/**
 * How many of a ranked list to draw: the usual counts as buttons, the last of them a number box
 * that starts at that count and takes any count up to `max` (what the export stores). The value
 * is a string, as kept in the URL.
 */
export function CountPicker({ label, value, presets, max, onChange, noun = 'items' }: {
  label: string; value: string; presets: readonly string[]; max: number; onChange: (v: string) => void; noun?: string
}) {
  const fixed = presets.slice(0, -1)
  const typed = !fixed.includes(value)
  // what the box shows: the chosen count when it is not one of the buttons, else the last preset
  const [draft, setDraft] = useState(typed ? value : presets[presets.length - 1])
  useEffect(() => { if (typed) setDraft(value) }, [value, typed])
  const take = (v: string) => { const n = Math.round(Number(v)); if (Number.isFinite(n) && n >= 1) onChange(String(Math.min(max, n))) }
  return (
    <span className="control" title={`Any number of ${noun}, 1–${max}: type it in the last box`}>
      {label}
      <span className="seg" role="radiogroup">
        {fixed.map((o) => (
          <button key={o} type="button" role="radio" aria-checked={o === value} className={o === value ? 'active' : ''} onClick={() => onChange(o)}>{o}</button>
        ))}
        <input aria-label={`${label}: number of ${noun}`} className={`seg-input${typed ? ' active' : ''}`} type="number" min={1} max={max} step={1}
          value={draft} onFocus={() => take(draft)} onChange={(e) => { setDraft(e.target.value); take(e.target.value) }}
          onBlur={() => { if (typed) setDraft(value) }} />
      </span>
    </span>
  )
}
