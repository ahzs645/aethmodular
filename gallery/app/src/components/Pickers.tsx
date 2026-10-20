import { createContext, useContext, type ReactNode } from 'react'
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

export function BaselinePicker({ value, onChange, options, label = 'Baseline', short = false, as = 'segmented', backgroundRank }: {
  value: string
  onChange: (method: string) => void
  /** defaults to the page's baselines (BaselineOptionsProvider) */
  options?: readonly string[]
  label?: string
  short?: boolean
  as?: 'segmented' | 'select'
  /** for exports that call every VIBES run "VIBES" */
  backgroundRank?: number | null
}) {
  const pageOptions = useBaselineOptions()
  const list = options ?? pageOptions
  const name = (m: string) => baselineName(m, { short, backgroundRank })
  return as === 'select'
    ? <Select label={label} value={value} options={[...list]} optionLabel={name} onChange={onChange} />
    : <Segmented label={label} value={value} options={list} optionLabel={name} onChange={onChange} />
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
 * How many of a ranked list to draw: the usual counts as buttons, or any count up to `max`
 * (what the export stores) typed in. The value is a string, as kept in the URL.
 */
export function CountPicker({ label, value, presets, max, onChange, noun = 'items' }: {
  label: string; value: string; presets: readonly string[]; max: number; onChange: (v: string) => void; noun?: string
}) {
  return <>
    <Segmented label={label} value={value} options={presets} onChange={onChange} />
    <label className="control" title={`Any number of ${noun}, 1–${max}`}>
      or<input aria-label={`${label}: number of ${noun}`} className="select" type="number" min={1} max={max} step={1} value={value} style={{ width: 70 }}
        onChange={(e) => { const n = e.target.valueAsNumber; if (Number.isFinite(n) && n >= 1) onChange(String(Math.min(max, Math.round(n)))) }} />
    </label>
  </>
}
