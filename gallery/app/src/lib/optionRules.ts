/**
 * The rules behind lib/chartOptions, as plain functions so they can be tested without React:
 * which options and choices a chart offers given the facts and the earlier values, and the
 * value each option resolves to. See chartOptions.tsx for how a chart declares them.
 */
import type { ReactNode } from 'react'

export type OptionValue = string | boolean
export type OptionContext = Record<string, unknown>
export interface OptionChoice {
  value: string
  label?: string | ((ctx: OptionContext) => string)
  when?: (ctx: OptionContext) => boolean
}
export interface OptionDef {
  key: string
  label?: string
  /** a string list or full choices; omit for a toggle */
  choices?: readonly (string | OptionChoice)[]
  toggle?: boolean
  /** required for a value option; a `node` has none */
  default?: OptionValue
  when?: (ctx: OptionContext) => boolean
  /** the value while the option is hidden (default: its default) */
  hidden?: OptionValue
  hiddenNote?: (ctx: OptionContext) => string | null
  title?: string
  /** 'select' for a dropdown; segmented buttons by default up to five choices */
  style?: 'segmented' | 'select'
  bind?: [OptionValue, (v: never) => void]
  persist?: boolean
  /** a control drawn by the chart, placed among the others; no value of its own */
  node?: (ctx: OptionContext) => ReactNode
  onChange?: (v: never) => void
}

const asChoice = (c: string | OptionChoice): OptionChoice => (typeof c === 'string' ? { value: c } : c)

/** One option after resolving: what the engine draws for it, if anything. */
export type ResolvedOption =
  | { kind: 'node'; def: OptionDef; ctx: OptionContext }
  | { kind: 'hidden'; def: OptionDef; value: OptionValue; note: string | null }
  | { kind: 'toggle'; def: OptionDef; value: boolean }
  | { kind: 'choice'; def: OptionDef; value: string; offered: { value: string; label: string }[] }

/** Resolve options in order; each sees the facts and the values resolved before it. */
export function resolveOptions(defs: readonly OptionDef[], raws: readonly OptionValue[], facts: OptionContext = {}) {
  const ctx: OptionContext = { ...facts }
  const values: Record<string, OptionValue> = {}
  const items: ResolvedOption[] = []
  defs.forEach((d, i) => {
    if (d.node) {
      if (!d.when || d.when(ctx)) items.push({ kind: 'node', def: d, ctx: { ...ctx } })
      return
    }
    let value: OptionValue
    if (d.when && !d.when(ctx)) {
      value = d.hidden ?? d.default ?? ''
      items.push({ kind: 'hidden', def: d, value, note: d.hiddenNote?.(ctx) ?? null })
    } else if (d.toggle) {
      value = !!raws[i]
      items.push({ kind: 'toggle', def: d, value })
    } else {
      const offered = (d.choices ?? []).map(asChoice).filter((c) => !c.when || c.when(ctx))
        .map((c) => ({ value: c.value, label: typeof c.label === 'function' ? c.label(ctx) : c.label ?? c.value }))
      // a chosen value that is no longer offered falls back to the default, else the first offered
      value = offered.some((c) => c.value === raws[i]) ? String(raws[i])
        : offered.find((c) => c.value === d.default)?.value ?? offered[0]?.value ?? String(d.default)
      items.push({ kind: 'choice', def: d, value, offered })
    }
    values[d.key] = value
    ctx[d.key] = value
  })
  return { values, items, ctx }
}
