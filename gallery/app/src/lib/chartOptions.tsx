import { useState } from 'react'
import { Combobox } from '@/components/Combobox'
import { Segmented, Toggle } from '@/components/ChartFrame'
import { useSearchParam } from '@/lib/url'
import type { AxisOption } from '@/components/AxisChoices'

/**
 * A chart's options, declared once with the rules for when each applies, instead of
 * hand-written conditions scattered through its JSX. Every chart that uses this resolves
 * and draws its controls the same way:
 *
 *   const o = useChartOptions('scatter', [
 *     { key: 'layout', label: 'Layout', choices: ['Combined', 'Per site', 'Per season'], default: 'Combined' },
 *     { key: 'fitLines', label: 'Fit lines', toggle: true, default: true },
 *     { key: 'grouping', label: 'Fit grouping', default: 'pooled', bind: [grouping, setGrouping],
 *       when: (c) => c.fitLines && c.estimators > 0 && c.layout !== 'Per season',
 *       hiddenNote: (c) => (c.layout === 'Per season' ? 'one fit per panel' : null),
 *       choices: [{ value: 'pooled', label: (c) => (c.layout === 'Per site' ? 'One fit per site' : 'One pooled fit') },
 *                 { value: 'site', when: (c) => c.layout === 'Combined' }, { value: 'season' }] },
 *   ], { estimators: methods.length })
 *   o.values.layout · <ChartFrame controls={o.controls()} …>
 *
 * Options resolve in order. Each sees the facts the chart passes (site count, calendars …)
 * and the values resolved before it, so a later option can depend on an earlier one.
 * - `when` on an option: shown only when true; otherwise it takes `hidden` (or its default),
 *   and `hiddenNote` can say why ("one fit per panel").
 * - `when` on a choice: offered only when true. A chosen value that is no longer offered
 *   resolves to the first one that is, so the chart never draws a combination it hides.
 * - `bind`: state the chart does not own (the gallery-wide fit grouping); `persist`: kept in
 *   the URL as `${chartId}.${key}`; otherwise the chart's own state.
 * - `node`: a control the engine does not draw itself (a field picker, a count), placed in order;
 *   drawn when `controls()` is called, so it may use values the chart computes after the hook.
 * - `onChange`: a side effect of the reader's change (clear a brush, reset a legend).
 * Figure (spec-driven charts) uses the same engine for its variants (choices that patch the spec).
 */
import { resolveOptions, type OptionContext, type OptionDef, type OptionValue } from '@/lib/optionRules'
export type { OptionChoice, OptionContext, OptionDef, OptionValue } from '@/lib/optionRules'

/** The resolved values, and `controls()` for the chart's ChartFrame. */
export function useChartOptions(chartId: string, defs: readonly OptionDef[], facts: OptionContext = {}) {
  // one state per option, in a fixed order: a mounted chart never changes its list of options
  const stored = defs.map((d) => useStoredOption(chartId, d)) // eslint-disable-line react-hooks/rules-of-hooks
  const { values, items, ctx } = resolveOptions(defs, stored.map((s) => s[0]), facts)
  const setter = (i: number) => (v: OptionValue) => { stored[i][1](v); (defs[i].onChange as ((v: OptionValue) => void) | undefined)?.(v) }

  const controls = () => <>{items.map((it) => {
    const d = it.def, i = defs.indexOf(d), set = setter(i)
    // an axis option is drawn by its axis title (axisChoices), not in the row
    if (d.axis && it.kind === 'choice') return null
    if (it.kind === 'node') return <span key={d.key} className="option-node">{d.node!(it.ctx)}</span>
    if (it.kind === 'hidden') return it.note ? <span key={d.key} className="control hint" title={d.title}>{it.note}</span> : null
    if (it.kind === 'toggle') return <Toggle key={d.key} label={d.label ?? d.key} checked={it.value} onChange={(x) => set(x)} title={d.title} />
    // a single remaining choice is not a choice: nothing to draw, the value is fixed
    if (it.offered.length < 2) return null
    const label = (v: string) => it.offered.find((c) => c.value === v)?.label ?? v
    return it.offered.length <= 5 && d.style !== 'select'
      ? <Segmented key={d.key} label={d.label} value={it.value} options={it.offered.map((c) => c.value)} optionLabel={label} onChange={(v) => set(v)} title={d.title} />
      : <Combobox key={d.key} label={d.label ?? ''} value={it.value} title={d.title} onChange={(v) => set(v)} options={it.offered.map((c) => ({ key: c.value, label: c.label }))} />
  })}</>
  // set an option from outside its control (a click on another chart picks the configuration)
  const set = (key: string, v: OptionValue) => { const i = defs.findIndex((d) => d.key === key); if (i >= 0) setter(i)(v) }
  // the axis options, for <AxisChoices {...o.axisChoices()}> around the plot (axes get pick="x" / "y"):
  // each axis offers only its own option's choices, labelled as in the row they left. A chart with
  // several panels names the options of each panel's axes: axisChoices({ x: 'axis', y: 'cvMetric' })
  const axisChoices = (pick?: { x?: string; y?: string }) => {
    const out: { options: AxisOption[]; x?: [string, (v: string) => void]; y?: [string, (v: string) => void]; noun: string } = { options: [], noun: 'choices' }
    for (const it of items) {
      const d = it.def
      if (!d.axis || it.kind !== 'choice') continue
      if (pick && pick[d.axis] !== d.key) continue
      const set = setter(defs.indexOf(d))
      out[d.axis] = [it.value, (v: string) => set(v)]
      out.options.push(...it.offered.map((c) => ({ key: c.value, label: c.label, group: d.label ?? d.key, roles: [d.axis!] })))
    }
    return out
  }
  /** the label an axis option's value is shown with (for the axis title) */
  const axisLabel = (key: string) => { const it = items.find((x) => x.def.key === key); return it?.kind === 'choice' ? it.offered.find((c) => c.value === it.value)?.label ?? it.value : '' }
  return { values, controls, set, ctx, axisChoices, axisLabel }
}

function useStoredOption(chartId: string, d: OptionDef): [OptionValue, (v: OptionValue) => void] {
  // the kind of storage is fixed per option, so the hooks below run in the same order every render
  if (d.bind) return d.bind as [OptionValue, (v: OptionValue) => void]
  if (d.persist) {
    const [v, set] = useSearchParam<string>(`${chartId}.${d.key}`, d.toggle ? (d.default ? '1' : '0') : String(d.default ?? '')) // eslint-disable-line react-hooks/rules-of-hooks
    return d.toggle ? [v === '1', (x) => set(x ? '1' : '0')] : [v, (x) => set(String(x))]
  }
  const [v, set] = useState<OptionValue>(d.default ?? '') // eslint-disable-line react-hooks/rules-of-hooks
  return [v, set]
}

/**
 * The fit-grouping option every scatter-type chart shares, bound to the gallery-wide setting.
 * Charts say only what differs: when fits are drawn at all, whether a fit per site still
 * splits anything (not inside one site's panel, or with one site in view), what the pooled
 * fit is called there, and a note for when the option is hidden.
 */
export function fitGroupingOption({ bind, when, offerSite = () => true, pooledLabel, hiddenNote }: {
  bind: [string, (v: never) => void]
  when?: (ctx: OptionContext) => boolean
  offerSite?: (ctx: OptionContext) => boolean
  pooledLabel?: (ctx: OptionContext) => string
  hiddenNote?: (ctx: OptionContext) => string | null
}): OptionDef {
  return {
    key: 'grouping', label: 'Fit grouping', style: 'select', default: 'pooled', hidden: 'pooled', bind, when, hiddenNote,
    title: 'Only the groupings that still split the drawn data are offered',
    choices: [
      { value: 'pooled', label: pooledLabel ?? 'One pooled fit' },
      { value: 'site', label: 'Separate fit per site', when: offerSite },
      { value: 'season', label: 'Separate fit per season' },
    ],
  }
}
