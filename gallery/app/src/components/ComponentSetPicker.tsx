import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import * as d3 from 'd3'
import { Combobox } from '@/components/Combobox'
import { Segmented } from '@/components/ChartFrame'
import { BandColor } from '@/components/GroupingDialog'
import { XIcon } from '@/components/Icons'
import { BASIS_LABEL, BUILT_IN_SETS, describeSet, LEGEND_ORDER_LABEL, mergeSets, newSetId, parseSets, readSavedSets, resolveSet, setsParam, writeSavedSets,
  type Basis, type BuiltInSet, type ComponentSetDef, type LegendOrder } from '@/lib/componentSets'
import { fmt } from '@/lib/stats'
import { inkOnFill } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'

/** The reader's component sets: the URL's (`csets`) and this browser's, and saving or deleting one in both. */
export function useComponentSets() {
  const [param, setParam] = useSearchParam<string>('csets', '')
  const list = useMemo(() => mergeSets(parseSets(param), readSavedSets()), [param])
  const store = useCallback((next: ComponentSetDef[]) => { setParam(setsParam(next)); writeSavedSets(next) }, [setParam])
  const save = useCallback((d: ComponentSetDef) => store(list.some((x) => x.id === d.id) ? list.map((x) => (x.id === d.id ? d : x)) : [...list, d]), [list, store])
  const remove = useCallback((id: string) => store(list.filter((x) => x.id !== id)), [list, store])
  return { list, save, remove }
}
export type ComponentSets = ReturnType<typeof useComponentSets>

/** a measured species under a component, over a set of filters */
export interface Underlying { field: string; mean: number; n: number }
/** what the picker and the editor need from the chart, per basis */
export interface BasisInfo {
  /** the components this view can stack, in the basis's order */
  available: string[]
  /** the filters the chart would draw on this basis (the editor's means and preview) */
  rows: FilterRow[]
  /** the meta whose field groups name the components' families */
  meta: MetaFile
  /** what a component is made from (the measurement, the method) */
  source: (field: string) => string
  /** the measured species underneath a component, over these filters (the reconstruction's; none for a measured species) */
  under?: (field: string, rows: FilterRow[]) => Underlying[]
}

/** where a component goes when it is added back: after the last component that comes before it in the basis's order */
export function insertInOrder(fields: string[], add: string, order: string[]): string[] {
  const at = order.indexOf(add)
  let i = 0
  fields.forEach((f, k) => { if (order.indexOf(f) < at) i = k + 1 })
  return [...fields.slice(0, i), add, ...fields.slice(i)]
}

type EditMode = 'new' | 'copy' | 'own'
/**
 * The set editor, opened from the picker or the chart: `open(def, mode, note)`. 'new' and 'copy' save a new set of
 * the reader's ('copy': edits to a ready-made set, which stays as it is); 'own' changes one of theirs.
 */
export function useSetEditor({ sets, bases, standardColor, onUse }: {
  sets: ComponentSets; bases: Partial<Record<Basis, BasisInfo>>
  standardColor: (field: string, basis: Basis) => string
  /** the chart switches to the saved set */
  onUse: (id: string) => void
}) {
  const [editing, setEditing] = useState<{ def: ComponentSetDef; mode: EditMode; note?: string } | null>(null)
  const open = useCallback((def: ComponentSetDef, mode: EditMode, note?: string) => setEditing({ def, mode, note }), [])
  const node: ReactNode = editing && <ComponentSetDialog start={editing.def} mode={editing.mode} note={editing.note} bases={bases} standardColor={standardColor}
    onClose={() => setEditing(null)}
    onSave={(d) => { sets.save(d); onUse(d.id); setEditing(null) }}
    onRemove={editing.mode === 'own' ? () => { sets.remove(editing.def.id); onUse(BUILT_IN_SETS.find((s) => bases[s.basis])!.id); setEditing(null) } : undefined} />
  return { open, node }
}

/**
 * The Stacked bars' Components: one picker for the ready-made sets (SPARTAN's reconstruction, the measured species)
 * and the reader's own, then + (a new set from what is showing) and Edit… (a ready-made set's edits save as the
 * reader's own). `showing`: the current set as drawn, less anything hidden in the legend.
 */
export function ComponentSetPicker({ value, onChange, sets, bases, editor, showing, hiddenNote }: {
  value: string
  onChange: (id: string) => void
  sets: ComponentSets
  /** the bases this view offers (no reconstruction where no filter has SPARTAN's balance) */
  bases: Partial<Record<Basis, BasisInfo>>
  editor: ReturnType<typeof useSetEditor>
  showing: ComponentSetDef
  /** said in the editor when the legend hides some of the set */
  hiddenNote?: string
}) {
  const builtIn = BUILT_IN_SETS.filter((s) => bases[s.basis])
  const own = sets.list.filter((s) => bases[s.basis])
  const current: ComponentSetDef | BuiltInSet | undefined = builtIn.find((s) => s.id === value) ?? own.find((s) => s.id === value)
  const isOwn = !!current && own.some((s) => s.id === current.id)
  const n = (s: ComponentSetDef | BuiltInSet) => resolveSet(s, bases[s.basis]!.available).length
  const options = [
    ...own.map((s) => ({ key: s.id, label: s.name, detail: describeSet(s, bases[s.basis]!.available), group: 'Your sets', disabled: n(s) === 0 && s.id !== value })),
    ...builtIn.map((s) => ({ key: s.id, label: s.name, detail: `${describeSet(s, bases[s.basis]!.available)} · ${s.note}`, group: BASIS_LABEL[s.basis], disabled: n(s) === 0 && s.id !== value })),
  ]
  return (
    <span className="grouping-picker">
      <Combobox label="Components" value={current ? value : (options[0]?.key ?? '')} options={options} minWidth={170} onChange={onChange}
        title="What the bars stack: SPARTAN's reconstruction (sums to PM2.5) or the measured species, a ready-made set or your own (+)" />
      <span className="grouping-actions">
        <button type="button" className="btn icon-btn" title="A new set of yours, starting from what is showing: components, names, order, colours and legend" aria-label="New component set"
          onClick={() => editor.open({ ...showing, id: newSetId(), name: '' }, 'new', hiddenNote)}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true"><path d="M8 3v10M3 8h10" /></svg>
        </button>
        {current && <button type="button" className="btn"
          title={isOwn ? 'Change this set: add or remove components, rename them, reorder, recolour, the legend' : `Edit what is showing; saving makes a set of yours (${current.name} stays as it is)`}
          onClick={() => (isOwn ? editor.open({ ...(current as ComponentSetDef), fields: showing.fields }, 'own', hiddenNote)
            : editor.open({ ...showing, id: newSetId(), name: `${current.name} (edited)` }, 'copy', hiddenNote))}>Edit…</button>}
      </span>
    </span>
  )
}

/**
 * Under the legend: the components the set leaves out, each one click from the stack, and those hidden in the legend,
 * one click from leaving the set. On a ready-made set these open the editor (saving makes a set of yours).
 */
export function ComponentSetBar({ notShown, hidden, nameOf, colorOf, onAdd, onDrop, onEdit, builtIn }: {
  notShown: string[]; hidden: string[]
  nameOf: (field: string) => string; colorOf: (field: string) => string
  onAdd: (field: string) => void; onDrop: () => void; onEdit: () => void
  /** a ready-made set: the buttons say they make a set of yours */
  builtIn: boolean
}) {
  return (
    <div className="cset-bar">
      {notShown.length > 0 && <span className="cset-bar-group"><span className="cset-bar-label">Not in this set</span>
        {notShown.map((f) => <button key={f} type="button" className="chip cset-add" onClick={() => onAdd(f)}
          title={builtIn ? `Add ${nameOf(f)}: opens the editor, saving makes a set of yours` : `Add ${nameOf(f)} to this set`}>
          <span className="swatch square" style={{ background: colorOf(f) }} />+ {nameOf(f)}</button>)}</span>}
      {hidden.length > 0 && <span className="cset-bar-group"><span className="cset-bar-label">Hidden in the legend: {hidden.map(nameOf).join(', ')}</span>
        <button type="button" className="btn quiet" onClick={onDrop} title={builtIn ? 'Opens the editor without them; saving makes a set of yours' : 'Take them out of this set'}>
          {builtIn ? 'Save what is showing…' : 'Remove from the set'}</button></span>}
      <button type="button" className="btn quiet" onClick={onEdit} title="Components, names, stacking order, colours and the legend">Edit components…</button>
    </div>
  )
}

function ComponentSetDialog({ start, mode, note, bases, standardColor, onSave, onRemove, onClose }: {
  start: ComponentSetDef; mode: EditMode; note?: string
  bases: Partial<Record<Basis, BasisInfo>>; standardColor: (field: string, basis: Basis) => string
  onSave: (d: ComponentSetDef) => void; onRemove?: () => void; onClose: () => void
}) {
  const [draft, setDraft] = useState<ComponentSetDef>(start)
  const set = (patch: Partial<ComponentSetDef>) => setDraft((d) => ({ ...d, ...patch }))
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !document.querySelector('.combobox-menu, .export-color-popover')) { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const info = bases[draft.basis]!
  const chosen = draft.fields.filter((f) => info.available.includes(f))
  const rest = info.available.filter((f) => !chosen.includes(f))
  const standard = (f: string) => standardColor(f, draft.basis)
  const colorOf = (f: string) => draft.colors?.[f] || standard(f)
  const nameOf = (f: string) => draft.labels?.[f]?.trim() || f
  // each component's mean over the filters in view (all sites pooled), and the measured species under it
  const stats = useMemo(() => new Map(info.available.map((f) => {
    const v = info.rows.map((r) => r[f]).filter((x): x is number => typeof x === 'number' && Number.isFinite(x))
    return [f, { n: v.length, mean: v.length ? d3.mean(v)! : NaN, under: info.under?.(f, info.rows) ?? [] }]
  })), [info])
  const pm = useMemo(() => d3.mean(info.rows, (r) => (typeof r['PM2.5 mass'] === 'number' ? (r['PM2.5 mass'] as number) : undefined)), [info])
  const familyOf = (f: string) => info.meta.field_groups.find((g) => g.fields.includes(f))?.label ?? (f.startsWith('OC') || f.startsWith('EC') ? 'Carbon' : 'Other')

  const toggle = (f: string) => set({ fields: chosen.includes(f) ? chosen.filter((x) => x !== f) : insertInOrder(chosen, f, info.available) })
  const move = (i: number, by: number) => {
    const j = i + by
    if (j < 0 || j >= chosen.length) return
    const next = [...chosen];[next[i], next[j]] = [next[j], next[i]]
    set({ fields: next })
  }
  const patchMap = (key: 'colors' | 'labels', f: string, v: string) => {
    const m = { ...draft[key] }
    if (v) m[f] = v
    else delete m[f]
    set({ [key]: m })
  }
  const switchBasis = (b: Basis) => { if (b !== draft.basis) set({ basis: b, fields: [...bases[b]!.available] }) }
  const startFrom = (s: BuiltInSet) => setDraft((d) => ({ ...d, basis: s.basis, fields: s.fields(bases[s.basis]!.available) }))
  const name = draft.name.trim() || `My ${draft.basis === 'recon' ? 'reconstruction' : 'species'} set`
  const legend: LegendOrder = draft.legend ?? 'foot'
  const offered = Object.keys(bases) as Basis[]

  // names must stay apart: from each other, and from any other measurement the filters carry (the chart stacks by name)
  const reserved = new Set(['id', 'site', 'code', 'date', 'season', 'month', 'year', 'PM2.5 mass', ...offered.flatMap((b) => bases[b]!.meta.fields), ...offered.flatMap((b) => bases[b]!.available)])
  const clash = (f: string) => {
    const l = draft.labels?.[f]?.trim()
    if (!l || l === f) return null
    if (chosen.some((g) => g !== f && nameOf(g) === l)) return 'another component has this name'
    if (reserved.has(l)) return 'a measurement already has this name'
    return null
  }
  const clashes = chosen.filter((f) => clash(f)).length
  const ok = chosen.length > 0 && clashes === 0
  const ownColours = Object.keys(draft.colors ?? {}).filter((f) => chosen.includes(f)).length
  const ownNames = Object.keys(draft.labels ?? {}).filter((f) => chosen.includes(f)).length

  // the preview: one bar of the chosen components' means over the view, foot of the stack on the left
  const parts = chosen.map((f) => ({ f, v: stats.get(f)?.mean ?? NaN })).filter((p) => p.v > 0)
  const total = d3.sum(parts, (p) => p.v)
  const legendItems = legend === 'top' ? [...chosen].reverse() : chosen
  const keep = (m?: Record<string, string>) => Object.fromEntries(Object.entries(m ?? {}).filter(([f, v]) => chosen.includes(f) && v.trim()))
  const title = mode === 'own' ? 'Edit component set' : mode === 'copy' ? 'Edit what is showing' : 'New component set'

  const row = (f: string, i: number | null) => {
    const st = stats.get(f)
    const under = (st?.under ?? []).filter((u) => u.n > 0)
    const why = clash(f)
    return (
      <div key={f} className={`cset-row${i === null ? ' off' : ''}`}>
        <input type="checkbox" checked={i !== null} onChange={() => toggle(f)} aria-label={i !== null ? `Leave out ${f}` : `Add ${f}`} />
        {i !== null
          ? <BandColor label={nameOf(f)} color={colorOf(f)} standard={standard(f)} onChange={(c) => patchMap('colors', f, c)} onStandard={() => patchMap('colors', f, '')} />
          : <span className="grouping-swatch" style={{ background: colorOf(f), cursor: 'default' }} aria-hidden="true" />}
        {i !== null
          ? <input className={`select cset-label${why ? ' invalid' : ''}`} value={draft.labels?.[f] ?? ''} placeholder={f} aria-label={`Name of ${f}`} aria-invalid={!!why}
            title={why ?? `Name in the legend, labels and tooltips (blank: ${f})`} onChange={(e) => patchMap('labels', f, e.target.value)} />
          : <span className="cset-name">{nameOf(f)}</span>}
        <span className="grouping-range">{familyOf(f)}</span>
        <span className="grouping-count">{fmt(st?.mean, 2)} µg/m³ · n {st?.n.toLocaleString()}</span>
        {i !== null
          ? <span className="cset-move">
            <button type="button" className="btn quiet x-btn" disabled={i === 0} onClick={() => move(i, -1)} aria-label={`Move ${f} down the stack`} title="Lower in the stack">↓</button>
            <button type="button" className="btn quiet x-btn" disabled={i === chosen.length - 1} onClick={() => move(i, 1)} aria-label={`Move ${f} up the stack`} title="Higher in the stack">↑</button>
          </span>
          : <span />}
        <span className="cset-source">
          {why && <strong className="cset-clash">{why}. </strong>}
          {nameOf(f) !== f && <>{f} · </>}{info.source(f)}
          {under.length > 0 && <> · <b>measured on these filters:</b> {under.map((u) => `${u.field} ${fmt(u.mean, 2)}`).join(' · ')} µg/m³</>}
        </span>
      </div>
    )
  }

  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="export-dialog ratio-dialog grouping-dialog cset-dialog" role="dialog" aria-modal="true" aria-label={title}>
        <header>
          <div><strong>{title}</strong>
            <span className="export-title">{mode === 'copy' ? 'saving makes a set of yours; the ready-made set stays as it is' : 'what the stacked bars stack: components, names, order, colours, legend'}</span></div>
          <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
        </header>
        {note && <p className="chart-note" style={{ margin: 0 }}>{note}</p>}
        <section className="ratio-presets">
          <div className="row-label">Start from</div>
          <div className="ratio-preset-list">
            {BUILT_IN_SETS.filter((s) => bases[s.basis]).map((s) => (
              <button key={s.id} type="button" className="chip ratio-preset" title={`${s.note} (keeps your names and colours)`} onClick={() => startFrom(s)}>{s.name}</button>
            ))}
          </div>
        </section>
        <section className="grouping-form">
          {offered.length > 1 && <Segmented label="Basis" value={draft.basis} options={offered} optionLabel={(b) => BASIS_LABEL[b]} onChange={switchBasis}
            title="SPARTAN reconstruction: the twelve components of SPARTAN's figure, which sum to each filter's PM2.5. Measured species: FTIR carbon, ions and elements as measured. One set takes one basis: the reconstruction's dust already holds the crustal elements, and its sodium is the sea-salt split" />}
          <label className="grouping-name"><span className="combobox-label">Set name</span>
            <input className="select" value={draft.name} placeholder={name} onChange={(e) => set({ name: e.target.value })} />
          </label>
        </section>
        <div className="cset-preview" role="img" aria-label="The chosen components' mean over the filters in view">
          <div className="cset-preview-bar">
            {parts.map((p) => <span key={p.f} style={{ flexGrow: p.v, background: colorOf(p.f), color: inkOnFill(colorOf(p.f)) }} title={`${nameOf(p.f)}: ${fmt(p.v, 2)} µg/m³ (${((p.v / total) * 100).toFixed(1)}%)`}>
              {p.v / total > 0.08 ? nameOf(p.f) : ''}</span>)}
          </div>
          <span className="grouping-range">{parts.length ? `mean over ${info.rows.length.toLocaleString()} filters in view, all sites pooled: ${fmt(total, 1)} µg/m³${pm ? ` of PM2.5 ${fmt(pm, 1)}` : ''}` : 'no component with a positive mean in view'}</span>
        </div>
        <section className="cset-list">
          <div className="row-label">In the stack, foot first <span className="field-multi-n">{chosen.length}</span>
            <span className="cset-hint">tick to add or remove · type a new name · click a swatch for its colour · arrows to reorder</span></div>
          {chosen.map((f, i) => row(f, i))}
          {!chosen.length && <p className="chart-note" style={{ margin: 0 }}>Tick at least one component below.</p>}
          {rest.length > 0 && <div className="row-label" style={{ marginTop: 6 }}>Not in the stack</div>}
          {rest.map((f) => row(f, null))}
          {draft.basis === 'measured' && <p className="chart-note" style={{ margin: 0 }}>Measured species stack family by family (carbon, ions, crustal elements, trace metals), so the order here is the order within each family.</p>}
        </section>
        <section className="cset-legend">
          <Segmented label="Legend" value={legend} options={['foot', 'top'] as const} optionLabel={(o) => LEGEND_ORDER_LABEL[o]}
            onChange={(v) => set({ legend: v === 'top' ? 'top' : undefined })}
            title="Foot first: the legend lists the stack from the bottom up, as SPARTAN's figure keys it. Top first: it reads down the bars" />
          <div className="legend cset-legend-preview" aria-label="Legend preview">
            {legendItems.map((f) => <span key={f} className="legend-item"><span className="swatch square" style={{ background: colorOf(f) }} />{nameOf(f)}</span>)}
          </div>
        </section>
        <div className="export-actions">
          {onRemove && <button type="button" className="btn" style={{ marginRight: 'auto' }} onClick={onRemove}>Delete set</button>}
          {ownNames > 0 && <button type="button" className="btn" onClick={() => set({ labels: {} })} title="Every component back to its own name">Reset names</button>}
          {ownColours > 0 && <button type="button" className="btn" onClick={() => set({ colors: {} })} title="Every component back to the chart's palette">Reset colours</button>}
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="button" className="btn primary" disabled={!ok} title={clashes ? 'Two names clash: see the highlighted rows' : undefined}
            onClick={() => onSave({ id: draft.id, name, basis: draft.basis, fields: chosen, colors: keep(draft.colors), labels: keep(draft.labels), ...(legend === 'top' ? { legend } : {}) })}>
            {mode === 'own' ? 'Save' : 'Create and use'}</button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
