import { useCallback, useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import * as d3 from 'd3'
import { Combobox } from '@/components/Combobox'
import { Segmented } from '@/components/ChartFrame'
import { BandColor } from '@/components/GroupingDialog'
import { XIcon } from '@/components/Icons'
import { BASIS_LABEL, BUILT_IN_SETS, describeSet, mergeSets, newSetId, parseSets, readSavedSets, resolveSet, setsParam, writeSavedSets,
  type Basis, type BuiltInSet, type ComponentSetDef } from '@/lib/componentSets'
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

/** what the picker and the editor need from the chart, per basis */
export interface BasisInfo {
  /** the components this view can stack, in the basis's order */
  available: string[]
  /** the filters the chart would draw on this basis (the preview's means) */
  rows: FilterRow[]
  /** the meta whose field groups name the components' families */
  meta: MetaFile
}

/**
 * The Stacked bars' Components: one picker for the ready-made sets (SPARTAN's reconstruction, the measured species)
 * and the reader's own, then + (a new set) and Edit… (Build on this… for a ready-made one, which saves a copy). The
 * editor chooses the basis, the components, their stacking order and their colours.
 */
export function ComponentSetPicker({ value, onChange, sets, bases, standardColor }: {
  value: string
  onChange: (id: string) => void
  sets: ReturnType<typeof useComponentSets>
  /** the bases this view offers (no reconstruction where no filter has SPARTAN's balance) */
  bases: Partial<Record<Basis, BasisInfo>>
  /** a component's colour in the chart's palette on a basis, what it shows without a colour of its own */
  standardColor: (field: string, basis: Basis) => string
}) {
  const [editing, setEditing] = useState<{ def: ComponentSetDef; copy: boolean; fresh: boolean } | null>(null)
  const builtIn = BUILT_IN_SETS.filter((s) => bases[s.basis])
  const own = sets.list.filter((s) => bases[s.basis])
  const current: ComponentSetDef | BuiltInSet | undefined = builtIn.find((s) => s.id === value) ?? own.find((s) => s.id === value)
  const isBuiltIn = !!current && builtIn.some((s) => s.id === current.id)
  const n = (s: ComponentSetDef | BuiltInSet) => resolveSet(s, bases[s.basis]!.available).length
  const options = [
    ...own.map((s) => ({ key: s.id, label: s.name, detail: describeSet(s, bases[s.basis]!.available), group: 'Your sets', disabled: n(s) === 0 && s.id !== value })),
    ...builtIn.map((s) => ({ key: s.id, label: s.name, detail: `${describeSet(s, bases[s.basis]!.available)} · ${s.note}`, group: BASIS_LABEL[s.basis], disabled: n(s) === 0 && s.id !== value })),
  ]
  // a ready-made set opens as the reader's copy, with its components as this view resolves them
  const copyOf = (s: ComponentSetDef | BuiltInSet): ComponentSetDef => ({ id: newSetId(), name: `${s.name} (my set)`, basis: s.basis,
    fields: resolveSet(s, bases[s.basis]!.available), ...('colors' in s && s.colors ? { colors: { ...s.colors } } : {}) })
  return (
    <span className="grouping-picker">
      <Combobox label="Components" value={current ? value : (options[0]?.key ?? '')} options={options} minWidth={170} onChange={onChange}
        title="What the bars stack: SPARTAN's reconstruction (sums to PM2.5) or the measured species, a ready-made set or your own (+)" />
      <span className="grouping-actions">
        <button type="button" className="btn icon-btn" title="Make your own set: components, stacking order and colours" aria-label="New component set"
          onClick={() => setEditing({ def: copyOf(current ?? builtIn[0]), copy: true, fresh: true })}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true"><path d="M8 3v10M3 8h10" /></svg>
        </button>
        {current && (isBuiltIn
          ? <button type="button" className="btn" title="Open this set in the editor; saving makes a set of yours" onClick={() => setEditing({ def: copyOf(current), copy: true, fresh: false })}>Build on this…</button>
          : <button type="button" className="btn" title="Change this set: components, order, colours" onClick={() => setEditing({ def: current as ComponentSetDef, copy: false, fresh: false })}>Edit…</button>)}
      </span>
      {editing && <ComponentSetDialog start={editing.def} isNew={editing.copy} fresh={editing.fresh} bases={bases} standardColor={standardColor} onClose={() => setEditing(null)}
        onSave={(d) => { sets.save(d); onChange(d.id); setEditing(null) }}
        onRemove={editing.copy ? undefined : () => { sets.remove(editing.def.id); onChange(BUILT_IN_SETS.find((s) => bases[s.basis])!.id); setEditing(null) }} />}
    </span>
  )
}

function ComponentSetDialog({ start, isNew, fresh, bases, standardColor, onSave, onRemove, onClose }: {
  start: ComponentSetDef; isNew: boolean; fresh: boolean
  bases: Partial<Record<Basis, BasisInfo>>; standardColor: (field: string, basis: Basis) => string
  onSave: (d: ComponentSetDef) => void; onRemove?: () => void; onClose: () => void
}) {
  const [draft, setDraft] = useState<ComponentSetDef>(() => (fresh ? { ...start, name: '' } : start))
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
  // each component's mean over the filters in view (all sites pooled), for the rows and the preview bar
  const stats = useMemo(() => new Map(info.available.map((f) => {
    const v = info.rows.map((r) => r[f]).filter((x): x is number => typeof x === 'number' && Number.isFinite(x))
    return [f, { n: v.length, mean: v.length ? d3.mean(v)! : NaN }]
  })), [info])
  const pm = useMemo(() => d3.mean(info.rows, (r) => (typeof r['PM2.5 mass'] === 'number' ? (r['PM2.5 mass'] as number) : undefined)), [info])
  const familyOf = (f: string) => info.meta.field_groups.find((g) => g.fields.includes(f))?.label ?? (f.startsWith('OC') || f.startsWith('EC') ? 'Carbon' : 'Other')

  const toggle = (f: string) => set({ fields: chosen.includes(f) ? chosen.filter((x) => x !== f) : [...chosen, f] })
  const move = (i: number, by: number) => {
    const j = i + by
    if (j < 0 || j >= chosen.length) return
    const next = [...chosen];[next[i], next[j]] = [next[j], next[i]]
    set({ fields: next })
  }
  const setColor = (f: string, c: string) => {
    const colors = { ...draft.colors }
    if (c) colors[f] = c
    else delete colors[f]
    set({ colors })
  }
  const switchBasis = (b: Basis) => { if (b !== draft.basis) set({ basis: b, fields: [...bases[b]!.available] }) }
  const startFrom = (s: BuiltInSet) => setDraft((d) => ({ ...d, basis: s.basis, fields: s.fields(bases[s.basis]!.available), name: d.name || '' }))
  const name = draft.name.trim() || `My ${draft.basis === 'recon' ? 'reconstruction' : 'species'} set`
  const ok = chosen.length > 0
  const ownColours = Object.keys(draft.colors ?? {}).filter((f) => chosen.includes(f)).length
  const offered = (Object.keys(bases) as Basis[])

  // the preview: one bar of the chosen components' means over the view, foot of the stack on the left
  const parts = chosen.map((f) => ({ f, v: stats.get(f)?.mean ?? NaN })).filter((p) => p.v > 0)
  const total = d3.sum(parts, (p) => p.v)

  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="export-dialog ratio-dialog grouping-dialog cset-dialog" role="dialog" aria-modal="true" aria-label={isNew ? 'New component set' : 'Edit a component set'}>
        <header>
          <div><strong>{isNew ? (fresh ? 'New component set' : 'Your copy of a ready-made set') : 'Edit component set'}</strong>
            <span className="export-title">what the stacked bars stack, in which order and colours</span></div>
          <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
        </header>
        <section className="ratio-presets">
          <div className="row-label">Start from</div>
          <div className="ratio-preset-list">
            {BUILT_IN_SETS.filter((s) => bases[s.basis]).map((s) => (
              <button key={s.id} type="button" className="chip ratio-preset" title={s.note} onClick={() => startFrom(s)}>{s.name}</button>
            ))}
          </div>
        </section>
        <section className="grouping-form">
          {offered.length > 1 && <Segmented label="Basis" value={draft.basis} options={offered} optionLabel={(b) => BASIS_LABEL[b]} onChange={switchBasis}
            title="SPARTAN reconstruction: the twelve components of SPARTAN's figure, which sum to each filter's PM2.5. Measured species: FTIR carbon, ions and elements as measured. One set takes one basis: the reconstruction's dust already holds the crustal elements, and its sodium is the sea-salt split" />}
          <label className="grouping-name"><span className="combobox-label">Name</span>
            <input className="select" value={draft.name} placeholder={name} onChange={(e) => set({ name: e.target.value })} />
          </label>
        </section>
        <div className="cset-preview" role="img" aria-label="The chosen components' mean over the filters in view">
          <div className="cset-preview-bar">
            {parts.map((p) => <span key={p.f} style={{ flexGrow: p.v, background: colorOf(p.f), color: inkOnFill(colorOf(p.f)) }} title={`${p.f}: ${fmt(p.v, 2)} µg/m³ (${((p.v / total) * 100).toFixed(1)}%)`}>
              {p.v / total > 0.08 ? p.f : ''}</span>)}
          </div>
          <span className="grouping-range">{parts.length ? `mean over ${info.rows.length.toLocaleString()} filters in view, all sites pooled: ${fmt(total, 1)} µg/m³${pm ? ` of PM2.5 ${fmt(pm, 1)}` : ''}` : 'no component with a positive mean in view'}</span>
        </div>
        <section className="cset-list">
          <div className="row-label">In the stack, foot first <span className="field-multi-n">{chosen.length}</span></div>
          {chosen.map((f, i) => (
            <div key={f} className="cset-row">
              <input type="checkbox" checked onChange={() => toggle(f)} aria-label={`Leave out ${f}`} />
              <BandColor label={f} color={colorOf(f)} standard={standard(f)} onChange={(c) => setColor(f, c)} onStandard={() => setColor(f, '')} />
              <span className="cset-name">{f}{draft.colors?.[f] && <small> · own colour</small>}</span>
              <span className="grouping-range">{draft.basis === 'measured' ? familyOf(f) : ''}</span>
              <span className="grouping-count">{fmt(stats.get(f)?.mean, 2)} µg/m³ · n {stats.get(f)?.n.toLocaleString()}</span>
              <span className="cset-move">
                <button type="button" className="btn quiet x-btn" disabled={i === 0} onClick={() => move(i, -1)} aria-label={`Move ${f} down the stack`} title="Lower in the stack">↓</button>
                <button type="button" className="btn quiet x-btn" disabled={i === chosen.length - 1} onClick={() => move(i, 1)} aria-label={`Move ${f} up the stack`} title="Higher in the stack">↑</button>
              </span>
            </div>
          ))}
          {!chosen.length && <p className="chart-note" style={{ margin: 0 }}>Tick at least one component below.</p>}
          {rest.length > 0 && <div className="row-label" style={{ marginTop: 6 }}>Not in the stack</div>}
          {rest.map((f) => (
            <div key={f} className="cset-row off">
              <input type="checkbox" checked={false} onChange={() => toggle(f)} aria-label={`Add ${f}`} />
              <span className="grouping-swatch" style={{ background: colorOf(f), cursor: 'default' }} aria-hidden="true" />
              <span className="cset-name">{f}</span>
              <span className="grouping-range">{draft.basis === 'measured' ? familyOf(f) : ''}</span>
              <span className="grouping-count">{fmt(stats.get(f)?.mean, 2)} µg/m³ · n {stats.get(f)?.n.toLocaleString()}</span>
              <span />
            </div>
          ))}
          {draft.basis === 'measured' && <p className="chart-note" style={{ margin: 0 }}>Measured species stack family by family (carbon, ions, crustal elements, trace metals), so the order here is the order within each family.</p>}
        </section>
        <div className="export-actions">
          {onRemove && <button type="button" className="btn" style={{ marginRight: 'auto' }} onClick={onRemove}>Delete set</button>}
          {ownColours > 0 && <button type="button" className="btn" onClick={() => set({ colors: {} })} title="Every component back to the chart's palette">Reset colours</button>}
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="button" className="btn primary" disabled={!ok} onClick={() => onSave({ ...draft, name, fields: chosen, colors: Object.fromEntries(Object.entries(draft.colors ?? {}).filter(([f]) => chosen.includes(f))) })}>
            {isNew ? 'Create and use' : 'Save'}</button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
