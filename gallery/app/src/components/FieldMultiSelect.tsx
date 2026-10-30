import { useCallback, useEffect, useId, useMemo, useRef, useState, type RefObject } from 'react'
import { fieldHelp } from '@/lib/glossary'
import { pickerName } from '@/lib/pmfGroups'
import { createPortal } from 'react-dom'
import { chemText } from '@/lib/format'
import { prettyUnit } from '@/components/FieldSelect'
import type { MetaFile } from '@/lib/types'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'

/**
 * Several fields at once (a correlogram's species, a matrix's rows): a button that says how many
 * are chosen, opening a popover with presets on top, then a searchable, grouped checkbox list of
 * the fields this view has. A preset is a named list (only the fields in view count); ticking a
 * field by hand makes the selection the reader's own. The chart owns the state:
 *
 *   <FieldMultiSelect meta={meta} available={fields} value={chosen} preset={preset}
 *     presets={PRESETS} onChange={(fields, preset) => …} />
 */
export interface FieldPreset { key: string; label: string; fields: (available: string[]) => string[]; note?: string }

export function FieldMultiSelect({ label = 'Fields', meta, available, value, preset, presets, onChange, max }: {
  label?: string
  meta: MetaFile
  /** the fields that can be chosen here (e.g. those with enough values) */
  available: string[]
  value: string[]
  /** the preset the selection came from; null when chosen by hand */
  preset: string | null
  presets: FieldPreset[]
  onChange: (fields: string[], preset: string | null) => void
  /** a soft cap, said on the button and in the popover (the fields past it are not drawn) */
  max?: number
}) {
  const [open, setOpen] = useState(false)
  const btn = useRef<HTMLButtonElement>(null)
  const autoFocusSearch = useRef(false)
  const id = useId()
  const close = useCallback((restoreFocus = true) => {
    setOpen(false)
    if (restoreFocus) btn.current?.focus({ preventScroll: true })
  }, [])
  const name = preset ? presets.find((p) => p.key === preset)?.label : null
  return (
    <span className="control">
      <span className="combobox-label" id={`${id}-label`}>{label}</span>
      <button ref={btn} type="button" className={`combobox-trigger${open ? ' open' : ''}`} aria-haspopup="dialog" aria-expanded={open} aria-controls={id} aria-labelledby={`${id}-label`}
        onPointerDown={e => { autoFocusSearch.current = e.pointerType === 'mouse' }}
        onKeyDown={e => { if (e.key === 'ArrowDown') { e.preventDefault(); autoFocusSearch.current = true; setOpen(true) } }}
        onClick={e => { if (e.detail === 0) autoFocusSearch.current = true; setOpen(!open) }}
        title="Choose the fields: a preset, or tick them one by one">
        <span className="combobox-value">{name ?? 'Custom'} · {max && value.length > max ? `${max} of ${value.length} drawn` : value.length}</span>
        <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && <Popover id={id} label={label} anchor={btn} autoFocusSearch={autoFocusSearch.current} meta={meta} available={available} value={value} preset={preset} presets={presets} onChange={onChange} max={max} onClose={close} />}
    </span>
  )
}

function Popover({ id, label, anchor, autoFocusSearch, meta, available, value, preset, presets, onChange, max, onClose }: {
  id: string; label: string; anchor: RefObject<HTMLElement>; autoFocusSearch: boolean; meta: MetaFile; available: string[]; value: string[]; preset: string | null
  presets: FieldPreset[]; onChange: (fields: string[], preset: string | null) => void; max?: number; onClose: (restoreFocus?: boolean) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const search = useRef<HTMLInputElement>(null)
  const [q, setQ] = useState('')
  const place = useFloatingMenu(true, anchor, box, 500)
  const positioned = place !== null
  useEffect(() => {
    if (positioned) (autoFocusSearch ? search.current : box.current)?.focus({ preventScroll: true })
  }, [positioned, autoFocusSearch])
  useEffect(() => {
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node) && !anchor.current?.contains(e.target as Node)) onClose(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose() } }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [anchor, onClose])
  const chosen = new Set(value)
  const s = q.trim().toLowerCase()
  const groups = useMemo(() => meta.field_groups
    .map((g) => ({ label: g.label, fields: g.fields.filter((f) => available.includes(f) && (!s || `${f} ${g.label}`.toLowerCase().includes(s))) }))
    .filter((g) => g.fields.length), [meta.field_groups, available, s])
  const toggle = (f: string) => onChange(chosen.has(f) ? value.filter((x) => x !== f) : [...value, f], null)
  const setGroup = (fs: string[], on: boolean) => onChange(on ? [...value, ...fs.filter((f) => !chosen.has(f))] : value.filter((f) => !fs.includes(f)), null)
  return createPortal(
    <div ref={box} id={id} tabIndex={-1} className="field-multi" role="dialog" aria-label="Choose fields" style={place ?? { position: 'fixed', visibility: 'hidden' }}>
      <div className="field-multi-title"><strong>{label}</strong><button type="button" className="btn" aria-label="Done choosing fields" onClick={() => onClose()}>Done</button></div>
      <div className="field-multi-presets">
        {presets.map((p) => {
          const n = p.fields(available).length
          return <button key={p.key} type="button" className={`chip${preset === p.key ? ' active' : ''}${n < 2 ? ' disabled' : ''}`} disabled={n < 2}
            title={`${p.note ?? p.label}: ${n} field${n === 1 ? '' : 's'} in view`} onClick={() => onChange(p.fields(available), p.key)}>{p.label}</button>
        })}
      </div>
      <div className="field-multi-head">
        <input ref={search} className="select" type="search" aria-label="Search fields" placeholder="Search fields…" value={q} onChange={(e) => setQ(e.target.value)} />
        <span className="field-multi-count">{value.length} chosen{max && value.length > max ? ` · the first ${max} are drawn` : ''}</span>
        <button type="button" className="btn quiet" onClick={() => onChange([], null)} disabled={!value.length}>Clear</button>
      </div>
      <div className="field-multi-list">
        {groups.map((g) => {
          const on = g.fields.filter((f) => chosen.has(f)).length
          return <div key={g.label} className="field-multi-group">
            <label className="field-multi-grouphead">
              <input type="checkbox" checked={on === g.fields.length} ref={(el) => { if (el) el.indeterminate = on > 0 && on < g.fields.length }}
                onChange={(e) => setGroup(g.fields, e.target.checked)} />
              {g.label} <span className="field-multi-n">{on}/{g.fields.length}</span>
            </label>
            {g.fields.map((f) => <label key={f} className="field-multi-item">
              <input type="checkbox" checked={chosen.has(f)} onChange={() => toggle(f)} />
              <span title={chemText(fieldHelp(f))}>{chemText(pickerName(f))}</span>
              {prettyUnit(meta.field_units?.[f]) && <span className="field-multi-n">{prettyUnit(meta.field_units?.[f])}</span>}
            </label>)}
          </div>
        })}
        {!groups.length && <p className="site-empty">No field matches “{q}”.</p>}
      </div>
    </div>,
    document.body,
  )
}
