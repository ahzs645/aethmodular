import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { chemText } from '@/lib/format'
import type { ComboOption } from '@/components/Combobox'

/**
 * Combobox's several-at-once twin: the same trigger, and a list of ticks floating on the page
 * (portal, so a dialog or panel never clips it). Arrows move through the list, Space or Enter
 * ticks, Esc closes; lists longer than SEARCH_FROM get Combobox's search box. The trigger names
 * what is ticked ("N chosen" past two).
 */
const SEARCH_FROM = 10
const fold = (s: string) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '')

export function MultiCombobox<T extends string>({ label, value, options, onChange, title, placeholder = 'Choose…' }: {
  label: string
  value: readonly T[]
  options: readonly ComboOption<T>[]
  onChange: (value: T[]) => void
  title?: string
  placeholder?: string
}) {
  const id = useId()
  const root = useRef<HTMLDivElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)
  const [place, setPlace] = useState<CSSProperties | null>(null)
  const [active, setActive] = useState(0)
  const [query, setQuery] = useState('')
  const searchable = options.length > SEARCH_FROM
  const visible = options.map((_, i) => i).filter((i) => !query || fold(`${options[i].label} ${options[i].group ?? ''}`).includes(fold(query)))

  useEffect(() => {
    if (!open) { setQuery(''); return }
    const dismiss = (e: PointerEvent) => { const t = e.target as Node; if (!root.current?.contains(t) && !menu.current?.contains(t)) setOpen(false) }
    document.addEventListener('pointerdown', dismiss)
    return () => document.removeEventListener('pointerdown', dismiss)
  }, [open])
  useLayoutEffect(() => {
    if (!open) { setPlace(null); return }
    const update = () => {
      const r = trigger.current?.getBoundingClientRect()
      if (!r) return
      const up = window.innerHeight - r.bottom < 260 && r.top > window.innerHeight - r.bottom
      const w = menu.current?.offsetWidth ?? r.width
      setPlace({ position: 'fixed', left: Math.max(8, Math.min(r.left, window.innerWidth - w - 8)), minWidth: r.width, ...(up ? { bottom: window.innerHeight - r.top + 4 } : { top: r.bottom + 4 }) })
    }
    update()
    const onScroll = (e: Event) => { if (!menu.current?.contains(e.target as Node)) update() }
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', update)
    return () => { window.removeEventListener('scroll', onScroll, true); window.removeEventListener('resize', update) }
  }, [open])
  // the focus follows the active row (or the search box, while typing)
  useEffect(() => {
    if (!open || !place) return
    if (searchable && document.activeElement === menu.current?.querySelector('input[type=search]')) return
    menu.current?.querySelector<HTMLElement>(`#${CSS.escape(`${id}-option-${active}`)}`)?.focus()
  }, [open, place, active, id, searchable])

  const toggle = (key: T) => onChange(value.includes(key) ? value.filter((k) => k !== key) : [...value, key])
  const close = () => { setOpen(false); trigger.current?.focus() }
  const move = (by: number) => {
    if (!visible.length) return
    const at = visible.indexOf(active)
    setActive(visible[at < 0 ? 0 : (at + by + visible.length) % visible.length])
  }
  const onListKey = (e: KeyboardEvent) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); move(e.key === 'ArrowDown' ? 1 : -1) }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close() }
    else if (e.key === 'Tab') setOpen(false)
    else if ((e.key === 'Enter' || e.key === ' ') && (e.target as HTMLElement).getAttribute('role') === 'option') { e.preventDefault(); toggle(options[active].key) }
  }
  const names = options.filter((o) => value.includes(o.key)).map((o) => o.label)

  return <div className="combobox-control" ref={root} title={title}>
    {label && <span className="combobox-label" id={`${id}-label`}>{label}</span>}
    <div className="combobox">
      <button ref={trigger} type="button" className={`combobox-trigger${open ? ' open' : ''}`} aria-haspopup="listbox" aria-expanded={open} aria-controls={`${id}-list`}
        aria-labelledby={label ? `${id}-label` : undefined} aria-label={label ? undefined : title}
        onClick={() => { setActive(Math.max(0, options.findIndex((o) => value.includes(o.key)))); setOpen(!open) }}
        onKeyDown={(e) => { if (!open && (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); setActive(0); setOpen(true) } }}>
        <span className="combobox-value">{chemText(names.length === 0 ? placeholder : names.length <= 2 ? names.join(', ') : `${names.length} chosen`)}</span>
        <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && createPortal(<div className="combobox-menu floating" role="listbox" aria-multiselectable="true" id={`${id}-list`} ref={menu}
        style={place ?? { position: 'fixed', visibility: 'hidden' }} onKeyDown={onListKey}
        aria-labelledby={label ? `${id}-label` : undefined} aria-label={label ? undefined : title}>
        {searchable && <div className="combobox-search" role="presentation">
          <input type="search" value={query} placeholder="Search…" aria-label={`Search ${label || title || 'the options'}`} autoFocus
            onChange={(e) => { setQuery(e.target.value); const q = fold(e.target.value); const first = options.findIndex((o) => fold(`${o.label} ${o.group ?? ''}`).includes(q)); if (first >= 0) setActive(first) }} />
        </div>}
        {searchable && !visible.length && <div className="combobox-empty" role="presentation">No matches</div>}
        {visible.map((index, at) => { const o = options[index]; return <div key={o.key} role="presentation">
          {o.group && o.group !== options[visible[at - 1]]?.group && <div className="combobox-group" role="presentation">{o.group}</div>}
          <div role="option" tabIndex={-1} id={`${id}-option-${index}`} aria-selected={value.includes(o.key)} title={o.title}
            className={`combobox-option multi${index === active ? ' active' : ''}`}
            onMouseEnter={() => setActive(index)} onMouseDown={(e) => e.preventDefault()} onClick={() => { setActive(index); toggle(o.key) }}>
            <span className={`multi-tick${value.includes(o.key) ? ' on' : ''}`} aria-hidden="true">{value.includes(o.key) ? '✓' : ''}</span>
            <span className="combobox-text">{chemText(o.label)}</span>
          </div>
        </div> })}
      </div>, document.body)}
    </div>
  </div>
}
