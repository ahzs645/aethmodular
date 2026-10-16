import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { chemText } from '@/lib/format'

export interface ComboOption<T extends string> {
  key: T
  label: string
  /** heading the option sits under (consecutive options share one heading) */
  group?: string
  /** muted text at the right edge, e.g. a count */
  hint?: string
}

/**
 * The gallery's one dropdown: a styled, select-only combobox, compact enough to sit in a
 * chart's control row (the height of a native select) and used for every picker, so they
 * all look and behave alike. Focus stays on the trigger while browsing; arrows, Home/End,
 * Enter/Space and type-ahead (several letters) work as in a native select. Long lists
 * scroll, can carry group headings, and open upwards when there is no room below. The list
 * floats on the page (a portal at the trigger's place), so a dialog, sidebar or chart frame
 * that clips its content never cuts it off.
 */
export function Combobox<T extends string>({ label, value, options, onChange, title, minWidth }: {
  label: string
  value: T
  options: readonly ComboOption<T>[]
  onChange: (value: T) => void
  title?: string
  /** a floor for the trigger width, so a row of pickers does not jitter as values change */
  minWidth?: number
}) {
  const id = useId()
  const root = useRef<HTMLDivElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [place, setPlace] = useState<CSSProperties | null>(null)
  const found = options.findIndex((option) => option.key === value)
  const selected = Math.max(0, found)
  const [active, setActive] = useState(selected)
  const typed = useRef({ text: '', at: 0 })

  useEffect(() => {
    if (!open) return
    const dismiss = (event: PointerEvent) => {
      const t = event.target as Node
      if (!root.current?.contains(t) && !menu.current?.contains(t)) setOpen(false)
    }
    document.addEventListener('pointerdown', dismiss)
    return () => document.removeEventListener('pointerdown', dismiss)
  }, [open])
  // the list sits under the trigger (above it when the window has no room below), inside the
  // window's edges, and follows the trigger while the page or a panel scrolls
  useLayoutEffect(() => {
    if (!open) { setPlace(null); return }
    const trigger = root.current?.querySelector('.combobox-trigger')
    if (!trigger) return
    const update = () => {
      const r = trigger.getBoundingClientRect()
      const up = window.innerHeight - r.bottom < 260 && r.top > window.innerHeight - r.bottom
      const w = menu.current?.offsetWidth ?? r.width
      const left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8))
      setPlace({ position: 'fixed', left, minWidth: r.width, ...(up ? { bottom: window.innerHeight - r.top + 4 } : { top: r.bottom + 4 }) })
    }
    update()
    const onScroll = (e: Event) => { if (!menu.current?.contains(e.target as Node)) update() }
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', update)
    return () => { window.removeEventListener('scroll', onScroll, true); window.removeEventListener('resize', update) }
  }, [open])
  useEffect(() => {
    if (open) menu.current?.querySelector(`#${CSS.escape(`${id}-option-${active}`)}`)?.scrollIntoView({ block: 'nearest' })
  }, [open, active, id])

  const choose = (index: number) => {
    onChange(options[index].key)
    setOpen(false)
  }
  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowUp':
        event.preventDefault()
        setActive(open ? (active + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length : selected)
        setOpen(true)
        break
      case 'Home':
      case 'End':
        event.preventDefault()
        setActive(event.key === 'Home' ? 0 : options.length - 1)
        setOpen(true)
        break
      case 'Enter':
      case ' ':
        event.preventDefault()
        if (open) choose(active)
        else { setActive(selected); setOpen(true) }
        break
      case 'Escape':
        if (open) { event.preventDefault(); event.stopPropagation(); setOpen(false) }
        break
      case 'Tab':
        setOpen(false)
        break
      default:
        if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
          // type-ahead: letters typed within a short pause build one search string
          const now = Date.now()
          typed.current = { text: (now - typed.current.at < 700 ? typed.current.text : '') + event.key.toLowerCase(), at: now }
          const match = options.findIndex((option) => option.label.toLowerCase().startsWith(typed.current.text))
          if (match >= 0) { event.preventDefault(); setActive(match); setOpen(true) }
        }
    }
  }

  return <div className="combobox-control" ref={root} title={title} onBlur={(event) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false)
  }}>
    {label && <span className="combobox-label" id={`${id}-label`}>{label}</span>}
    <div className="combobox">
      <button type="button" role="combobox" className={`combobox-trigger${open ? ' open' : ''}`} style={minWidth ? { minWidth } : undefined}
        aria-labelledby={label ? `${id}-label` : undefined} aria-label={label ? undefined : title}
        aria-expanded={open} aria-haspopup="listbox"
        aria-controls={`${id}-list`} aria-activedescendant={open ? `${id}-option-${active}` : undefined}
        onKeyDown={onKeyDown} onClick={() => { setActive(selected); setOpen(!open) }}>
        <span className="combobox-value">{chemText(found >= 0 ? options[found].label : value)}</span>
        <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && createPortal(<div className="combobox-menu floating" role="listbox" id={`${id}-list`} ref={menu}
        style={place ?? { position: 'fixed', visibility: 'hidden' }}
        aria-labelledby={label ? `${id}-label` : undefined} aria-label={label ? undefined : title}>
        {options.map((option, index) => <div key={option.key} role="presentation">
          {option.group && option.group !== options[index - 1]?.group && <div className="combobox-group" role="presentation">{option.group}</div>}
          <button type="button" role="option" tabIndex={-1}
            id={`${id}-option-${index}`} aria-selected={option.key === value}
            className={`combobox-option${index === active ? ' active' : ''}`}
            onMouseEnter={() => setActive(index)} onMouseDown={(event) => event.preventDefault()} onClick={() => choose(index)}>
            <span className="combobox-check" aria-hidden="true">{option.key === value ? '✓' : ''}</span>
            <span className="combobox-text">{chemText(option.label)}</span>
            {option.hint && <span className="combobox-hint">{option.hint}</span>}
          </button>
        </div>)}
      </div>, document.body)}
    </div>
  </div>
}
