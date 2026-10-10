import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'

/** A styled, select-only combobox; focus stays on the trigger while browsing options. */
export function Combobox<T extends string>({ label, value, options, onChange }: {
  label: string
  value: T
  options: readonly { key: T; label: string }[]
  onChange: (value: T) => void
}) {
  const id = useId()
  const root = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const selected = Math.max(0, options.findIndex(option => option.key === value))
  const [active, setActive] = useState(selected)

  useEffect(() => {
    if (!open) return
    const dismiss = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', dismiss)
    return () => document.removeEventListener('pointerdown', dismiss)
  }, [open])

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
          const match = options.findIndex(option => option.label.toLowerCase().startsWith(event.key.toLowerCase()))
          if (match >= 0) { event.preventDefault(); setActive(match); setOpen(true) }
        }
    }
  }

  return <div className="combobox-control" ref={root} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false)
  }}>
    <span className="combobox-label" id={`${id}-label`}>{label}</span>
    <div className="combobox">
      <button type="button" role="combobox" className={`combobox-trigger${open ? ' open' : ''}`}
        aria-labelledby={`${id}-label`} aria-expanded={open} aria-haspopup="listbox"
        aria-controls={`${id}-list`} aria-activedescendant={open ? `${id}-option-${active}` : undefined}
        onKeyDown={onKeyDown} onClick={() => { setActive(selected); setOpen(!open) }}>
        <span>{options[selected].label}</span>
        <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </button>
      {open && <div className="combobox-menu" role="listbox" id={`${id}-list`} aria-labelledby={`${id}-label`}>
        {options.map((option, index) => <button type="button" role="option" tabIndex={-1}
          key={option.key} id={`${id}-option-${index}`} aria-selected={option.key === value}
          className={`combobox-option${index === active ? ' active' : ''}`}
          onMouseEnter={() => setActive(index)} onMouseDown={event => event.preventDefault()} onClick={() => choose(index)}>
          <span>{option.label}</span><span className="combobox-check" aria-hidden="true">{option.key === value ? '✓' : ''}</span>
        </button>)}
      </div>}
    </div>
  </div>
}
