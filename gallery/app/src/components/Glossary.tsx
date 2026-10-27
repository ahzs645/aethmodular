import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { TERMS } from '@/lib/glossary'
import { BookIcon } from '@/components/Icons'

/** The header's Glossary button and its dialog: every term the gallery uses, searchable. */
export function GlossaryButton() {
  const [open, setOpen] = useState(false)
  return <>
    <button type="button" className="btn quiet icon-btn" onClick={() => setOpen(true)} aria-label="Glossary" title="Glossary: what FTIR, HIPS, TOR, MAC, IMPROVE, SPARTAN … mean"><BookIcon /></button>
    {open && <GlossaryDialog onClose={() => setOpen(false)} />}
  </>
}

/** the glossary itself; components/HeaderMore opens it from the phone's ⋯ menu */
export function GlossaryDialog({ onClose }: { onClose: () => void }) {
  const [q, setQ] = useState('')
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => {
    input.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])
  const shown = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? TERMS.filter((t) => `${t.term} ${t.long ?? ''} ${t.text}`.toLowerCase().includes(s)) : TERMS
  }, [q])
  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="export-dialog glossary-dialog" role="dialog" aria-modal="true" aria-label="Glossary">
        <header>
          <div><strong>Glossary</strong><span className="export-title">the terms behind the charts</span></div>
          <button type="button" className="btn quiet" onClick={onClose} aria-label="Close" title="Close (Esc)">✕</button>
        </header>
        <input ref={input} className="select" type="search" placeholder="Search terms…" value={q} onChange={(e) => setQ(e.target.value)} />
        <dl className="glossary-list">
          {shown.map((t) => (
            <div key={t.term}>
              <dt>{t.term}{t.long && <span className="glossary-long"> · {t.long}</span>}</dt>
              <dd>{t.text}</dd>
            </div>
          ))}
          {!shown.length && <p className="site-empty">No term matches “{q}”.</p>}
        </dl>
      </div>
    </div>,
    document.body,
  )
}
