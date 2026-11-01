import { useMemo, useRef, useState } from 'react'
import { Dialog } from '@/components/Dialog'
import { TERMS } from '@/lib/glossary'
import { BookIcon, XIcon } from '@/components/Icons'

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
  const shown = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? TERMS.filter((t) => `${t.term} ${t.long ?? ''} ${t.text}`.toLowerCase().includes(s)) : TERMS
  }, [q])
  return (
    <Dialog onClose={onClose} className="export-dialog glossary-dialog" label="Glossary" initialFocus={input}>
        <header>
          <div><strong>Glossary</strong><span className="export-title">the terms behind the charts</span></div>
          <button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close" title="Close (Esc)"><XIcon /></button>
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
    </Dialog>
  )
}
