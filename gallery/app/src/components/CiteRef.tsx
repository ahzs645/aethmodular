import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { SeasonSource } from '@/siteSeasons'

/**
 * A numbered reference, "(1)": pointing at it shows the full citation in a card, and clicking
 * the number (or the card's title) opens the paper. The card sits in a portal so the sidebar's
 * scroll box cannot clip it, and stays open while the pointer crosses into it.
 */
export function CiteRef({ n, source }: { n: number; source: SeasonSource }) {
  const ref = useRef<HTMLAnchorElement>(null)
  const [at, setAt] = useState<{ left: number; top: number } | null>(null)
  const timer = useRef<number>(0)
  const show = () => {
    window.clearTimeout(timer.current)
    const r = ref.current?.getBoundingClientRect()
    if (r) setAt({ left: Math.max(8, Math.min(r.left, window.innerWidth - 328)), top: r.bottom + 4 })
  }
  const hide = () => { timer.current = window.setTimeout(() => setAt(null), 150) }
  useEffect(() => () => window.clearTimeout(timer.current), [])
  return (
    <>
      <a ref={ref} className={`cite-ref${source.paper ? '' : ' agency'}`} href={source.url} target="_blank" rel="noreferrer"
        aria-label={`Reference ${n}: ${source.cite}`}
        onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide} onClick={(e) => e.stopPropagation()}>({n})</a>
      {at && createPortal(
        <div className="cite-pop" role="tooltip" style={{ left: at.left, top: at.top }} onMouseEnter={show} onMouseLeave={hide}>
          <a href={source.url} target="_blank" rel="noreferrer" className="cite-pop-title">{source.cite}</a>
          {source.title !== source.cite && <span>{source.title}</span>}
          {source.journal && <em>{source.journal}</em>}
          {source.quote && <q>{source.quote}</q>}
          {!source.paper && <span className="cite-pop-note">Agency source, not a peer-reviewed paper</span>}
        </div>,
        document.body,
      )}
    </>
  )
}
