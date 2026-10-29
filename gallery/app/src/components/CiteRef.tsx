import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type { SeasonSource, SourceChecked, SourceMatch, SourceScope } from '@/siteSeasons'

// the audit's qualifications in words (docs/site-seasonality-network.md)
const MATCH: Record<SourceMatch, string> = {
  exact: 'Same months',
  rounded: 'Months rounded to whole months',
  partial: 'Covers part of the year',
  inferred: 'Some months inferred',
  unverified: 'Month passage not yet read',
}
const SCOPE: Record<SourceScope, string> = {
  local: 'this city',
  regional: 'regional',
  national: 'national',
  transferred: "another city's study",
}
const CHECKED: Record<SourceChecked, string> = {
  'full text': 'checked in the full text',
  'indexed passage': 'checked in an indexed passage',
  abstract: 'checked in the abstract only',
  'not read': 'not yet read',
}

/**
 * A marker that shows a card on hover or focus. The card sits in a portal so the sidebar's scroll box
 * cannot clip it, and stays open while the pointer crosses into it.
 */
function Popover({ label, ariaLabel, className, href, children }: { label: ReactNode; ariaLabel: string; className: string; href?: string; children: ReactNode }) {
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
      <a ref={ref} className={className} href={href} target={href ? '_blank' : undefined} rel="noreferrer" tabIndex={0} aria-label={ariaLabel}
        onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide} onClick={(e) => e.stopPropagation()}>{label}</a>
      {at && createPortal(
        <div className="cite-pop" role="tooltip" style={{ left: at.left, top: at.top }} onMouseEnter={show} onMouseLeave={hide}>{children}</div>,
        document.body,
      )}
    </>
  )
}

/**
 * A numbered reference, "(1)": pointing at it shows the full citation and how far it carries for this
 * site (same months? this city? what was read?); clicking the number (or the card's title) opens it.
 */
export function CiteRef({ n, source }: { n: number; source: SeasonSource }) {
  const tags = source.paper
    ? [source.match && MATCH[source.match], source.scope && SCOPE[source.scope], source.checked && CHECKED[source.checked]].filter(Boolean).join(' · ')
    : ''
  // anything short of the same months, from this region, read in full is worth noticing
  const weak = source.paper && ((!!source.match && source.match !== 'exact') || source.scope === 'transferred'
    || source.checked === 'abstract' || source.checked === 'not read')
  return (
    <Popover label={`(${n})`} ariaLabel={`Reference ${n}: ${source.cite}`} className={`cite-ref${source.paper ? '' : ' agency'}`} href={source.url}>
      <a href={source.url} target="_blank" rel="noreferrer" className="cite-pop-title">{source.cite}</a>
      {source.title !== source.cite && <span>{source.title}</span>}
      {source.journal && <em>{source.journal}</em>}
      {tags && <span className={`cite-pop-tags${weak ? ' warn' : ''}`}>{tags}</span>}
      {source.monthsMatch && <span className="cite-pop-note">{source.monthsMatch}</span>}
      {source.quote && <q>{source.quote}</q>}
      {!source.paper && <span className="cite-pop-note">Agency or other source, not a peer-reviewed paper</span>}
    </Popover>
  )
}

/** "(i)" after a site's references: the caveats on its calendar as a whole. */
export function CalendarNoteRef({ note, site }: { note: string; site: string }) {
  return (
    <Popover label="(i)" ariaLabel={`About ${site || 'this'} calendar`} className="cite-ref agency">
      <span className="cite-pop-title">{site ? `${site}: about this calendar` : 'About this calendar'}</span>
      <span>{note}</span>
    </Popover>
  )
}
