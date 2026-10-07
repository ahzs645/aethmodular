import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useSearchParam } from '@/lib/url'
import { useHeaderHeight } from '@/hooks/useSidePanel'

interface Entry { id: string; label: string }

/**
 * A sticky table of contents for the long research tabs. It reads the chart
 * titles (`section.panel h2`) out of its own content after render, so a new
 * ChartFrame appears here without being registered anywhere. Wide screens get
 * a side column that marks the section in view; narrow ones a sticky jump menu.
 */
export function PageToc({ children }: { children: ReactNode }) {
  const mainRef = useRef<HTMLDivElement>(null)
  const jumpRef = useRef<HTMLDivElement>(null)
  const [navigation, setNavigation] = useSearchParam<'shown' | 'hidden'>('toc', 'shown', ['shown', 'hidden'])
  const showNavigation = navigation === 'shown'
  const [entries, setEntries] = useState<Entry[]>([])
  const [active, setActive] = useState<string | null>(null)
  // sit just below the sticky app header, whatever height its tab row wraps to
  const top = useHeaderHeight()

  // collect headings; charts load their data asynchronously, so re-scan on DOM changes
  useEffect(() => {
    const root = mainRef.current
    if (!root) return
    const scan = () => {
      const next: Entry[] = []
      root.querySelectorAll<HTMLElement>('section.panel').forEach((sec, i) => {
        const h = sec.querySelector('h2')
        const label = h?.firstChild?.textContent?.trim() || h?.textContent?.trim()
        if (!label) return
        if (!sec.id) sec.id = `sec-${i}-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40)}`
        next.push({ id: sec.id, label })
      })
      setEntries((prev) => (prev.length === next.length && prev.every((e, i) => e.id === next[i].id && e.label === next[i].label) ? prev : next))
    }
    scan()
    const mo = new MutationObserver(scan)
    mo.observe(root, { childList: true, subtree: true })
    return () => mo.disconnect()
  }, [])

  // the active entry is the last section whose top has passed under the header
  useEffect(() => {
    const onScroll = () => {
      let current: string | null = entries[0]?.id ?? null
      for (const e of entries) {
        const el = document.getElementById(e.id)
        if (el && el.getBoundingClientRect().top - top - jumpH() - 24 <= 0) current = e.id
      }
      setActive(current)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [entries, top, showNavigation])

  // on narrow screens the sticky jump menu also covers the top of the page
  const jumpH = () => (jumpRef.current && jumpRef.current.offsetParent ? jumpRef.current.offsetHeight : 0)
  const go = (id: string) => {
    const el = document.getElementById(id)
    if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - top - jumpH() - 12, behavior: 'smooth' })
  }

  return (
    <div className={`toc-layout${entries.length > 1 && showNavigation ? ' has-toc' : ''}`}>
      <div ref={mainRef} style={{ minWidth: 0 }}>
        {entries.length > 1 && (
          <div ref={jumpRef} className="toc-controls" style={{ top }}>
            {showNavigation && <label className="toc-select control">
              Jump to
              <select className="select" value={active ?? ''} onChange={(e) => go(e.target.value)}>
                {entries.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}
              </select>
            </label>}
            <button type="button" className="btn quiet" aria-expanded={showNavigation}
              title={showNavigation ? 'Hide section navigation and give charts the full width' : 'Show section navigation'}
              onClick={() => setNavigation(showNavigation ? 'hidden' : 'shown')}>
              {showNavigation ? 'Hide sidebar' : 'Show sidebar'}
            </button>
          </div>
        )}
        {children}
      </div>
      {entries.length > 1 && showNavigation && (
        <nav className="toc" style={{ top: top + 8 }} aria-label="Sections on this tab">
          <p className="toc-title">On this tab</p>
          <ol>
            {entries.map((e) => (
              <li key={e.id}>
                {/* a button, not an #anchor: the URL hash already holds the app state */}
                <button type="button" className={`toc-link ${active === e.id ? 'active' : ''}`} onClick={() => go(e.id)}>{e.label}</button>
              </li>
            ))}
          </ol>
        </nav>
      )}
    </div>
  )
}
