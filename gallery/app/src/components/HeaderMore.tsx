import { useEffect, useRef, useState, type ReactNode } from 'react'
import { BookIcon, ViewsIcon } from '@/components/Icons'
import { GlossaryDialog } from '@/components/Glossary'
import { ViewsPanel } from '@/components/ViewsMenu'

/**
 * On a phone the header keeps the page menu and Search; the display toggles (`display`), Glossary
 * and Views fold into this ⋯ menu (styles.css shows it at 560 px and below, where the separate
 * buttons are hidden). Views opens in the same popover, the glossary in its own dialog as from the
 * header, the chart display settings in theirs (the menu stays open behind them).
 */
export function HeaderMore({ display }: { display?: ReactNode }) {
  const [open, setOpen] = useState<null | 'menu' | 'views'>(null)
  const [glossary, setGlossary] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    // the display toggles' settings popover floats outside the menu: working in it keeps the menu open
    const onDown = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node) && !(e.target as Element).closest?.('.display-menu')) setOpen(null) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !document.querySelector('.display-menu')) setOpen(null) }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey) }
  }, [open])
  return (
    <div className="views-menu header-more" ref={box}>
      <button type="button" className={`btn quiet icon-btn${open ? ' on' : ''}`} aria-haspopup="menu" aria-expanded={!!open} aria-label="More"
        title="Display, glossary and views" onClick={() => setOpen(open ? null : 'menu')}>⋯</button>
      {open === 'menu' && (
        <div className="views-pop header-more-pop" role="menu" aria-label="More">
          {display && <>
            <div className="views-section">Display</div>
            <div className="header-more-display">{display}</div>
            <div className="header-more-rule" />
          </>}
          <button type="button" role="menuitem" className="views-item header-more-item" onClick={() => { setOpen(null); setGlossary(true) }}>
            <span className="header-more-name"><BookIcon /> Glossary</span><span className="views-note">what FTIR, HIPS, TOR, MAC … mean</span>
          </button>
          <button type="button" role="menuitem" className="views-item header-more-item" onClick={() => setOpen('views')}>
            <span className="header-more-name"><ViewsIcon /> Views</span><span className="views-note">presets, saved views, copy link, embed</span>
          </button>
        </div>
      )}
      {open === 'views' && <div className="views-pop" role="dialog" aria-label="Views"><ViewsPanel onClose={() => setOpen(null)} /></div>}
      {glossary && <GlossaryDialog onClose={() => setGlossary(false)} />}
    </div>
  )
}
