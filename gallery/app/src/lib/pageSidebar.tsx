import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/**
 * The left sidebar on tabs without the Filters panel (Baselines, Meeting, Blanks): App draws
 * the dock and a slot in it; a page puts its own controls there with `<SidebarControls>`,
 * so a page's switches (a version, a cohort, the page-wide baseline) sit with the other
 * filters instead of in a prose panel on the page. Without a sidebar (an embed, a share
 * build that drops it) the controls render where they are written, so they never vanish.
 */
const Ctx = createContext<{ node: HTMLElement | null; setNode: (n: HTMLElement | null) => void; note: string | null; setNote: (n: string | null) => void }>({ node: null, setNode: () => {}, note: null, setNote: () => {} })

export function PageSidebarProvider({ children }: { children: ReactNode }) {
  const [node, setNode] = useState<HTMLElement | null>(null)
  const [note, setNote] = useState<string | null>(null)
  return <Ctx.Provider value={{ node, setNode, note, setNote }}>{children}</Ctx.Provider>
}

/** what a page shows on the sidebar's edge tab (SidebarControls `tabNote`), so its setting reads while the sidebar is closed */
export const usePageSidebarNote = () => useContext(Ctx).note

/** where App's sidebar takes the page's controls */
export function PageSidebarSlot() {
  const { setNode } = useContext(Ctx)
  return <div ref={setNode} className="page-sidebar-slot" />
}

/**
 * a page's controls, in the sidebar when there is one. `tabNote` goes on the sidebar's edge tab: a page whose
 * setting drives every chart (the Similarity target) keeps it in view while the sidebar is closed, which matters
 * in the single-chart view, where the charts' own options sit on the right and the page's on the left
 */
export function SidebarControls({ title, tabNote, children }: { title?: string; tabNote?: string; children: ReactNode }) {
  const { node, setNote } = useContext(Ctx)
  useEffect(() => {
    if (!tabNote) return
    setNote(tabNote)
    return () => setNote(null)
  }, [tabNote, setNote])
  const body = <section className="filter-section page-sidebar-section">
    {title && <div className="row-label page-sidebar-title">{title}</div>}
    <div className="page-sidebar-controls">{children}</div>
  </section>
  return node ? createPortal(body, node) : <div className="panel">{body}</div>
}
