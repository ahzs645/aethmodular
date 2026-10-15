import { createContext, useContext, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/**
 * The left sidebar on tabs without the Filters panel (Baselines, Meeting, Blanks): App draws
 * the dock and a slot in it; a page puts its own controls there with `<SidebarControls>`,
 * so a page's switches (a version, a cohort, the page-wide baseline) sit with the other
 * filters instead of in a prose panel on the page. Without a sidebar (an embed, a share
 * build that drops it) the controls render where they are written, so they never vanish.
 */
const Ctx = createContext<{ node: HTMLElement | null; setNode: (n: HTMLElement | null) => void }>({ node: null, setNode: () => {} })

export function PageSidebarProvider({ children }: { children: ReactNode }) {
  const [node, setNode] = useState<HTMLElement | null>(null)
  return <Ctx.Provider value={{ node, setNode }}>{children}</Ctx.Provider>
}

/** where App's sidebar takes the page's controls */
export function PageSidebarSlot() {
  const { setNode } = useContext(Ctx)
  return <div ref={setNode} className="page-sidebar-slot" />
}

/** a page's controls, in the sidebar when there is one */
export function SidebarControls({ title, children }: { title?: string; children: ReactNode }) {
  const { node } = useContext(Ctx)
  const body = <section className="filter-section page-sidebar-section">
    {title && <div className="row-label page-sidebar-title">{title}</div>}
    <div className="page-sidebar-controls">{children}</div>
  </section>
  return node ? createPortal(body, node) : <div className="panel">{body}</div>
}
