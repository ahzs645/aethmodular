import { createContext, useContext, useEffect, useId, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/**
 * A page's own controls (a channel, a topic, a version, the page-wide baseline), in the side dock with the other
 * filters instead of in a prose panel on the page. A page writes them in `<SidebarControls>` and they land:
 *
 *  - in the dock's page tab (after Samples and Plot) on tabs with the Samples panel: SubsetBar draws the tab
 *    when a page has registered controls (or offers the site focus), named by the page (`tab`, default "Page");
 *  - in App's sidebar slot on tabs without the Samples panel (Baselines, Meeting, Blanks);
 *  - where they are written when there is neither (an embed, a share build that drops the dock), so they never vanish.
 */
type Reg = { id: string; tab: string }
type Ctx = {
  node: HTMLElement | null; setNode: (n: HTMLElement | null) => void
  tabNode: HTMLElement | null; setTabNode: (n: HTMLElement | null) => void
  plotNode: HTMLElement | null; setPlotNode: (n: HTMLElement | null) => void
  samplesNode: HTMLElement | null; setSamplesNode: (n: HTMLElement | null) => void
  note: string | null; setNote: (n: string | null) => void
  regs: Reg[]; setRegs: (f: (r: Reg[]) => Reg[]) => void
}
const Ctx = createContext<Ctx>({ node: null, setNode: () => {}, tabNode: null, setTabNode: () => {}, plotNode: null, setPlotNode: () => {}, samplesNode: null, setSamplesNode: () => {}, note: null, setNote: () => {}, regs: [], setRegs: () => {} })

export function PageSidebarProvider({ children }: { children: ReactNode }) {
  const [node, setNode] = useState<HTMLElement | null>(null)
  const [tabNode, setTabNode] = useState<HTMLElement | null>(null)
  const [plotNode, setPlotNode] = useState<HTMLElement | null>(null)
  const [samplesNode, setSamplesNode] = useState<HTMLElement | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [regs, setRegs] = useState<Reg[]>([])
  return <Ctx.Provider value={{ node, setNode, tabNode, setTabNode, plotNode, setPlotNode, samplesNode, setSamplesNode, note, setNote, regs, setRegs }}>{children}</Ctx.Provider>
}

/** what a page shows on the sidebar's edge tab (SidebarControls `tabNote`), so its setting reads while the sidebar is closed */
export const usePageSidebarNote = () => useContext(Ctx).note

/** the page tab's label while the open page has controls for it (the first `tab` given), else null */
export function usePageTab(): string | null {
  const { regs } = useContext(Ctx)
  return regs.length ? regs[0].tab : null
}

/** where App's sidebar takes the page's controls (tabs without the Samples panel) */
export function PageSidebarSlot() {
  const { setNode } = useContext(Ctx)
  return <div ref={setNode} className="page-sidebar-slot" />
}

/** where the dock's page tab takes them (SubsetBar); kept mounted while the tab is hidden so the controls keep their state */
export function PageTabSlot({ hidden }: { hidden: boolean }) {
  const { setTabNode } = useContext(Ctx)
  return <div ref={setTabNode} className="page-sidebar-slot" hidden={hidden} />
}

/** where the dock's Plot tab takes a page's unit controls (`PlotControls`), under its MAC row (SubsetBar) */
export function PlotTabSlot() {
  const { setPlotNode } = useContext(Ctx)
  return <div ref={setPlotNode} className="page-sidebar-slot" />
}

/** where the dock's Samples tab takes a page's sample presets (`SampleControls`), under its summary (SubsetBar) */
export function SamplesTabSlot() {
  const { setSamplesNode } = useContext(Ctx)
  return <div ref={setSamplesNode} className="page-sidebar-slot" />
}

/**
 * a page's presets for the samples (sites, calendar, dates, exclusions: Samples settings the page knows a
 * reproduction needs), at the top of the dock's Samples tab. Like PlotControls, they render nothing while that tab
 * is shut, and in place where there is no dock.
 */
export function SampleControls({ children }: { children: ReactNode }) {
  const { node, tabNode, samplesNode } = useContext(Ctx)
  if (samplesNode) return createPortal(children, samplesNode)
  return tabNode || node ? null : <div className="panel">{children}</div>
}

/**
 * a page's own unit choice that hangs on the MAC (HIPS as Fabs or as BC = Fabs / MAC), in the dock's Plot tab under
 * the MAC row instead of the Page tab. The Plot tab draws its options only while open, so while it is shut (and the
 * slot gone) these render nothing where there is a dock; with no dock at all (an embed) they are drawn in place.
 */
export function PlotControls({ children }: { children: ReactNode }) {
  const { node, tabNode, plotNode } = useContext(Ctx)
  if (plotNode) return createPortal(children, plotNode)
  return tabNode || node ? null : <div className="panel">{children}</div>
}

/**
 * a page's controls, in the dock when there is one. `title` heads the group; `tab` names the dock's page tab
 * (default "Page"). `tabNote` goes on the sidebar's edge tab: a page whose setting drives every chart (the
 * Similarity target) keeps it in view while the sidebar is closed, which matters in the single-chart view,
 * where the charts' own options sit on the right and the page's on the left
 */
export function SidebarControls({ title, tab = 'Page', tabNote, children }: { title?: string; tab?: string; tabNote?: string; children: ReactNode }) {
  const { node, tabNode, setNote, setRegs } = useContext(Ctx)
  const id = useId()
  useEffect(() => {
    setRegs((r) => [...r, { id, tab }])
    return () => setRegs((r) => r.filter((x) => x.id !== id))
  }, [id, tab, setRegs])
  useEffect(() => {
    if (!tabNote) return
    setNote(tabNote)
    return () => setNote(null)
  }, [tabNote, setNote])
  const body = <section className="filter-section page-sidebar-section">
    {title && <div className="row-label page-sidebar-title">{title}</div>}
    <div className="page-sidebar-controls">{children}</div>
  </section>
  const target = tabNode ?? node
  return target ? createPortal(body, target) : <div className="panel">{body}</div>
}
