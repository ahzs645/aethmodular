import { createContext, useContext, useState, type ReactNode } from 'react'
import { Segmented } from '@/components/ChartFrame'
import { BASIC_PANEL_AXES, PANEL_AXES, collapsePanelAxes, type PanelAxes, type PanelDim } from '@/lib/axes'

/**
 * Panel axes, the gallery's one setting for every chart that draws several crossplot panels
 * (per site, per season, side by side): Shared gives every panel the same range so slopes and
 * levels compare by eye; Own fits each panel to its own data. App provides it (the sidebar's Axes
 * row, `pax` in the URL); each multi-panel chart shows the same control in its own options while
 * it draws more than one panel, so changing it on one chart changes it on all.
 */
const Ctx = createContext<{ value: PanelAxes; set: (v: PanelAxes) => void } | null>(null)

export function PanelAxesProvider({ value, onChange, children }: { value: PanelAxes; onChange: (v: PanelAxes) => void; children: ReactNode }) {
  return <Ctx.Provider value={{ value, set: onChange }}>{children}</Ctx.Provider>
}

/** the gallery's setting; outside App (a test, a standalone page) each chart keeps its own */
export function usePanelAxes(): [PanelAxes, (v: PanelAxes) => void] {
  const c = useContext(Ctx)
  const [own, setOwn] = useState<PanelAxes>('Shared')
  return c ? [c.value, c.set] : [own, setOwn]
}

export const PANEL_AXES_TITLE = 'Several panels: Shared gives every panel the same range, so slopes and levels compare by eye across panels · Per site (per-season layout): the seasons of one site share a range, so seasons compare within a site · Own zooms each panel to its own data (tick values then differ between panels). One setting for every multi-panel chart (also under Axes in the sidebar)'

/**
 * The standard control; a chart includes it only while it draws more than one panel. `dims` are what its panels
 * split by: with both site and season (the scatterplot's per-season grid) it offers Shared / Per site / Own;
 * otherwise Shared / Own, the current setting shown as what it amounts to on this chart.
 */
export function PanelAxesControl({ dims = [] }: { dims?: PanelDim[] }) {
  const [value, set] = usePanelAxes()
  if (dims.length >= 2) return <Segmented label="Panel axes" value={value} options={PANEL_AXES} onChange={set} title={PANEL_AXES_TITLE} />
  return <Segmented label="Panel axes" value={collapsePanelAxes(value, dims[0] ?? null)} options={BASIC_PANEL_AXES} onChange={set} title={PANEL_AXES_TITLE} />
}
