import { createContext, useContext, useId, useLayoutEffect, useMemo, useRef, useSyncExternalStore, type ReactNode } from 'react'
import { panelAxesKey, type PanelAxes, type PanelDim } from '@/lib/axes'
import { unionExtent, type Extent } from '@/lib/facets'
import { PanelAxesControl, usePanelAxes } from '@/lib/panelAxes'
import type { FitGrouping } from '@/lib/siteRegression'

/**
 * The panel group: what the panels of one figure share, provided by both layouts (components/Panels for a composite,
 * components/FacetGrid for small multiples) so every multi-panel figure behaves the same way.
 *
 *  - Axis ranges. With `share` on, each shared chart inside (Crossplot, SweepPlot, SpectrumPlot) files its own data
 *    range here and draws the union of its range group instead (useGroupDomain). The range groups follow the gallery's
 *    Panel axes setting (`pax`, the sidebar's Axes row, lib/panelAxes): Shared, Per site (panels with the same `site`)
 *    or Own. A chart's own `domain`/`yDomain` props still win. `share` may give a shape per axis
 *    (`{ y: ([lo]) => [Math.max(0.5, lo - 0.02), 1] }`): the chart then draws exactly that range, made from the union.
 *  - One fit grouping. Crossplots inside draw the group's `grouping` when it is set; otherwise each hoists its fit
 *    grouping box to the group (useGroupControl), which draws one box for all of them.
 *  - Hoisted controls. Any chart can hand the group a control (useGroupControl): it is drawn once, in the group's
 *    control row above the panels, from what every panel registered.
 *
 * The legend slot (`legend`) and `show`/`order` live on the layouts themselves.
 */

/** an axis range made from the union of the panels' own ranges */
export type AxisShape = (union: Extent) => Extent
/** which axes the panels share: both, one, or per axis `true` (the chart pads and rounds the union) or a shape */
export type ShareAxes = boolean | 'x' | 'y' | { x?: boolean | AxisShape; y?: boolean | AxisShape }
type Axis = 'x' | 'y'
type Shapes = Partial<Record<Axis, true | AxisShape>>

const normalise = (s: ShareAxes | undefined): Shapes | null => {
  if (!s) return null
  if (s === true) return { x: true, y: true }
  if (s === 'x' || s === 'y') return { [s]: true }
  const o: Shapes = {}
  for (const a of ['x', 'y'] as const) if (s[a]) o[a] = s[a] === true ? true : (s[a] as AxisShape)
  return Object.keys(o).length ? o : null
}

interface ControlReg { id: string; value: unknown; render: (values: unknown[]) => ReactNode }

/** registrations live outside React state, so filing a range re-renders only the panels whose union changed */
class GroupStore {
  private regs = new Map<string, { key: string; extent: Extent }>()
  private unions = new Map<string, Extent | null>()
  private controls = new Map<string, ControlReg>()
  private listeners = new Set<() => void>()
  private controlListeners = new Set<() => void>()
  controlVersion = 0

  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  subscribeControls = (fn: () => void) => { this.controlListeners.add(fn); return () => { this.controlListeners.delete(fn) } }
  get = (key: string): Extent | null => this.unions.get(key) ?? null

  private recompute(key: string) {
    const next = unionExtent([...this.regs.values()].filter((r) => r.key === key).map((r) => r.extent))
    const prev = this.unions.get(key) ?? null
    if (prev === next || (prev && next && prev[0] === next[0] && prev[1] === next[1])) return false
    this.unions.set(key, next)
    return true
  }
  set(id: string, key: string, extent: Extent | null) {
    const old = this.regs.get(id)
    if (extent) this.regs.set(id, { key, extent }); else this.regs.delete(id)
    let changed = this.recompute(key)
    if (old && old.key !== key) changed = this.recompute(old.key) || changed
    if (changed) this.listeners.forEach((fn) => fn())
  }
  setControl(regId: string, reg: ControlReg | null) {
    if (reg) this.controls.set(regId, reg); else if (!this.controls.delete(regId)) return
    this.controlVersion++
    this.controlListeners.forEach((fn) => fn())
  }
  /** each control id once: the render function last registered, given every panel's value */
  controlNodes(): ReactNode[] {
    const byId = new Map<string, ControlReg[]>()
    for (const r of this.controls.values()) byId.set(r.id, [...(byId.get(r.id) ?? []), r])
    return [...byId.entries()].map(([id, regs]) => <span key={id} className="panel-group-control">{regs[regs.length - 1].render(regs.map((r) => r.value))}</span>)
  }
}

interface GroupValue {
  store: GroupStore
  shapes: Shapes | null
  mode: PanelAxes
  /** a fit grouping every crossplot inside draws; undefined: the gallery's, with one hoisted control */
  grouping?: FitGrouping | 'none'
}
const GroupCtx = createContext<GroupValue | null>(null)
/** the panel a chart sits in: its key (Own) and site (Per site) */
const SlotCtx = createContext<{ key: string; site?: string } | null>(null)

/** the group a layout provides (components/Panels, FacetGrid): call once per layout, render <PanelGroupProvider> */
export function usePanelGroupValue({ share, grouping }: { share?: ShareAxes; grouping?: FitGrouping | 'none' }): GroupValue {
  const store = useRef<GroupStore>()
  if (!store.current) store.current = new GroupStore()
  const [mode] = usePanelAxes()
  // a shape passed inline is a new function every render; the group keeps the first per axis kind
  const shapeKey = typeof share === 'object' && share ? `x:${typeof share.x}|y:${typeof share.y}` : String(share)
  const shapes$ = useRef(share)
  shapes$.current = share
  const shapes = useMemo(() => {
    const s = normalise(shapes$.current)
    if (!s) return null
    // call through the ref, so the latest inline shape is used without a new context value each render
    const live = (a: Axis): true | AxisShape | undefined => s[a] === true ? true : s[a] ? (u: Extent) => {
      const cur = normalise(shapes$.current)?.[a]
      return typeof cur === 'function' ? cur(u) : u
    } : undefined
    return { x: live('x'), y: live('y') }
  }, [shapeKey]) // eslint-disable-line react-hooks/exhaustive-deps
  return useMemo(() => ({ store: store.current!, shapes, mode, grouping }), [shapes, mode, grouping])
}

export const PanelGroupProvider = ({ value, children }: { value: GroupValue; children: ReactNode }) =>
  <GroupCtx.Provider value={value}>{children}</GroupCtx.Provider>

/** one panel of a group (the layouts wrap each panel, and its ⤢ copy, in one) */
export function PanelSlot({ panelKey, site, children }: { panelKey: string; site?: string; children: ReactNode }) {
  const v = useMemo(() => ({ key: panelKey, site }), [panelKey, site])
  return <SlotCtx.Provider value={v}>{children}</SlotCtx.Provider>
}

/** inside a panel group (Panels or FacetGrid)? */
export const useInPanelGroup = () => useContext(GroupCtx) !== null

const noSubscribe = () => () => {}

/**
 * A shared chart's range on one axis: files this chart's own data range (`own`, null for none or when the chart has a
 * fixed range) with its group, and returns the range to draw. `final` is true when the group shaped it (draw it as it
 * is); otherwise pad and round it the way the chart pads its own. Null outside a sharing group, under Own, or before
 * the first panel has filed a range: draw your own.
 */
export function useGroupDomain(axis: Axis, own: Extent | null): { domain: Extent; final: boolean } | null {
  const g = useContext(GroupCtx), slot = useContext(SlotCtx)
  const id = useId()
  const shape = g?.shapes?.[axis]
  // Own with no shape changes nothing: skip the round trip
  const active = !!g && !!slot && !!shape && !(g.mode === 'Own' && shape === true)
  const key = active ? `${panelAxesKey(g!.mode, slot!)}|${axis}` : ''
  const lo = own?.[0], hi = own?.[1]
  const ok = active && lo != null && hi != null && Number.isFinite(lo) && Number.isFinite(hi)
  useLayoutEffect(() => {
    if (!g) return
    g.store.set(id, key, ok ? [lo!, hi!] : null)
  }, [g, id, key, ok, lo, hi])
  useLayoutEffect(() => () => g?.store.set(id, '', null), [g, id])
  const union = useSyncExternalStore(g ? g.store.subscribe : noSubscribe, () => (active && g ? g.store.get(key) : null))
  return useMemo(() => {
    if (!active || !union) return null
    return typeof shape === 'function' ? { domain: shape(union), final: true } : { domain: union, final: false }
  }, [active, union, shape])
}

/**
 * Hoist a control to the group: every panel registers its `value` under `id`, and the group draws
 * `render(values)` once, in its control row. Returns true when hoisted (inside a group): the chart then leaves
 * out its own copy. `value` null withdraws this panel's registration (nothing to offer).
 *
 *   const hoisted = useGroupControl('fit-grouping', { site: multiSite }, (vs) => <FitGroupingControl only={…} />)
 *   {!hoisted && <FitGroupingControl … />}
 */
export function useGroupControl<T>(id: string, value: T | null, render: (values: T[]) => ReactNode): boolean {
  const g = useContext(GroupCtx)
  const reg = useId()
  const render$ = useRef(render)
  render$.current = render
  const sig = value === null ? null : JSON.stringify(value)
  useLayoutEffect(() => {
    if (!g) return
    g.store.setControl(reg, sig === null ? null : { id, value: JSON.parse(sig), render: (vs) => render$.current(vs as T[]) })
  }, [g, reg, id, sig])
  useLayoutEffect(() => () => g?.store.setControl(reg, null), [g, reg])
  return !!g
}

/** the group's fit grouping for the crossplots inside (undefined: none set, use the gallery's) */
export const useGroupGrouping = () => useContext(GroupCtx)?.grouping

/**
 * The group's control row, above the panels: the controls the charts hoisted (one each) and, while ranges are
 * shared across several panels, the Panel axes setting. A `.frame-controls` row: left out of exports.
 */
export function PanelGroupControls({ value, axesControl, dims, extra }: { value: GroupValue; axesControl: boolean; dims?: PanelDim[]; extra?: ReactNode }) {
  const v = useSyncExternalStore(value.store.subscribeControls, () => value.store.controlVersion)
  const nodes = useMemo(() => value.store.controlNodes(), [value.store, v]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!nodes.length && !axesControl && !extra) return null
  return <div className="frame-controls panel-group-controls">
    {extra}
    {axesControl && <PanelAxesControl dims={dims} />}
    {nodes}
  </div>
}
