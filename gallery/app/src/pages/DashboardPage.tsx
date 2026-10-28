import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { FrameExtrasContext, Segmented, type FrameExtras } from '@/components/ChartFrame'
import { FieldSelect } from '@/components/FieldSelect'
import { AxisFieldsProvider, type AxisRole } from '@/components/AxisFieldPopover'
import { DashHeader, type DashFilters, type SheetActions } from '@/components/dashboard/DashHeader'
import { dashboardChart, dashboardCharts, startFields, STARTER, type DashboardChart, type DashboardData } from '@/components/dashboard/registry'
import { useHeaderHeight } from '@/hooks/useSidePanel'
import { useMediaQuery } from '@/hooks/useMediaQuery'
import { ChartSite, ChartFit, LEGEND_MODES, TileLegend, type LegendMode } from '@/lib/chartFit'
import { addTile, arrange, COLS, nextId, nextTileId, placeTile, removeTile, stepTile, ROWS, rowsUsed, tileKeyPrefix, updateTile, type DashState, type Scope, type Sheet, type Tile } from '@/lib/dashboard'
import { clearScopedParams, copyScopedParams, ParamScope, useSearchParam } from '@/lib/url'
import { XIcon } from '@/components/Icons'

const GAP = 10
/** at or below this width the dashboard is one column and the page scrolls (Grafana and Observable do the same) */
const PHONE = 820
/** a grid row's height on a phone, where the window no longer sets it */
const PHONE_ROW = 60

type Role = 'grid' | 'hero' | 'rail' | 'strip' | 'max' | 'phone'
type Drag = { id: string; mode: 'move' | 'resize'; px: number; py: number; tile: Tile }

/**
 * The Dashboard tab: the reader's own pages of gallery charts, fitted to the window. Three
 * scopes (one site, compare sites, the network), each with named sheets of tiles. The header
 * holds the scope's site picker and the common filters beside the page menu (components/
 * dashboard/DashHeader); the full Filters panel is one click away, so every tab's filtering
 * applies.
 *
 * Grid: tiles placed on a 12 × 12 grid the size of the window. In Edit the grid's cells show;
 * drag a tile by its bar to place it, or its corner to resize: the tiles in the way move down
 * and everything floats up (lib/dashboard arrange, as in Grafana / react-grid-layout).
 * Focus: headline numbers on top, one chart large, the rest in a rail. A tile's ⋯ (on hover)
 * holds its name, actions and options. On a phone: one column in reading order, the page
 * scrolls. State is in the URL (lib/dashboard), so Views → Save / Share keep a dashboard.
 */
export function DashboardPage({ data, dash, setDash, filters, onScope, onPickSite, display }: {
  data: DashboardData
  dash: DashState
  setDash: (d: DashState) => void
  filters: DashFilters | null
  onScope: (s: Scope) => void
  onPickSite: (code: string) => void
  display?: ReactNode
}) {
  const { scope, layout } = dash
  const sheets = dash.sheets[scope] ?? starterSheets(scope, data)
  const active = Math.min(dash.sheet, sheets.length - 1)
  const sheet = sheets[active]
  const tiles = sheet.tiles
  const setSheets = (next: Sheet[] | null, sheetIdx = active) => setDash({ ...dash, sheet: sheetIdx, sheets: { ...dash.sheets, [scope]: next } })
  const setTiles = (next: Tile[]) => setSheets(sheets.map((s, i) => (i === active ? { ...s, tiles: next } : s)))
  const [edit, setEdit] = useState(false)
  const [adding, setAdding] = useState(false)
  const [maxId, setMaxId] = useState<string | null>(null)
  const [focusId, setFocusId] = useState<string | null>(null)
  const phone = usePhone()
  useEffect(() => { setMaxId(null); setFocusId(null) }, [scope, active])

  // presenting: no header and no tile chrome, the window (full screen where the browser allows);
  // → / ← (or Page Down / Up, space) step through the scope's sheets, Esc ends it
  const [present, setPresent] = useState(false)
  const startPresent = () => {
    setEdit(false); setMaxId(null); setPresent(true)
    document.documentElement.requestFullscreen?.().catch(() => { /* not allowed here: present in the window */ })
  }
  const stopPresent = () => { setPresent(false); if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {}) }
  const goSheet = (by: number) => setDash({ ...dash, sheet: (active + by + sheets.length) % sheets.length })
  useEffect(() => {
    if (!present) return
    const html = document.documentElement
    html.classList.add('dash-present')
    const key = (e: KeyboardEvent) => {
      if ((e.target as Element)?.closest?.('input, textarea, [role="listbox"]')) return
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); goSheet(1) }
      else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); goSheet(-1) }
      else if (e.key === 'Escape') stopPresent()
    }
    // Esc in full screen leaves full screen without a keydown: end presenting with it
    const left = () => { if (!document.fullscreenElement) setPresent(false) }
    window.addEventListener('keydown', key)
    document.addEventListener('fullscreenchange', left)
    return () => { html.classList.remove('dash-present'); window.removeEventListener('keydown', key); document.removeEventListener('fullscreenchange', left) }
  }) // eslint-disable-line react-hooks/exhaustive-deps

  // the page never scrolls, except on a phone; the grid fills the window under the header
  useEffect(() => {
    document.documentElement.classList.add('dash-mode')
    return () => document.documentElement.classList.remove('dash-mode')
  }, [])
  useEffect(() => {
    if (!maxId) return
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setMaxId(null) }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [maxId])
  // the header's slot (App's Shell) appears after the first commit
  const [slot, setSlot] = useState<HTMLElement | null>(null)
  useEffect(() => setSlot(document.getElementById('dash-bar')), [])

  const top = useHeaderHeight()
  const mainRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 1200, h: 700 })
  useLayoutEffect(() => {
    const el = mainRef.current
    if (!el) return
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight })
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const rowPx = phone ? PHONE_ROW : Math.max(36, (box.h - (ROWS - 1) * GAP) / ROWS)
  const colPx = (box.w - (COLS - 1) * GAP) / COLS
  const span = (n: number, unit: number) => n * unit + (n - 1) * GAP

  // ---- arranging: the tile follows the pointer in whole cells; the rest make room as it goes
  const [drag, setDrag] = useState<Drag | null>(null)
  const [preview, setPreview] = useState<Tile[] | null>(null)
  const startDrag = (id: string, mode: Drag['mode'], e: React.PointerEvent) => {
    const t = tiles.find((x) => x.id === id)
    if (!t || e.button !== 0) return
    e.preventDefault()
    setDrag({ id, mode, px: e.clientX, py: e.clientY, tile: t })
  }
  useEffect(() => {
    if (!drag) return
    const step = (e: PointerEvent) => {
      const dx = Math.round((e.clientX - drag.px) / (colPx + GAP)), dy = Math.round((e.clientY - drag.py) / (rowPx + GAP))
      const t = drag.tile
      return drag.mode === 'move' ? placeTile(tiles, drag.id, t.x + dx, Math.max(0, t.y + dy))
        : updateTile(tiles, drag.id, { w: Math.max(1, t.w + dx), h: Math.max(1, t.h + dy) })
    }
    const move = (e: PointerEvent) => setPreview(step(e))
    const up = (e: PointerEvent) => { setTiles(step(e)); setPreview(null); setDrag(null) }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up, { once: true })
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up) }
  }, [drag]) // eslint-disable-line react-hooks/exhaustive-deps
  const shown = preview ?? tiles
  // the keyboard: arrows move the focused tile a cell, Shift + arrows resize it, Delete removes it;
  // the tile keeps the focus as it moves (its bar refocuses after the grid re-renders)
  const [kbFocus, setKbFocus] = useState<string | null>(null)
  const keyMove = (t: Tile, e: React.KeyboardEvent) => {
    const d = ({ ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] } as Record<string, [number, number]>)[e.key]
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); remove(t.id); return }
    if (!d) return
    e.preventDefault()
    setKbFocus(t.id)
    const dir = d[0] < 0 ? 'left' : d[0] > 0 ? 'right' : d[1] < 0 ? 'up' : 'down'
    setTiles(e.shiftKey ? updateTile(tiles, t.id, { w: Math.max(1, t.w + d[0]), h: Math.max(1, t.h + d[1]) }) : stepTile(tiles, t.id, dir))
  }

  // ---- tiles and sheets
  const prefix = (id: string, sh = sheet) => tileKeyPrefix(scope, sh.id, id)
  const add = (chart: DashboardChart) => {
    setTiles(addTile(tiles, { kind: chart.kind, w: chart.size[0], h: chart.size[1], fields: startFields(chart, data) }))
    setAdding(false)
  }
  const remove = (id: string) => { clearScopedParams(prefix(id)); setTiles(removeTile(tiles, id)) }
  const duplicate = (t: Tile) => {
    copyScopedParams(prefix(t.id), prefix(nextTileId(tiles)))
    setTiles(addTile(tiles, { kind: t.kind, w: t.w, h: t.h, fields: t.fields }))
  }
  const sheetActions: SheetActions = {
    names: sheets.map((s) => s.name),
    active,
    onPick: (i) => setDash({ ...dash, sheet: i }),
    onAdd: () => { setSheets([...sheets, { id: nextId('s', sheets), name: `Sheet ${sheets.length + 1}`, tiles: [] }], sheets.length); setAdding(true) },
    onDuplicate: () => {
      const id = nextId('s', sheets)
      sheet.tiles.forEach((t) => copyScopedParams(prefix(t.id), tileKeyPrefix(scope, id, t.id)))
      setSheets([...sheets, { ...sheet, id, name: `${sheet.name} copy` }], sheets.length)
    },
    onRename: (name) => setSheets(sheets.map((s, i) => (i === active ? { ...s, name } : s))),
    onDelete: () => {
      clearScopedParams(`${scope}.${sheet.id}.`)
      setSheets(sheets.filter((_, i) => i !== active), Math.max(0, active - 1))
    },
    onReset: () => { sheets.forEach((s) => clearScopedParams(`${scope}.${s.id}.`)); setSheets(null, 0) },
  }

  // focus: the headline numbers in a strip across the top, one chart large, the rest in a rail
  const strip = tiles.filter((t) => t.kind === 'stat')
  const charts = tiles.filter((t) => t.kind !== 'stat')
  const hero = charts.find((t) => t.id === focusId) ?? charts[0]
  const rail = charts.filter((t) => t !== hero)
  const stripH = strip.length ? span(2, rowPx) : 0
  const bodyH = box.h - (stripH ? stripH + GAP : 0)
  const railN = Math.min(rail.length, 3)
  const railH = railN ? Math.max(150, (bodyH - (railN - 1) * GAP) / railN) : 0

  const tileView = (t: Tile, w: number, h: number, role: Role) => {
    const chart = dashboardChart(t.kind)
    const style: React.CSSProperties | undefined = role === 'grid' ? { gridColumn: `${t.x + 1} / span ${t.w}`, gridRow: `${t.y + 1} / span ${t.h}` }
      : role === 'rail' ? { height: h, flex: 'none' } : role === 'strip' ? { height: h, flex: '1 1 0' } : role === 'phone' ? { height: h } : undefined
    if (!chart) return <div key={t.id} className="dash-tile panel" style={style}>
      <p className="frame-empty">A chart this gallery no longer has (“{t.kind}”).</p>
      <button type="button" className="btn" onClick={() => remove(t.id)}>Remove</button>
    </div>
    return <TileView key={`${scope}-${sheet.id}-${t.id}`} scopePrefix={prefix(t.id)} tile={t} chart={chart} data={data} width={w} height={h} style={style}
      edit={edit && role === 'grid'} role={role} dragging={drag?.id === t.id}
      onMoveStart={(e) => startDrag(t.id, 'move', e)} onResizeStart={(e) => startDrag(t.id, 'resize', e)}
      onKey={(e) => keyMove(t, e)} keepFocus={kbFocus === t.id}
      onChange={(patch) => setTiles(updateTile(tiles, t.id, patch))}
      onDuplicate={() => duplicate(t)} onRemove={() => remove(t.id)}
      onMaximize={() => setMaxId(maxId === t.id ? null : t.id)} onFocus={role === 'rail' ? () => setFocusId(t.id) : undefined} />
  }
  const maxTile = maxId ? tiles.find((t) => t.id === maxId) : undefined
  const reach = rowsUsed(shown)
  const gridVars = { ['--col' as string]: `${colPx}px`, ['--row' as string]: `${rowPx}px`, ['--gap' as string]: `${GAP}px` }

  let body: ReactNode
  if (maxTile) body = <div className="dash-max">{tileView(maxTile, box.w, box.h, 'max')}</div>
  else if (phone) body = <div className="dash-phone" style={{ gap: GAP }}>
    {(layout === 'focus' && hero ? [...strip, hero, ...rail] : tiles).map((t) => tileView(t, box.w, Math.max(120, t.h * PHONE_ROW), 'phone'))}
  </div>
  else if (layout === 'grid') body = (
    <div className={`dash-grid${edit ? ' editing' : ''}`} style={{ gridAutoRows: rowPx, gap: GAP, ...gridVars, minHeight: edit ? span(Math.max(ROWS, reach + 2), rowPx) : undefined }}>
      {shown.map((t) => tileView(t, span(t.w, colPx), span(t.h, rowPx), 'grid'))}
      {edit && <div className="dash-fold" style={{ top: ROWS * (rowPx + GAP) - GAP / 2 }} title="The bottom of the window: tiles below it scroll"><span>window</span></div>}
    </div>
  )
  else body = (
    <div className="dash-focus-wrap" style={{ gap: GAP }}>
      {strip.length > 0 && <div className="dash-strip" style={{ gap: GAP, height: stripH }}>
        {strip.map((t) => tileView(t, (box.w - (strip.length - 1) * GAP) / strip.length, stripH, 'strip'))}</div>}
      {hero && <div className="dash-focus" style={{ gap: GAP, height: bodyH }}>
        <div className="dash-hero" style={{ flexBasis: rail.length ? '68%' : '100%' }}>{tileView(hero, rail.length ? box.w * 0.68 : box.w, bodyH, 'hero')}</div>
        {rail.length > 0 && <div className="dash-rail" style={{ gap: GAP }}>{rail.map((t) => tileView(t, box.w * 0.32 - GAP, railH, 'rail'))}</div>}
      </div>}
    </div>
  )

  return (
    <>
      {slot && createPortal(<DashHeader meta={data.meta} filters={filters} scope={scope} onScope={onScope}
        onPickSite={onPickSite} layout={layout} onLayout={(l) => setDash({ ...dash, layout: l })} edit={edit && !phone} onEdit={setEdit}
        onAdd={() => setAdding(true)} rowsUsed={reach} sheets={sheetActions} display={display} onPresent={phone ? undefined : startPresent} />, slot)}
      <div ref={mainRef} className={`dash-main${edit ? ' editing' : ''}${phone ? ' phone' : ''}`}
        style={phone ? undefined : { height: present ? 'calc(100vh - 20px)' : `calc(100vh - ${top}px - 12px)` }}>
        {tiles.length === 0 && <div className="dash-empty">
          <p>“{sheet.name}” has no charts yet.</p>
          <span><button type="button" className="btn" onClick={() => setAdding(true)}>+ Add chart</button></span>
        </div>}
        {body}
      </div>
      {adding && <ChartCatalog data={data} onPick={add} onClose={() => setAdding(false)} />}
      {present && <div className="dash-present-bar" role="toolbar" aria-label="Presenting">
        <button type="button" className="btn quiet" onClick={() => goSheet(-1)} disabled={sheets.length < 2} aria-label="Previous sheet">‹</button>
        <span>{sheet.name} <span className="dash-count">{active + 1}/{sheets.length}</span></span>
        <button type="button" className="btn quiet" onClick={() => goSheet(1)} disabled={sheets.length < 2} aria-label="Next sheet">›</button>
        <button type="button" className="btn quiet x-btn" onClick={stopPresent} title="Stop presenting (Esc)"><XIcon /></button>
      </div>}
    </>
  )
}

const starterSheets = (scope: Scope, data: DashboardData): Sheet[] => STARTER[scope].map((s, i) => ({
  id: `s${i + 1}`,
  name: s.name,
  tiles: arrange(s.tiles.reduce<Tile[]>((acc, t) => {
    const chart = dashboardChart(t.kind)
    // a starter's fields where this network has them all, else the chart's own start fields
    const known = !t.fields || data.meta.fields.length === 0 || t.fields.every((f) => data.meta.fields.includes(f))
    return chart ? addTile(acc, { kind: t.kind, w: t.w, h: t.h, fields: (known && t.fields) || startFields(chart, data) }) : acc
  }, [])),
}))

/** true once the element is within a screen of the viewport (always true when not `lazy`) */
function useNear(ref: React.RefObject<HTMLElement>, lazy: boolean) {
  const [near, setNear] = useState(!lazy)
  useEffect(() => {
    if (!lazy || near || !ref.current) return
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) setNear(true) }, { rootMargin: '100% 0px' })
    io.observe(ref.current)
    return () => io.disconnect()
  }, [lazy, near, ref])
  return near
}

const usePhone = () => useMediaQuery(`(max-width: ${PHONE}px)`)

/** the Add chart menu: every registered kind, under the tab it comes from */
function ChartCatalog({ data, onPick, onClose }: { data: DashboardData; onPick: (c: DashboardChart) => void; onClose: () => void }) {
  const groups = useMemo(() => {
    const out = new Map<string, DashboardChart[]>()
    for (const c of dashboardCharts()) if (!c.available || c.available(data)) out.set(c.group, [...(out.get(c.group) ?? []), c])
    return [...out]
  }, [data])
  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])
  const fieldsNote = (c: DashboardChart) => (c.fields === 'pair' ? 'x and y' : c.fields === 'single' ? 'one field' : c.fields === 'trio' ? 'three fields' : '')
  return (
    <div className="map-dialog-backdrop" onPointerDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="panel dash-catalog" role="dialog" aria-label="Add a chart">
        <header className="frame-head">
          <div><h2>Add a chart</h2><p className="subtitle">It goes in the first free space, smaller if that is all there is, or takes half of the largest tile when the sheet is full. Drag it anywhere in Edit; change its fields from its ⋯ menu.</p></div>
          <div className="frame-actions"><button type="button" className="btn quiet x-btn" onClick={onClose} aria-label="Close"><XIcon /></button></div>
        </header>
        <div className="dash-catalog-groups">
          {groups.map(([group, charts]) => (
            <section key={group}>
              <h3>{group}</h3>
              {charts.map((c) => (
                <button key={c.kind} type="button" className="dash-card" onClick={() => onPick(c)}>
                  <strong>{c.label}</strong>
                  <span>{c.blurb}</span>
                  {fieldsNote(c) && <em>{fieldsNote(c)}</em>}
                </button>
              ))}
            </section>
          ))}
        </div>
      </div>
    </div>
  )
}

/**
 * One tile: the chart in its compact ChartFrame (name, actions and options behind ⋯),
 * fitted to the tile. The tile tells the chart a height (lib/chartFit) and corrects it after
 * each draw by what is left over, so legends and readouts of any height fill the tile; a
 * chart that cannot shrink enough scrolls inside its tile, never the page.
 */
function TileView({ tile, chart, data, width, height, style, scopePrefix, edit, role, dragging, onMoveStart, onResizeStart, onKey, keepFocus, onChange, onDuplicate, onRemove, onMaximize, onFocus }: {
  tile: Tile
  chart: DashboardChart
  data: DashboardData
  width: number
  height: number
  style?: React.CSSProperties
  scopePrefix: string
  edit: boolean
  role: Role
  dragging: boolean
  onMoveStart: (e: React.PointerEvent) => void
  onResizeStart: (e: React.PointerEvent) => void
  onKey: (e: React.KeyboardEvent) => void
  /** the tile the keyboard is arranging: its bar takes the focus back after each move */
  keepFocus: boolean
  onChange: (patch: Partial<Omit<Tile, 'id'>>) => void
  onDuplicate: () => void
  onRemove: () => void
  onMaximize: () => void
  onFocus?: () => void
}) {
  const siteName = useMemo(() => scopePrefix.startsWith('site.') && new Set(data.rows.map(r => r.site)).size === 1 ? data.rows[0].site : null, [scopePrefix, data.rows])
  const n = chart.fields === 'pair' ? 2 : chart.fields === 'single' ? 1 : chart.fields === 'trio' ? 3 : 0
  const start = startFields(chart, data)
  const fields = Array.from({ length: n }, (_, i) => tile.fields[i] ?? start[i])
  // x and y never show the same field: picking the other one's field swaps the two (as in App)
  const setField = (i: number, f: string) => {
    const next = [...fields]
    const other = next.indexOf(f)
    if (other >= 0 && other !== i) next[other] = next[i]
    next[i] = f
    onChange({ fields: next })
  }
  // the tile's axis titles open the field popover on the tile's own fields
  const axisValue = { x: fields[0] ?? '', y: fields[1] ?? '', field: fields[0] ?? '' }
  const axisSet = (role: AxisRole, f: string) => setField(role === 'y' ? 1 : 0, f)

  // where the chart's legends go (⋯ menu), kept with the tile's other URL keys
  const [legend, setLegend] = useSearchParam<LegendMode>(`${scopePrefix}legend`, 'auto', LEGEND_MODES)
  const legendReserve = Math.round(Math.min(220, Math.max(130, width * 0.28)))

  // ---- fit: a first guess, then corrected by the frame's leftover (or overflow) after each draw
  const minFit = chart.kind === 'stat' || chart.kind === 'sitesummary' ? 16 : 120
  const ref = useRef<HTMLDivElement>(null)
  const [fit, setFit] = useState(() => Math.max(minFit, height - 70))
  const tries = useRef(0)
  const key = `${width}|${height}|${tile.kind}|${fields.join('|')}|${legend}`
  const lastKey = useRef(key)
  if (lastKey.current !== key) { lastKey.current = key; tries.current = 0 }
  // new rows (a filter, the data still arriving) start the correction over, in this render: an
  // effect would run after the measuring below and leave a chart sized for the loading state
  const lastData = useRef<[unknown, unknown]>([data.rows, data.meta])
  if (lastData.current[0] !== data.rows || lastData.current[1] !== data.meta) { lastData.current = [data.rows, data.meta]; tries.current = 0 }
  // a chart that loads its own data (the daily weather, the wind roses) redraws by itself, without
  // this tile rendering again: when its contents change the tile measures again, and the correction
  // starts over if it had run out (at most 12 times in any 5 s, so a chart that redraws on every
  // correction cannot loop)
  const [, redraw] = useState(0)
  useEffect(() => {
    // the tile, not its panel: a chart whose data is still coming draws no frame at first
    const root = ref.current
    if (!root) return
    const resets: number[] = []
    const mo = new MutationObserver(() => {
      const now = performance.now()
      while (resets.length && now - resets[0] > 5000) resets.shift()
      if (resets.length >= 12) return
      resets.push(now)
      if (tries.current >= 6) tries.current = 0
      redraw((n) => n + 1)
    })
    mo.observe(root, { childList: true, subtree: true })
    return () => mo.disconnect()
  }, [tile.kind])
  useLayoutEffect(() => {
    const panel = ref.current?.querySelector(':scope > .panel') as HTMLElement | null
    if (!panel || tries.current >= 6) return
    const top = panel.getBoundingClientRect().top
    let bottom = top
    for (const c of panel.children) {
      const el = c as HTMLElement
      if (getComputedStyle(el).position === 'absolute') continue
      bottom = Math.max(bottom, el.getBoundingClientRect().bottom)
    }
    const pad = parseFloat(getComputedStyle(panel).paddingBottom) || 0
    // overflowing: the panel's own overflow, which counts margins that spill past the last child;
    // otherwise the room left under the last child
    const over = panel.scrollHeight - panel.clientHeight
    const slack = over > 0 ? -over : panel.clientHeight - (bottom - top + pad + panel.scrollTop)
    if (Math.abs(slack) <= 2) return
    // never taller than the tile: past that a chart only scrolls
    const next = Math.min(Math.max(minFit, height), Math.max(minFit, Math.round(fit + slack)))
    if (next === fit) return
    tries.current += 1
    setFit(next)
  })

  // a short name for the bar and the menu
  const subtitle = chart.fields === 'pair' ? `${fields[1]} vs ${fields[0]}` : chart.fields === 'single' && chart.kind !== 'stat' ? fields[0] : undefined
  const menu = (close: () => void) => <>
    {onFocus && <button type="button" className="btn quiet" onClick={() => { close(); onFocus() }}>◱ Show large</button>}
    <button type="button" className="btn quiet" onClick={() => { close(); onMaximize() }}>{role === 'max' ? '⤡ Back' : '⤢ Full window'}</button>
    <button type="button" className="btn quiet" onClick={() => { close(); onDuplicate() }}>⧉ Duplicate</button>
    <button type="button" className="btn quiet" onClick={() => { close(); onRemove() }}>✕ Remove</button>
    {chart.kind !== 'stat' && chart.kind !== 'sitecards' && <span className="tile-legend-mode">
      <Segmented label="Legend" value={legend} options={[...LEGEND_MODES]} onChange={setLegend}
        optionLabel={(m) => ({ auto: 'Auto', bottom: 'Bottom', right: 'Right', off: 'Off' }[m as LegendMode])}
        title="Auto: the first few and +N more · Bottom: every item · Right: a column beside the chart (where the chart has its legend with the plot) · Off" />
    </span>}
  </>
  const pickers: ReactNode = n === 0 ? null : <span className="dash-fields">
    {chart.fields === 'pair'
      ? <><FieldSelect label="x" value={fields[0]} meta={data.meta} onChange={(f) => setField(0, f)} />
        <FieldSelect label="y" value={fields[1]} meta={data.meta} onChange={(f) => setField(1, f)} /></>
      : fields.map((f, i) => <FieldSelect key={i} label={n === 1 ? 'Measurement' : `Field ${i + 1}`} value={f} meta={data.meta} onChange={(v) => setField(i, v)} />)}
  </span>
  const barRef = useRef<HTMLDivElement>(null)
  useEffect(() => { if (edit && keepFocus && document.activeElement !== barRef.current) barRef.current?.focus() })
  const editBar = edit ? <div ref={barRef} className="tile-edit-bar" onPointerDown={onMoveStart} onKeyDown={onKey} tabIndex={0} role="button"
    aria-label={`${chart.label}, ${tile.w} by ${tile.h} at column ${tile.x + 1}, row ${tile.y + 1}. Arrow keys move it, Shift and arrows resize, Delete removes`}
    title="Drag to move · arrow keys move, Shift + arrows resize, Delete removes">
    <span aria-hidden>⠿</span> <strong>{chart.label}</strong>{(subtitle ?? (chart.kind === 'stat' ? fields[0] : null)) && <span className="frame-sub">{subtitle ?? fields[0]}</span>}
    <span className="tile-size">{tile.w} × {tile.h}</span>
  </div> : undefined
  const extras: FrameExtras = { compact: true, menu, controls: pickers, subtitle, editBar }

  // on a phone (one long column) a tile draws once it comes near the screen, and then stays
  const near = useNear(ref, role === 'phone')

  // did the chart draw its own frame? if not, the tile draws one with its menu
  const [framed, setFramed] = useState(true)
  useLayoutEffect(() => { const has = !!ref.current?.querySelector(':scope > .panel:not(.dash-fallback)'); if (has !== framed) setFramed(has) })

  return (
    <div ref={ref} className={`dash-tile legend-${legend}${dragging ? ' dragging' : ''}${edit ? ' editing' : ''}${chart.kind === 'stat' ? ' stat' : ''}`}
      style={{ ...style, ['--legend-w' as string]: `${legendReserve}px` }}>
      <ParamScope prefix={scopePrefix}>
        <FrameExtrasContext.Provider value={extras}>
          <AxisFieldsProvider meta={data.meta} value={axisValue} set={axisSet}>
            <ChartSite name={siteName}><TileLegend mode={legend} reserve={legendReserve}><ChartFit height={fit}>
              {near ? chart.render(data, fields.map(data.eff)) : <section className="panel compact"><p className="frame-empty">{chart.label}</p></section>}
            </ChartFit></TileLegend></ChartSite>
          </AxisFieldsProvider>
        </FrameExtrasContext.Provider>
      </ParamScope>
      {!framed && <section className="panel compact dash-fallback">
        <div className="tile-head">{editBar}</div>
        <p className="frame-empty"><strong>{chart.label}</strong>: nothing to draw for this subset.</p>
        <div className="tile-menu-actions">{menu(() => {})}</div>
        {pickers && <div className="frame-controls">{pickers}</div>}
      </section>}
      {edit && <span className="dash-resize" onPointerDown={onResizeStart} title="Drag to resize" aria-hidden />}
    </div>
  )
}
