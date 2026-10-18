/**
 * The Dashboard tab's state: which scope it shows (one site, several, the whole network),
 * how the tiles are laid out (a grid that fits the window, or one chart in focus with the
 * rest in a rail), and each scope's sheets, each a named set of tiles. All of it lives in
 * the URL hash, so Views → Save / Share / Embed keep a whole dashboard:
 *
 *   dscope=site&dsheet=1&dsite=Overview:s1:t1:stat:0:0:3:2:PM2.5%20mass,t2:…;Carbon:s2:…
 *
 * A scope's param is its sheets joined by ';', each `name:id:tiles`; a tile is
 * `id:kind:x:y:w:h:field…` on a 12-column grid whose 12 rows fill the window; names and
 * fields are URI-encoded like the ratios param. A tile's chart controls are the chart's own
 * URL keys under `<scope>.<sheet>.<tile>.` (lib/url ParamScope), so two scatterplots never
 * share one. Pure functions only; pages/DashboardPage draws it.
 */

export const SCOPES = ['site', 'sites', 'network'] as const
export type Scope = (typeof SCOPES)[number]
export const SCOPE_LABEL: Record<Scope, string> = { site: 'Site', sites: 'Compare', network: 'Network' }

export const LAYOUTS = ['grid', 'focus'] as const
export type Layout = (typeof LAYOUTS)[number]

/** the grid: 12 columns, and 12 rows that fill the window (more rows scroll) */
export const COLS = 12
export const ROWS = 12

export interface Tile {
  id: string
  /** a chart kind from components/dashboard/registry */
  kind: string
  /** column and row of the top-left cell (0-based), and the columns and rows it spans */
  x: number
  y: number
  w: number
  h: number
  /** the tile's fields: [x, y] for a crossplot, [field] for a one-measurement chart, [] otherwise */
  fields: string[]
}

export interface Sheet { id: string; name: string; tiles: Tile[] }

export interface DashState {
  scope: Scope
  layout: Layout
  /** the open sheet of the open scope (an index into its sheets) */
  sheet: number
  /** each scope's sheets; null = that scope's starter sheets */
  sheets: Record<Scope, Sheet[] | null>
}

const SCOPE_KEY: Record<Scope, string> = { site: 'dsite', sites: 'dsites', network: 'dnet' }
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const int = (v: string | undefined, fallback: number, lo: number, hi: number) => {
  const n = Number(v)
  return v !== undefined && v !== '' && Number.isInteger(n) ? clamp(n, lo, hi) : fallback
}

// ---- tiles

export const parseTiles = (v: string): Tile[] => {
  const tiles: Tile[] = []
  for (const part of v.split(',')) {
    const [id, kind, x, y, w, h, ...fields] = part.split(':')
    if (!id || !kind || tiles.some((t) => t.id === id)) continue
    try {
      const ww = int(w, 6, 1, COLS)
      tiles.push({ id, kind, x: int(x, 0, 0, COLS - ww), y: int(y, 0, 0, 999), w: ww, h: int(h, 4, 1, 24), fields: fields.map(decodeURIComponent) })
    } catch { /* a mangled field: drop the tile, keep the rest */ }
  }
  return arrange(tiles)
}

export const tilesParam = (tiles: Tile[]): string =>
  tiles.map((t) => [t.id, t.kind, t.x, t.y, t.w, t.h, ...t.fields.map(encodeURIComponent)].join(':')).join(',')

// ---- sheets and the whole state

export const parseSheets = (v: string | null): Sheet[] | null => {
  if (v == null) return null
  const sheets: Sheet[] = []
  for (const part of v.split(';')) {
    const [name, id, ...rest] = part.split(':')
    if (!id || sheets.some((s) => s.id === id)) continue
    try { sheets.push({ id, name: decodeURIComponent(name) || 'Sheet', tiles: rest.length ? parseTiles(rest.join(':')) : [] }) } catch { /* skip */ }
  }
  return sheets.length ? sheets : null
}

export const sheetsParam = (sheets: Sheet[] | null): string | null =>
  sheets === null ? null : sheets.map((s) => `${encodeURIComponent(s.name)}:${s.id}:${tilesParam(s.tiles)}`).join(';')

export const parseDash = (p: URLSearchParams): DashState => ({
  scope: (SCOPES as readonly string[]).includes(p.get('dscope') ?? '') ? p.get('dscope') as Scope : 'sites',
  layout: p.get('dlay') === 'focus' ? 'focus' : 'grid',
  sheet: int(p.get('dsheet') ?? undefined, 0, 0, 99),
  sheets: Object.fromEntries(SCOPES.map((s) => [s, parseSheets(p.get(SCOPE_KEY[s]))])) as Record<Scope, Sheet[] | null>,
})

/** the hash params for a dashboard; the defaults (compare sites, grid, first sheet, starter sheets) are left out */
export const dashParams = (d: DashState): Record<string, string | null> => ({
  dscope: d.scope === 'sites' ? null : d.scope,
  dlay: d.layout === 'grid' ? null : d.layout,
  dsheet: d.sheet ? String(d.sheet) : null,
  ...Object.fromEntries(SCOPES.map((s) => [SCOPE_KEY[s], sheetsParam(d.sheets[s])])),
})

/** s1, t1 …: the lowest number not in use, so a removed one's URL keys are free again */
export const nextId = (prefix: string, used: { id: string }[]): string => {
  const ids = new Set(used.map((t) => t.id))
  let n = 1
  while (ids.has(`${prefix}${n}`)) n += 1
  return `${prefix}${n}`
}
export const nextTileId = (tiles: Tile[]) => nextId('t', tiles)

// ---- the grid: positions, collisions, gravity

const hits = (a: Pick<Tile, 'x' | 'y' | 'w' | 'h'>, b: Pick<Tile, 'x' | 'y' | 'w' | 'h'>) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
const reading = (a: Tile, b: Tile) => a.y - b.y || a.x - b.x

/**
 * Settle a layout the way Grafana and react-grid-layout do: the tile being moved or resized
 * (`moved`) keeps the place it was dropped; every other tile that now overlaps is pushed down
 * below it; then everything floats up as far as it can (no holes). Returns the tiles in
 * reading order (top to bottom, left to right), which is also the order Focus and phones use.
 */
export function arrange(tiles: Tile[], moved?: string): Tile[] {
  const fit = (t: Tile): Tile => ({ ...t, w: clamp(t.w, 1, COLS), x: clamp(t.x, 0, COLS - clamp(t.w, 1, COLS)), y: Math.max(0, t.y), h: Math.max(1, t.h) })
  const first = tiles.filter((t) => t.id === moved).map(fit)
  const placed: Tile[] = [...first]
  for (const t of tiles.filter((t) => t.id !== moved).map(fit).sort(reading)) {
    const c = { ...t }
    for (let o = placed.find((p) => hits(p, c)); o; o = placed.find((p) => hits(p, c))) c.y = o.y + o.h
    placed.push(c)
  }
  // gravity: in reading order (the moved tile first among equals), each floats up until it touches one above
  const out: Tile[] = []
  for (const t of placed.sort((a, b) => reading(a, b) || (a.id === moved ? -1 : b.id === moved ? 1 : 0))) {
    const c = { ...t }
    while (c.y > 0 && !out.some((o) => hits(o, { ...c, y: c.y - 1 }))) c.y -= 1
    while (out.some((o) => hits(o, c))) c.y += 1
    out.push(c)
  }
  return out.sort(reading)
}

/** the first place a w × h tile fits without moving anything (top to bottom, left to right) */
export function firstFit(tiles: Tile[], w: number, h: number): { x: number; y: number } {
  const ww = clamp(w, 1, COLS)
  for (let y = 0; ; y++) for (let x = 0; x + ww <= COLS; x++) if (!tiles.some((t) => hits(t, { x, y, w: ww, h }))) return { x, y }
}

/** rows the tiles reach down to; more than ROWS means the grid scrolls */
export const rowsUsed = (tiles: Tile[]) => tiles.reduce((m, t) => Math.max(m, t.y + t.h), 0)

/** the smallest a new tile is shrunk to before a sheet's tile is split to make room for it */
const MIN_W = 3, MIN_H = 3
/** a grid cell's width over its height on a typical laptop window (1280 × 800: ~95 × 52 px) */
const CELL_ASPECT = 1.8

/**
 * Where a new w × h tile goes without leaving the window: the first gap it fits in; else the
 * largest gap it fits in shrunk (to no less than 3 × 3); else the largest tile on the sheet is
 * split in half (across its longer side on screen) and the new tile takes one half, the way tiling window
 * managers make room. Only a sheet with nothing left to split puts it below the window.
 * Returns the sheet (with a split tile resized) and the new tile's place and size.
 */
export function placeNew(tiles: Tile[], w: number, h: number, rows = ROWS): { tiles: Tile[]; x: number; y: number; w: number; h: number } {
  const ww = clamp(w, 1, COLS), hh = Math.max(1, h)
  const inside = (x: number, y: number, cw: number, ch: number) => y + ch <= rows && !tiles.some((t) => hits(t, { x, y, w: cw, h: ch }))
  const scan = (cw: number, ch: number) => {
    for (let y = 0; y + ch <= rows; y++) for (let x = 0; x + cw <= COLS; x++) if (inside(x, y, cw, ch)) return { x, y }
    return null
  }
  const exact = scan(ww, hh)
  if (exact) return { tiles, ...exact, w: ww, h: hh }
  // shrunk: the largest area that fits, keeping as much of the asked width as it can
  let best: { x: number; y: number; w: number; h: number } | null = null
  for (let cw = ww; cw >= Math.min(MIN_W, ww); cw--) for (let ch = hh; ch >= Math.min(MIN_H, hh); ch--) {
    if (best && cw * ch <= best.w * best.h) continue
    const at = scan(cw, ch)
    if (at) best = { ...at, w: cw, h: ch }
  }
  if (best) return { tiles, ...best }
  // split the largest tile that halves into two tiles of at least the minimum size
  const splittable = tiles.filter((t) => t.y + t.h <= rows && (t.w >= 2 * MIN_W || t.h >= 2 * MIN_H))
    .sort((a, b) => b.w * b.h - a.w * a.h || reading(a, b))
  const big = splittable[0]
  if (big) {
    const across = big.w >= 2 * MIN_W && (big.w * CELL_ASPECT >= big.h || big.h < 2 * MIN_H)
    const keep = across ? Math.ceil(big.w / 2) : Math.ceil(big.h / 2)
    const resized = tiles.map((t) => (t.id !== big.id ? t : across ? { ...t, w: keep } : { ...t, h: keep }))
    return across
      ? { tiles: resized, x: big.x + keep, y: big.y, w: big.w - keep, h: big.h }
      : { tiles: resized, x: big.x, y: big.y + keep, w: big.w, h: big.h - keep }
  }
  return { tiles, ...firstFit(tiles, ww, hh), w: ww, h: hh }
}

export const addTile = (tiles: Tile[], tile: Omit<Tile, 'id' | 'x' | 'y'> & Partial<Pick<Tile, 'x' | 'y'>>): Tile[] => {
  const id = nextTileId(tiles)
  if (tile.x !== undefined && tile.y !== undefined) return arrange([...tiles, { ...tile, x: tile.x, y: tile.y, id }], id)
  const p = placeNew(tiles, tile.w, tile.h)
  return arrange([...p.tiles, { ...tile, x: p.x, y: p.y, w: p.w, h: p.h, id }], id)
}

export const removeTile = (tiles: Tile[], id: string): Tile[] => arrange(tiles.filter((t) => t.id !== id))

/** change a tile (fields, or place and size: then the others make room) */
export const updateTile = (tiles: Tile[], id: string, patch: Partial<Omit<Tile, 'id'>>): Tile[] => {
  const next = tiles.map((t) => (t.id === id ? { ...t, ...patch } : t))
  return 'x' in patch || 'y' in patch || 'w' in patch || 'h' in patch ? arrange(next, id) : next
}

/**
 * Move a tile to (x, y). Dropped exactly on a tile of the same size, the two swap places, so
 * two tiles trade without a hole or a push off the window; anywhere else (or onto a tile of
 * another size, where a swap would overlap) the others make room as in `arrange`.
 */
export function placeTile(tiles: Tile[], id: string, x: number, y: number): Tile[] {
  const moved = tiles.find((t) => t.id === id)
  if (!moved) return tiles
  const nx = clamp(x, 0, COLS - moved.w), ny = Math.max(0, y)
  if (nx === moved.x && ny === moved.y) return tiles
  const other = tiles.find((t) => t.id !== id && t.x === nx && t.y === ny && t.w === moved.w && t.h === moved.h)
  const next = tiles.map((t) => (t.id === id ? { ...t, x: nx, y: ny } : other && t.id === other.id ? { ...t, x: moved.x, y: moved.y } : t))
  return arrange(next, id)
}

/**
 * One keyboard step: the tile trades places with the tile next to it on that side (for →, the
 * one that starts where this one ends and shares a row), as reordering does; with nothing
 * there it moves one cell. A plain one-cell step would just float back up (↓) or push the
 * neighbour down below everything (→).
 */
export function stepTile(tiles: Tile[], id: string, dir: 'left' | 'right' | 'up' | 'down'): Tile[] {
  const t = tiles.find((x) => x.id === id)
  if (!t) return tiles
  const rows = (o: Tile) => o.y < t.y + t.h && t.y < o.y + o.h
  const cols = (o: Tile) => o.x < t.x + t.w && t.x < o.x + o.w
  const others = tiles.filter((o) => o.id !== id).sort(reading)
  const n = dir === 'right' ? others.find((o) => o.x === t.x + t.w && rows(o))
    : dir === 'left' ? others.find((o) => o.x + o.w === t.x && rows(o))
      : dir === 'down' ? others.find((o) => o.y === t.y + t.h && cols(o))
        : others.find((o) => o.y + o.h === t.y && cols(o))
  if (!n) return dir === 'left' || dir === 'right' ? placeTile(tiles, id, t.x + (dir === 'right' ? 1 : -1), t.y)
    : placeTile(tiles, id, t.x, t.y + (dir === 'down' ? 1 : -1))
  const swap = dir === 'right' ? { t: { x: n.x + n.w - t.w }, n: { x: t.x } }
    : dir === 'left' ? { t: { x: n.x }, n: { x: n.x + t.w } }
      : dir === 'down' ? { t: { y: n.y + n.h - t.h }, n: { y: t.y } }
        : { t: { y: n.y }, n: { y: n.y + t.h } }
  return arrange(tiles.map((o) => (o.id === id ? { ...o, ...swap.t } : o.id === n.id ? { ...o, ...swap.n } : o)), id)
}

/** the URL query keys a tile's charts wrote, to clear when the tile goes */
export const tileKeyPrefix = (scope: Scope, sheet: string, id: string) => `${scope}.${sheet}.${id}.`
