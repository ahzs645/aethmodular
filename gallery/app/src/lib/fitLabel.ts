import { FONT } from './theme.ts'

let measurer: CanvasRenderingContext2D | null = null
/** the drawn width of a label at a size, so fits follow the text instead of a guessed character width */
export function textWidth(text: string, size: number, weight = 400, family = FONT.family) {
  measurer ??= document.createElement('canvas').getContext('2d')
  if (!measurer) return text.length * size * 0.6
  measurer.font = `${weight} ${size}px ${family}`
  return measurer.measureText(text).width
}

export interface GutterOptions {
  /** font size in px; 11, the tick-label size */
  size?: number
  weight?: number
  family?: string
  /** added to the widest label (tick length, gap to the plot) */
  pad?: number
  min?: number
  max?: number
}

/**
 * The room a column of labels needs, measured as drawn rather than guessed from a character count:
 * the widest label at `size` plus `pad`, kept within [min, max]. Use it for left gutters of
 * category rows, label columns and "does this label fit" tests (docs/gallery-chart-standards.md §1).
 *
 *   const left = labelGutter(sites.map((s) => s.site), { size: 11, pad: 12, min: 60, max: 180 })
 */
export function labelGutter(labels: Iterable<string>, { size = 11, weight = 400, family = FONT.family, pad = 0, min = 0, max = Infinity }: GutterOptions = {}): number {
  let w = 0
  for (const l of labels) w = Math.max(w, textWidth(l, size, weight, family))
  return Math.min(max, Math.max(min, Math.ceil(w + pad)))
}

/** Whether a label fits in `room` px at `size` (a measured `text.length * 6 <= room`). */
export const fitsWidth = (text: string, room: number, size = 11, weight = 400, family = FONT.family) => textWidth(text, size, weight, family) <= room

/**
 * A label cut to `room` px with an ellipsis ("Organic carbon (T…"), or unchanged when it fits.
 * Keeps at least `minChars` characters so a row never shows a bare "…". Pair it with a native
 * `<title>` holding the full text where nothing else shows it.
 */
export function clipText(text: string, room: number, { size = 11, weight = 400, family = FONT.family, minChars = 1 }: { size?: number; weight?: number; family?: string; minChars?: number } = {}): string {
  if (textWidth(text, size, weight, family) <= room) return text
  let s = text
  while (s.length > minChars && textWidth(`${s.trimEnd()}…`, size, weight, family) > room) s = s.slice(0, -1)
  return `${s.trimEnd()}…`
}

/** Greedy word wrap at a font size; null when a single word is wider than the room. */
function wrap(text: string, size: number, room: number, weight: number): string[] | null {
  const lines: string[] = []
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const last = lines.length - 1
    if (last >= 0 && textWidth(`${lines[last]} ${word}`, size, weight) <= room) lines[last] += ` ${word}`
    else if (textWidth(word, size, weight) <= room) lines.push(word)
    else return null
  }
  return lines
}

function ellipsis(text: string, size: number, room: number, weight: number) {
  if (textWidth(text, size, weight) <= room) return text
  let s = text
  while (s.length > 1 && textWidth(`${s}…`, size, weight) > room) s = s.slice(0, -1)
  return s.length > 1 ? `${s.trimEnd()}…` : null
}

export interface FittedLabel { size: number; lines: string[]; lineHeight: number }
interface FitOptions { max?: number; min?: number; weight?: number; leading?: number; maxLines?: number }

/**
 * Fit a label inside a box: largest size first on one line, then wrapped onto
 * more lines, shrinking down to `min`; at `min` the first line is cut with an
 * ellipsis. `roomFor(lines, lineHeight)` gives the width available to a block
 * of that many lines (constant for a rectangle, narrower for a circle).
 * Returns null when not even a few characters fit.
 */
function fit(text: string, height: number, roomFor: (n: number, lh: number) => number, { max = 10.5, min = 7.5, weight = 400, leading = 1.15, maxLines = 3 }: FitOptions): FittedLabel | null {
  for (let size = max; size >= min; size -= 0.5) {
    const lh = size * leading
    for (let n = 1; n <= maxLines && n * lh <= height; n++) {
      const room = roomFor(n, lh)
      if (room <= 0) break
      const lines = wrap(text, size, room, weight)
      if (lines && lines.length <= n) return { size, lines, lineHeight: lh }
    }
  }
  const lh = min * leading
  if (lh > height) return null
  const cut = ellipsis(text, min, roomFor(1, lh), weight)
  return cut && cut.length >= 5 ? { size: min, lines: [cut], lineHeight: lh } : null
}

/** Fit a label inside a w×h rectangle (padding already taken off). */
export const fitInRect = (text: string, w: number, h: number, opts: FitOptions = {}) => fit(text, h, () => w, opts)

/** Fit a label centred in a circle of radius r: a block of n lines gets the chord width at its top and bottom edges. */
export const fitInCircle = (text: string, r: number, opts: FitOptions & { pad?: number } = {}) => {
  const pad = opts.pad ?? 3
  return fit(text, 2 * (r - pad), (n, lh) => {
    const half = (n * lh) / 2
    return half >= r - pad ? 0 : 2 * Math.sqrt((r - pad) ** 2 - half ** 2)
  }, opts)
}

interface PointLabel { key: string; text: string; x: number; y: number }
interface LabelBox { left: number; right: number; top: number; bottom: number }

/**
 * Direct labels near points: prefer above/right, turn left at the edge, then
 * move the text to the nearest free row. Points stay fixed. Both labels and
 * point markers are obstacles; the result includes a connector for moved text.
 * Labels that cannot fit stay in the caller's legend/tooltip instead.
 */
export function placePointLabels(points: readonly PointLabel[], width: number, height: number,
  { size = 10.5, weight = 600, radius = 10, gap = 4 }: { size?: number; weight?: number; radius?: number; gap?: number } = {}) {
  const overlaps = (a: LabelBox, b: LabelBox) => a.left < b.right + gap && b.left < a.right + gap
    && a.top < b.bottom + gap && b.top < a.bottom + gap
  const obstacles: LabelBox[] = points.map(p => ({ left: p.x - radius, right: p.x + radius, top: p.y - radius, bottom: p.y + radius }))
  const placed: { key: string; text: string; x: number; y: number; anchor: 'start' | 'end';
    connector: { x1: number; y1: number; x2: number; y2: number } | null }[] = []
  const descent = size * 0.25, pad = 2, offset = radius + gap + 2
  for (const p of [...points].sort((a, b) => a.y - b.y || a.x - b.x || a.key.localeCompare(b.key))) {
    const w = textWidth(p.text, size, weight)
    if (w + pad * 2 > width || size + descent + pad * 2 > height) continue
    const preferred = p.y - 9
    const end = p.x + offset + w > width - pad
    const sides = end ? ['end', 'start'] as const : ['start', 'end'] as const
    let best: { box: LabelBox; x: number; y: number; anchor: 'start' | 'end'; score: number } | null = null
    for (const anchor of sides) {
      const left = Math.max(pad, Math.min(width - w - pad, anchor === 'start' ? p.x + offset : p.x - offset - w))
      const ys = [preferred, size + pad, height - descent - pad,
        ...obstacles.flatMap(b => [b.top - gap - descent, b.bottom + gap + size])]
      for (const y of ys) {
        const box = { left, right: left + w, top: y - size, bottom: y + descent }
        if (box.top < pad || box.bottom > height - pad || obstacles.some(b => overlaps(box, b))) continue
        const dx = (anchor === 'start' ? left : left + w) - p.x
        const score = (y - preferred) ** 2 + dx ** 2 + (anchor === sides[0] ? 0 : offset ** 2)
        if (!best || score < best.score) best = { box, x: anchor === 'start' ? left : left + w, y, anchor, score }
      }
    }
    if (!best) continue
    const x2 = Math.max(best.box.left, Math.min(best.box.right, p.x))
    const y2 = Math.max(best.box.top, Math.min(best.box.bottom, p.y))
    const distance = Math.hypot(x2 - p.x, y2 - p.y)
    const moved = Math.abs(best.y - preferred) > 0.5
    const connector = moved && distance > radius + 2 ? {
      x1: p.x + (x2 - p.x) * (radius + 1) / distance,
      y1: p.y + (y2 - p.y) * (radius + 1) / distance, x2, y2,
    } : null
    placed.push({ key: p.key, text: p.text, x: best.x, y: best.y, anchor: best.anchor, connector })
    obstacles.push(best.box)
  }
  return placed
}
