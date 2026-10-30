import { FONT } from '@/lib/theme'

let measurer: CanvasRenderingContext2D | null = null
/** the drawn width of a label at a size, so fits follow the text instead of a guessed character width */
export function textWidth(text: string, size: number, weight = 400, family = FONT.family) {
  measurer ??= document.createElement('canvas').getContext('2d')
  if (!measurer) return text.length * size * 0.6
  measurer.font = `${weight} ${size}px ${family}`
  return measurer.measureText(text).width
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
