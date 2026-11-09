export interface ExportTextLine {
  text: string
  rect: DOMRect
}

/** Preserve the browser's line wrapping when HTML evidence is redrawn as SVG.
 * A Range bounding box spans every wrapped line; drawing its whole text once
 * would overflow that box and force the PDF exporter to shrink it to one line.
 */
export function textLines(node: Text): ExportTextLine[] {
  const raw = node.data
  const first = raw.search(/\S/), end = raw.trimEnd().length
  if (first < 0) return []
  const range = node.ownerDocument.createRange()
  const line = (start: number, stop: number): ExportTextLine => {
    range.setStart(node, start)
    range.setEnd(node, stop)
    return { text: raw.slice(start, stop).replace(/\s+/g, ' '), rect: range.getBoundingClientRect() }
  }
  const whole = line(first, end)
  const rects = [...range.getClientRects()].filter(r => r.width > 0 && r.height > 0)
  if (rects.length <= 1) return whole.rect.width > 0 ? [whole] : []

  const lines: ExportTextLine[] = []
  let start = -1, stop = -1, top = 0, bottom = 0
  // Only wrapped nodes need per-glyph measurement. Keep UTF-16 surrogate pairs
  // intact; collapsed whitespace has no glyph and must not choose a line.
  for (let offset = first; offset < end;) {
    const size = raw.codePointAt(offset)! > 0xffff ? 2 : 1
    const next = offset + size
    if (!/\s/.test(raw.slice(offset, next))) {
      range.setStart(node, offset)
      range.setEnd(node, next)
      const rect = range.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        // Fallback fonts (for example an emoji beside scientific units) can
        // have different ascenders on the same line. Use vertical overlap.
        const overlap = Math.min(rect.bottom, bottom) - Math.max(rect.top, top)
        if (start >= 0 && overlap < Math.min(rect.height, bottom - top) / 2) {
          lines.push(line(start, stop))
          start = -1
        }
        if (start < 0) { start = offset; top = rect.top; bottom = rect.bottom }
        stop = next
      }
    }
    offset = next
  }
  if (start >= 0) lines.push(line(start, stop))
  return lines
}
