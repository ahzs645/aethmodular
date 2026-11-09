import assert from 'node:assert/strict'
import test from 'node:test'
import { textLines } from './exportText.ts'

/** A measured text node with explicit browser line fragments. The fixture fixes
 * layout independently of the extraction code, including collapsed whitespace.
 */
function measuredText(lines) {
  const raw = lines.join('\n'), glyphs = []
  let offset = 0
  lines.forEach((line, row) => {
    let column = 0
    for (const char of line) {
      const width = /\s/.test(char) ? 0 : 8
      const tallerFallbackFont = char.length === 2
      glyphs.push({ start: offset, end: offset + char.length,
        left: 20 + column, top: 30 + row * 18 - (tallerFallbackFont ? 2 : 0),
        right: 20 + column + width, bottom: 44 + row * 18, width, height: tallerFallbackFont ? 16 : 14 })
      offset += char.length
      column += /\s/.test(char) ? 4 : width
    }
    offset += 1 // the supplied line break
  })
  const union = (parts) => {
    if (!parts.length) return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }
    const left = Math.min(...parts.map(p => p.left)), top = Math.min(...parts.map(p => p.top))
    const right = Math.max(...parts.map(p => p.right)), bottom = Math.max(...parts.map(p => p.bottom))
    return { left, top, right, bottom, width: right - left, height: bottom - top }
  }
  return {
    data: raw,
    ownerDocument: { createRange: () => {
      let start = 0, end = raw.length
      const selected = () => glyphs.filter(g => g.start < end && g.end > start && g.width)
      return {
        setStart: (_node, value) => { start = value }, setEnd: (_node, value) => { end = value },
        getBoundingClientRect: () => union(selected()),
        getClientRects: () => [...new Set(selected().map(g => g.top))].map(top => union(selected().filter(g => g.top === top))),
      }
    } },
  }
}

test('wrapped evidence retains each measured line, its full wording and its own PDF width', () => {
  const evidence = ['TOR EC reference · µg/filter', 'fixed 18 components · 233 filters', 'EPO source evaluation withheld']
  const lines = textLines(measuredText(evidence))
  assert.deepEqual(lines.map(l => l.text), evidence)
  assert.deepEqual(lines.map(l => l.rect.top), [30, 48, 66])
  assert.ok(lines.every(l => l.rect.height === 14))
  assert.ok(lines.every(l => l.rect.width < 260), 'each line fits the measured figure width')
})

test('one-line labels keep their position while collapsed leading and trailing whitespace is omitted', () => {
  const [line] = textLines(measuredText(['  median  absolute error  ']))
  assert.equal(line.text, 'median absolute error')
  assert.equal(line.rect.left, 28)
  assert.equal(line.rect.top, 30)
  assert.equal(line.rect.height, 14)
})

test('wrapped scientific symbols and supplementary Unicode characters are retained intact', () => {
  const labels = ['λ = 2.96 · µg/m³', '🧪 corrected source comparison']
  assert.deepEqual(textLines(measuredText(labels)).map(l => l.text), labels)
  assert.deepEqual(textLines(measuredText(['   ', ' \t '])), [])
})
