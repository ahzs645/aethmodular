import type { ReactNode } from 'react'

/**
 * The little Markdown the meeting specs and exporters write: paragraphs
 * (blank-line separated), "- " bullet lists (indented lines continue an
 * item), **bold**, *italic* and `code`. Anything else is plain text; no HTML
 * is interpreted, so exported text cannot inject markup.
 */
export function Markdown({ text }: { text: string }) {
  const blocks = text.trim().split(/\n\s*\n/)
  return <>{blocks.map((b, i) => {
    const lines = b.split('\n')
    if (lines[0].trimStart().startsWith('- ')) {
      const items: string[] = []
      for (const l of lines) {
        if (l.trimStart().startsWith('- ')) items.push(l.trimStart().slice(2))
        else if (items.length) items[items.length - 1] += ' ' + l.trim()
      }
      return <ul key={i} style={{ margin: '4px 0 8px 18px', lineHeight: 1.55 }}>{items.map((it, j) => <li key={j}>{inline(it)}</li>)}</ul>
    }
    return <p key={i} style={{ margin: '4px 0 8px', lineHeight: 1.55 }}>{inline(lines.map((l) => l.trim()).join(' '))}</p>
  })}</>
}

function inline(s: string): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g
  let last = 0, k = 0
  for (let m = re.exec(s); m; m = re.exec(s)) {
    if (m.index > last) out.push(s.slice(last, m.index))
    const t = m[0]
    out.push(t.startsWith('**') ? <strong key={k++}>{t.slice(2, -2)}</strong>
      : t.startsWith('`') ? <code key={k++}>{t.slice(1, -1)}</code>
      : <em key={k++}>{t.slice(1, -1)}</em>)
    last = m.index + t.length
  }
  if (last < s.length) out.push(s.slice(last))
  return out
}
