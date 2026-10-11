/**
 * Download a chart as SVG or PNG. The decks in this repo are built from
 * figures, so a chart that can only be screenshotted is a chart that will be
 * redrawn in matplotlib; this is what lets the gallery feed a slide directly.
 *
 * The SVG is serialised with an explicit white background and the app font
 * declared inline, because a bare <svg> opened outside the page has neither.
 */
const FONT = 'ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

function serialise(svg: SVGSVGElement): { xml: string; width: number; height: number } {
  const clone = svg.cloneNode(true) as SVGSVGElement
  // a composed panel is not in the DOM, so its client size is 0 and the attributes carry it
  const width = svg.clientWidth || Number(svg.getAttribute('width')) || 800
  const height = svg.clientHeight || Number(svg.getAttribute('height')) || 500
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('width', String(width))
  clone.setAttribute('height', String(height))
  clone.setAttribute('font-family', FONT)
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  bg.setAttribute('width', '100%')
  bg.setAttribute('height', '100%')
  bg.setAttribute('fill', '#ffffff')
  clone.insertBefore(bg, clone.firstChild)
  return { xml: new XMLSerializer().serializeToString(clone), width, height }
}

const NS = 'http://www.w3.org/2000/svg'
/** parts of a panel that are UI, not figure */
const SKIP = '.frame-head, .frame-controls, .tooltip, button, select, input, [data-export="skip"]'

/** HTML that is part of the figure: legends and facet titles (add data-export="text" to anything else) */
const HTML_TEXT = '.legend, .facet-title, [data-export="text"]'

const visible = (el: Element) => {
  const cs = getComputedStyle(el)
  return cs.display !== 'none' && cs.visibility !== 'hidden' && Number(cs.opacity) > 0 && el.getClientRects().length > 0
}

/**
 * The whole figure in a panel as one SVG: every chart <svg> (facets, strips,
 * colour ramps) at the place it sits on screen, plus the HTML legends' and facet
 * titles' swatches and text redrawn as SVG. The panel's title, controls, buttons and tooltips
 * are left out. docs/gallery-chart-standards.md §1: export the whole panel,
 * not its first facet.
 */
export function composePanel(panel: HTMLElement): SVGSVGElement | null {
  const svgs = [...panel.querySelectorAll('svg')].filter((s) =>
    !s.closest(SKIP) && !s.parentElement?.closest('svg') && visible(s))
  const legends = [...panel.querySelectorAll<HTMLElement>(HTML_TEXT)].filter((l) => !l.closest(SKIP) && visible(l))
  if (!svgs.length) return null

  type Part = { rect: DOMRect; build: (x: number, y: number) => Element[] }
  const parts: Part[] = svgs.map((svg) => ({
    rect: svg.getBoundingClientRect(),
    build: (x, y) => {
      const r = svg.getBoundingClientRect()
      const clone = svg.cloneNode(true) as SVGSVGElement
      clone.removeAttribute('style')
      clone.removeAttribute('class')
      clone.setAttribute('x', String(x))
      clone.setAttribute('y', String(y))
      clone.setAttribute('width', String(r.width))
      clone.setAttribute('height', String(r.height))
      return [clone]
    },
  }))

  // legend and facet-title text and CSS swatches; their <svg> marks are already in `svgs`
  for (const legend of legends) {
    for (const sw of legend.querySelectorAll<HTMLElement>('.swatch')) {
      if (!visible(sw)) continue
      const rect = sw.getBoundingClientRect()
      parts.push({ rect, build: (x, y) => {
        const cs = getComputedStyle(sw)
        const el = document.createElementNS(NS, 'rect')
        const dashed = sw.classList.contains('dashed'), ring = sw.classList.contains('ring')
        const h = dashed ? 2 : rect.height
        el.setAttribute('x', String(x)); el.setAttribute('y', String(dashed ? y - 1 : y))
        el.setAttribute('width', String(rect.width)); el.setAttribute('height', String(h))
        el.setAttribute('rx', String(Math.min(parseFloat(cs.borderTopLeftRadius) || 0, rect.width / 2)))
        if (ring) {
          el.setAttribute('fill', 'none'); el.setAttribute('stroke', cs.borderTopColor); el.setAttribute('stroke-width', '1.5')
        } else {
          el.setAttribute('fill', dashed ? cs.borderTopColor : cs.backgroundColor)
          if (cs.opacity !== '1') el.setAttribute('opacity', cs.opacity)
        }
        return [el]
      } })
    }
    const walker = document.createTreeWalker(legend, NodeFilter.SHOW_TEXT)
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = n.textContent?.replace(/\s+/g, ' ').trim()
      const host = n.parentElement
      if (!text || !host || host.closest('svg') || host.closest(SKIP) || !visible(host)) continue
      const range = document.createRange()
      range.selectNodeContents(n)
      const rect = range.getBoundingClientRect()
      if (!rect.width) continue
      parts.push({ rect, build: (x, y) => {
        const cs = getComputedStyle(host)
        const el = document.createElementNS(NS, 'text')
        el.setAttribute('x', String(x))
        el.setAttribute('y', String(y + rect.height / 2))
        el.setAttribute('dominant-baseline', 'central')
        el.setAttribute('font-size', cs.fontSize)
        el.setAttribute('fill', cs.color)
        if (cs.fontWeight !== '400' && cs.fontWeight !== 'normal') el.setAttribute('font-weight', cs.fontWeight)
        const op = Number(host.closest('.legend-item') ? getComputedStyle(host.closest('.legend-item')!).opacity : 1)
        if (op < 1) el.setAttribute('opacity', String(op))
        if (host.closest('.legend-item.off')) el.setAttribute('text-decoration', 'line-through')
        el.textContent = text
        return [el]
      } })
    }
  }

  const pad = 8
  const x0 = Math.min(...parts.map((p) => p.rect.left)) - pad
  const y0 = Math.min(...parts.map((p) => p.rect.top)) - pad
  const x1 = Math.max(...parts.map((p) => p.rect.right)) + pad
  const y1 = Math.max(...parts.map((p) => p.rect.bottom)) + pad
  const out = document.createElementNS(NS, 'svg') as SVGSVGElement
  out.setAttribute('width', String(Math.ceil(x1 - x0)))
  out.setAttribute('height', String(Math.ceil(y1 - y0)))
  out.setAttribute('viewBox', `0 0 ${Math.ceil(x1 - x0)} ${Math.ceil(y1 - y0)}`)
  for (const p of parts) for (const el of p.build(p.rect.left - x0, p.rect.top - y0)) out.appendChild(el)
  return out
}

function download(href: string, filename: string) {
  const a = document.createElement('a')
  a.href = href
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
}

const slug = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'chart'

export function downloadSvg(svg: SVGSVGElement, title: string) {
  const { xml } = serialise(svg)
  const blob = new Blob([xml], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  download(url, `${slug(title)}.svg`)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function downloadPng(svg: SVGSVGElement, title: string, scale = 2) {
  const { xml, width, height } = serialise(svg)
  const blob = new Blob([xml], { type: 'image/svg+xml;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const img = new Image()
  img.onload = () => {
    const canvas = document.createElement('canvas')
    canvas.width = width * scale
    canvas.height = height * scale
    const ctx = canvas.getContext('2d')!
    ctx.scale(scale, scale)
    ctx.drawImage(img, 0, 0)
    URL.revokeObjectURL(url)
    download(canvas.toDataURL('image/png'), `${slug(title)}.png`)
  }
  img.src = url
}

/** Download a table as CSV; cells are quoted only when they need it. */
export function downloadCsv(header: string[], rows: (string | number | null)[][], title: string) {
  const cell = (v: string | number | null) => {
    const s = v == null ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const text = [header, ...rows].map((r) => r.map(cell).join(',')).join('\n') + '\n'
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }))
  download(url, `${slug(title)}.csv`)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
