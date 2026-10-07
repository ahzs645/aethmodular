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
