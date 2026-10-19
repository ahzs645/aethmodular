import { paletteSeason, type ExportSeason, type PaletteSeason } from './exportSeasons'
import { collectExportPalette, paintColor, type ExportPalette, type ExportSite } from './exportColors'
/**
 * Download a chart as PNG, JPEG, SVG or PDF. The decks in this repo are built
 * from figures, so a chart that can only be screenshotted is a chart that will
 * be redrawn in matplotlib; this is what lets the gallery feed a slide directly.
 *
 * Three steps, so the export dialog (components/ExportDialog) can preview and
 * size every option before anything is saved:
 *   composeFigure  clone the panel's figure once: chart parts and legend items, at page positions
 *   layoutFigure   place the legend (as on screen, right, top, bottom, or none) into one <svg>
 *   renderFigure   serialise that <svg> to the chosen format and scale
 *
 * The SVG is serialised with an explicit background and the app font declared
 * inline, because a bare <svg> opened outside the page has neither.
 */
const FONT = 'ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
const NS = 'http://www.w3.org/2000/svg'
/** parts of a panel that are UI, not figure */
const SKIP = '.frame-head, .frame-controls, .tooltip, button, select, input, [data-export="skip"]'
/** HTML that is part of the figure: legends and facet titles (add data-export="text" to anything else) */
const HTML_TEXT = '.legend, .facet-title, [data-export="text"]'

export type ExportFormat = 'png' | 'jpeg' | 'svg' | 'pdf'
export type LegendPlacement = 'shown' | 'right' | 'bottom' | 'top' | 'none'
export type ExportBackground = 'white' | 'transparent'

interface Box { left: number; top: number; right: number; bottom: number }
/** a piece of the figure, already built at its page position */
interface Piece { els: Element[]; box: Box; season?: PaletteSeason; detail?: string; off?: boolean }
export interface Figure {
  /** axes, marks, facet titles: everything that stays where it is */
  chart: Piece[]
  /** each HTML legend on the panel, split into its items so they can be re-flowed */
  legends: Piece[][]
  palette: ExportPalette
}

const visible = (el: Element) => {
  const cs = getComputedStyle(el)
  return cs.display !== 'none' && cs.visibility !== 'hidden' && Number(cs.opacity) > 0 && el.getClientRects().length > 0
}
const boxOf = (r: DOMRect): Box => ({ left: r.left, top: r.top, right: r.right, bottom: r.bottom })
const union = (boxes: Box[]): Box => ({
  left: Math.min(...boxes.map((b) => b.left)), top: Math.min(...boxes.map((b) => b.top)),
  right: Math.max(...boxes.map((b) => b.right)), bottom: Math.max(...boxes.map((b) => b.bottom)),
})

/**
 * The whole figure in a panel: every chart <svg> (facets, strips, colour ramps),
 * plus the HTML legends' and facet titles' swatches and text redrawn as SVG.
 * The panel's title, controls, buttons and tooltips are left out.
 * docs/gallery-chart-standards.md §1: export the whole panel, not its first facet.
 */
export function composeFigure(panel: HTMLElement, sites: readonly ExportSite[] = [], seasons: readonly ExportSeason[] = []): Figure | null {
  const svgs = [...panel.querySelectorAll('svg')].filter((s) =>
    !s.closest(SKIP) && !s.parentElement?.closest('svg') && visible(s))
  if (!svgs.length) return null
  const found = [...panel.querySelectorAll<HTMLElement>(HTML_TEXT)].filter((l) => !l.closest(SKIP) && visible(l))
  // a legend inside a legend would otherwise be walked, and its text drawn, twice
  const texts = found.filter((t) => !found.some((o) => o !== t && o.contains(t)))

  // a legend's items: .legend-item where there is one, else the legend's child that holds the part
  const legendEls = [...new Set(texts.map((t) => t.closest<HTMLElement>('.legend')).filter((l): l is HTMLElement => !!l))]
  const items = new Map<HTMLElement, Map<Element, Piece[]>>(legendEls.map((l) => [l, new Map()]))
  const chart: Piece[] = []
  const place = (source: Element, piece: Piece) => {
    // the outermost legend that was collected (a legend can wrap another)
    let legend: HTMLElement | null = null
    for (let l = source.closest<HTMLElement>('.legend'); l; l = l.parentElement?.closest<HTMLElement>('.legend') ?? null) if (items.has(l)) legend = l
    const slot = legend && items.get(legend)
    if (!legend || !slot) { chart.push(piece); return }
    let item: Element | null = source.closest('.legend-item')
    if (!item || !legend.contains(item)) {
      item = source
      while (item.parentElement && item.parentElement !== legend) item = item.parentElement
    }
    slot.set(item, [...(slot.get(item) ?? []), piece])
  }

  const annotate = (source: Element, target: Element) => {
    const method = source.closest('[data-regression-color]')?.getAttribute('data-regression-color')
    if (method) target.setAttribute('data-regression-color', method)
    const series = source.closest('[data-export-series]')
    if (series) {
      const name = series.getAttribute('data-export-series')!, native = series.getAttribute('data-export-color')
      target.setAttribute('data-export-series', name)
      target.setAttribute('data-export-color', native ?? '')
      if (seasons.some(s => s.name === name)) target.setAttribute('data-export-season', name)
      const site = sites.find(s => s.name === name && paintColor(s.color) === paintColor(native))
      if (site) {
        target.setAttribute('data-export-site', site.code)
        target.setAttribute('data-export-color', native!)
        target.setAttribute('data-export-legend-site', site.code)
      }
    }
    const scope = source.closest('[data-export-site], [data-fit-site]')
    if (scope) {
      target.setAttribute('data-export-site', scope.getAttribute('data-export-site') ?? scope.getAttribute('data-fit-site')!)
      if (scope.hasAttribute('data-export-color')) target.setAttribute('data-export-color', scope.getAttribute('data-export-color')!)
    }
  }

  for (const svg of svgs) {
    const r = svg.getBoundingClientRect()
    const clone = svg.cloneNode(true) as SVGSVGElement
    clone.removeAttribute('style')
    clone.removeAttribute('class')
    clone.setAttribute('x', String(r.left))
    clone.setAttribute('y', String(r.top))
    clone.setAttribute('width', String(r.width))
    clone.setAttribute('height', String(r.height))
    // UI inside a chart (the axis pickers' ▾) is not part of the figure
    for (const el of clone.querySelectorAll('[data-export="skip"]')) el.remove()
    // marks carry filter ids for the right-click menu; a figure has no use for them
    for (const el of [clone, ...clone.querySelectorAll('[data-fids], [data-fid]')]) { el.removeAttribute('data-fids'); el.removeAttribute('data-fid'); el.removeAttribute('data-ftitle') }
    place(svg, { els: [clone], box: boxOf(r) })
  }

  for (const host of texts) {
    for (const sw of host.querySelectorAll<HTMLElement>('.swatch')) {
      if (!visible(sw)) continue
      const rect = sw.getBoundingClientRect()
      const cs = getComputedStyle(sw)
      const el = document.createElementNS(NS, 'rect')
      const dashed = sw.classList.contains('dashed'), ring = sw.classList.contains('ring')
      const h = dashed ? 2 : rect.height
      el.setAttribute('x', String(rect.left)); el.setAttribute('y', String(dashed ? rect.top - 1 : rect.top))
      el.setAttribute('width', String(rect.width)); el.setAttribute('height', String(h))
      el.setAttribute('rx', String(Math.min(parseFloat(cs.borderTopLeftRadius) || 0, rect.width / 2)))
      if (ring) {
        el.setAttribute('fill', 'none'); el.setAttribute('stroke', cs.borderTopColor); el.setAttribute('stroke-width', '1.5')
      } else {
        el.setAttribute('fill', dashed ? cs.borderTopColor : cs.backgroundColor)
        if (cs.opacity !== '1') el.setAttribute('opacity', cs.opacity)
      }
      annotate(sw, el)
      place(sw, { els: [el], box: boxOf(rect) })
    }
    const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT)
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = n.textContent?.replace(/\s+/g, ' ').trim()
      const parent = n.parentElement
      if (!text || !parent || parent.closest('svg') || parent.closest(SKIP) || !visible(parent)) continue
      const range = document.createRange()
      range.selectNodeContents(n)
      const rect = range.getBoundingClientRect()
      if (!rect.width) continue
      const cs = getComputedStyle(parent)
      const el = document.createElementNS(NS, 'text')
      el.setAttribute('x', String(rect.left))
      el.setAttribute('y', String(rect.top + rect.height / 2))
      el.setAttribute('dominant-baseline', 'central')
      el.setAttribute('font-size', cs.fontSize)
      el.setAttribute('fill', cs.color)
      if (cs.fontWeight !== '400' && cs.fontWeight !== 'normal') el.setAttribute('font-weight', cs.fontWeight)
      const item = parent.closest('.legend-item')
      const op = Number(item ? getComputedStyle(item).opacity : 1)
      if (op < 1) el.setAttribute('opacity', String(op))
      if (parent.closest('.legend-item.off')) el.setAttribute('text-decoration', 'line-through')
      el.textContent = text
      // its width on screen: the PDF draws it in Inter, which runs wider, and shrinks it to fit
      el.setAttribute('data-w', rect.width.toFixed(1))
      annotate(parent, el)
      place(parent, { els: [el], box: boxOf(rect) })
    }
  }

  const legends = legendEls.map((l) => [...items.get(l)!.values()].filter((ps) => ps.length)
    .map((ps) => {
      const els = ps.flatMap(p => p.els)
      const name = els.find(el => el.hasAttribute('data-export-season'))?.getAttribute('data-export-season')
      const season = seasons.find(s => s.name === name)
      const detail = els.filter(el => el.tagName === 'text').map(el => el.textContent ?? '').find(t => /^n=\d+$/.test(t))
      return { els, box: union(ps.map(p => p.box)), season: season ? paletteSeason(season) : undefined, detail,
        off: els.some(el => el.getAttribute('text-decoration') === 'line-through') }
    }))
    .filter((l) => l.length)
  if (!chart.length) return null
  return { chart, legends, palette: collectExportPalette([...chart, ...legends.flat()].flatMap(p => p.els), sites, seasons) }
}

/** Consolidate legend keys only; chart marks and regression populations stay intact. */
function consolidateSeasonLegend(items: Piece[]): Piece[] {
  const groups = new Map<string, Piece[]>()
  for (const item of items) {
    // Keep hidden and shown entries separate: a crossed-out key must not describe visible marks.
    const key = item.season ? `${item.season.group}|${item.off}` : `other:${groups.size}`
    groups.set(key, [...(groups.get(key) ?? []), item])
  }
  return [...groups.values()].map(group => {
    const first = group[0], season = first.season
    if (!season) return first
    const label = first.els.find(el => el.tagName === 'text')
    if (!label) return first
    const text = label.cloneNode(true) as SVGTextElement
    const counts = group.map(item => item.detail && /^n=(\d+)$/.exec(item.detail))
    const detail = counts.every(Boolean) ? `n=${counts.reduce((sum, match) => sum + Number(match![1]), 0)}` : undefined
    text.textContent = season.groupLabel + (detail ? `  ${detail}` : '')
    const canvas = document.createElement('canvas'), ctx = canvas.getContext('2d')!
    ctx.font = `${text.getAttribute('font-weight') ?? '400'} ${text.getAttribute('font-size') ?? '12px'} ${FONT}`
    const width = ctx.measureText(text.textContent).width
    text.setAttribute('data-w', width.toFixed(1))
    const x = Number(text.getAttribute('x'))
    const els = [...first.els.filter(el => el.tagName !== 'text').map(el => el.cloneNode(true) as Element), text]
    return { ...first, els, detail, box: { ...first.box, right: Math.max(x + width, first.box.left + 12) } }
  })
}

/** One <svg> from a composed figure, with the legend where it was asked for. */
export function layoutFigure(fig: Figure, legend: LegendPlacement = 'shown', consolidateSeasons = false): SVGSVGElement {
  if (consolidateSeasons && fig.palette.seasons.length) {
    fig = { ...fig, legends: fig.legends.map(consolidateSeasonLegend) }
    if (legend === 'shown') legend = 'bottom'
  }
  const out = document.createElementNS(NS, 'svg') as SVGSVGElement
  const placed: { piece: Piece; dx: number; dy: number }[] = fig.chart.map((piece) => ({ piece, dx: 0, dy: 0 }))
  const chartBox = union(fig.chart.map((p) => p.box))
  const legendPieces = fig.legends.flat()

  if (legend === 'shown') {
    for (const piece of legendPieces) placed.push({ piece, dx: 0, dy: 0 })
  } else if (legend === 'right' && legendPieces.length) {
    // one item per row, legends one under another, top-aligned with the chart
    let y = chartBox.top
    const x = chartBox.right + 16
    for (const items of fig.legends) {
      for (const it of items) {
        placed.push({ piece: it, dx: x - it.box.left, dy: y - it.box.top })
        y += it.box.bottom - it.box.top + 5
      }
      y += 8
    }
  } else if ((legend === 'top' || legend === 'bottom') && legendPieces.length) {
    // items flow left to right and wrap at the chart's width; each legend starts a new row
    const maxW = Math.max(chartBox.right - chartBox.left, 240)
    const rows: { it: Piece; x: number; y: number }[] = []
    let y = 0
    for (const items of fig.legends) {
      let x = 0, rowH = 0
      for (const it of items) {
        const w = it.box.right - it.box.left, h = it.box.bottom - it.box.top
        if (x > 0 && x + w > maxW) { y += rowH + 5; x = 0; rowH = 0 }
        rows.push({ it, x, y })
        x += w + 16
        rowH = Math.max(rowH, h)
      }
      y += rowH + 8
    }
    const height = y - 8
    const top = legend === 'top' ? chartBox.top - 14 - height : chartBox.bottom + 14
    for (const r of rows) placed.push({ piece: r.it, dx: chartBox.left + r.x - r.it.box.left, dy: top + r.y - r.it.box.top })
  }

  const pad = 8
  const all = union(placed.map(({ piece, dx, dy }) => ({ left: piece.box.left + dx, right: piece.box.right + dx, top: piece.box.top + dy, bottom: piece.box.bottom + dy })))
  const w = Math.ceil(all.right - all.left + 2 * pad), h = Math.ceil(all.bottom - all.top + 2 * pad)
  out.setAttribute('width', String(w))
  out.setAttribute('height', String(h))
  out.setAttribute('viewBox', `0 0 ${w} ${h}`)
  for (const { piece, dx, dy } of placed) {
    const g = document.createElementNS(NS, 'g')
    g.setAttribute('transform', `translate(${dx - all.left + pad},${dy - all.top + pad})`)
    for (const el of piece.els) g.appendChild(el.cloneNode(true))
    out.appendChild(g)
  }
  return out
}

/** The figure exactly as it sits on screen (the old single-button export). */
export function composePanel(panel: HTMLElement): SVGSVGElement | null {
  const fig = composeFigure(panel)
  return fig ? layoutFigure(fig, 'shown') : null
}

function serialise(svg: SVGSVGElement, background: ExportBackground, fontFamily = FONT): { xml: string; width: number; height: number; node: SVGSVGElement } {
  const clone = svg.cloneNode(true) as SVGSVGElement
  // a composed figure is not in the DOM, so its client size is 0 and the attributes carry it
  const width = svg.clientWidth || Number(svg.getAttribute('width')) || 800
  const height = svg.clientHeight || Number(svg.getAttribute('height')) || 500
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('width', String(width))
  clone.setAttribute('height', String(height))
  clone.setAttribute('font-family', fontFamily)
  if (fontFamily === FONT) for (const el of clone.querySelectorAll('[data-w]')) el.removeAttribute('data-w')
  if (background === 'white') {
    const bg = document.createElementNS(NS, 'rect')
    bg.setAttribute('width', '100%')
    bg.setAttribute('height', '100%')
    bg.setAttribute('fill', '#ffffff')
    clone.insertBefore(bg, clone.firstChild)
  }
  return { xml: new XMLSerializer().serializeToString(clone), width, height, node: clone }
}

export interface RenderOpts { format: ExportFormat; scale: number; background: ExportBackground }
export const MIME: Record<ExportFormat, string> = { png: 'image/png', jpeg: 'image/jpeg', svg: 'image/svg+xml', pdf: 'application/pdf' }
/** browsers cap a canvas side near 16k px (Safari lower), so very large scales are refused */
export const MAX_CANVAS_SIDE = 16000

/** The file for one export option. */
export async function renderFigure(svg: SVGSVGElement, opts: RenderOpts): Promise<Blob> {
  // JPEG has no alpha: a transparent background would turn black
  const background = opts.format === 'jpeg' ? 'white' : opts.background
  if (opts.format === 'svg') return new Blob([serialise(svg, background).xml], { type: 'image/svg+xml;charset=utf-8' })
  if (opts.format === 'pdf') return renderPdf(svg, background)
  const { xml, width, height } = serialise(svg, background)
  const url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }))
  try {
    const img = new Image()
    await new Promise<void>((resolve, reject) => { img.onload = () => resolve(); img.onerror = () => reject(new Error('could not draw the figure')); img.src = url })
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(width * opts.scale)
    canvas.height = Math.round(height * opts.scale)
    if (Math.max(canvas.width, canvas.height) > MAX_CANVAS_SIDE) throw new Error('too large for the browser to draw')
    const ctx = canvas.getContext('2d')!
    ctx.scale(opts.scale, opts.scale)
    ctx.drawImage(img, 0, 0)
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('the browser could not encode the image'))), MIME[opts.format], 0.92))
  } finally {
    URL.revokeObjectURL(url)
  }
}

/**
 * A vector PDF (jsPDF + svg2pdf.js, loaded only when asked for). The page is the
 * figure's size at 96 px to the inch, so it prints at the size it shows on screen.
 * PDF's built-in fonts cannot draw ₂, −, ≈ or λ, so Inter (OFL) is embedded and
 * every text uses it.
 */
async function renderPdf(svg: SVGSVGElement, background: ExportBackground): Promise<Blob> {
  const [{ jsPDF }, { svg2pdf }, fonts] = await Promise.all([import('jspdf'), import('svg2pdf.js'), pdfFonts()])
  const copy = serialise(svg, background, 'Inter')
  const { node, height } = copy
  let { width } = copy
  for (const el of node.querySelectorAll('[font-family]')) el.setAttribute('font-family', 'Inter')
  for (const el of node.querySelectorAll<SVGElement>('[style*="font-family"]')) el.style.fontFamily = 'Inter'
  // svg2pdf looks a weight up as a jsPDF style ("600normal"); there are two faces, so name one of them
  for (const el of node.querySelectorAll('[font-weight]')) {
    const w = el.getAttribute('font-weight')!
    el.setAttribute('font-weight', w === 'bold' || Number(w) >= 600 ? 'bold' : 'normal')
  }
  for (const el of node.querySelectorAll('[font-style]')) el.setAttribute('font-style', 'normal')
  // labels pushed right by the wider font get the room they need at the page edge
  const extra = Math.ceil(await fitText(node, fonts))
  if (extra > 0) {
    width += extra
    node.setAttribute('width', String(width))
    node.setAttribute('viewBox', `0 0 ${width} ${height}`)
  }
  const pt = 0.75
  const doc = new jsPDF({ unit: 'pt', format: [width * pt, height * pt], orientation: width > height ? 'landscape' : 'portrait', compress: true })
  doc.addFileToVFS('Inter-Regular.ttf', fonts.regular)
  doc.addFont('Inter-Regular.ttf', 'Inter', 'normal')
  doc.addFileToVFS('Inter-SemiBold.ttf', fonts.bold)
  doc.addFont('Inter-SemiBold.ttf', 'Inter', 'bold')
  doc.setFont('Inter', 'normal')
  // svg2pdf measures text and reads computed styles, so the copy has to be in the document while it converts
  const host = document.createElement('div')
  host.style.cssText = 'position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none'
  host.appendChild(node)
  document.body.appendChild(host)
  try {
    await svg2pdf(node, doc, { x: 0, y: 0, width: width * pt, height: height * pt })
  } finally {
    host.remove()
  }
  return doc.output('blob')
}

/**
 * Legend and facet text was placed where the screen font put it; Inter runs wider, so a
 * label would run into its count or the next label. Each text that no longer fits its
 * on-screen width is shrunk a little (at most 10 %), and whatever still overflows pushes
 * the rest of its legend item (each item is one <g>) to the right.
 */
async function fitText(node: SVGSVGElement, fonts: { regular: string; bold: string }): Promise<number> {
  const texts = [...node.querySelectorAll<SVGTextElement>('text[data-w]')]
  if (!texts.length) return 0
  let widest = 0
  const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
  const faces = [new FontFace('InterExport', bytes(fonts.regular), { weight: '400' }), new FontFace('InterExport', bytes(fonts.bold), { weight: '600' })]
  await Promise.all(faces.map((f) => f.load()))
  faces.forEach((f) => document.fonts.add(f))
  const ctx = document.createElement('canvas').getContext('2d')!
  const groups = new Set(texts.map((t) => t.parentElement!))
  for (const g of groups) {
    const kids = [...g.children].filter((el) => el.hasAttribute('x')).sort((a, b) => Number(a.getAttribute('x')) - Number(b.getAttribute('x')))
    let shift = 0
    for (const el of kids) {
      const x = Number(el.getAttribute('x'))
      if (shift) el.setAttribute('x', String(x + shift))
      const w = Number(el.getAttribute('data-w'))
      if (el.tagName !== 'text' || !w) continue
      const size = parseFloat(el.getAttribute('font-size') ?? '12')
      ctx.font = `${el.getAttribute('font-weight') === 'bold' ? 600 : 400} ${size}px InterExport`
      const need = ctx.measureText(el.textContent ?? '').width
      if (need > w * 1.01) {
        const k = Math.max(0.9, w / need)
        el.setAttribute('font-size', (size * k).toFixed(2))
        shift += need * k - w
      }
    }
    widest = Math.max(widest, shift)
  }
  for (const t of texts) t.removeAttribute('data-w')
  faces.forEach((f) => document.fonts.delete(f))
  return widest
}

let fontCache: Promise<{ regular: string; bold: string }> | null = null
function pdfFonts() {
  fontCache ??= (async () => {
    const [{ default: regularUrl }, { default: boldUrl }] = await Promise.all([
      import('@expo-google-fonts/inter/400Regular/Inter_400Regular.ttf?url'),
      import('@expo-google-fonts/inter/600SemiBold/Inter_600SemiBold.ttf?url'),
    ])
    const b64 = async (url: string) => {
      const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer())
      let s = ''
      for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
      return btoa(s)
    }
    const [regular, bold] = await Promise.all([b64(regularUrl), b64(boldUrl)])
    return { regular, bold }
  })()
  fontCache.catch(() => { fontCache = null })
  return fontCache
}

function download(href: string, filename: string) {
  const a = document.createElement('a')
  a.href = href
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
}

export const slug = (s: string) =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'chart'

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  download(url, filename)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/** Download a table as CSV; cells are quoted only when they need it. */
export function downloadCsv(header: string[], rows: (string | number | null)[][], title: string) {
  const cell = (v: string | number | null) => {
    const s = v == null ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const text = [header, ...rows].map((r) => r.map(cell).join(',')).join('\n') + '\n'
  downloadBlob(new Blob([text], { type: 'text/csv' }), `${slug(title)}.csv`)
}
