import { paletteSeason, type ExportSeason, type PaletteSeason } from './exportSeasons'
import { collectExportPalette, fixedColours, paintColor, printPaint, replacePaint, type ExportPalette, type ExportSite } from './exportColors'
import { CELL_INK, INK_LIGHT, inkOnFill, isDark } from './theme'
/**
 * Download a chart as PNG, JPEG, SVG or PDF. The decks in this repo are built
 * from figures, so a chart that can only be screenshotted is a chart that will
 * be redrawn in matplotlib; this is what lets the gallery feed a slide directly.
 *
 * Three steps, so the export dialog (components/ExportDialog) can preview and
 * size every option before anything is saved:
 *   composeFigure  clone the panel's figure once: chart parts and legend items, at page positions
 *   layoutFigure   place the legend (as on screen, right, top, bottom, inside the plot, or none) into one <svg>,
 *                  and for a publication figure the fit statistics inside the plot too (PubLayout)
 *   renderFigure   serialise that <svg> to the chosen format and scale, or to a physical width (a journal column)
 *
 * The SVG is serialised with an explicit background and the app font declared
 * inline, because a bare <svg> opened outside the page has neither.
 */
const FONT = 'ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
/**
 * Typefaces for a publication figure. Journals ask for Arial or Helvetica (Elsevier, ACS, IEEE artwork
 * guides); some house styles are serif. SVG and PNG use the font installed here; the PDF can embed only
 * Inter (the one font file the app ships), whichever is chosen.
 */
export const PUB_FONTS: { key: string; label: string; css: string }[] = [
  { key: 'arial', label: 'Arial', css: 'Arial, Helvetica, sans-serif' },
  { key: 'helvetica', label: 'Helvetica', css: 'Helvetica, Arial, sans-serif' },
  { key: 'times', label: 'Times New Roman', css: '"Times New Roman", Times, serif' },
  { key: 'calibri', label: 'Calibri', css: 'Calibri, Carlito, Arial, sans-serif' },
  { key: 'screen', label: 'As on screen', css: FONT },
]
const NS = 'http://www.w3.org/2000/svg'
/** parts of a panel that are UI, not figure */
const SKIP = '.frame-head, .frame-controls, .tooltip, button, select, input, [data-export="skip"]'
/** HTML that is part of the figure: legends and facet titles (add data-export="text" to anything else) */
const HTML_TEXT = '.legend, .facet-title, [data-export="text"]'

export type ExportFormat = 'png' | 'jpeg' | 'svg' | 'pdf'
export type LegendPlacement = 'shown' | 'right' | 'bottom' | 'top' | 'inside' | 'none'
export type ExportBackground = 'white' | 'transparent'

interface Box { left: number; top: number; right: number; bottom: number }
/** a piece of the figure, already built at its page position */
interface Piece { els: Element[]; box: Box; season?: PaletteSeason; detail?: string; off?: boolean
  /** a legend's reading hint (Explain mode), not a key: a publication figure leaves it out */
  note?: boolean }
/** a line of a fit's statistics, handed over by components/RegressionOverlay when the export sets them inside the plot */
interface StatsLine { text: string; color: string; method?: string; name?: boolean }
/** a plot area (a chart's clipped axes box) and where its marks are, to find its emptiest corners */
interface Plot { box: Box; points: [number, number][]; stats: { site?: string; color?: string; lines: StatsLine[] }[] }
export interface Figure {
  /** axes, marks, facet titles: everything that stays where it is */
  chart: Piece[]
  /** each HTML legend on the panel, split into its items so they can be re-flowed */
  legends: Piece[][]
  palette: ExportPalette
  plots: Plot[]
  /** the figure's text sizes in px: the usual one (tick labels) and the smallest */
  text: { median: number; min: number }
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

  const plots: Plot[] = []
  const sizes: number[] = []
  // a map's WebGL basemap (components/GeoMap keeps its drawing buffer): its pixels as an <image>, first so
  // the markers' SVGs sit on top. It is a raster in the theme on screen; lib/themeCss cannot relight it.
  for (const canvas of panel.querySelectorAll<HTMLCanvasElement>('canvas.maplibregl-canvas')) {
    if (canvas.closest(SKIP) || !visible(canvas)) continue
    let href: string
    try { href = canvas.toDataURL('image/png') } catch { continue }
    const r = canvas.getBoundingClientRect()
    const img = document.createElementNS(NS, 'image')
    img.setAttribute('x', String(r.left)); img.setAttribute('y', String(r.top))
    img.setAttribute('width', String(r.width)); img.setAttribute('height', String(r.height))
    img.setAttribute('preserveAspectRatio', 'none')
    img.setAttribute('href', href)
    chart.push({ els: [img], box: boxOf(r) })
  }
  for (const svg of svgs) {
    const r = svg.getBoundingClientRect()
    const areas = plotAreas(svg)
    plots.push(...areas)
    const stats = [...svg.querySelectorAll('[data-export-stats] > g')].map((g) => ({
      site: g.getAttribute('data-fit-site') ?? undefined, color: g.getAttribute('data-export-color') ?? undefined,
      lines: [...g.querySelectorAll('text')].map((t) => ({ text: t.textContent ?? '', color: t.getAttribute('fill') ?? '#000', method: t.getAttribute('data-regression-color') ?? undefined, name: t.getAttribute('data-line') === 'name' })),
    }))
    // the svg's own fits go with its largest plot area (a facet's summary with its facet)
    if (stats.length && areas.length) areas.reduce((a, b) => (area(b.box) > area(a.box) ? b : a)).stats.push(...stats)
    for (const t of svg.querySelectorAll('text')) if (!t.closest('[data-export-stats], [data-export="skip"]') && t.getClientRects().length) sizes.push(parseFloat(getComputedStyle(t).fontSize))
    const clone = svg.cloneNode(true) as SVGSVGElement
    for (const el of clone.querySelectorAll('[data-export-stats]')) el.remove()
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
      // a diamond swatch is a rotated square: draw the diamond, not its bounding box
      if (sw.classList.contains('diamond')) {
        const d = document.createElementNS(NS, 'path'), cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2
        d.setAttribute('d', `M${cx},${rect.top}L${rect.right},${cy}L${cx},${rect.bottom}L${rect.left},${cy}Z`)
        d.setAttribute('fill', cs.backgroundColor)
        annotate(sw, d)
        place(sw, { els: [d], box: boxOf(rect) })
        continue
      }
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
      sizes.push(parseFloat(cs.fontSize))
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
      place(parent, { els: [el], box: boxOf(rect), note: !!parent.closest('.legend-note') })
    }
  }

  const legends = legendEls.map((l) => [...items.get(l)!.values()].filter((ps) => ps.length)
    .map((ps) => {
      const els = ps.flatMap(p => p.els)
      const name = els.find(el => el.hasAttribute('data-export-season'))?.getAttribute('data-export-season')
      const season = seasons.find(s => s.name === name)
      const detail = els.filter(el => el.tagName === 'text').map(el => el.textContent ?? '').find(t => /^n=\d+$/.test(t))
      return { els, box: union(ps.map(p => p.box)), season: season ? paletteSeason(season) : undefined, detail, note: ps.every((p) => p.note),
        off: els.some(el => el.getAttribute('text-decoration') === 'line-through') }
    }))
    .filter((l) => l.length)
  if (!chart.length) return null
  const dark = isDark() || panel.dataset.chartTheme === 'dark'
  const categories = [...panel.querySelectorAll('[data-export-fixed-colours]')].flatMap(el =>
    (el.getAttribute('data-export-fixed-colours') ?? '').split(/\s+/).map(color => ({ color })))
  const fixed = fixedColours(sites, seasons, categories)
  if (dark) for (const p of [...chart, ...legends.flat()]) for (const el of p.els) printColours(el, fixed)
  if (dark) for (const pl of plots) for (const st of pl.stats) for (const l of st.lines) { const light = printPaint(l.color, fixed); if (light) l.color = light }
  for (const p of chart) for (const el of p.els) for (const d of [el, ...el.querySelectorAll('[data-dot]')]) d.removeAttribute('data-dot')
  const sorted = sizes.filter((v) => v > 0).sort((a, b) => a - b)
  const text = { median: sorted[Math.floor(sorted.length / 2)] ?? 11, min: sorted[0] ?? 11 }
  return { chart, legends, plots, text, palette: collectExportPalette([...chart, ...legends.flat()].flatMap(p => p.els), sites, seasons) }
}

const area = (b: Box) => (b.right - b.left) * (b.bottom - b.top)

/**
 * A chart's plot areas: the boxes its marks are clipped to (every crossplot, time series and bar chart clips
 * its marks to the axes), at page positions, with sample points along everything drawn inside them, so an
 * inset legend or fit readout can take the emptiest corner. Gridlines and axes run along the edges or across
 * the whole box, so they are left out of the count.
 */
function plotAreas(svg: SVGSVGElement): Plot[] {
  const boxes: Box[] = []
  for (const user of svg.querySelectorAll<SVGGraphicsElement>('[clip-path]')) {
    const id = /url\(\s*['"]?#([^'")]+)/.exec(user.getAttribute('clip-path') ?? '')?.[1]
    const rect = id ? document.getElementById(id)?.querySelector('rect') : null
    const ctm = user.getScreenCTM()
    if (!rect || !ctm) continue
    const x = Number(rect.getAttribute('x') ?? 0), y = Number(rect.getAttribute('y') ?? 0)
    const w = Number(rect.getAttribute('width')), h = Number(rect.getAttribute('height'))
    if (!(w > 60 && h > 60)) continue
    const a = new DOMPoint(x, y).matrixTransform(ctm), b = new DOMPoint(x + w, y + h).matrixTransform(ctm)
    const box = { left: Math.min(a.x, b.x), top: Math.min(a.y, b.y), right: Math.max(a.x, b.x), bottom: Math.max(a.y, b.y) }
    if (!boxes.some((o) => Math.abs(o.left - box.left) < 2 && Math.abs(o.top - box.top) < 2 && Math.abs(o.right - box.right) < 2 && Math.abs(o.bottom - box.bottom) < 2)) boxes.push(box)
  }
  // nested clips (a fit band inside the marks' clip) are the same plot
  const plots = boxes.filter((b) => !boxes.some((o) => o !== b && area(o) > area(b) && o.left <= b.left + 1 && o.top <= b.top + 1 && o.right >= b.right - 1 && o.bottom >= b.bottom - 1))
  if (!plots.length) return []
  const inside = (b: Box, x: number, y: number) => x > b.left + 2 && x < b.right - 2 && y > b.top + 2 && y < b.bottom - 2
  const points: [number, number][] = []
  const marks = [...svg.querySelectorAll<SVGGraphicsElement>('circle, ellipse, rect, path, polygon, polyline')]
    .filter((el) => !el.closest('defs, clipPath, [data-export-stats], .regression-summary, [data-export="skip"]') && el.getClientRects().length)
  const perPath = Math.max(6, Math.min(40, Math.floor(6000 / Math.max(1, marks.length))))
  for (const el of marks) {
    const ctm = el.getScreenCTM()
    if (!ctm) continue
    if (el instanceof SVGGeometryElement && !(el instanceof SVGCircleElement) && !(el instanceof SVGEllipseElement)) {
      let len = 0
      try { len = el.getTotalLength() } catch { /* no geometry */ }
      if (len > 0) {
        for (let i = 0; i <= perPath; i++) { const p = el.getPointAtLength((len * i) / perPath).matrixTransform(ctm); points.push([p.x, p.y]) }
        continue
      }
    }
    const r = el.getBoundingClientRect()
    points.push([(r.left + r.right) / 2, (r.top + r.bottom) / 2])
  }
  return plots.map((box) => ({ box, points: points.filter(([x, y]) => inside(box, x, y)), stats: [] }))
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

/**
 * What a publication figure shows. The fit statistics' lines are only there when the chart draws them
 * (the 95 % intervals need the gallery's fit uncertainty on); these choose among them.
 */
export interface PubShow {
  n: boolean; r2: boolean; equation: boolean; method: boolean; slopeCI: boolean; interceptCI: boolean; ciNote: boolean
  counts: boolean; fitLine: boolean; band: boolean; identity: boolean
}
export const PUB_SHOW: PubShow = { n: true, r2: true, equation: true, method: true, slopeCI: true, interceptCI: true, ciNote: false,
  counts: true, fitLine: true, band: true, identity: true }

/**
 * A publication figure: the fit statistics inside the plot (or above it, as on screen, or left out),
 * gridlines optional, one typeface (`font`, CSS) for every text, an optional panel letter, "(a)", and
 * the pieces of information chosen in `show`.
 */
export interface PubLayout { stats: 'inside' | 'above' | 'off'; grid: boolean; label: string; font: string; show: PubShow }

/** a legend item or facet title without its "n=…" count, its box narrowed to what is left */
function withoutCount(it: Piece): Piece {
  const els = it.els.filter((el) => !(el.tagName === 'text' && /^n=\d+$/.test(el.textContent ?? '')))
  if (els.length === it.els.length) return it
  const right = Math.max(it.box.left, ...els.map((el) => Number(el.getAttribute('x') ?? it.box.left) + Number(el.getAttribute(el.tagName === 'text' ? 'data-w' : 'width') ?? 0)))
  return { ...it, els, box: { ...it.box, right } }
}

/** a legend item drawn k times its size, from its top left */
function scaled(it: Piece, k: number): Piece {
  const { left: L, top: T } = it.box
  const g = document.createElementNS(NS, 'g')
  g.setAttribute('transform', `translate(${L},${T}) scale(${+k.toFixed(3)}) translate(${-L},${-T})`)
  for (const el of it.els) g.appendChild(el.cloneNode(true))
  return { ...it, els: [g], box: { left: L, top: T, right: L + (it.box.right - L) * k, bottom: T + (it.box.bottom - T) * k } }
}

/**
 * One <svg> from a composed figure, with the legend where it was asked for. `legendScale` sizes a moved
 * legend (inside, right, top, bottom): 1 is its size on the page, or in a publication figure the figure's
 * own text size. A legend left as on screen keeps its size, since the chart sits around it.
 */
export function layoutFigure(fig: Figure, legend: LegendPlacement = 'shown', consolidateSeasons = false, pub?: PubLayout, legendScale = 1): SVGSVGElement {
  if (consolidateSeasons && fig.palette.seasons.length) {
    fig = { ...fig, legends: fig.legends.map(consolidateSeasonLegend) }
    if (legend === 'shown') legend = 'bottom'
  }
  if (pub) {
    // the legend's reading hints go, and the counts if asked
    const keep = (it: Piece) => (pub.show.counts ? it : withoutCount(it))
    fig = { ...fig,
      chart: pub.show.counts ? fig.chart : fig.chart.map(withoutCount).filter((p) => p.els.length),
      legends: fig.legends.map((items) => items.filter((it) => !it.note).map(keep)).filter((items) => items.length) }
  }
  if (legend !== 'shown') {
    // a publication legend starts at the figure's text size (the page's legend is set a little larger than the ticks)
    const sizes = fig.legends.flat().flatMap((it) => it.els.filter((el) => el.tagName === 'text').map((el) => parseFloat(el.getAttribute('font-size') ?? '0'))).filter((v) => v > 0)
    const k = (pub && sizes.length ? fig.text.median / Math.max(...sizes) : 1) * legendScale
    if (Math.abs(k - 1) > 0.01) fig = { ...fig, legends: fig.legends.map((items) => items.map((it) => scaled(it, k))) }
  }
  const out = document.createElementNS(NS, 'svg') as SVGSVGElement
  const placed: { piece: Piece; dx: number; dy: number }[] = fig.chart.map((piece) => ({ piece, dx: 0, dy: 0 }))
  const chartBox = union(fig.chart.map((p) => p.box))
  const legendPieces = fig.legends.flat()

  // insets: the fit statistics first (one per plot that has them), then the legend, each in the emptiest corner left
  const taken: Box[] = []
  if (pub?.stats === 'inside') for (const plot of fig.plots) {
    if (!plot.stats.length) continue
    const inset = statsInset(plot, fig.text.median, pub.font, pub.show)
    if (!inset) continue
    const at = corner(plot, inset.w, inset.h, ['tl', 'tr', 'bl', 'br'], taken)
    // too big for the plot: above it, where the summary sits on screen
    const box = at ?? { left: plot.box.left, top: plot.box.top - inset.h - 6, right: plot.box.left + inset.w, bottom: plot.box.top - 6 }
    taken.push(box)
    placed.push({ piece: { els: inset.draw(box.left, box.top), box }, dx: 0, dy: 0 })
  }
  if (legend === 'inside') {
    // one plot to put it in; a figure of several facets keeps its legend at the right
    const plot = fig.plots.length === 1 ? fig.plots[0] : null
    const itemsW = Math.max(0, ...legendPieces.map((it) => it.box.right - it.box.left))
    const itemsH = legendPieces.reduce((h, it) => h + it.box.bottom - it.box.top + 3, 0) - 3 + 4 * (fig.legends.length - 1)
    const pad = 5
    const at = plot && legendPieces.length ? corner(plot, itemsW + 2 * pad, itemsH + 2 * pad, ['br', 'tl', 'tr', 'bl'], taken) : null
    if (at) {
      taken.push(at)
      placed.push({ piece: { els: [insetBackground(at)], box: at }, dx: 0, dy: 0 })
      let y = at.top + pad
      for (const items of fig.legends) {
        for (const it of items) {
          placed.push({ piece: it, dx: at.left + pad - it.box.left, dy: y - it.box.top })
          y += it.box.bottom - it.box.top + 3
        }
        y += 4
      }
    } else legend = plot ? 'bottom' : 'right'
  }
  if (pub?.label.trim()) {
    const size = Math.round(fig.text.median * 1.3)
    const t = document.createElementNS(NS, 'text')
    const top = chartBox.top - size - 4
    t.setAttribute('x', String(chartBox.left)); t.setAttribute('y', String(top + size / 2))
    t.setAttribute('dominant-baseline', 'central'); t.setAttribute('font-size', String(size)); t.setAttribute('font-weight', 'bold'); t.setAttribute('fill', '#000')
    t.textContent = pub.label.trim()
    placed.push({ piece: { els: [t], box: { left: chartBox.left, top, right: chartBox.left + textWidth(t.textContent, size, pub.font, 'bold'), bottom: top + size } }, dx: 0, dy: 0 })
  }

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
  if (pub) {
    // gridlines: every chart draws them as lines in the grid colour (light, as composed)
    const grid = paintColor(INK_LIGHT.grid)
    if (!pub.grid) for (const el of out.querySelectorAll('line')) if (paintColor(el.getAttribute('stroke')) === grid) el.remove()
    if (!pub.show.fitLine) for (const el of out.querySelectorAll('[data-regression]')) el.remove()
    if (!pub.show.band) for (const el of out.querySelectorAll('[data-fit-uncertainty]')) el.remove()
    if (!pub.show.identity) for (const el of out.querySelectorAll('[data-identity]')) el.remove()
    // one typeface for the whole figure (the summary and counts are monospace on screen)
    for (const el of out.querySelectorAll('[font-family]')) el.removeAttribute('font-family')
    for (const el of out.querySelectorAll<SVGElement>('[style*="font-family"]')) el.style.removeProperty('font-family')
    out.setAttribute('data-font', pub.font)
  }
  return out
}

const measure = document.createElement('canvas').getContext('2d')
const textWidth = (text: string, size: number, font: string, weight = 'normal') => {
  if (!measure) return text.length * size * 0.55
  measure.font = `${weight} ${size}px ${font}`
  return measure.measureText(text).width
}

function insetBackground(b: Box) {
  const r = document.createElementNS(NS, 'rect')
  r.setAttribute('x', String(b.left)); r.setAttribute('y', String(b.top))
  r.setAttribute('width', String(b.right - b.left)); r.setAttribute('height', String(b.bottom - b.top))
  r.setAttribute('rx', '2'); r.setAttribute('fill', '#ffffff'); r.setAttribute('fill-opacity', '0.88')
  r.setAttribute('stroke', '#c8c8c8'); r.setAttribute('stroke-width', '0.6')
  return r
}

/**
 * A plot's fit statistics as a compact readout in the figure's own text size and typeface: the site,
 * n and R², each fit's equation and its 95 % intervals, as `show` asks. The resampling note (which CI)
 * is off by default: a journal wants methods in the caption. Null when nothing is left to show.
 */
function statsLine(text: string, show: PubShow): string | null {
  if (/^n = /.test(text)) {
    const parts = text.split(' · ').filter((p) => (/^n = /.test(p) ? show.n : /^R²/.test(p) ? show.r2 : true))
    return parts.length ? parts.join(', ') : null
  }
  if (/^95% CI ·|^CI unavailable/.test(text)) return show.ciNote ? text.replace(' · ', ': ') : null
  if (/^95% slope/.test(text)) return show.slopeCI ? text : null
  if (/^95% intercept/.test(text)) return show.interceptCI ? text : null
  if (/\by = /.test(text)) return show.equation ? (show.method ? text : text.slice(text.indexOf('y = '))) : null
  return text
}

function statsInset(plot: Plot, size: number, font: string, show: PubShow) {
  const fs = Math.max(8, Math.round(size * 0.95))
  const step = Math.round(fs * 1.25)
  const lines = plot.stats.flatMap((st) => st.lines
    .map((l) => ({ ...l, text: l.name ? l.text : statsLine(l.text, show), site: st.site, siteColor: st.color }))
    .filter((l): l is typeof l & { text: string } => l.text !== null))
  // a site's name alone says nothing
  if (!lines.some((l) => !l.name)) return null
  const padX = 5, padY = 4
  const w = Math.ceil(Math.max(...lines.map((l) => textWidth(l.text, fs, font, l.name ? 'bold' : 'normal')))) + 2 * padX
  const h = lines.length * step + 2 * padY - (step - fs)
  const draw = (x: number, y: number) => {
    const els: Element[] = [insetBackground({ left: x, top: y, right: x + w, bottom: y + h })]
    lines.forEach((l, i) => {
      const t = document.createElementNS(NS, 'text')
      t.setAttribute('x', String(x + padX)); t.setAttribute('y', String(y + padY + i * step + fs / 2))
      t.setAttribute('dominant-baseline', 'central'); t.setAttribute('font-size', String(fs)); t.setAttribute('fill', l.color)
      if (l.name) t.setAttribute('font-weight', 'bold')
      if (l.method) t.setAttribute('data-regression-color', l.method)
      if (l.site) { t.setAttribute('data-export-site', l.site); if (l.siteColor) t.setAttribute('data-export-color', l.siteColor) }
      t.textContent = l.text
      els.push(t)
    })
    return els
  }
  return { w, h, draw }
}

/**
 * The emptiest corner of a plot for a w × h box, clear of boxes already placed. Null when it does not fit,
 * or when every corner would cover data (a few samples of a line, or about 1 % of the marks): an inset
 * that hides points is worse than one outside the axes.
 */
function corner(plot: Plot, w: number, h: number, order: ('tl' | 'tr' | 'bl' | 'br')[], taken: Box[]): Box | null {
  const m = 6, b = plot.box
  if (w > b.right - b.left - 2 * m || h > b.bottom - b.top - 2 * m) return null
  const at = (c: string): Box => {
    const left = c[1] === 'l' ? b.left + m : b.right - m - w
    const top = c[0] === 't' ? b.top + m : b.bottom - m - h
    return { left, top, right: left + w, bottom: top + h }
  }
  const overlaps = (p: Box, q: Box) => p.left < q.right && q.left < p.right && p.top < q.bottom && q.top < p.bottom
  let best: Box | null = null, bestScore = Infinity
  for (const c of order) {
    const box = at(c)
    if (taken.some((t) => overlaps(t, box))) continue
    const near = { left: box.left - 3, top: box.top - 3, right: box.right + 3, bottom: box.bottom + 3 }
    const score = plot.points.reduce((n, [x, y]) => n + (x >= near.left && x <= near.right && y >= near.top && y <= near.bottom ? 1 : 0), 0)
    if (score < bestScore) { best = box; bestScore = score }
  }
  return bestScore <= Math.max(2, plot.points.length * 0.01) ? best : null
}

/** The figure exactly as it sits on screen (the old single-button export). */
export function composePanel(panel: HTMLElement): SVGSVGElement | null {
  const fig = composeFigure(panel)
  return fig ? layoutFigure(fig, 'shown') : null
}

/** A figure composed on the dark page, redrawn in the light theme's colours (lib/theme printColour); site and season colours stay. */
const PAINTS = ['fill', 'stroke', 'stop-color', 'flood-color', 'color']
const SHAPES = new Set(['rect', 'path', 'circle', 'polygon', 'ellipse'])
function printColours(root: Element, fixed: ReadonlySet<string>) {
  const recoloured = new Set<Element>()
  const swap = (el: Element, paint: string) => {
    const light = printPaint(paint, fixed)
    if (!light) return paint
    recoloured.add(el)
    return replacePaint(paint, paintColor(paint)!, light)
  }
  const all = [root, ...root.querySelectorAll('*')]
  for (const el of all) {
    for (const a of PAINTS) {
      const v = el.getAttribute(a)
      if (v) el.setAttribute(a, swap(el, v))
    }
    const style = (el as SVGElement).style
    if (style) for (const a of PAINTS) {
      const v = style.getPropertyValue(a)
      if (v) style.setProperty(a, swap(el, v))
    }
  }
  // a label on a heat-map cell was given white or dark ink for the cell as the dark page showed it
  // (lib/theme inkOn): pick it again for the cell's light colour
  for (const el of all) {
    const ink = paintColor(el.getAttribute('fill'))
    if (el.tagName !== 'text' || recoloured.has(el) || (ink !== '#ffffff' && ink !== CELL_INK)) continue
    let cell = el.previousElementSibling
    while (cell && !SHAPES.has(cell.tagName)) cell = cell.previousElementSibling
    const fill = cell ? paintColor(cell.getAttribute('fill')) : null
    if (fill) el.setAttribute('fill', inkOnFill(fill))
  }
}

function serialise(svg: SVGSVGElement, background: ExportBackground, fontFamily = svg.getAttribute('data-font') ?? FONT, widthMm?: number): { xml: string; width: number; height: number; node: SVGSVGElement } {
  const clone = svg.cloneNode(true) as SVGSVGElement
  // a composed figure is not in the DOM, so its client size is 0 and the attributes carry it
  const width = svg.clientWidth || Number(svg.getAttribute('width')) || 800
  const height = svg.clientHeight || Number(svg.getAttribute('height')) || 500
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('width', widthMm ? `${widthMm}mm` : String(width))
  clone.setAttribute('height', widthMm ? `${+(widthMm * height / width).toFixed(2)}mm` : String(height))
  // a physical size keeps the drawing's px as its coordinates
  if (widthMm) clone.setAttribute('viewBox', `0 0 ${width} ${height}`)
  clone.removeAttribute('data-font')
  clone.setAttribute('font-family', fontFamily)
  if (fontFamily !== 'Inter') for (const el of clone.querySelectorAll('[data-w]')) el.removeAttribute('data-w')
  if (background === 'white') {
    const bg = document.createElementNS(NS, 'rect')
    bg.setAttribute('width', '100%')
    bg.setAttribute('height', '100%')
    bg.setAttribute('fill', '#ffffff')
    clone.insertBefore(bg, clone.firstChild)
  }
  return { xml: new XMLSerializer().serializeToString(clone), width, height, node: clone }
}

/** `widthMm` (a journal column): the file is that wide on paper; a raster is then drawn at `dpi` instead of `scale` */
export interface RenderOpts { format: ExportFormat; scale: number; background: ExportBackground; widthMm?: number; dpi?: number }

/** pixels per figure px for a raster at a physical width and dpi, or the plain scale */
export const rasterScale = (widthPx: number, opts: Pick<RenderOpts, 'scale' | 'widthMm' | 'dpi'>) =>
  opts.widthMm && opts.dpi ? (opts.dpi * opts.widthMm / 25.4) / widthPx : opts.scale
export const MIME: Record<ExportFormat, string> = { png: 'image/png', jpeg: 'image/jpeg', svg: 'image/svg+xml', pdf: 'application/pdf' }
/** browsers cap a canvas side near 16k px (Safari lower), so very large scales are refused */
export const MAX_CANVAS_SIDE = 16000

/** The file for one export option. */
export async function renderFigure(svg: SVGSVGElement, opts: RenderOpts): Promise<Blob> {
  // JPEG has no alpha: a transparent background would turn black
  const background = opts.format === 'jpeg' ? 'white' : opts.background
  if (opts.format === 'svg') return new Blob([serialise(svg, background, undefined, opts.widthMm).xml], { type: 'image/svg+xml;charset=utf-8' })
  if (opts.format === 'pdf') return renderPdf(svg, background, opts.widthMm)
  const { xml, width, height } = serialise(svg, background)
  const scale = rasterScale(width, opts)
  const url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }))
  try {
    const img = new Image()
    await new Promise<void>((resolve, reject) => { img.onload = () => resolve(); img.onerror = () => reject(new Error('could not draw the figure')); img.src = url })
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(width * scale)
    canvas.height = Math.round(height * scale)
    if (Math.max(canvas.width, canvas.height) > MAX_CANVAS_SIDE) throw new Error('too large for the browser to draw')
    const ctx = canvas.getContext('2d')!
    ctx.scale(scale, scale)
    ctx.drawImage(img, 0, 0)
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('the browser could not encode the image'))), MIME[opts.format], 0.92))
  } finally {
    URL.revokeObjectURL(url)
  }
}

/**
 * A vector PDF (jsPDF + svg2pdf.js, loaded only when asked for). The page is the
 * figure's size at 96 px to the inch, so it prints at the size it shows on screen,
 * or `widthMm` wide (a journal column) with everything scaled to it.
 * PDF's built-in fonts cannot draw ₂, −, ≈ or λ, so Inter (OFL) is embedded and
 * every text uses it.
 */
async function renderPdf(svg: SVGSVGElement, background: ExportBackground, widthMm?: number): Promise<Blob> {
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
  const pt = widthMm ? (widthMm / 25.4 * 72) / width : 0.75
  const doc = new jsPDF({ unit: 'pt', format: [width * pt, height * pt], orientation: width > height ? 'landscape' : 'portrait', compress: true })
  doc.addFileToVFS('Inter-Regular.ttf', fonts.regular)
  doc.addFont('Inter-Regular.ttf', 'Inter', 'normal')
  doc.addFileToVFS('Inter-SemiBold.ttf', fonts.bold)
  doc.addFont('Inter-SemiBold.ttf', 'Inter', 'bold')
  doc.setFont('Inter', 'normal')
  // svg2pdf measures text and reads computed styles, so the copy has to be in the document while it converts
  const host = document.createElement('div')
  host.style.cssText = 'position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none'
  // lib/themeCss leaves a [data-print] copy alone, so the dark page's CSS does not recolour the PDF
  host.dataset.print = ''
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
