import { EMBED_CHART_MESSAGE, type EmbedChartMessage } from './embedChart.ts'
import { deckSlides, deckText, type Slide } from './sectionDeck.ts'
import { itemHref, sectionUrl, type Section, type SectionItem } from './sections.ts'
import { downloadBlob, renderPdfPages, slug } from './export.ts'
import { EXPORT_CHART, type DrawnChart } from '../components/EmbedExportBridge.tsx'

export type SectionFormat = 'pdf' | 'pptx'
export interface ExportProgress { done: number; total: number; title: string }
export interface ExportResult { missing: string[] }

/** the size a chart is drawn at for a page or slide */
const FRAME = { width: 1280, height: 760 }
const FIND_MS = 45000

/** One chart at a time, in a frame of its own off the screen: what is on screen may have been let go, and is card-sized. */
async function drawItem(item: SectionItem): Promise<DrawnChart | null> {
  const frame = document.createElement('iframe')
  frame.style.cssText = `position:fixed;left:-20000px;top:0;width:${FRAME.width}px;height:${FRAME.height}px;border:0;pointer-events:none`
  frame.setAttribute('aria-hidden', 'true')
  frame.tabIndex = -1
  try {
    const found = new Promise<boolean>((resolve) => {
      const giveUp = window.setTimeout(() => { window.removeEventListener('message', hear); resolve(false) }, FIND_MS)
      const hear = (e: MessageEvent<EmbedChartMessage>) => {
        if (e.source !== frame.contentWindow || e.data?.type !== EMBED_CHART_MESSAGE || e.data.active !== item.chart) return
        window.clearTimeout(giveUp)
        window.removeEventListener('message', hear)
        resolve(true)
      }
      window.addEventListener('message', hear)
    })
    frame.src = itemHref(item, true)
    document.body.appendChild(frame)
    if (!await found) return null
    const host = frame.contentWindow as unknown as Record<string, () => Promise<DrawnChart | null>>
    // the bridge mounts with the page, a moment after the chart's frame
    for (let i = 0; i < 100 && !host[EXPORT_CHART]; i++) await new Promise((r) => setTimeout(r, 100))
    return await host[EXPORT_CHART]?.() ?? null
  } finally { frame.remove() }
}

async function deck(slides: Slide[], drawn: Map<string, DrawnChart>): Promise<Blob> {
  const { default: PptxGenJS } = await import('pptxgenjs')
  const pptx = new PptxGenJS()
  pptx.defineLayout({ name: 'HOUSE', width: 13.333, height: 7.5 })
  pptx.layout = 'HOUSE'
  const ink = '22252A', font = 'Arial', m = 0.5, W = 13.333 - 2 * m
  for (const s of slides) {
    const slide = pptx.addSlide()
    slide.background = { color: 'FFFFFF' }
    slide.addNotes(s.notes)
    if (s.kind === 'title' || s.kind === 'divider') {
      slide.addText(s.title || ' ', { x: m, y: 2.6, w: W, h: 1.2, fontSize: 34, bold: true, color: ink, fontFace: font, valign: 'middle' })
      if (s.body) slide.addText(s.body, { x: m, y: 3.9, w: W, h: 1.4, fontSize: 17, color: ink, fontFace: font, valign: 'top' })
      continue
    }
    slide.addText(s.title || ' ', { x: m, y: 0.3, w: W, h: 0.7, fontSize: 20, bold: true, color: ink, fontFace: font, valign: 'top' })
    if (s.kind === 'text') { slide.addText(s.body ?? '', { x: m, y: 1.2, w: W, h: 5.7, fontSize: 17, color: ink, fontFace: font, valign: 'top' }); continue }
    const fig = s.item && drawn.get(s.item.id)
    if (!fig) { slide.addText('This chart could not be drawn for the export.', { x: m, y: 3.2, w: W, h: 0.6, fontSize: 17, color: '8F8C84', fontFace: font, align: 'center' }); continue }
    // centred under the title, its shape kept
    const top = 1.1, room = 7.5 - top - 0.3, k = Math.min(W / fig.width, room / fig.height)
    const w = fig.width * k, h = fig.height * k
    slide.addImage({ data: fig.jpeg, x: (13.333 - w) / 2, y: top + (room - h) / 2, w, h })
  }
  return (await pptx.write({ outputType: 'blob' })) as Blob
}

function pdf(slides: Slide[], drawn: Map<string, DrawnChart>) {
  return renderPdfPages(slides.map((s) => {
    const fig = s.item && drawn.get(s.item.id)
    const note = s.item?.note ? deckText(s.item.note) : undefined
    if (s.kind === 'chart') return { title: s.title, svg: fig?.svg, body: fig ? note : 'This chart could not be drawn for the export.' }
    return { title: s.title, body: s.body, large: s.kind !== 'text' }
  }))
}

/** Draw every chart, one at a time, and save the section as one file. Says which charts could not be drawn. */
export async function exportSection(section: Section, format: SectionFormat, onStep: (p: ExportProgress) => void, cancelled: () => boolean): Promise<ExportResult | null> {
  const slides = deckSlides(section, await sectionUrl(section).catch(() => undefined))
  const charts = slides.filter((s) => s.item)
  const drawn = new Map<string, DrawnChart>()
  const missing: string[] = []
  for (const [i, s] of charts.entries()) {
    if (cancelled()) return null
    onStep({ done: i, total: charts.length, title: s.title })
    const fig = await drawItem(s.item!).catch(() => null)
    if (fig) drawn.set(s.item!.id, fig); else missing.push(s.title)
  }
  if (cancelled()) return null
  onStep({ done: charts.length, total: charts.length, title: '' })
  const blob = format === 'pdf' ? await pdf(slides, drawn) : await deck(slides, drawn)
  downloadBlob(blob, `${slug(section.name) || 'section'}.${format}`)
  return { missing }
}
