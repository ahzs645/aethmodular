import { useEffect } from 'react'
import { composeFigure, layoutFigure, renderFigure } from '@/lib/export'
import { setExportRender, settled } from '@/lib/exportRender'
import { useExportSeasons, useExportSites } from '@/lib/exportSites'

/** One chart as a section's export needs it, from its embed. */
export interface DrawnChart { svg: string; jpeg: string; width: number; height: number }
export const EXPORT_CHART = '__galleryExportChart'
/** a chart with its data still coming has no drawing to copy yet */
const DRAW_MS = 20000
const WAIT_MS = 150

const drawn = (panel: Element) => !!panel.querySelector('.frame-chart svg, .frame-chart canvas')

/**
 * Inside a section's embed: hands the page around it this chart, drawn for a file the way the chart's own export
 * dialog draws it at its defaults (lib/export, lib/exportRender). JPEG for a slide: a canvas's PNG always has an
 * alpha channel, which the house deck style rules out for PowerPoint. Null when the chart draws nothing to copy.
 */
export function EmbedExportBridge() {
  const sites = useExportSites()
  const seasons = useExportSeasons()
  useEffect(() => {
    const host = window as unknown as Record<string, () => Promise<DrawnChart | null>>
    host[EXPORT_CHART] = async () => {
      const panel = document.querySelector<HTMLElement>('[data-single-active], .dash-max .panel')
      if (!panel) return null
      for (const end = Date.now() + DRAW_MS; !drawn(panel) && Date.now() < end;) await new Promise((r) => setTimeout(r, WAIT_MS))
      setExportRender({ panel, width: null, statsInside: false })
      try {
        await settled(panel)
        const fig = composeFigure(panel, sites, seasons)
        if (!fig) return null
        const svg = layoutFigure(fig, 'shown')
        const blob = await renderFigure(svg, { format: 'jpeg', scale: 3, background: 'white' })
        const jpeg = await new Promise<string>((resolve, reject) => {
          const r = new FileReader()
          r.onload = () => resolve(r.result as string)
          r.onerror = () => reject(new Error('could not read the image'))
          r.readAsDataURL(blob)
        })
        // as drawn, not as a file: the PDF's own copy needs the on-screen text widths a saved SVG leaves out (lib/export pdfCopy)
        return { svg: new XMLSerializer().serializeToString(svg), jpeg, width: Number(svg.getAttribute('width')), height: Number(svg.getAttribute('height')) }
      } finally { setExportRender(null) }
    }
    return () => { delete host[EXPORT_CHART] }
  }, [sites, seasons])
  return null
}
