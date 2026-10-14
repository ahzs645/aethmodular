import { useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { createPortal } from 'react-dom'
import { Combobox } from '@/components/Combobox'
import { Segmented } from '@/components/ChartFrame'
import {
  composeFigure, downloadBlob, layoutFigure, renderFigure, slug, MAX_CANVAS_SIDE,
  type ExportBackground, type ExportFormat, type Figure, type LegendPlacement,
} from '@/lib/export'

/**
 * One export control per chart: the button names the format it will save (PNG by
 * default, or whatever this reader used last), and opens a dialog with a live
 * preview of the file, the format, the resolution, where the legend goes and the
 * background, and the size of every option before anything is downloaded.
 */
const FORMATS: { key: ExportFormat; label: string; note: string }[] = [
  { key: 'png', label: 'PNG', note: 'raster · slides, docs' },
  { key: 'jpeg', label: 'JPEG', note: 'raster · smallest, no transparency' },
  { key: 'svg', label: 'SVG', note: 'vector · Illustrator, Inkscape' },
  { key: 'pdf', label: 'PDF', note: 'vector · print, posters, LaTeX' },
]
const SCALES: { key: string; scale: number; label: string }[] = [
  { key: '1', scale: 1, label: '1× · 96 dpi' },
  { key: '2', scale: 2, label: '2× · 192 dpi' },
  { key: '3', scale: 3, label: '3× · 288 dpi' },
  { key: '300', scale: 300 / 96, label: '300 dpi · print' },
  { key: '4', scale: 4, label: '4× · 384 dpi' },
  { key: '600', scale: 600 / 96, label: '600 dpi · poster' },
]
const LEGENDS: LegendPlacement[] = ['shown', 'right', 'bottom', 'top', 'none']
const LEGEND_LABEL: Record<LegendPlacement, string> = { shown: 'As on screen', right: 'Right', bottom: 'Bottom', top: 'Top', none: 'None' }

interface Settings { format: ExportFormat; scale: string; legend: LegendPlacement; background: ExportBackground }
const DEFAULTS: Settings = { format: 'png', scale: '2', legend: 'shown', background: 'white' }
const KEY = 'gallery.export'
// the reader's last choices: a per-browser convenience, so the button says what it will save
function loadSettings(): Settings {
  try {
    const s = { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') } as Settings
    return {
      format: FORMATS.some((f) => f.key === s.format) ? s.format : DEFAULTS.format,
      scale: SCALES.some((x) => x.key === s.scale) ? s.scale : DEFAULTS.scale,
      legend: LEGENDS.includes(s.legend) ? s.legend : DEFAULTS.legend,
      background: s.background === 'transparent' ? 'transparent' : 'white',
    }
  } catch { return DEFAULTS }
}
function saveSettings(s: Settings) {
  try { localStorage.setItem(KEY, JSON.stringify(s)) } catch { /* private window: nothing kept */ }
}

export const fileSize = (n: number) => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`)

export function ExportButton({ panelRef, title }: { panelRef: RefObject<HTMLElement>; title: string }) {
  const [open, setOpen] = useState(false)
  const [format, setFormat] = useState<ExportFormat>(() => loadSettings().format)
  return (
    <>
      <button type="button" className="btn quiet export-btn" aria-haspopup="dialog" aria-expanded={open}
        title="Export the whole figure (every facet and the legend): choose the format, resolution and legend, with a preview and file sizes"
        onClick={() => setOpen(true)}>
        {format.toUpperCase()}
        <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
      </button>
      {open && panelRef.current && <ExportDialog panel={panelRef.current} title={title}
        onClose={(saved) => { setOpen(false); if (saved) setFormat(saved) }} />}
    </>
  )
}

type Rendered = { blob: Blob | null; error: string | null }

function ExportDialog({ panel, title, onClose }: { panel: HTMLElement; title: string; onClose: (saved?: ExportFormat) => void }) {
  // cloned once, when the dialog opens: later re-renders of the chart do not change the export
  const fig = useMemo<Figure | null>(() => composeFigure(panel), [panel])
  const [s, setS] = useState<Settings>(loadSettings)
  const set = (patch: Partial<Settings>) => setS((cur) => { const next = { ...cur, ...patch }; saveSettings(next); return next })
  const hasLegend = !!fig && fig.legends.length > 0
  const legend = hasLegend ? s.legend : 'none'
  const svg = useMemo(() => (fig ? layoutFigure(fig, legend) : null), [fig, legend])
  const width = Number(svg?.getAttribute('width') ?? 0), height = Number(svg?.getAttribute('height') ?? 0)
  const scale = SCALES.find((x) => x.key === s.scale)!.scale
  const raster = s.format === 'png' || s.format === 'jpeg'

  // every format at the current settings, so the sizes sit next to the choices and switching is instant
  const [out, setOut] = useState<Partial<Record<ExportFormat, Rendered>>>({})
  useEffect(() => {
    if (!svg) return
    let live = true
    setOut({})
    const timer = setTimeout(async () => {
      // the chosen format first (SVG before PDF, as the PDF's preview), then the others for their sizes
      const first: ExportFormat[] = s.format === 'pdf' ? ['svg', 'pdf'] : [s.format]
      const order = [...first, ...(['png', 'jpeg', 'svg', 'pdf'] as ExportFormat[]).filter((f) => !first.includes(f))]
      for (const format of order) {
        let r: Rendered
        try { r = { blob: await renderFigure(svg, { format, scale, background: s.background }), error: null } } catch (e) { r = { blob: null, error: e instanceof Error ? e.message : String(e) } }
        if (!live) return
        setOut((cur) => ({ ...cur, [format]: r }))
      }
    }, 120)
    return () => { live = false; clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [svg, scale, s.background])

  // the preview is the file itself for PNG / JPEG / SVG; a PDF previews as its SVG
  const current = out[s.format]
  const previewBlob = s.format === 'pdf' ? out.svg?.blob : current?.blob
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!previewBlob) { setPreviewUrl(null); return }
    const url = URL.createObjectURL(previewBlob)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [previewBlob])

  const dialogRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    dialogRef.current?.querySelector<HTMLButtonElement>('.export-download')?.focus()
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const rasterSize = out[s.format]?.blob?.size
  // other resolutions scale roughly with the pixel count; marked ≈ because compression is not linear
  const scaleHint = (k: number) => {
    const px = `${Math.round(width * k).toLocaleString()} × ${Math.round(height * k).toLocaleString()} px`
    if (Math.max(width, height) * k > MAX_CANVAS_SIDE) return `${px} · too large`
    if (!raster || !rasterSize) return px
    return `${px} · ${k === scale ? '' : '≈ '}${fileSize(Math.round(rasterSize * (k / scale) ** 2))}`
  }
  const inches = (px: number) => (raster ? (px * scale) / 300 : px / 96).toFixed(1)
  const save = () => {
    if (!current?.blob) return
    downloadBlob(current.blob, `${slug(title)}.${s.format === 'jpeg' ? 'jpg' : s.format}`)
    onClose(s.format)
  }

  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div ref={dialogRef} className="export-dialog" role="dialog" aria-modal="true" aria-label={`Export ${title}`}>
        <header>
          <div>
            <strong>Export</strong>
            <span className="export-title">{title}</span>
          </div>
          <button type="button" className="btn quiet" onClick={() => onClose()} aria-label="Close" title="Close (Esc)">✕</button>
        </header>
        {!fig ? <p className="frame-empty">This panel has no figure to export.</p> : (
          <div className="export-body">
            <div className={`export-preview ${s.background === 'transparent' && s.format !== 'jpeg' ? 'checker' : ''}`}>
              {previewUrl ? <img src={previewUrl} alt={`Preview of the ${s.format.toUpperCase()} export`} /> : <span className="export-wait">Rendering…</span>}
            </div>
            <div className="export-options">
              <fieldset className="export-formats">
                <legend>Format</legend>
                {FORMATS.map((f) => {
                  const r = out[f.key]
                  return (
                    <label key={f.key} className={s.format === f.key ? 'on' : ''}>
                      <input type="radio" name="export-format" checked={s.format === f.key} onChange={() => set({ format: f.key })} />
                      <span className="export-format-name">{f.label}</span>
                      <span className="export-format-note">{f.note}</span>
                      <span className="export-format-size">{r ? (r.blob ? fileSize(r.blob.size) : '–') : '…'}</span>
                    </label>
                  )
                })}
              </fieldset>
              <div className="export-row">
                <Combobox label="Resolution" value={s.scale} onChange={(v) => set({ scale: v })}
                  title={raster ? 'Pixel density of the image; 300 dpi is the usual print requirement' : 'Vector formats have no resolution; this only applies to PNG and JPEG'}
                  options={SCALES.map((x) => ({ key: x.key, label: x.label, hint: scaleHint(x.scale) }))} />
              </div>
              <div className="export-row">
                <Segmented label="Legend" value={legend} options={hasLegend ? LEGENDS : ['none'] as LegendPlacement[]} optionLabel={(o) => LEGEND_LABEL[o]}
                  onChange={(v) => set({ legend: v })} title={hasLegend ? 'Where the legend sits in the file; the chart on the page is not changed' : 'This chart has no legend'} />
              </div>
              <div className="export-row">
                <Segmented label="Background" value={s.format === 'jpeg' ? 'white' : s.background} options={['white', 'transparent'] as ExportBackground[]}
                  onChange={(v) => set({ background: v })} title={s.format === 'jpeg' ? 'JPEG has no transparency; it is always white' : 'Transparent suits a coloured slide or poster background'} />
              </div>
              <p className="export-facts">
                {raster
                  ? <>{Math.round(width * scale).toLocaleString()} × {Math.round(height * scale).toLocaleString()} px · prints at {inches(width)} × {inches(height)} in at 300 dpi</>
                  : <>{width} × {height} px canvas · vector, sharp at any size{s.format === 'pdf' ? ` · page ${(width / 96).toFixed(1)} × ${(height / 96).toFixed(1)} in, Inter font embedded` : ''}</>}
              </p>
              {current?.error && <p className="export-error">Could not make this file: {current.error}.</p>}
              <div className="export-actions">
                <button type="button" className="btn quiet" onClick={() => onClose()}>Cancel</button>
                <button type="button" className="btn primary export-download" disabled={!current?.blob} onClick={save}>
                  Download {s.format.toUpperCase()}{current?.blob ? ` · ${fileSize(current.blob.size)}` : ''}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}
