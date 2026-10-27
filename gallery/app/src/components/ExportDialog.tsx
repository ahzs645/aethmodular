import { useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { createPortal } from 'react-dom'
import { ExportColorPicker } from '@/components/ExportColorOptions'
import { useExportSeasons, useExportSites } from '@/lib/exportSites'
import { applyExportColors, DEFAULT_EXPORT_COLORS, readExportColors, type ExportColors } from '@/lib/exportColors'
import { Combobox } from '@/components/Combobox'
import { Segmented } from '@/components/ChartFrame'
import { setExportRender, settled } from '@/lib/exportRender'
import {
  composeFigure, downloadBlob, layoutFigure, rasterScale, renderFigure, slug, MAX_CANVAS_SIDE,
  PUB_FONTS, PUB_SHOW, type ExportBackground, type ExportFormat, type Figure, type LegendPlacement, type PubLayout, type PubShow,
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
const LEGENDS: LegendPlacement[] = ['shown', 'inside', 'right', 'bottom', 'top', 'none']
const LEGEND_LABEL: Record<LegendPlacement, string> = { shown: 'As on screen', inside: 'Inside', right: 'Right', bottom: 'Bottom', top: 'Top', none: 'None' }

/**
 * Publication style. Widths are Elsevier's artwork columns (90 / 140 / 190 mm; ACS and IEEE are within a
 * few mm), and text should print at 7 pt or more (6 pt for sub- and superscripts). For a column the chart
 * is redrawn at the width where its usual text (the tick labels) comes out at the chosen size (8 pt by default), so the axes re-flow
 * for the narrower figure instead of the whole picture being shrunk; the file is then exactly that wide.
 * Resolution for a raster at that size: 300 dpi halftone, 500–600 dpi combination art, 1000 dpi line art.
 */
type Style = 'screen' | 'publication'
const COLUMNS: { key: string; label: string; note: string }[] = [
  { key: 'screen', label: 'As on screen', note: 'the chart\'s own width' },
  { key: '90', label: 'Single column · 90 mm', note: '3.5 in' },
  { key: '140', label: '1.5 column · 140 mm', note: '5.5 in' },
  { key: '190', label: 'Double column · 190 mm', note: '7.5 in' },
]
const DPIS: { key: string; label: string }[] = [
  { key: '300', label: '300 dpi · halftone' },
  { key: '600', label: '600 dpi · combination' },
  { key: '1000', label: '1000 dpi · line art' },
]
const TEXT_PTS = ['7', '8', '9', '10']
const MIN_PT = 7
/** what a publication figure can show or leave out, in the order the dialog lists them */
const SHOW_ITEMS: { key: keyof PubShow; label: string; title: string }[] = [
  { key: 'n', label: 'n', title: 'The number of pairs in the fit' },
  { key: 'r2', label: 'R²', title: 'The coefficient of determination' },
  { key: 'equation', label: 'Equation', title: 'y = slope · x + intercept, for each fit' },
  { key: 'method', label: 'Fit name', title: 'The method before the equation (Deming, OLS, York)' },
  { key: 'slopeCI', label: 'Slope CI', title: "The slope's 95 % interval (when the gallery's fit uncertainty is on, under Plot options)" },
  { key: 'interceptCI', label: 'Intercept CI', title: "The intercept's 95 % interval (when the gallery's fit uncertainty is on)" },
  { key: 'ciNote', label: 'CI method', title: 'How the intervals were resampled; usually said in the caption instead' },
  { key: 'counts', label: 'Counts', title: 'The n=… after each legend entry and facet title' },
  { key: 'fitLine', label: 'Fit line', title: 'The fitted line(s)' },
  { key: 'band', label: 'CI band', title: "The fit's shaded 95 % band" },
  { key: 'identity', label: '1:1 line', title: 'The dashed 1:1 line on a crossplot' },
]
const STATS = ['inside', 'above', 'off'] as const
const STATS_LABEL = { inside: 'Inside', above: 'Above', off: 'Off' }

interface Settings {
  format: ExportFormat; scale: string; legend: LegendPlacement; background: ExportBackground; colors: ExportColors
  style: Style; column: string; dpi: string; stats: PubLayout['stats']; grid: boolean; label: string
  font: string; textPt: string; show: PubShow
  /** a moved legend's size, in % (100: as on the page, or the figure's text size in a publication figure) */
  legendSize: number
}
const DEFAULTS: Settings = { format: 'png', scale: '2', legend: 'shown', background: 'white', colors: DEFAULT_EXPORT_COLORS,
  style: 'screen', column: '90', dpi: '600', stats: 'inside', grid: false, label: '', font: 'arial', textPt: '8', show: PUB_SHOW, legendSize: 100 }
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
      colors: readExportColors(s.colors),
      style: s.style === 'publication' ? 'publication' : 'screen',
      column: COLUMNS.some((c) => c.key === s.column) ? s.column : DEFAULTS.column,
      dpi: DPIS.some((d) => d.key === s.dpi) ? s.dpi : DEFAULTS.dpi,
      stats: STATS.includes(s.stats) ? s.stats : DEFAULTS.stats,
      grid: typeof s.grid === 'boolean' ? s.grid : DEFAULTS.grid,
      label: typeof s.label === 'string' ? s.label.slice(0, 12) : '',
      font: PUB_FONTS.some((f) => f.key === s.font) ? s.font : DEFAULTS.font,
      textPt: TEXT_PTS.includes(s.textPt) ? s.textPt : DEFAULTS.textPt,
      show: Object.fromEntries(Object.entries(PUB_SHOW).map(([k, v]) => [k, typeof s.show?.[k as keyof PubShow] === 'boolean' ? s.show[k as keyof PubShow] : v])) as unknown as PubShow,
      legendSize: Number.isFinite(s.legendSize) ? Math.min(160, Math.max(60, s.legendSize)) : 100,
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
  // while the dialog is open the chart is drawn for the file (lib/exportRender): the dialog sets how, and
  // closing it gives the page its chart back, as does the chart leaving the page with its export open
  const held = useRef(false)
  const close = (saved?: ExportFormat) => {
    setOpen(false)
    if (held.current) { held.current = false; setExportRender(null) }
    if (saved) setFormat(saved)
  }
  useEffect(() => () => { if (held.current) setExportRender(null) }, [])
  const start = () => { held.current = true; setOpen(true) }
  return (
    <>
      <button type="button" className="btn quiet export-btn" aria-haspopup="dialog" aria-expanded={open}
        title="Export the whole figure (every facet and the legend): choose the format, resolution and legend, with a preview and file sizes"
        onClick={start}>
        {format.toUpperCase()}
        <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
      </button>
      {open && panelRef.current && <ExportDialog panel={panelRef.current} title={title} onClose={close} />}
    </>
  )
}

type Rendered = { blob: Blob | null; error: string | null }

function ExportDialog({ panel, title, onClose }: { panel: HTMLElement; title: string; onClose: (saved?: ExportFormat) => void }) {
  const sites = useExportSites()
  const seasons = useExportSeasons()
  const [s, setS] = useState<Settings>(loadSettings)
  const set = (patch: Partial<Settings>) => setS((cur) => { const next = { ...cur, ...patch }; saveSettings(next); return next })
  const pub = s.style === 'publication'
  const widthMm = pub && s.column !== 'screen' ? Number(s.column) : undefined

  // The chart is drawn for the file while the dialog is open (lib/exportRender): as on the page, or for a
  // publication at a column's width with its fit statistics handed over to go inside the plot. Copied once
  // per drawing (and the palette with it): hovering or re-rendering the chart afterwards changes nothing.
  // The first copy is always as on screen, which gives the chart's text size for the column widths.
  const [fig, setFig] = useState<Figure | null | undefined>(undefined)
  const base = useRef<number | null>(null)
  const drawing = pub ? `${s.column}|${s.stats}|${s.textPt}` : 'screen'
  useEffect(() => {
    let live = true
    ;(async () => {
      if (base.current === null) {
        setExportRender({ panel, width: null, statsInside: false })
        await settled(panel)
        if (!live) return
        const first = composeFigure(panel, sites, seasons)
        base.current = first?.text.median ?? 11
        if (!pub) { setFig(first); return }
      }
      // the column's width in pt, at the chosen size for the chart's usual text; the dialog's pad of 8 px a side comes off
      const width = widthMm ? (widthMm / 25.4 * 72) * base.current / Number(s.textPt) - 16 : null
      setExportRender({ panel, width, statsInside: pub && s.stats !== 'above' })
      await settled(panel, width)
      if (live) setFig(composeFigure(panel, sites, seasons))
    })()
    return () => { live = false }
  }, [drawing]) // eslint-disable-line react-hooks/exhaustive-deps

  const hasLegend = !!fig && fig.legends.length > 0
  // inside needs a single plot to sit in; the screen style keeps the legend where the page has it
  const legends = LEGENDS.filter((l) => l !== 'inside' || (pub && fig?.plots.length === 1))
  const legend = !hasLegend ? 'none' : legends.includes(s.legend) ? s.legend : pub && legends.includes('inside') ? 'inside' : 'shown'
  const svg = useMemo(() => {
    if (!fig) return null
    const font = PUB_FONTS.find((f) => f.key === s.font)!.css
    const clone = layoutFigure(fig, legend, s.colors.consolidateSeasons, pub ? { stats: s.stats, grid: s.grid, label: s.label, font, show: s.show } : undefined, s.legendSize / 100)
    applyExportColors(clone, fig.palette, s.colors)
    return clone
  }, [fig, legend, s.colors, pub, s.stats, s.grid, s.label, s.font, s.show, s.legendSize])
  const width = Number(svg?.getAttribute('width') ?? 0), height = Number(svg?.getAttribute('height') ?? 0)
  const dpi = Number(s.dpi)
  const scale = rasterScale(width || 1, { scale: SCALES.find((x) => x.key === s.scale)!.scale, widthMm, dpi })
  const raster = s.format === 'png' || s.format === 'jpeg'
  // what the text prints at: the figure's px over its physical width (96 px to the inch otherwise)
  const ptPerPx = widthMm && width ? (widthMm / 25.4 * 72) / width : 0.75
  const textPt = fig ? { median: fig.text.median * ptPerPx, min: fig.text.min * ptPerPx } : null

  // every format at the current settings, so the sizes sit next to the choices and switching is instant
  const request = useMemo(() => ({ svg, scale, background: s.background, widthMm, dpi: widthMm ? dpi : undefined }), [svg, scale, s.background, widthMm, dpi])
  const [rendered, setRendered] = useState<{ request: typeof request; out: Partial<Record<ExportFormat, Rendered>> } | null>(null)
  // Never download an old color/size variant while its replacement is still rendering.
  const out = rendered?.request === request ? rendered.out : {}
  useEffect(() => {
    if (!svg) return
    let live = true
    const timer = setTimeout(async () => {
      // the chosen format first (SVG before PDF, as the PDF's preview), then the others for their sizes
      const first: ExportFormat[] = s.format === 'pdf' ? ['svg', 'pdf'] : [s.format]
      const order = [...first, ...(['png', 'jpeg', 'svg', 'pdf'] as ExportFormat[]).filter((f) => !first.includes(f))]
      for (const format of order) {
        let r: Rendered
        try { r = { blob: await renderFigure(svg, { format, scale, background: s.background, widthMm: request.widthMm, dpi: request.dpi }), error: null } } catch (e) { r = { blob: null, error: e instanceof Error ? e.message : String(e) } }
        if (!live) return
        setRendered(cur => ({ request, out: { ...(cur?.request === request ? cur.out : {}), [format]: r } }))
      }
    }, 120)
    return () => { live = false; clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request])

  // the preview is the file itself for PNG / JPEG / SVG; a PDF previews as its SVG
  const current = out[s.format]
  const previewBlob = s.format === 'pdf' ? out.svg?.blob : current?.blob
  const [preview, setPreview] = useState<{ blob: Blob; url: string } | null>(null)
  const previewUrl = preview && preview.blob === previewBlob ? preview.url : null
  useEffect(() => {
    if (!previewBlob) return
    const url = URL.createObjectURL(previewBlob)
    setPreview({ blob: previewBlob, url })
    return () => URL.revokeObjectURL(url)
  }, [previewBlob])

  const dialogRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !dialogRef.current?.querySelector('.export-color-popover')) { e.stopPropagation(); onClose() } }
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
        {fig === undefined ? <p className="frame-empty">Drawing the figure…</p> : !fig ? <p className="frame-empty">This panel has no figure to export.</p> : (
          <div className="export-body">
            <div className={`export-preview ${s.background === 'transparent' && s.format !== 'jpeg' ? 'checker' : ''}`}>
              {previewUrl ? <img src={previewUrl} alt={`Preview of the ${s.format.toUpperCase()} export`} /> : <span className="export-wait">Rendering…</span>}
            </div>
            <div className="export-options">
              <div className="export-settings">
                <div className="export-row">
                  <Segmented label="Style" value={s.style} options={['screen', 'publication'] as Style[]} optionLabel={(o) => (o === 'screen' ? 'Screen' : 'Publication')}
                    onChange={(v) => set({ style: v, ...(v === 'publication' && s.legend === 'shown' ? { legend: 'inside' as LegendPlacement } : {}) })}
                    title="Screen: the chart as it is on the page. Publication: a journal figure: a column's width with text near 8 pt, the legend and fit statistics inside the plot, Arial / Helvetica, gridlines optional, a panel letter" />
                </div>
                {pub && <>
                  <div className="export-row">
                    <Combobox label="Width" value={s.column} onChange={(v) => set({ column: v })}
                      title="A journal column (Elsevier's artwork widths; ACS and IEEE are within a few mm). The chart is redrawn so its text prints near 8 pt at that width"
                      options={COLUMNS.map((c) => ({ key: c.key, label: c.label, hint: c.note }))} />
                  </div>
                  <div className="export-row">
                    <Segmented label="Text" value={s.textPt} options={TEXT_PTS} optionLabel={(o) => `${o} pt`}
                      onChange={(v) => set({ textPt: v })} title="The size the tick labels print at; the rest of the text keeps its proportion. Journals ask for 7 pt or more" />
                  </div>
                  <div className="export-row">
                    <Combobox label="Font" value={s.font} onChange={(v) => set({ font: v })}
                      title="Every text in the figure. SVG and PNG use the font installed on this computer; the PDF embeds Inter whichever is chosen"
                      options={PUB_FONTS.map((f) => ({ key: f.key, label: f.label }))} />
                  </div>
                  <div className="export-row">
                    <Segmented label="Fit stats" value={s.stats} options={[...STATS]} optionLabel={(o) => STATS_LABEL[o]}
                      onChange={(v) => set({ stats: v })} title="Inside: n, R², each fit's equation and 95 % intervals in the plot's emptiest corner (the resampling note belongs in the caption). Above: as on screen. Off: left out" />
                  </div>
                  <div className="export-row">
                    <Segmented label="Gridlines" value={s.grid ? 'on' : 'off'} options={['on', 'off']} optionLabel={(o) => (o === 'on' ? 'On' : 'Off')}
                      onChange={(v) => set({ grid: v === 'on' })} title="Most journals print without gridlines; keep them for reading values off the plot" />
                  </div>
                  <fieldset className="export-show">
                    <legend>Show</legend>
                    {SHOW_ITEMS.map((it) => (
                      <label key={it.key} className={`export-check${s.show[it.key] ? ' on' : ''}`} title={it.title}>
                        <input type="checkbox" checked={s.show[it.key]} onChange={(e) => set({ show: { ...s.show, [it.key]: e.target.checked } })} />
                        {it.label}
                      </label>
                    ))}
                  </fieldset>
                  <div className="export-row">
                    <label className="export-label-field" title="A panel letter above the plot's top left, for a multi-panel figure">
                      <span className="combobox-label">Panel label</span>
                      <input type="text" value={s.label} maxLength={12} placeholder="(a)" onChange={(e) => set({ label: e.target.value })} />
                    </label>
                  </div>
                </>}
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
                  {widthMm
                    ? <Combobox label="Resolution" value={s.dpi} onChange={(v) => set({ dpi: v })}
                      title={raster ? 'Dots per inch at the column width: 300 for photos and heat maps, 600 for plots with text, 1000 for line art' : 'Vector formats have no resolution; this only applies to PNG and JPEG'}
                      options={DPIS.map((d) => ({ key: d.key, label: d.label, hint: scaleHint(rasterScale(width || 1, { scale: 1, widthMm, dpi: Number(d.key) })) }))} />
                    : <Combobox label="Resolution" value={s.scale} onChange={(v) => set({ scale: v })}
                      title={raster ? 'Pixel density of the image; 300 dpi is the usual print requirement' : 'Vector formats have no resolution; this only applies to PNG and JPEG'}
                      options={SCALES.map((x) => ({ key: x.key, label: x.label, hint: scaleHint(x.scale) }))} />}
                </div>
                <div className="export-row">
                  <Segmented label="Legend" value={legend} options={hasLegend ? legends : ['none'] as LegendPlacement[]} optionLabel={(o) => LEGEND_LABEL[o]}
                    onChange={(v) => set({ legend: v })} title={hasLegend ? 'Where the legend sits in the file; the chart on the page is not changed' : 'This chart has no legend'} />
                </div>
                {hasLegend && <div className="export-row">
                  <label className="export-range" title={legend === 'shown' ? 'Move the legend (Inside, Right, Bottom, Top) to size it; as on screen the chart is laid out around it' : pub ? "The legend's size; 100 % is the figure's own text size" : "The legend's size; 100 % is its size on the page"}>
                    <span className="combobox-label">Legend size</span>
                    <input type="range" min={60} max={160} step={5} value={s.legendSize} disabled={legend === 'shown'}
                      onChange={(e) => set({ legendSize: Number(e.target.value) })} />
                    <span className="export-range-value">{s.legendSize} %</span>
                    {s.legendSize !== 100 && legend !== 'shown' && <button type="button" className="btn quiet" onClick={() => set({ legendSize: 100 })}>Reset</button>}
                  </label>
                </div>}
                <div className="export-row">
                  <Segmented label="Background" value={s.format === 'jpeg' ? 'white' : s.background} options={['white', 'transparent'] as ExportBackground[]}
                    onChange={(v) => set({ background: v })} title={s.format === 'jpeg' ? 'JPEG has no transparency; it is always white' : 'Transparent suits a coloured slide or poster background'} />
                </div>
                <ExportColorPicker palette={fig.palette} value={s.colors} onChange={colors => set({ colors })} />
                <p className="export-facts">
                  {widthMm
                    ? <>{widthMm} × {Math.round(widthMm * height / width)} mm{raster ? ` · ${Math.round(width * scale).toLocaleString()} × ${Math.round(height * scale).toLocaleString()} px at ${dpi} dpi` : s.format === 'pdf' ? ` · vector, Inter font embedded${s.font !== 'screen' ? ' (the PDF can only embed Inter; SVG and PNG use the chosen font)' : ''}` : ' · vector'}</>
                    : raster
                    ? <>{Math.round(width * scale).toLocaleString()} × {Math.round(height * scale).toLocaleString()} px · prints at {inches(width)} × {inches(height)} in at 300 dpi</>
                    : <>{width} × {height} px canvas · vector, sharp at any size{s.format === 'pdf' ? ` · page ${(width / 96).toFixed(1)} × ${(height / 96).toFixed(1)} in, Inter font embedded` : ''}</>}
                  {pub && textPt && <><br />Text {textPt.median.toFixed(1)} pt, smallest {textPt.min.toFixed(1)} pt{widthMm ? '' : ' at 96 px to the inch'}
                    {textPt.min < MIN_PT - 0.05 && <span className="export-warn"> · under the usual {MIN_PT} pt minimum{widthMm && s.column !== '190' ? ': try a wider column' : ''}</span>}</>}
                </p>
                {current?.error && <p className="export-error">Could not make this file: {current.error}.</p>}
              </div>
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
