import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Toggle } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import { ColorLegend } from '@/components/ColorLegend'
import type { Config } from './common'

/** spec_curve.json, written by gallery/data/export_spec_curve.py */
export interface SpecCurveFile {
  n: number
  n_pass: number
  guardrails: string
  choices: string[]
  /** optional headings: rows start..end-1 sit under title */
  groups?: { title: string; start: number; end: number }[]
  masks: string[]
  ok: string
  intercept: number[]
  slope: (number | null)[]
  heldout: (number | null)[]
  cut: (number | null)[]
  k: (number | null)[]
  categories: { co: string[]; sp: string[]; sel: string[]; mo: string[]; lot: string[] }
  co: number[]
  sp: number[]
  sel: number[]
  mo: number[]
  lot: number[]
}

const BASE = `${import.meta.env.BASE_URL}data`
const MODE_LABEL: Record<string, string> = { site_heldout: 'protocol A', app: 'protocol B', app_fmm: 'protocol B2' }
const TOP_H = 250
const ROW_H = 17
const GROUP_H = 18
const GAP = 16
const BAND = 0.5
const labelW = (s: string) => s.length * 6.3
/** strip heat map: share of specifications using a choice, white (none) to dark red (all) */
const HEAT = (t: number) => d3.interpolateReds(0.12 + 0.88 * t)

/**
 * The specification curve (Simonsohn et al. 2020) over the whole Addis
 * search: every scored configuration, every k, sorted by the Deming intercept
 * it produces, with the analytic choices that produced it underneath. The
 * interactive twin of research/ftir_ec_phase3/output/plots/pathways/
 * 01_specification_curve.png; row selection, guardrails and the choice list
 * come from the same spec_curve.py module, so the counts agree.
 *
 * 20k columns do not fit in 1,000 px, so each pixel of the choice strip is the
 * share of the specifications in it that use the choice (ink = all of them),
 * rather than whichever single column a raster happens to sample.
 */
export function SpecCurve({
  selected, onSelect, src = 'spec_curve.json', id = 'spec-curve', title, subtitle, provenance, selectHint = 'click to load into the k-sweep below',
}: {
  selected?: Config
  onSelect?: (c: Config) => void
  /** another export in the same schema (e.g. meeting/spec_curve_20260917.json) */
  src?: string
  id?: string
  title?: string
  subtitle?: (f: SpecCurveFile | null) => string
  provenance?: string
  selectHint?: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [data, setData] = useState<SpecCurveFile | null | 'missing'>(null)
  const [required, setRequired] = useState<Set<number>>(new Set())
  const [passOnly, setPassOnly] = useState(false)
  const [cursor, setCursor] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    fetch(`${BASE}/${src}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => !cancelled && setData(d))
      .catch(() => !cancelled && setData('missing'))
    return () => { cancelled = true }
  }, [src])

  const file = data && data !== 'missing' ? data : null

  // the specifications left after the choice and guardrail filters, still in intercept order
  const shown = useMemo(() => {
    if (!file) return [] as number[]
    const out: number[] = []
    for (let i = 0; i < file.n; i++) {
      if (passOnly && file.ok[i] !== '1') continue
      let keep = true
      for (const c of required) if (file.masks[c][i] !== '1') { keep = false; break }
      if (keep) out.push(i)
    }
    return out
  }, [file, required, passOnly])

  const labelPad = file ? Math.ceil(Math.max(...file.choices.map(labelW))) + 22 : 180
  const margin = { top: 14, right: 20, bottom: 44, left: labelPad }
  const innerW = Math.max(240, width - margin.left - margin.right)
  const nRows = file?.choices.length ?? 0
  const stripTop = TOP_H + GAP
  // rows grouped under headings (preprocessing / cohort / calibration settings) when the export says so
  const groups = file?.groups ?? []
  const groupOf = (c: number) => groups.findIndex((g) => c >= g.start && c < g.end)
  const rowY = (c: number) => c * ROW_H + (groups.length ? (groupOf(c) + 1) * GROUP_H : 0)
  const stripH = nRows * ROW_H + groups.length * GROUP_H
  const height = margin.top + stripTop + stripH + margin.bottom

  const m = shown.length
  const x = d3.scaleLinear().domain([0, Math.max(1, m)]).range([0, innerW])

  // y follows the figure: the bulk of the curve, with the far tail clamped to the bottom edge
  const y = useMemo(() => {
    const ys = shown.map((i) => file!.intercept[i])
    const lo = (d3.quantile(ys, 0.003) ?? -1) - 0.6
    const hi = Math.max(2.6, (d3.quantile(ys, 0.997) ?? 0) + 1.9)
    return d3.scaleLinear().domain([lo, hi]).range([TOP_H, 0])
  }, [shown, file])

  // the curve, sampled at ~2 points per pixel; it is monotone, so nothing is lost
  const curve = useMemo(() => {
    if (!file || !m) return { line: '', area: '' }
    const step = Math.max(1, Math.floor(m / (innerW * 2)))
    const pts: [number, number][] = []
    for (let j = 0; j < m; j += step) pts.push([j + 0.5, file.intercept[shown[j]]])
    pts.push([m - 0.5, file.intercept[shown[m - 1]]])
    const [yLo] = y.domain()
    const clampY = (v: number) => y(Math.max(yLo, v))
    const line = d3.line<[number, number]>().x((p) => x(p[0])).y((p) => clampY(p[1]))(pts) ?? ''
    const area = d3.area<[number, number]>().x((p) => x(p[0])).y0(y(0)).y1((p) => clampY(p[1]))(pts) ?? ''
    return { line, area }
  }, [file, shown, m, innerW, x, y])

  // the choice strips as a heat map: one pixel column per screen pixel, coloured by the share
  // of the specifications there that use the choice (white = none, dark red = all)
  const strip = useMemo(() => {
    if (!file || !m || typeof document === 'undefined') return null
    const cols = Math.max(1, Math.min(m, Math.round(innerW)))
    const out: string[] = []
    for (let c = 0; c < nRows; c++) {
      const canvas = document.createElement('canvas')
      canvas.width = cols
      canvas.height = 1
      const ctx = canvas.getContext('2d')
      if (!ctx) return null
      const img = ctx.createImageData(cols, 1)
      const mask = file.masks[c]
      for (let b = 0; b < cols; b++) {
        const j0 = Math.floor((b * m) / cols)
        const j1 = Math.max(j0 + 1, Math.floor(((b + 1) * m) / cols))
        let used = 0
        for (let j = j0; j < j1; j++) if (mask[shown[j]] === '1') used++
        const t = used / (j1 - j0)
        const col = t === 0 ? d3.rgb(255, 255, 255) : d3.rgb(HEAT(t))
        const o = b * 4
        img.data[o] = col.r
        img.data[o + 1] = col.g
        img.data[o + 2] = col.b
        img.data[o + 3] = 255
      }
      ctx.putImageData(img, 0, 0)
      out.push(canvas.toDataURL())
    }
    return out
  }, [file, shown, m, innerW, nRows])

  // derived once per filter change, not on every cursor move
  const marks = useMemo(() => {
    if (!file) return { passIdx: [] as [number, number][], selIdx: [] as [number, number][], nearZero: 0 }
    const passIdx: [number, number][] = []
    const selIdx: [number, number][] = []
    let nearZero = 0
    shown.forEach((i, j) => {
      if (file.ok[i] === '1') passIdx.push([i, j])
      if (
        selected && file.categories.co[file.co[i]] === selected.co && file.cut[i] === selected.cut &&
        file.categories.sel[file.sel[i]] === (selected.sel ?? 'raw') && file.categories.sp[file.sp[i]] === selected.sp
      ) selIdx.push([i, j])
      if (Math.abs(file.intercept[i]) <= BAND) nearZero++
    })
    return { passIdx, selIdx, nearZero }
  }, [file, shown, selected])

  // helpers below run only once the file has loaded (the body is empty until then)
  const f = file as SpecCurveFile
  const cfgOf = (i: number): Config => ({
    co: f.categories.co[f.co[i]],
    cut: f.cut[i],
    sel: f.categories.sel[f.sel[i]],
    sp: f.categories.sp[f.sp[i]],
  })
  const describe = (i: number) => {
    const c = cfgOf(i)
    const mode = f.categories.mo[f.mo[i]]
    const lot = f.categories.lot[f.lot[i]]
    return `${c.co}${c.cut !== null ? '-' + c.cut : ''}${c.sel && c.sel !== 'raw' ? ' (sel ' + c.sel + ')' : ''} × ${c.sp} · ${MODE_LABEL[mode] ?? mode} · k ${f.k[i] ?? '?'}${lot !== 'all' ? ' · training lot ' + lot : ''}`
  }

  const { passIdx, selIdx, nearZero } = marks
  const [yLo] = y.domain()

  const locate = (e: React.MouseEvent) => {
    const box = (e.currentTarget as SVGRectElement).getBoundingClientRect()
    const j = Math.floor(x.invert(e.clientX - box.left))
    return j >= 0 && j < m ? j : null
  }
  const onMove = (e: React.MouseEvent) => {
    const j = locate(e)
    setCursor(j)
    if (j === null) return tip.hide()
    const i = shown[j]
    tip.show(e, [
      `#${(j + 1).toLocaleString()} of ${m.toLocaleString()}`,
      describe(i),
      `Deming intercept ${fmt(f.intercept[i], 2)} µg/m³ · slope ${fmt(f.slope[i], 2)}`,
      `held-out TOR R² ${fmt(f.heldout[i], 2)} · ${f.ok[i] === '1' ? 'passes every guardrail' : 'fails a guardrail'}`,
      ...(onSelect ? [selectHint] : []),
    ])
  }
  const toggleChoice = (c: number) => setRequired((s) => { const n = new Set(s); if (n.has(c)) n.delete(c); else n.add(c); return n })

  return (
    <ChartFrame
      id={id}
      title={title ?? 'Specification curve — every Addis configuration, sorted by the intercept it produces'}
      subtitle={subtitle ? subtitle(file) : `All ${file ? f.n.toLocaleString() + ' ' : ''}scored Addis specifications (every cohort, cutoff, baseline, protocol and k), sorted by Deming intercept at MAC 10, with the analytic choices underneath: a darker strip means more of the specifications at that position used the choice. Click a choice to keep only specifications that use it; click several to intersect. Red points pass every guardrail${file ? ': ' + f.guardrails : ''}.`}
      provenance={provenance ?? 'calibration_explorer batch results via research/ftir_ec_phase3/scripts/spec_curve.py · figure twin: pathways/01_specification_curve.png'}
      controls={
        <>
          <Toggle label="guardrail passers only" checked={passOnly} onChange={setPassOnly} />
          {required.size > 0 && <button type="button" className="btn quiet" onClick={() => setRequired(new Set())}>clear choices</button>}
          {file && <span className="control">{m.toLocaleString()} of {f.n.toLocaleString()} shown · {f.n_pass} pass</span>}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {data === null ? (
          <Empty>Loading the specification curve…</Empty>
        ) : !file ? (
          <Empty>No spec_curve.json. Run <code>python gallery/data/export_spec_curve.py</code>.</Empty>
        ) : m === 0 ? (
          <Empty>No specification uses every selected choice{passOnly ? ' and passes the guardrails' : ''}.</Empty>
        ) : (
          <svg width={width} height={height} fontFamily={FONT.family}>
            <defs><clipPath id={`${id}-clip`}><rect width={innerW} height={TOP_H} /></clipPath></defs>
            <g transform={`translate(${margin.left},${margin.top})`}>
              {/* ---- the curve */}
              <YAxis scale={y} x={0} label="Deming intercept, MAC 10 (µg/m³)" gridWidth={innerW} tickCount={6} />
              <rect x={0} y={y(BAND)} width={innerW} height={y(-BAND) - y(BAND)} fill={INK.good} fillOpacity={0.1} />
              <g clipPath={`url(#${id}-clip)`}>
                <path d={curve.area} fill={INK.accent} fillOpacity={0.14} />
                <path d={curve.line} fill="none" stroke={INK.accent} strokeWidth={1.4} />
              </g>
              <line x1={0} x2={innerW} y1={y(0)} y2={y(0)} stroke={INK.text} strokeWidth={1} />
              {selIdx.map(([i, j]) => (
                <circle key={`s${i}`} cx={x(j + 0.5)} cy={y(Math.max(yLo, f.intercept[i]))} r={3.4} fill="none" stroke={INK.text} strokeOpacity={0.7} strokeWidth={1.1} pointerEvents="none" />
              ))}
              {passIdx.map(([i, j]) => (
                <circle key={`p${i}`} cx={x(j + 0.5)} cy={y(f.intercept[i])} r={2.3} fill={INK.negative} pointerEvents="none" />
              ))}
              <text x={6} y={14} fontSize={11.5} fill={INK.text}>
                {m.toLocaleString()} {m === f.n ? 'Addis configurations' : `of ${f.n.toLocaleString()} configurations`}, sorted by intercept
              </text>
              <text x={innerW - 4} y={y(BAND) - 6} textAnchor="end" fontSize={10.5} fill={INK.good}>
                within {BAND} of zero: {nearZero.toLocaleString()}
              </text>
              <g transform={`translate(${innerW - 4},${TOP_H - 10})`} fontSize={10.5} textAnchor="end">
                <text fill={INK.text}>
                  <tspan fill={INK.negative}>●</tspan> passes every guardrail ({passIdx.length} of {m.toLocaleString()})
                  {selIdx.length > 0 && <><tspan dx={10}>○</tspan> selected configuration ({selIdx.length})</>}
                </text>
              </g>

              {/* ---- the choice strip */}
              <g transform={`translate(0,${stripTop})`}>
                {strip && strip.map((href, c) => (
                  <image key={c} href={href} x={0} y={rowY(c)} width={innerW} height={ROW_H - 1.5} preserveAspectRatio="none" style={{ imageRendering: 'pixelated' }} />
                ))}
                {groups.map((g) => (
                  <text key={g.title} x={-labelPad + 4} y={rowY(g.start) - 5} fontSize={11.5} fontWeight={700} fill={INK.text}>{g.title}</text>
                ))}
                {f.choices.map((label, c) => {
                  const on = required.has(c)
                  return (
                    <g key={label} style={{ cursor: 'pointer' }} onClick={() => toggleChoice(c)}>
                      {on && <rect x={-labelPad + 4} y={rowY(c) + 1} width={labelPad - 8} height={ROW_H - 2} rx={3} fill={INK.accent} fillOpacity={0.12} />}
                      <text x={-8} y={rowY(c) + ROW_H / 2} dy="0.32em" textAnchor="end" fontSize={11} fontWeight={on ? 600 : 400} fill={on ? INK.accent : INK.text}>
                        {label}
                      </text>
                      <title>{on ? 'click to drop this filter' : 'click to keep only specifications that use this choice'}</title>
                    </g>
                  )
                })}
                <XAxis scale={x} y={stripH} label="specification, ordered by the intercept it produces" tickCount={8} format={(v: number) => d3.format(',')(v)} />
              </g>

              {/* ---- cursor across both panels, and the hit area */}
              {cursor !== null && <line x1={x(cursor + 0.5)} x2={x(cursor + 0.5)} y1={0} y2={stripTop + stripH} stroke={INK.text} strokeOpacity={0.5} pointerEvents="none" />}
              <rect
                x={0} y={0} width={innerW} height={TOP_H} fill="transparent" style={{ cursor: onSelect ? 'pointer' : 'default' }}
                onMouseMove={onMove}
                onMouseLeave={() => { setCursor(null); tip.hide() }}
                onClick={(e) => { const j = locate(e); if (j !== null && onSelect) onSelect(cfgOf(shown[j])) }}
              />
            </g>
          </svg>
        )}
        {file && m > 0 && (
          <div className="legend">
            <ColorLegend scale={d3.scaleSequential((t: number) => HEAT(t)).domain([0, 100])} width={180}
              label="strip colour: % of the specifications at that position that use the choice" format={(v) => `${v.toFixed(0)}%`} />
          </div>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
