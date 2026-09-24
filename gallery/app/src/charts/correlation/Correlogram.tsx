import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { RESTATES, isBookkeeping, sameProduct } from '@/fieldLineage'
import { fmt, regression } from '@/lib/stats'
import { INK, RAMP_DIVERGING, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const ENCODINGS = ['Colour', 'Circle size', 'Scatter matrix'] as const
const ALL_FAMILIES = 'Top 6 by coverage'

/**
 * Correlogram — the chemistry correlation matrix, 14 figures in the estate.
 * AGENTS.md asks for R² when ranking relationship strength and signed r only
 * where direction matters; here direction is the whole point, so cells carry
 * signed r and the tooltip adds R².
 *
 * Restatements (the HIPS unit conversion and public FTIR carbon duplicates)
 * are dropped because their near-perfect correlations are not new findings.
 *
 * `Scatter matrix` is react-graph-gallery's own correlogram: the pairwise
 * scatterplots of a handful of species with their histograms on the diagonal.
 * The r-matrix says *how much* two species co-vary; the scatter matrix shows
 * *how* — a single Beijing dust day can carry a correlation on its own, and
 * only the cloud shows it.
 */
export function Correlogram({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const [siteCode, setSiteCode] = useState('ETAD')
  const [encoding, setEncoding] = useState<(typeof ENCODINGS)[number]>('Colour')
  const [family, setFamily] = useState(ALL_FAMILIES)
  const lg = useLegend()

  const siteCodes = useMemo(
    () => ['All', ...meta.sites.filter((s) => rows.some((r) => r.code === s.code)).map((s) => s.code)],
    [meta.sites, rows]
  )
  const effectiveSite = siteCodes.includes(siteCode) ? siteCode : siteCodes[0] ?? 'All'
  const sub = useMemo(() => (effectiveSite === 'All' ? rows : rows.filter((r) => r.code === effectiveSite)), [rows, effectiveSite])

  // Computed from the live subset rather than a precomputed file, so the
  // season/site/date filters above actually move these numbers.
  const block = useMemo(() => {
    const fields = meta.fields
      .filter((f) => !(f in RESTATES) && !isBookkeeping(f))
      .filter((f) => sub.filter((r) => typeof r[f] === 'number').length >= 20)
    const matrix = fields.map((fa) =>
      fields.map((fb) => {
        if (sameProduct(fa, fb)) return null
        const xs: number[] = []
        const ys: number[] = []
        for (const r of sub) {
          const a = r[fa]
          const b = r[fb]
          if (typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b)) {
            xs.push(a)
            ys.push(b)
          }
        }
        const st = regression(xs, ys)
        return st ? st.r : null
      })
    )
    const n: Record<string, number> = {}
    for (const f of fields) n[f] = sub.filter((r) => typeof r[f] === 'number').length
    const name = effectiveSite === 'All' ? 'All sites' : meta.sites.find((s) => s.code === effectiveSite)?.name ?? effectiveSite
    return { code: effectiveSite, name, fields, matrix, n }
  }, [sub, effectiveSite, meta.fields, meta.sites])

  // the scatter matrix cannot show 28 species: pick a family, or the six best-covered
  const families = [ALL_FAMILIES, ...meta.field_groups.map((g) => g.label)]
  const splomFields = useMemo(() => {
    // uncertainties and detection limits are bookkeeping, not species
    const measured = (f: string) => !f.includes('uncertainty') && !f.includes('MDL')
    const pool = family === ALL_FAMILIES
      ? block.fields.filter(measured).sort((a, b) => (block.n[b] ?? 0) - (block.n[a] ?? 0))
      : (meta.field_groups.find((g) => g.label === family)?.fields ?? []).filter((f) => block.fields.includes(f) && measured(f))
    return pool.slice(0, 6)
  }, [family, block, meta.field_groups])

  const labelPad = 128
  const size = Math.min(Math.max(320, width - labelPad - 40), 660)
  const cell = size / Math.max(1, block.fields.length)

  const color = useMemo(
    () => d3.scaleLinear<string>().domain([-1, -0.66, -0.33, 0, 0.33, 0.66, 1]).range(RAMP_DIVERGING).clamp(true),
    []
  )

  return (
    <ChartFrame
      id="correlogram"
      title="Correlogram — how the chemistry co-varies"
      subtitle="Pearson r between species at one site. Blue is negative, red positive. The in-house and ChemSpec EC feeds both report FTIR EC, so the duplicate ChemSpec carbon fields are omitted. 'Scatter matrix' shows the clouds behind a few cells, with each species' histogram on the diagonal."
      provenance="stands in for 14 correlogram + heatmap figures · react-graph-gallery.com/correlogram"
      controls={
        <>
          <Select label="site" value={block.code} options={siteCodes} onChange={setSiteCode} optionLabel={(c) => (c === 'All' ? 'All sites' : meta.sites.find((s) => s.code === c)?.name ?? c)} />
          <Segmented label="encoding" value={encoding} options={ENCODINGS} onChange={setEncoding} />
          {encoding === 'Scatter matrix' ? (
            <Select label="species" value={family} options={families} onChange={setFamily} title="At most six species per matrix" />
          ) : (
            <span className="control">{block.fields.length} species</span>
          )}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap centered" style={{ overflowX: 'auto' }}>
        {block.fields.length < 2 ? (
          <p className="frame-empty">Fewer than two species with ≥20 filters in this subset.</p>
        ) : encoding === 'Scatter matrix' ? (
          splomFields.length < 2 ? (
            <Empty>Fewer than two species of that family have ≥20 filters here.</Empty>
          ) : (
            <ScatterMatrix rows={sub.filter((r) => lg.show(r.site))} fields={splomFields} width={Math.min(width, 760)} tip={tip} hl={hl} units={meta.field_units} dim={lg.dim} />
          )
        ) : (
          <svg width={size + labelPad + 30} height={size + labelPad + 20} className="animated">
            <g transform={`translate(${labelPad},${labelPad})`}>
              {block.fields.map((fa, i) =>
                block.fields.map((fb, j) => {
                  const r = block.matrix[i][j]
                  if (r === null) return null
                  const cx = j * cell
                  const cy = i * cell
                  const rad = (cell / 2 - 1.5) * Math.sqrt(Math.abs(r))
                  return (
                    <g
                      key={`${i}-${j}`}
                      onMouseEnter={(e) =>
                        tip.show(e, [fa, `vs ${fb}`, `r = ${fmt(r, 3)} · R² = ${fmt(r * r, 3)}`, `n = ${Math.min(block.n[fa] ?? 0, block.n[fb] ?? 0)}`])
                      }
                      onMouseLeave={tip.hide}
                    >
                      <rect x={cx} y={cy} width={cell} height={cell} fill={encoding === 'Colour' ? color(r) : '#fff'} stroke="#fff" strokeWidth={1} />
                      {encoding === 'Circle size' && <circle cx={cx + cell / 2} cy={cy + cell / 2} r={Math.max(1, rad)} fill={color(r)} />}
                      {cell > 26 && encoding === 'Colour' && (
                        <text
                          x={cx + cell / 2} y={cy + cell / 2} dy="0.34em" textAnchor="middle"
                          fontSize={Math.min(10, cell / 3.4)} fill={Math.abs(r) > 0.55 ? '#fff' : INK.text}
                          fontFamily={FONT.mono} pointerEvents="none"
                        >
                          {r.toFixed(2)}
                        </text>
                      )}
                    </g>
                  )
                })
              )}
              {block.fields.map((f, i) => (
                <text key={`r${i}`} x={-8} y={i * cell + cell / 2} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>
                  {f}
                </text>
              ))}
              {block.fields.map((f, j) => (
                <text key={`c${j}`} transform={`translate(${j * cell + cell / 2},-8) rotate(-52)`} fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>
                  {f}
                </text>
              ))}
            </g>
          </svg>
        )}
        <div className="legend">
          {encoding === 'Scatter matrix' ? (
            <Legend items={meta.sites.filter((s) => sub.some((r) => r.site === s.name)).map((s) => ({ label: s.name, color: s.color }))} {...lg.props} note="diagonal: histogram of the species · off-diagonal: the pair, Deming line (λ=1), r" />
          ) : (
            <ColorLegend scale={color} label="Pearson r" width={180} ticks={5} format={(v) => (v > 0 ? `+${v}` : String(v))} note="Repeated fields omitted: HIPS BC is Fabs ÷ MAC; ChemSpec EC/OC repeat the FTIR products" />
          )}
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** Pairwise scatter matrix for a few species, histograms on the diagonal. */
function ScatterMatrix({ rows, fields, width, tip, hl, units, dim }: {
  rows: FilterRow[]
  fields: string[]
  width: number
  tip: ReturnType<typeof useTooltip>
  hl: ReturnType<typeof useHighlight>
  units: Record<string, string>
  dim: (site: string) => number
}) {
  const n = fields.length
  const pad = 6
  // the row labels sit in the left gutter: size it from the longest name
  const labelW = Math.max(60, Math.ceil(Math.max(...fields.map((f) => f.length)) * 6.4) + 12)
  const cellS = Math.max(70, Math.floor((width - labelW - 20 - pad * (n - 1)) / n))
  const scales = useMemo(
    () =>
      fields.map((f) => {
        const vals = rows.map((r) => r[f]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)).sort(d3.ascending)
        // 1–99 % so a single spike does not push the cloud into a corner
        const lo = Math.min(0, d3.quantile(vals, 0.01) ?? 0)
        const hi = d3.quantile(vals, 0.99) ?? 1
        return d3.scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([0, cellS]).nice()
      }),
    [rows, fields, cellS]
  )
  const W = labelW + n * cellS + (n - 1) * pad + 10
  const H = n * cellS + (n - 1) * pad + labelW
  const unitOf = (f: string) => (units[f] ? units[f].replace('ug/m3', 'µg/m³').replace('Mm-1', 'Mm⁻¹') : '')

  return (
    <svg width={W} height={H} className="animated">
      <g transform={`translate(${labelW},0)`}>
        {fields.map((fy, i) =>
          fields.map((fx, j) => {
            const ox = j * (cellS + pad)
            const oy = i * (cellS + pad)
            const sx = scales[j]
            const sy = scales[i]
            if (i === j) {
              const vals = rows.map((r) => r[fx]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
              const bins = d3.bin().domain(sx.domain() as [number, number]).thresholds(18)(vals)
              const ymax = d3.max(bins, (b) => b.length) ?? 1
              return (
                <g key={`${i}-${j}`} transform={`translate(${ox},${oy})`}>
                  <rect width={cellS} height={cellS} fill="#fafbfc" stroke={INK.grid} />
                  {bins.map((b, k) => (
                    <rect key={k} x={sx(b.x0 ?? 0)} y={cellS - (b.length / ymax) * (cellS - 16)} width={Math.max(0, sx(b.x1 ?? 0) - sx(b.x0 ?? 0) - 0.5)} height={(b.length / ymax) * (cellS - 16)} fill={INK.muted} fillOpacity={0.5} />
                  ))}
                  <text x={cellS / 2} y={11} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>{fx}</text>
                  <text x={cellS / 2} y={22} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.family}>{unitOf(fx)} · n={vals.length}</text>
                </g>
              )
            }
            const pts = rows
              .map((r) => ({ row: r, x: r[fx], y: r[fy] }))
              .filter((p): p is { row: FilterRow; x: number; y: number } => typeof p.x === 'number' && typeof p.y === 'number' && Number.isFinite(p.x) && Number.isFinite(p.y))
            // Deming (λ=1) rather than y-on-x least squares: both species are measured with error
            const st = regression(pts.map((p) => p.x), pts.map((p) => p.y), { errorsInVariables: true })
            const [d0, d1] = sx.domain() as [number, number]
            const yv = (v: number) => Math.max(sy.domain()[0] as number, Math.min(sy.domain()[1] as number, v))
            return (
              <g key={`${i}-${j}`} transform={`translate(${ox},${oy})`}>
                <rect width={cellS} height={cellS} fill="#fff" stroke={INK.grid} />
                {pts.map((p) => {
                  const f = focusStyle(p.row.id, hl.focusId, { r: 2, opacity: 0.55 }, hl.selected)
                  const cx = Math.max(0, Math.min(cellS, sx(p.x)))
                  const cy = Math.max(0, Math.min(cellS, cellS - sy(p.y)))
                  return (
                    <circle
                      key={p.row.id} cx={cx} cy={cy} r={f.r} fill={p.row.color as string} fillOpacity={f.opacity * dim(p.row.site)} stroke={f.stroke} strokeWidth={f.strokeWidth}
                      style={{ cursor: 'pointer' }}
                      onMouseEnter={(e) => { hl.setHover(p.row.id); tip.show(e, [p.row.id, `${p.row.site} · ${p.row.date}`, `${fx}: ${fmt(p.x)}`, `${fy}: ${fmt(p.y)}`]) }}
                      onMouseLeave={() => { hl.setHover(null); tip.hide() }}
                      onClick={() => hl.openSample(p.row.id)}
                    />
                  )
                })}
                {st && st.demingSlope !== null && st.demingIntercept !== null && (
                  <>
                    <line x1={sx(d0)} y1={cellS - sy(yv(st.demingSlope * d0 + st.demingIntercept))} x2={sx(d1)} y2={cellS - sy(yv(st.demingSlope * d1 + st.demingIntercept))} stroke={INK.deming} strokeWidth={1.2} strokeDasharray="5 2" pointerEvents="none" />
                    <text x={cellS - 4} y={11} textAnchor="end" fontSize={9.5} fill={Math.abs(st.r) > 0.6 ? INK.text : INK.muted} fontFamily={FONT.mono} pointerEvents="none">r={st.r.toFixed(2)}</text>
                  </>
                )}
              </g>
            )
          })
        )}
        {fields.map((f, i) => (
          <text key={`l${i}`} x={-8} y={i * (cellS + pad) + cellS / 2} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>{f}</text>
        ))}
        {fields.map((f, j) => (
          <text key={`b${j}`} x={j * (cellS + pad) + cellS / 2} y={n * (cellS + pad) - pad + 16} textAnchor="middle" fontSize={10.5} fill={INK.text} fontFamily={FONT.family}>{f}</text>
        ))}
      </g>
    </svg>
  )
}
