import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Segmented, Select } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { byConstruction, DERIVED_FROM } from '@/lib/derived'
import { fmt, regression } from '@/lib/stats'
import { INK, RAMP_DIVERGING, FONT } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'

const ENCODINGS = ['Colour', 'Circle size'] as const

/**
 * Correlogram — the chemistry correlation matrix, 14 figures in the estate.
 * AGENTS.md asks for R² when ranking relationship strength and signed r only
 * where direction matters; here direction is the whole point, so cells carry
 * signed r and the tooltip adds R².
 *
 * Fields derived from another by arithmetic (HIPS BC = HIPS Fabs / MAC) are
 * dropped: their r = 1.000 is a unit conversion, not a finding.
 */
export function Correlogram({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [siteCode, setSiteCode] = useState('ETAD')
  const [encoding, setEncoding] = useState<(typeof ENCODINGS)[number]>('Colour')

  const siteCodes = useMemo(
    () => ['All', ...meta.sites.filter((s) => rows.some((r) => r.code === s.code)).map((s) => s.code)],
    [meta.sites, rows]
  )
  const effectiveSite = siteCodes.includes(siteCode) ? siteCode : siteCodes[0] ?? 'All'

  // Computed from the live subset rather than a precomputed file, so the
  // season/site/date filters above actually move these numbers.
  const block = useMemo(() => {
    const sub = effectiveSite === 'All' ? rows : rows.filter((r) => r.code === effectiveSite)
    const fields = meta.fields
      .filter((f) => !(f in DERIVED_FROM))
      .filter((f) => sub.filter((r) => typeof r[f] === 'number').length >= 20)
    const matrix = fields.map((fa) =>
      fields.map((fb) => {
        if (byConstruction(fa, fb)) return null
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
  }, [rows, effectiveSite, meta.fields, meta.sites])

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
      subtitle="Pearson r between every pair of measured species at one site. Blue is negative, red positive. The FTIR/TOR EC pair sits at r ≈ 1.000 — that is a calibration relationship, not independent method agreement (see AGENTS.md)."
      provenance="stands in for 14 correlogram + heatmap figures · react-graph-gallery.com/correlogram"
      controls={
        <>
          <Select label="site" value={block.code} options={siteCodes} onChange={setSiteCode} />
          <Segmented label="encoding" value={encoding} options={ENCODINGS} onChange={setEncoding} />
          <span className="control">{block.fields.length} species</span>
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap" style={{ overflowX: 'auto' }}>
        {block.fields.length < 2 ? (
          <p className="frame-empty">Fewer than two species with ≥20 filters in this subset.</p>
        ) : (
          <svg width={size + labelPad + 30} height={size + labelPad + 20}>
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
          <span>r = −1</span>
          <span style={{ display: 'inline-block', width: 180, height: 10, borderRadius: 2, background: `linear-gradient(to right, ${RAMP_DIVERGING.join(',')})` }} />
          <span>+1</span>
          <span className="legend-note">HIPS BC omitted: it is HIPS Fabs ÷ MAC, r = 1 by construction</span>
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
