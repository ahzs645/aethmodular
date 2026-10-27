import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { SSA_BY_WAVELENGTH } from '@/lib/aeronet'
import { meanSe } from '@/lib/aeronetMath'
import type { FilterRow, MetaFile } from '@/lib/types'
import { colouring, num } from './shared'

const M = { top: 12, right: 18, bottom: 44, left: 56 }
const WAVELENGTHS = SSA_BY_WAVELENGTH.map(([w]) => w)

/**
 * The single-scattering albedo's spectral shape: mean SSA (± standard error) at 440, 675, 870 and 1020 nm
 * over the filter windows with an inversion, one line per season (one site) or per site. Falling with
 * wavelength is black carbon's signature; rising, dust's (its absorption is in the blue).
 */
export function AeronetSsaSpectrum({ rows, meta }: { rows: FilterRow[]; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const col = useMemo(() => colouring(rows, meta), [rows, meta])
  const lines = useMemo(() => {
    const withSsa = rows.filter((r) => num(r[SSA_BY_WAVELENGTH[0][1]]))
    return col.order(withSsa).map((k) => {
      const mine = withSsa.filter((r) => col.key(r) === k)
      return { k, n: mine.length, pts: SSA_BY_WAVELENGTH.map(([w, f]) => ({ w, ...meanSe(mine.map((r) => r[f] as number | null)) })) }
    }).filter((l) => l.n >= 3)
  }, [rows, col])

  const iw = Math.max(240, Math.min(width - M.left - M.right, 560)), ih = 280
  const x = d3.scaleLinear().domain([400, 1060]).range([0, iw])
  const lo = d3.min(lines, (l) => d3.min(l.pts, (p) => (p.mean ?? 1) - (p.se ?? 0))) ?? 0.8
  const hi = d3.max(lines, (l) => d3.max(l.pts, (p) => (p.mean ?? 0) + (p.se ?? 0))) ?? 1
  const y = d3.scaleLinear().domain([Math.max(0.5, lo - 0.01), Math.min(1, hi + 0.01)]).range([ih, 0]).nice()
  const path = d3.line<{ w: number; mean: number | null }>().defined((p) => p.mean !== null).x((p) => x(p.w)).y((p) => y(p.mean!))

  return (
    <ChartFrame id="aeronet-ssa-spectrum" title={`SSA spectral shape — by ${col.by}`}
      subtitle={`Mean single-scattering albedo at 440, 675, 870 and 1020 nm (almucantar inversions, each filter window's mean), with ± 1 standard error, one line per ${col.by} with at least 3 windows. SSA falling with wavelength marks black carbon (absorption flat in λ while scattering drops); rising marks dust (absorbing in the blue).`}
      provenance="notebooks/analysis/surface_columnar_linkage.ipynb section 8 (c) and surface_columnar_figures: SSA spectral shape by season">
      <div ref={wrapRef} className="chart-wrap">
        {lines.length === 0 ? <Empty>No {col.by} in view has 3 filter windows with an inversion.</Empty> : (
          <svg width={iw + M.left + M.right} height={ih + M.top + M.bottom} fontFamily={FONT.family}>
            <g transform={`translate(${M.left},${M.top})`}>
              <YAxis scale={y} x={0} gridWidth={iw} tickCount={6} label="Single-scattering albedo" labelX={-42} />
              <XAxis scale={x} y={ih} tickCount={6} label="Wavelength (nm)" labelDy={36} />
              {WAVELENGTHS.map((w) => <line key={w} x1={x(w)} x2={x(w)} y1={0} y2={ih} stroke={INK.grid} />)}
              {lines.map((l) => {
                const c = col.color(l.k)
                return (
                  <g key={l.k}>
                    <path d={path(l.pts) ?? ''} fill="none" stroke={c} strokeWidth={2} />
                    {l.pts.map((p) => p.mean === null ? null : (
                      <g key={p.w}>
                        {p.se !== null && <line x1={x(p.w)} x2={x(p.w)} y1={y(p.mean - p.se)} y2={y(p.mean + p.se)} stroke={c} strokeWidth={1.4} />}
                        <circle cx={x(p.w)} cy={y(p.mean)} r={4} fill={c} stroke={INK.panel} strokeWidth={1}
                          onMouseEnter={(e) => tip.show(e, [col.label(l.k), `SSA ${p.w} nm: ${fmt(p.mean, 3)} ± ${fmt(p.se, 3)} (SE)`, `${p.n} filter windows`])} onMouseLeave={tip.hide} />
                      </g>
                    ))}
                  </g>
                )
              })}
            </g>
          </svg>
        )}
        <div className="legend">
          {lines.map((l) => {
            const a = l.pts[0].mean, b = l.pts[l.pts.length - 1].mean
            return <span key={l.k} className="legend-item"><span className="swatch" style={{ background: col.color(l.k) }} />
              {col.label(l.k)} <span className="u">n={l.n}{a !== null && b !== null ? ` · 1020 − 440: ${b - a >= 0 ? '+' : ''}${fmt(b - a, 3)}` : ''}</span></span>
          })}
        </div>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
