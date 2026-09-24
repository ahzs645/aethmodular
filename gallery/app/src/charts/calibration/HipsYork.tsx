import { useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { XAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, MARGIN, FONT } from '@/lib/theme'
import type { CalibRunsFile, YorkFit } from '@/lib/types'

const VARIANTS = ['deployed', 'lot_lin', 'lot_quad'] as const
const VARIANT_LABEL: Record<string, string> = { deployed: 'deployed blank line', lot_lin: 'lot blank, linear', lot_quad: 'lot blank, quadratic' }
const VARIANT_COLOR: Record<string, string> = { deployed: '#1f2933', lot_lin: '#2b6cb0', lot_quad: '#c026d3' }

/**
 * HIPS instrument diagnostics: per-site weighted York (errors-in-variables)
 * fits of predicted EC against Fabs, under three blank-line conventions for
 * the HIPS transmittance. If a site's intercept moves with the blank line, the
 * offset is partly an instrument-processing artefact; if it does not, it is
 * in the aerosol. OFFSET_ADJUDICATION_2026-08-23 is the write-up to cite.
 */
export function HipsYork({ runs }: { runs: CalibRunsFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [what, setWhat] = useState<'intercept' | 'slope'>('intercept')
  const lg = useLegend()
  const shown = VARIANTS.filter((v) => lg.show(VARIANT_LABEL[v]))

  const rows = (runs.hips.york?.rows ?? []).filter((r) => r.fits && !r.error)
  const LEFT = 104
  const innerW = Math.max(240, width - LEFT - MARGIN.right)
  const rowH = 44
  const innerH = rows.length * rowH
  const vals = rows.flatMap((r) => shown.map((v) => r.fits![v]).filter((f): f is YorkFit => !!f && !('error' in f)).flatMap((f) => [f[what] - f[`${what}_se`], f[what] + f[`${what}_se`]]))
  const ext = d3.extent(vals) as [number, number]
  const x = d3.scaleLinear().domain([Math.min(what === 'slope' ? 0.8 : 0, ext[0] ?? 0), Math.max(what === 'slope' ? 1.2 : 0, ext[1] ?? 1)]).range([0, innerW]).nice()
  const y = d3.scaleBand<string>().domain(rows.map((r) => r.site)).range([0, innerH]).padding(0.3)

  return (
    <ChartFrame
      id="hips-york"
      title="HIPS diagnostics — does the offset move with the blank line?"
      subtitle={`Weighted York fits of predicted EC against HIPS Fabs at every SPARTAN site under three blank-line conventions, ± 1 SE. Calibration: ${runs.hips.config ? runs.hips.config.replace('AIRSpec', 'Spline baseline').replace(/k=(\d+)/, '$1 PLS factors') : 'see explorer'}. A site whose intercept is the same under all three carries the offset in the aerosol, not in the transmittance processing.`}
      provenance="calibration_explorer /api/hips_york · HIPS tab · OFFSET_ADJUDICATION_2026-08-23.md"
      controls={<Segmented value={what} options={['intercept', 'slope'] as const} onChange={setWhat} />}
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length === 0 ? (
          <Empty>No HIPS York rows exported.</Empty>
        ) : (
          <svg width={width} height={innerH + MARGIN.top + MARGIN.bottom}>
            <g transform={`translate(${LEFT},${MARGIN.top})`}>
              {rows.map((r) => (
                <text key={r.site} x={-10} y={(y(r.site) ?? 0) + y.bandwidth() / 2} dy="0.32em" textAnchor="end" fontSize={11.5} fill={runs.targets[r.site]?.color ?? INK.text} fontWeight={600} fontFamily={FONT.family}>
                  {runs.targets[r.site]?.site ?? r.site}
                </text>
              ))}
              <XAxis scale={x} y={innerH} label={what === 'intercept' ? 'York intercept (µg/m³)' : 'York slope'} />
              <line x1={x(what === 'slope' ? 1 : 0)} x2={x(what === 'slope' ? 1 : 0)} y1={0} y2={innerH} stroke={INK.axis} strokeDasharray="4 3" />
              {rows.map((r) =>
                VARIANTS.map((v, k) => {
                  const f = r.fits![v]
                  if (!f || 'error' in f || !lg.show(VARIANT_LABEL[v])) return null
                  const cy = (y(r.site) ?? 0) + (y.bandwidth() * (k + 0.5)) / VARIANTS.length
                  const val = f[what]
                  const se = f[`${what}_se`]
                  return (
                    <g key={v} onMouseEnter={(e) => tip.show(e, [`${runs.targets[r.site]?.site ?? r.site} · ${VARIANT_LABEL[v]}`, `${what} ${fmt(val, 3)} ± ${fmt(se, 3)}`, `slope ${fmt(f.slope, 3)} · intercept ${fmt(f.intercept, 3)} · κ ${fmt(f.kappa, 2)}`, `${r.n_matched} of ${r.n} filters matched · ${((r.frac_below_blank_r1 ?? 0) * 100).toFixed(0)} % below the blank R1 range`])} onMouseLeave={tip.hide} opacity={lg.dim(VARIANT_LABEL[v])}>
                      <line x1={x(val - se)} x2={x(val + se)} y1={cy} y2={cy} stroke={VARIANT_COLOR[v]} strokeWidth={1.5} />
                      <circle cx={x(val)} cy={cy} r={4} fill={VARIANT_COLOR[v]} />
                    </g>
                  )
                })
              )}
            </g>
          </svg>
        )}
        <Legend items={VARIANTS.map((v) => ({ label: VARIANT_LABEL[v], color: VARIANT_COLOR[v] }))} {...lg.props} note="bars = ± 1 SE" />
        {runs.hips.blanks && (
          <table className="placement" style={{ maxWidth: 720 }}>
            <thead><tr><th>lot</th><th>deployed blank line (intercept, slope)</th><th>n blanks</th><th>rms lin / quad</th><th>R1 range</th><th>τ₀ mean ± sd</th></tr></thead>
            <tbody>
              {Object.entries(runs.hips.blanks).map(([id, L]) => (
                <tr key={id}>
                  <td className="f">{String(L.lot ?? id.split('|')[0])}</td>
                  <td className="v">{typeof L.deployed_intercept === 'number' ? `${fmt(L.deployed_intercept, 1)}, ${fmt(L.deployed_slope as number, 3)}` : id}</td>
                  <td className="v">{String(L.n ?? '')}</td>
                  <td className="v">{fmt(L.rms_lin as number, 1)} / {fmt(L.rms_quad as number, 1)}</td>
                  <td className="v">{fmt(L.r1_min as number, 0)} – {fmt(L.r1_max as number, 0)}</td>
                  <td className="v">{fmt(L.tau0_mean as number, 4)} ± {fmt(L.tau0_sd as number, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
