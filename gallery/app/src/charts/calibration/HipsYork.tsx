import { useRegressions } from '@/lib/regressionPrefs'
import { useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { AxisChoices } from '@/components/AxisChoices'
import { Legend, useLegend } from '@/components/Legend'
import { IntervalStrip } from '@/components/IntervalStrip'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { useMac } from '@/lib/mac'
import { fmt } from '@/lib/stats'
import { INK, MARGIN } from '@/lib/theme'
import type { CalibRunsFile, YorkFit } from '@/lib/types'

const VARIANTS = ['deployed', 'lot_lin', 'lot_quad'] as const
const VARIANT_LABEL: Record<string, string> = { deployed: 'deployed blank line', lot_lin: 'lot blank, linear', lot_quad: 'lot blank, quadratic' }
const VARIANT_COLOR: Record<string, string> = { get deployed() { return INK.text }, lot_lin: '#2b6cb0', lot_quad: '#c026d3' }

/**
 * HIPS instrument diagnostics: per-site weighted York (errors-in-variables)
 * fits of predicted EC against Fabs, under three blank-line conventions for
 * the HIPS transmittance. If a site's intercept moves with the blank line, the
 * offset is partly an instrument-processing artefact; if it does not, it is
 * in the aerosol. OFFSET_ADJUDICATION_2026-08-23 is the write-up to cite.
 *
 * The fits are against Fabs / MAC at the export's MAC (hips.york.mac). Rescaling x and its errors by
 * one factor rescales the York slope and its SE by the inverse and leaves the intercept, so the slope
 * follows the MAC set under Filters (lib/mac) without a refit.
 */
export function HipsYork({ runs }: { runs: CalibRunsFile }) {
  const { showCI } = useRegressions()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [what, setWhat] = useState<'intercept' | 'slope'>('intercept')
  const lg = useLegend()
  const shown = VARIANTS.filter((v) => lg.show(VARIANT_LABEL[v]))

  const gallery = useMac()
  const k = gallery.mac / (runs.hips.york?.mac ?? gallery.exportMac)
  // the slope (and its SE) at the reader's MAC; the intercept does not move
  const rows = (runs.hips.york?.rows ?? []).filter((r) => r.fits && !r.error).map((r) => k === 1 ? r : {
    ...r, fits: Object.fromEntries(Object.entries(r.fits!).map(([v, f]) => [v, !f || 'error' in f ? f : { ...f, slope: f.slope * k, slope_se: f.slope_se * k }])) as typeof r.fits })
  const LEFT = 104
  const innerW = Math.max(240, width - LEFT - MARGIN.right)
  const rowH = 44
  const vals = rows.flatMap((r) => shown.map((v) => r.fits![v]).filter((f): f is YorkFit => !!f && !('error' in f)).flatMap((f) => [f[what] - (showCI && Number.isFinite(f[`${what}_se`]) ? 1.96 * f[`${what}_se`] : 0), f[what] + (showCI && Number.isFinite(f[`${what}_se`]) ? 1.96 * f[`${what}_se`] : 0)]))
  const ext = d3.extent(vals) as [number, number]
  // the slope view always spans 0.8–1.2 so agreement stays in frame
  const x = d3.scaleLinear().domain([Math.min(what === 'slope' ? 0.8 : 0, ext[0] ?? 0), Math.max(what === 'slope' ? 1.2 : 0, ext[1] ?? 1)]).range([0, innerW]).nice()

  return (
    <ChartFrame
      id="hips-york"
      title="HIPS diagnostics — does the offset move with the blank line?"
      subtitle={`Weighted York fits of predicted EC against HIPS Fabs ÷ ${gallery.label} at every SPARTAN site under three blank-line conventions${showCI ? ", with approximate 95% normal intervals (±1.96 SE)" : ""}. Calibration: ${runs.hips.config ? runs.hips.config.replace('AIRSpec', 'Spline baseline').replace(/k=(\d+)/, '$1 PLS factors') : 'see explorer'}. A site whose intercept is the same under all three carries the offset in the aerosol, not in the transmittance processing.`}
      provenance="calibration_explorer /api/hips_york · HIPS tab · OFFSET_ADJUDICATION_2026-08-23.md"
    >
      <div ref={wrapRef} className="chart-wrap">
        {rows.length === 0 ? (
          <Empty>No HIPS York rows exported.</Empty>
        ) : (
          // the x readout (intercept or slope) is picked from the axis title, as on the other charts
          <AxisChoices noun="readouts" x={[what, (v) => setWhat(v as 'intercept' | 'slope')]}
            options={[{ key: 'intercept', label: 'York intercept (µg/m³)', group: 'York fit', roles: ['x'] }, { key: 'slope', label: 'York slope', group: 'York fit', roles: ['x'] }]}>
          <IntervalStrip labelWidth={LEFT} rowHeight={rowH} minPanelWidth={240}
            panels={[{ label: what === 'intercept' ? 'York intercept (µg/m³)' : 'York slope', reference: what === 'slope' ? 1 : 0, domain: x.domain() as [number, number], pick: true }]}
            blocks={[{ key: 'sites', rows: rows.map((r) => {
              const fits = shown.map((v) => [v, r.fits![v]] as const).filter((e): e is readonly [typeof e[0], YorkFit] => !!e[1] && !('error' in e[1]))
              return { key: r.site, label: runs.targets[r.site]?.site ?? r.site, labelColor: runs.targets[r.site]?.color,
                tip: [`${runs.targets[r.site]?.site ?? r.site}`,
                  ...fits.map(([v, f]) => `${VARIANT_LABEL[v]}: ${what} ${fmt(f[what], 3)}${showCI ? ` · approx. 95% CI [${fmt(f[what] - 1.96 * f[`${what}_se`], 3)}, ${fmt(f[what] + 1.96 * f[`${what}_se`], 3)}]` : ''} (slope ${fmt(f.slope, 3)} · intercept ${fmt(f.intercept, 3)} · κ ${fmt(f.kappa, 2)})`),
                  `${r.n_matched} of ${r.n} filters matched · ${((r.frac_below_blank_r1 ?? 0) * 100).toFixed(0)} % below the blank R1 range`],
                // one mark per blank-line convention, stacked within the site's row
                marks: [VARIANTS.map((v, k) => {
                  const f = r.fits![v]
                  if (!f || 'error' in f || !lg.show(VARIANT_LABEL[v])) return null
                  const se = f[`${what}_se`], ok = showCI && Number.isFinite(se) && se >= 0
                  return { mid: f[what], lo: ok ? f[what] - 1.96 * se : null, hi: ok ? f[what] + 1.96 * se : null, color: VARIANT_COLOR[v], r: 4, dy: (k - (VARIANTS.length - 1) / 2) * 10 }
                }).filter((mk): mk is NonNullable<typeof mk> => mk !== null)] }
            }) }]} />
          </AxisChoices>
        )}
        <Legend items={VARIANTS.map((v) => ({ label: VARIANT_LABEL[v], color: VARIANT_COLOR[v] }))} {...lg.props} note={showCI ? "bars = approximate 95% normal CI (±1.96 SE); missing SE has no bar" : "fit uncertainty hidden"} />
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
