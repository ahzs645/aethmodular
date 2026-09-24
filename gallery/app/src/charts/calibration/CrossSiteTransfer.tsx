import { useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT } from '@/lib/theme'
import type { CalibRun, CalibRunsFile } from '@/lib/types'
import { cohortColor, fmtFit, metricRow, presetName } from './common'
import { PREPROCESSING, label } from '@/lib/labels'

/** Ethiopian sites first (the calibration's own target, then its neighbour), then the other SPARTAN sites. */
const SITE_ORDER = ['addis', 'etbi', 'indh', 'chts', 'uspa']
const AXES = ['per site', 'shared'] as const
const DOT_METRICS = ['intercept', 'slope', 'R²'] as const
/** The explorer's own threshold for "partially out of domain" (CROSS_SITE_EVALUATION_2026-08-22.md). */
const EXTRAP_FLAG = 20
const SPECTRA_SHAPE: Record<string, d3.SymbolType> = { raw: d3.symbolCircle, airspec: d3.symbolSquare, deriv2: d3.symbolDiamond }

const TRANSFER_TIP = (
  <>
    <p>Every calibration here is trained on IMPROVE filters with thermal (TOR) EC, then applied unchanged to each SPARTAN site. No SPARTAN site has thermal EC, so the site readouts compare the prediction with HIPS Fabs ÷ MAC, an optical equivalent. They show how the calibration transfers, not EC accuracy at the site.</p>
    <p>A slope mixes the site's true MAC with calibration error, so read the intercept ordering before the slopes. Score OOD is the share of site filters beyond the training 95th percentile in the model's score space (Reggente 2016). Q OOD is the share with unmodelled spectral variation beyond that percentile. Above {EXTRAP_FLAG}% the site is partly outside the training domain.</p>
    <p>For citation, the per-filter weighted York fits in research/ftir_ec_phase3/OFFSET_ADJUDICATION_2026-08-23.md supersede these Deming numbers. The HIPS diagnostics chart on this tab shows them for ocec-450 × Spline baseline.</p>
    <p>Fits are Deming (errors in both variables) throughout; the IMPROVE cross-validation row reports R² and RMSE only, because the explorer exports its slope and intercept from a y-on-x fit.</p>
  </>
)

/**
 * One IMPROVE-trained calibration read out at every SPARTAN site with FTIR
 * spectra and HIPS Fabs: the explorer's Sites tab. Three views share the
 * picker: the per-site crossplots, a summary table against the IMPROVE
 * cross-validation score (calibration set), and whether the site ordering holds across all the
 * exported calibrations.
 */
export function CrossSiteTransfer({ runs }: { runs: CalibRunsFile }) {
  const presets = runs.presets.filter((p) => runs.runs.some((r) => r.preset === p.key))
  const [presetKey, setPresetKey] = useState(presets.some((p) => p.key === 'ocec800_airspec') ? 'ocec800_airspec' : presets[0]?.key ?? '')
  const [mac, setMac] = useState(runs.mac_value)

  const sites = useMemo(() => {
    const present = [...new Set(runs.runs.map((r) => r.target))]
    return [...SITE_ORDER.filter((s) => present.includes(s)), ...present.filter((s) => !SITE_ORDER.includes(s)).sort()]
  }, [runs.runs])
  const siteRuns = sites.map((s) => runs.runs.find((r) => r.preset === presetKey && r.target === s)).filter((r): r is CalibRun => !!r)
  const macs = useMemo(() => [...new Set(runs.runs.flatMap((r) => r.metrics.map((m) => m.MAC)).filter((v): v is number => v !== null))].sort((a, b) => a - b), [runs.runs])
  const preset = presets.find((p) => p.key === presetKey)

  if (!presets.length) return null
  const controls = (
    <>
      <Select label="calibration" value={presetKey} options={presets.map((p) => p.key)} optionLabel={(k) => { const p = presets.find((q) => q.key === k); return p ? presetName(p) : k }} onChange={setPresetKey} />
      <Segmented label="MAC" value={String(mac)} options={macs.map(String)} onChange={(v) => setMac(Number(v))} />
    </>
  )
  return (
    <>
      <SiteMultiples runs={runs} siteRuns={siteRuns} mac={mac} controls={controls} presetLabel={preset ? presetName(preset) : presetKey} />
      <TransferTable runs={runs} siteRuns={siteRuns} mac={mac} presetLabel={preset ? presetName(preset) : presetKey} />
      <OrderingAcrossCalibrations runs={runs} sites={sites} presetKey={presetKey} mac={mac} onPick={setPresetKey} />
    </>
  )
}

/** The Deming fit (agreed with Ann, 23 Sep 2026: no y-on-x least squares in reader-facing views). */
function lineOf(run: CalibRun, mac: number) {
  const m = metricRow(run.metrics, 'all', run.ref_kind === 'fabs' ? mac : null)
  if (!m) return null
  return { m, slope: m.deming_slope, intercept: m.deming_intercept }
}

function SiteMultiples({ runs, siteRuns, mac, controls, presetLabel }: {
  runs: CalibRunsFile; siteRuns: CalibRun[]; mac: number; controls: React.ReactNode; presetLabel: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [axes, setAxes] = useState<(typeof AXES)[number]>('per site')

  const pts = siteRuns.map((run) => {
    const e = run.eval
    const out: { x: number; y: number; i: number }[] = []
    for (let i = 0; i < e.pred.length; i++) {
      const x = run.ref_kind === 'fabs' ? e.ref[i] / mac : e.ref[i]
      if (Number.isFinite(x) && Number.isFinite(e.pred[i])) out.push({ x, y: e.pred[i], i })
    }
    return out
  })
  const domainOf = (p: { x: number; y: number }[]): [number, number] => [
    Math.min(0, d3.min(p, (d) => Math.min(d.x, d.y)) ?? 0),
    Math.max(d3.max(p, (d) => d.x) ?? 1, d3.max(p, (d) => d.y) ?? 1) * 1.06,
  ]
  const shared = domainOf(pts.flat())

  const gap = 18
  // as many columns as fit, then balanced so five sites wrap 3 + 2 rather than 4 + 1
  const maxCols = Math.max(1, Math.min(siteRuns.length, Math.floor((width + gap) / 230)))
  const cols = Math.ceil(siteRuns.length / Math.ceil(siteRuns.length / maxCols))
  const cellW = (width - (cols - 1) * gap) / cols
  const m = { l: 42, r: 8, t: 40, b: 58 }
  const S = Math.max(150, Math.min(300, cellW - m.l - m.r))

  return (
    <ChartFrame
      id="cross-site-multiples"
      title={`Every site under one calibration — ${presetLabel}`}
      subtitle="One panel per SPARTAN site: the calibration's predicted FTIR EC against HIPS Fabs ÷ MAC, with the 1:1 line (dashed) and the Deming fit from the explorer's metrics for every filter at that site (each site is a test set the model never saw)."
      provenance="calibration_explorer /api/run for each target · test set 'all'"
      tip={TRANSFER_TIP}
      controls={<>{controls}<Segmented label="axes" value={axes} options={AXES} onChange={setAxes} /></>}
    >
      <div ref={wrapRef} className="chart-wrap">
        {!siteRuns.length ? (
          <Empty>No site readouts exported for this calibration.</Empty>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap, justifyItems: 'center' }}>
            {siteRuns.map((run, k) => {
              const t = runs.targets[run.target]
              const dom = axes === 'shared' ? shared : domainOf(pts[k])
              const x = d3.scaleLinear().domain(dom).range([0, S]).nice()
              const y = d3.scaleLinear().domain(x.domain()).range([S, 0])
              const [d0, d1] = x.domain()
              const fit = lineOf(run, mac)
              // clip the fitted line to the square
              const seg = fit && fit.slope !== null && fit.intercept !== null ? (() => {
                const f = (v: number) => fit.slope! * v + fit.intercept!
                const xs = [d0, d1, (d0 - fit.intercept!) / fit.slope!, (d1 - fit.intercept!) / fit.slope!]
                  .filter((v) => Number.isFinite(v) && v >= d0 && v <= d1 && f(v) >= d0 - 1e-9 && f(v) <= d1 + 1e-9)
                  .sort((a, b) => a - b)
                return xs.length >= 2 ? { x1: x(xs[0]), y1: y(f(xs[0])), x2: x(xs[xs.length - 1]), y2: y(f(xs[xs.length - 1])) } : null
              })() : null
              const flagged = (run.extrap_pct ?? 0) > EXTRAP_FLAG
              const color = t?.color ?? INK.neutral
              return (
                <svg key={run.target} width={S + m.l + m.r} height={S + m.t + m.b} fontFamily={FONT.family}>
                  <text x={m.l} y={14} fontSize={12.5} fontWeight={600} fill={color}>{t?.site ?? run.target}<tspan fill={INK.muted} fontWeight={400}>{`  ${t?.code ?? ''} · n = ${pts[k].length}`}</tspan></text>
                  <text x={m.l} y={29} fontSize={10.5} fill={flagged ? '#b45309' : INK.muted}>
                    {`score OOD ${fmt(run.extrap_pct, 1)} %${flagged ? ' · partly out of domain' : ''}`}
                  </text>
                  <g transform={`translate(${m.l},${m.t})`}>
                    <rect width={S} height={S} fill="none" stroke={flagged ? '#f59e0b' : 'none'} strokeDasharray="4 3" />
                    <YAxis scale={y} x={0} gridWidth={S} tickCount={4} label={k % cols === 0 ? 'predicted EC (µg/m³)' : undefined} />
                    <XAxis scale={x} y={S} tickCount={4} />
                    <line x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} stroke={INK.identity} strokeDasharray="5 4" />
                    {pts[k].map((p) => (
                      <circle
                        key={p.i} cx={x(p.x)} cy={y(p.y)} r={2.6} fill={color} fillOpacity={0.45}
                        onMouseEnter={(e) => tip.show(e, [
                          run.eval.id[p.i] ?? `filter #${p.i + 1}`,
                          `${t?.site ?? run.target} · ${run.eval.date?.[p.i] ?? 'no date'} · ${run.eval.group[p.i]}`,
                          `HIPS Fabs ÷ MAC ${mac}: ${fmt(p.x, 3)} µg/m³`,
                          `predicted EC: ${fmt(p.y, 3)} µg/m³ · residual ${fmt(p.y - p.x, 3)}`,
                        ])}
                        onMouseLeave={tip.hide}
                      />
                    ))}
                    {seg && <line {...seg} stroke={INK.deming} strokeWidth={1.8} strokeDasharray="7 3" />}
                    <text x={S / 2} y={S + 34} textAnchor="middle" fontSize={10.5} fill={INK.muted}>{`HIPS Fabs ÷ MAC ${mac} (µg/m³)`}</text>
                    <text x={S / 2} y={S + 51} textAnchor="middle" fontSize={11} fontFamily={FONT.mono} fill={INK.deming}>
                      {`Deming ${fmtFit(fit?.slope ?? null, fit?.intercept ?? null)} · R² ${fmt(fit?.m.R2 ?? null, 2)}`}
                    </text>
                  </g>
                </svg>
              )
            })}
          </div>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

function TransferTable({ runs, siteRuns, mac, presetLabel }: { runs: CalibRunsFile; siteRuns: CalibRun[]; mac: number; presetLabel: string }) {
  const held = siteRuns.find((r) => r.heldout)?.heldout ?? null
  const train = siteRuns[0]
  return (
    <ChartFrame
      id="cross-site-table"
      title={`Transfer summary — ${presetLabel}`}
      subtitle="The calibration-set check first (IMPROVE cross-validation: whole IMPROVE sites held out of training, scored against thermal EC), then the same calibration on each SPARTAN test set against HIPS Fabs ÷ MAC."
      tip={TRANSFER_TIP}
      exportable={false}
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="census-table">
          <thead>
            <tr><th>Applied to</th><th>Reference</th><th>n</th><th>Deming slope</th><th>Deming intercept</th><th>R²</th><th>RMSE</th><th>Score OOD</th><th>Q OOD</th><th>Median prediction</th><th>Negative</th></tr>
          </thead>
          <tbody>
            {held && train && (
              <tr style={{ background: 'var(--accent-soft)' }}>
                <td><strong>IMPROVE cross-validation (calibration set)</strong><div style={{ fontSize: 11, color: 'var(--ink-muted)' }}>{`trained on ${train.n_train.toLocaleString()} filters, ${train.n_train_sites} sites · ${train.k} PLS factors`}</div></td>
                <td>TOR EC (µg/filter)</td>
                <td>—</td>
                {/* the explorer exports this row's slope/intercept from a y-on-x fit only; no Deming values to show */}
                <td title="not exported as a Deming fit">—</td>
                <td title="not exported as a Deming fit">—</td>
                <td>{fmt(held.R2, 3)}</td>
                <td>{fmt(held.RMSE, 3)}</td>
                <td>—</td><td>—</td><td>—</td><td>—</td>
              </tr>
            )}
            {siteRuns.map((run) => {
              const t = runs.targets[run.target]
              const fit = lineOf(run, mac)
              const flagged = (run.extrap_pct ?? 0) > EXTRAP_FLAG
              return (
                <tr key={run.target}>
                  <td><span style={{ color: t?.color, fontWeight: 600 }}>{t?.site ?? run.target}</span> <span style={{ color: 'var(--ink-muted)' }}>{t?.code}</span></td>
                  <td>{run.ref_kind === 'fabs' ? `HIPS Fabs ÷ ${mac}` : 'EC'}</td>
                  <td>{fit?.m.n ?? run.eval.id.length}</td>
                  <td>{fmt(fit?.slope ?? null, 3)}</td>
                  <td style={{ fontWeight: (fit?.intercept ?? 0) < -1 ? 600 : 400 }}>{fmt(fit?.intercept ?? null, 3)}</td>
                  <td>{fmt(fit?.m.R2 ?? null, 3)}</td>
                  <td>{fmt(fit?.m.RMSE ?? null, 3)}</td>
                  <td style={{ color: flagged ? '#b45309' : undefined, fontWeight: flagged ? 600 : 400 }}>{`${fmt(run.extrap_pct, 1)} %`}</td>
                  <td>{`${fmt(run.q_residual_pct, 1)} %`}</td>
                  <td>{fmt(run.plausibility?.median ?? null, 2)}</td>
                  <td>{run.plausibility?.negative_pct != null ? `${fmt(run.plausibility.negative_pct, 1)} %` : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="chart-note">Intercept and RMSE are in the reference's units: µg/filter for the IMPROVE row, µg/m³ at the SPARTAN sites. Bold intercepts are below −1 µg/m³; an amber score OOD is above {EXTRAP_FLAG} %.</p>
    </ChartFrame>
  )
}

function OrderingAcrossCalibrations({ runs, sites, presetKey, mac, onPick }: {
  runs: CalibRunsFile; sites: string[]; presetKey: string; mac: number; onPick: (k: string) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [metric, setMetric] = useState<(typeof DOT_METRICS)[number]>('intercept')

  const rows = runs.presets.flatMap((p) => sites.map((s) => {
    const run = runs.runs.find((r) => r.preset === p.key && r.target === s)
    const fit = run ? lineOf(run, mac) : null
    const v = !fit ? null : metric === 'intercept' ? fit.intercept : metric === 'slope' ? fit.slope : fit.m.R2
    return { preset: p, site: s, v }
  })).filter((r): r is typeof r & { v: number } => r.v !== null && Number.isFinite(r.v))

  const left = 110, right = 24, rowH = 44
  const iw = Math.max(200, width - left - right)
  const ih = sites.length * rowH
  const ext = d3.extent(rows, (r) => r.v) as [number, number]
  const ref = metric === 'slope' ? 1 : metric === 'intercept' ? 0 : null
  const x = d3.scaleLinear().domain([Math.min(ext[0] ?? 0, ref ?? Infinity), Math.max(ext[1] ?? 1, ref ?? -Infinity)]).range([0, iw]).nice()
  const y = d3.scaleBand().domain(sites).range([0, ih]).padding(0.3)
  const spectra = [...new Set(runs.presets.map((p) => p.spectra))]
  const cohorts = [...new Set(runs.presets.map((p) => p.cohort))]

  return (
    <ChartFrame
      id="cross-site-ordering"
      title="Does the site ordering hold across calibrations?"
      subtitle="Each dot is one exported calibration read out at one site, coloured by cohort, shaped by preprocessing; the ringed dot is the calibration picked above. Click a dot to pick that calibration. A site whose dots stay together is robust to the calibration choice."
      tip={TRANSFER_TIP}
      controls={<Segmented label="readout" value={metric} options={DOT_METRICS} onChange={setMetric} />}
    >
      <div ref={wrapRef} className="chart-wrap">
        <svg width={width} height={ih + 96} fontFamily={FONT.family}>
          <g transform={`translate(${left},12)`}>
            <XAxis scale={x} y={ih} label={`Test set ${metric === 'R²' ? 'R²' : `Deming ${metric}`}${metric === 'intercept' ? ' (µg/m³)' : ''} · MAC ${mac}`} />
            {x.ticks(6).map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={ih} stroke={INK.grid} />)}
            {ref !== null && <line x1={x(ref)} x2={x(ref)} y1={0} y2={ih} stroke={INK.axis} strokeDasharray="4 3" />}
            {sites.map((s) => {
              const t = runs.targets[s]
              return (
                <g key={s}>
                  <text x={-12} y={(y(s) ?? 0) + y.bandwidth() / 2} dy="0.35em" textAnchor="end" fontSize={12} fontWeight={600} fill={t?.color ?? INK.text}>{t?.site ?? s}</text>
                  <line x1={0} x2={iw} y1={(y(s) ?? 0) + y.bandwidth() / 2} y2={(y(s) ?? 0) + y.bandwidth() / 2} stroke={INK.grid} />
                </g>
              )
            })}
            {rows.map((r, i) => {
              const picked = r.preset.key === presetKey
              // spread the nine calibrations across the row so overlapping dots stay visible
              const jitter = ((runs.presets.indexOf(r.preset) / Math.max(1, runs.presets.length - 1)) - 0.5) * y.bandwidth() * 0.8
              const cy = (y(r.site) ?? 0) + y.bandwidth() / 2 + jitter
              const path = d3.symbol().type(SPECTRA_SHAPE[r.preset.spectra] ?? d3.symbolTriangle).size(picked ? 110 : 55)() ?? ''
              return (
                <path
                  key={i} d={path} transform={`translate(${x(r.v)},${cy})`}
                  fill={cohortColor(r.preset.cohort)} fillOpacity={picked ? 1 : 0.7}
                  stroke={picked ? INK.text : '#fff'} strokeWidth={picked ? 2 : 0.8}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onPick(r.preset.key)}
                  onMouseEnter={(e) => tip.show(e, [presetName(r.preset), `${runs.targets[r.site]?.site ?? r.site}: ${metric} ${fmt(r.v, 3)}`, picked ? 'picked above' : 'click to pick this calibration'])}
                  onMouseLeave={tip.hide}
                />
              )
            })}
          </g>
          <g transform={`translate(${left},${ih + 66})`} fontSize={11} fill={INK.text}>
            {cohorts.map((c, i) => (
              <g key={c} transform={`translate(${i * 105},0)`}><circle cx={5} cy={-4} r={5} fill={cohortColor(c)} /><text x={14}>{c.replace('_', ' ')}</text></g>
            ))}
            {spectra.map((sp, i) => (
              <g key={sp} transform={`translate(${cohorts.length * 105 + 20 + i * 125},0)`}>
                <path d={d3.symbol().type(SPECTRA_SHAPE[sp] ?? d3.symbolTriangle).size(55)() ?? ''} transform="translate(5,-4)" fill={INK.muted} />
                <text x={14}>{label(PREPROCESSING, sp)}</text>
              </g>
            ))}
          </g>
        </svg>
        {tip.node}
      </div>
    </ChartFrame>
  )
}
