import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { XAxis, YAxis } from '@/components/Axes'
import { ColorLegend } from '@/components/ColorLegend'
import { HeatCell } from '@/components/HeatCell'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/format'
import { FONT, INK, RAMP_SEQUENTIAL } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { CHANNELS, SCREEN_TEXT, channelColor, instColor, logTick, type Screen, type Series } from './data'
import { SIGMA, channelCorrelation, powerLawExponent, relativeSpectra, summarize, type Day, type Summary } from './stats'

type Kind = 'absorption' | 'bcc'
const KIND_LABEL: Record<Kind, string> = { absorption: 'Absorption', bcc: 'BCc (notebooks)' }
interface Curve { key: string; label: string; color: string; spec: Summary[]; aae: number | null; n: number }
const M = { top: 12, right: 64, bottom: 46, left: 62 }

/** each channel's median and IQR of the day-by-day ratio to 880 nm, and the AAE of the median absorption spectrum */
function curve(label: string, color: string, days: readonly Day[], kind: Kind, lambda: number[]): Curve | null {
  const rel = relativeSpectra(days, kind)
  const spec = CHANNELS.map((c) => summarize(rel[c]))
  if (spec.some((s) => !s)) return null
  const abs = kind === 'absorption' ? spec : CHANNELS.map((c) => summarize(relativeSpectra(days, 'absorption')[c]))
  return { key: label, label, color, spec: spec as Summary[], aae: powerLawExponent(lambda, abs.map((s) => s!.median)), n: rel.IR.length }
}

/**
 * The spectral shape: each wavelength's absorption (or BCc) relative to 880 nm, by season at one site
 * or by site, with the power laws AAE = 1 and 2 for reference. Multi_Site_Analysis_Fixed §13 and
 * _FollowUp §7 compared UV and IR BCc (the UV/IR ratio, the UV-vs-IR crossplot); BCc divides the
 * absorption by each channel's cross-section, so a BCc ratio near one already means AAE near one
 * (optics.aae_from_columns). The absorption view is the corrected one; the BCc view is the notebooks'.
 */
export function SpectralShape({ series, meta, screen, by }: { series: Series[]; meta: MetaFile; screen: Screen; by: 'season' | 'site' }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [kind, setKind] = useSearchParam<Kind>('aeosp', 'absorption', ['absorption', 'bcc'])
  const leg = useLegend()
  const lambda = CHANNELS.map((c) => meta.wavelengths_nm[c])
  const curves = useMemo(() => {
    if (by === 'site') return series.map((s) => curve(s.inst.code === s.inst.siteCode ? s.inst.site : s.inst.label, instColor(meta, s.inst), s.days, kind, lambda)).filter((c): c is Curve => !!c)
    const s = series[0]
    if (!s) return []
    return [...seasonsForSite(meta, s.inst.site).map((se) => curve(seasonOnly(se.name), se.color, s.days.filter((d) => d.season?.name === se.name), kind, lambda)),
      curve('All days', INK.text, s.days, kind, lambda)].filter((c): c is Curve => !!c)
  }, [series, meta, kind, by]) // eslint-disable-line react-hooks/exhaustive-deps
  // absorption falls with wavelength for any aerosol: a channel below the next one is the file, not the air
  const odd = kind === 'absorption' ? curves.flatMap((c) => CHANNELS.slice(0, -1).filter((_, i) => c.spec[i].median < c.spec[i + 1].median).map((ch) => `${c.label} (${ch})`)) : []
  const corr = useMemo(() => (by === 'season' && series[0] ? channelCorrelation(series[0].days) : null), [series, by])

  const matrixW = corr ? 230 : 0
  const pw = Math.max(260, Math.min(640, width - (corr && width > 600 ? matrixW + 24 : 0)))
  const iw = pw - M.left - M.right, ih = 260
  const x = d3.scaleLog().domain([350, 950]).range([0, iw])
  // the reference power laws through 1 at 880 nm, in the view's units
  const ref = (aae: number, i: number) => (lambda[i] / 880) ** -aae * (kind === 'bcc' ? SIGMA.IR / SIGMA[CHANNELS[i]] : 1)
  const shown = curves.filter((c) => leg.show(c.key))
  const vals = [...shown.flatMap((c) => c.spec.flatMap((s) => [s.q25, s.q75])), ...lambda.flatMap((_, i) => [ref(1, i), ref(2, i)]), 1]
  const y = d3.scaleLog().domain([(d3.min(vals) ?? 0.5) * 0.92, (d3.max(vals) ?? 3) * 1.08]).range([ih, 0])
  const pts = (get: (i: number) => number) => CHANNELS.map((_, i) => [x(lambda[i]), y(get(i))] as [number, number])
  const line = d3.line()
  const band = (c: Curve) => d3.area<number>().x((i) => x(lambda[i])).y0((i) => y(c.spec[i].q25)).y1((i) => y(c.spec[i].q75))(d3.range(CHANNELS.length)) ?? ''
  const what = kind === 'absorption' ? 'b_abs(λ) / b_abs(880 nm)' : 'BCc(λ) / BCc(IR)'
  const color = d3.scaleSequential(d3.interpolateRgbBasis(RAMP_SEQUENTIAL)).domain([0.5, 1]).clamp(true)

  const site = series[0]?.inst.site
  return (
    <ChartFrame id={`aeth-overview-spectrum-${by}`} title={by === 'season' ? `${site} · spectral shape by season — absorption at five wavelengths relative to 880 nm` : 'Spectral shape by site — absorption at five wavelengths relative to 880 nm'}
      subtitle={`For each day with all five channels positive, ${kind === 'absorption' ? 'the absorption at each wavelength (BCc × the MA350 cross-section) divided by the 880 nm absorption' : "each channel's BCc divided by IR BCc, the notebooks' UV/IR ratio extended to every channel"}; the line is the median over days, the band the interquartile range, on log–log axes. Dashed: the power laws AAE = 1 (black carbon) and 2 through 880 nm${kind === 'bcc' ? ', converted to BCc ratios through the cross-sections: BCc divides out most of black carbon\'s λ⁻¹, which is why the notebooks\' UV/IR ratio sits near one' : ''}. The legend's AAE is a power-law fit to the median absorption spectrum (all five wavelengths), not the median of daily AAEs (the Sources page).${by === 'season' ? " Right: Pearson r between the channels' daily BCc (Multi_Site_Analysis_FollowUp 7a)." : ''} Screen: ${SCREEN_TEXT[screen]}.`}
      provenance="Multi_Site_Analysis_Fixed.ipynb §13 and Multi_Site_Analysis_FollowUp.ipynb §7, 7a (UV/IR ratio, correlation matrix); absorption form as research/ftir_hips_chem/scripts/optics.py · gallery/data/export_aethalometer.py"
      controls={<Segmented label="Show" value={kind} options={['absorption', 'bcc'] as Kind[]} onChange={setKind} optionLabel={(o) => KIND_LABEL[o]} />}>
      <div ref={wrapRef} className="chart-wrap">
        {!curves.length ? <Empty>No days with all five channels in this span.</Empty> : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'flex-start' }}>
            <svg width={pw} height={ih + M.top + M.bottom} fontFamily={FONT.family}>
              <g transform={`translate(${M.left},${M.top})`}>
                <YAxis scale={y as unknown as d3.ScaleLinear<number, number>} x={0} gridWidth={iw} tickCount={5} label={what} labelX={-46} format={(v) => logTick(v)} />
                {[1, 2].map((a) => <g key={a} pointerEvents="none">
                  <path d={line(pts((i) => ref(a, i))) ?? ''} fill="none" stroke={INK.identity} strokeWidth={1.2} strokeDasharray="5 4" />
                  <text x={x(lambda[0]) - 4} y={y(ref(a, 0))} dy="0.32em" textAnchor="end" fontSize={10} fontFamily={FONT.mono} fill={INK.muted}>AAE {a}</text>
                </g>)}
                {shown.map((c) => <g key={c.key} opacity={leg.dim(c.key)}>
                  <path d={band(c)} fill={c.color} fillOpacity={0.12} />
                  <path d={line(pts((i) => c.spec[i].median)) ?? ''} fill="none" stroke={c.color} strokeWidth={2} />
                  {CHANNELS.map((ch, i) => <circle key={ch} cx={x(lambda[i])} cy={y(c.spec[i].median)} r={3.5} fill={channelColor(ch)} stroke={c.color} strokeWidth={1.2}
                    onMouseEnter={(e) => { leg.setHover(c.key); tip.show(e, [`${c.label} · ${ch} (${lambda[i]} nm)`, `${what}: median ${fmt(c.spec[i].median, 3)} [IQR ${fmt(c.spec[i].q25, 3)}–${fmt(c.spec[i].q75, 3)}]`, `${c.n} days`, `AAE of the median spectrum ${fmt(c.aae, 2)}`]) }}
                    onMouseLeave={() => { leg.setHover(null); tip.hide() }} />)}
                </g>)}
                <XAxis scale={x as unknown as d3.ScaleLinear<number, number>} y={ih} tickCount={6} label="Wavelength (nm)" format={(v) => d3.format('~g')(v)} />
                {CHANNELS.map((ch, i) => <text key={ch} x={x(lambda[i])} y={-2} textAnchor="middle" fontSize={9.5} fill={channelColor(ch)}>{ch}</text>)}
              </g>
            </svg>
            {corr && corr.n >= 3 && (
              <div>
                <svg width={matrixW} height={matrixW + 10} fontFamily={FONT.family}>
                  <text x={4} y={12} fontSize={12} fontWeight={600} fill={INK.text}>Channels' daily BCc, r (n = {corr.n})</text>
                  <g transform="translate(46,22)">
                    {CHANNELS.map((a, i) => <g key={a}>
                      <text x={-6} y={i * 34 + 17} dy="0.32em" textAnchor="end" fontSize={11} fill={channelColor(a)}>{a}</text>
                      <text x={i * 34 + 17} y={5 * 34 + 14} textAnchor="middle" fontSize={11} fill={channelColor(a)}>{a}</text>
                      {CHANNELS.map((b, j) => {
                        const r = corr.r[i][j]
                        return <HeatCell key={b} x={j * 34} y={i * 34} width={33} height={33} fill={r === null ? null : color(r)} label={r === null ? '—' : fmt(r, 2)}
                          onEnter={(e) => tip.show(e, [`${a} against ${b}`, `r = ${fmt(r, 3)} over ${corr.n} days with all five channels`])} onLeave={tip.hide} />
                      })}
                    </g>)}
                  </g>
                </svg>
                <div className="legend"><ColorLegend scale={color} width={150} label="r" format={(v) => fmt(v, 2)} note="below 0.5 drawn as 0.5" /></div>
              </div>
            )}
          </div>
        )}
        {odd.length > 0 && <Note>Absorption should fall with wavelength, but the median reads below the next longer wavelength at {odd.join(', ')}: a problem with that channel in the daily file, not an aerosol spectrum, and it pulls the five-wavelength AAE fit.</Note>}
        <Legend items={curves.map((c) => ({ label: c.key, color: c.color, shape: 'line' as const, detail: `AAE ${fmt(c.aae, 2)} · ${c.n} d` }))} {...leg.props} />
        {tip.node}
      </div>
    </ChartFrame>
  )
}
