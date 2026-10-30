import { useMemo, type ReactNode } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Segmented } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { SpectrumPlot } from '@/components/SpectrumPlot'
import { f2, f3 } from '@/lib/format'
import { num } from '@/lib/table'
import { themed, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SpectralExport } from '../../../research/pages/spectralCommon'
import { rows } from '../../common'
import { sharedRuns, vipBandShare, vipOverlap } from './hips'

/** one VIP comparison the export can carry: its table, the two models (columns of the wide table) and their k */
export interface VipSet { key: string; label: string; table: string; models: [string, string]; k: [number | null, number | null]; cite: string }

const SHADES = ['bands', 'VIP ≥ 1 in both'] as const
/** the notebook bands, short enough to sit over a 150 cm⁻¹ band */
const SHORT: Record<string, string> = {
  'O-H / N-H region': 'O–H / N–H', 'aliphatic C-H stretch': 'C–H', 'carbonyl region': 'C=O',
  'amide / aromatic / N-H region': 'amide', 'fingerprint region': 'fingerprint',
}
/** alternate fills, so the three touching bands below 1850 cm⁻¹ read as three */
const FILLS = [() => themed('#e8eef6', '#1f2b3a'), () => themed('#f1ece0', '#33302a')]
const SHARED = () => themed('#8e44ad', '#c39bd3')

/**
 * ftir_07 [5] and ftir_08 [12]: two models' VIP profiles on the 2,722-point wavenumber axis (high to low, as the
 * notebooks drew it), VIP = 1 dashed, with either the five broad bands pls_transfer.summarize_vip_bands sums VIP²
 * over (ftir_07 [7] / ftir_08 [12] tables) or ftir_07's lower panel, the first model's VIP filled where both models
 * reach VIP ≥ 1. The overlap and band shares are recomputed here from the drawn profiles, against the notebook's.
 */
export function VipSpectra({ data, sets, colors, provenance, missing }: {
  data: SpectralExport; sets: VipSet[]; colors: Record<string, string>; provenance: ReactNode; missing: ReactNode
}) {
  const present = sets.filter((s) => data.tables[s.table])
  const [setKey, setSet] = useSearchParam('ml-model-form-vip-profiles.set', present[0]?.key ?? '', present.map((s) => s.key))
  const [shade, setShade] = useSearchParam<(typeof SHADES)[number]>('ml-model-form-vip-profiles.shade', 'bands', SHADES)
  const set = present.find((s) => s.key === setKey) ?? present[0]
  const edges = rows(data, 'vip_band_edges')

  const profile = useMemo(() => {
    const t = set ? data.tables[set.table] as Record<string, number[]> : null
    if (!t || !set) return null
    const [a, b] = set.models.map((m) => t[m])
    return { wn: t.wn, a, b, overlap: vipOverlap(a, b) }
  }, [data, set])

  if (!set || !profile) return <ChartFrame id="ml-model-form-vip-profiles" title="VIP profiles — written only on the machine with the Drive" exportable={false}>{missing}</ChartFrame>

  const { wn, a, b, overlap } = profile
  const [ma, mb] = set.models
  const shown = rows(data, 'vip_overlap').find((r) => String(r.comparison).startsWith(set.cite.split(' ')[0]))
  const bandRows = rows(data, 'vip_bands')
  const bands = edges.map((e, i) => ({ lo: num(e.low) ?? 0, hi: num(e.high) ?? 0, label: SHORT[String(e.band)] ?? String(e.band), fill: FILLS[i % 2]() }))
  // the band shares the notebook tabulated, recomputed from the drawn profile (they match to the export's 4 decimals)
  const shareGap = Math.max(0, ...edges.flatMap((e) => set.models.map((m, j) => {
    const nb = num(bandRows.find((r) => r.model === m && r.band === e.band)?.vip2_share)
    return nb == null ? 0 : Math.abs(vipBandShare(wn, j ? b : a, num(e.low) ?? 0, num(e.high) ?? 0) - nb)
  })))
  const runs = sharedRuns(a, b)
  const label = (m: string, k: number | null) => `${m}${k != null ? ` (k = ${k})` : ''}`
  const agree = shown && overlap.top_n_overlap === num(shown.top_n_overlap) && overlap.important_a_n === num(shown.important_a_n) && overlap.important_b_n === num(shown.important_b_n)

  return (
    <ChartFrame id="ml-model-form-vip-profiles"
      title={`${ma} and ${mb} VIP profiles: Spearman r ${f3(overlap.spearman_r)}, ${overlap.top_n_overlap === 0 ? 'no' : overlap.top_n_overlap} top-200 wavenumber${overlap.top_n_overlap === 1 ? '' : 's'} in common — ${set.cite}`}
      subtitle={`Variable importance in projection (VIP; Takahama et al. 2019, Eq. 14) of each PLS model at every one of the ${wn.length.toLocaleString()} wavenumbers, highest wavenumber on the left as the notebooks drew it. VIP ≥ 1 (dashed) marks a wavenumber carrying more than its share of the explained response variance: ${overlap.important_a_n.toLocaleString()} for ${ma}, ${overlap.important_b_n.toLocaleString()} for ${mb} (Jaccard ${f2(overlap.important_jaccard)}). ${shade === 'bands'
        ? 'Shaded: the five broad bands the notebook sums squared VIP over (the chart below); the gaps between them (1850–2800 cm⁻¹, above 3600) belong to no band.'
        : `Shaded: ${ma}'s VIP where both models reach VIP ≥ 1 (ftir_07 [5]'s lower panel), ${runs.length} stretch${runs.length === 1 ? '' : 'es'}.`} VIP grows with the number of PLS factors, so compare shapes, not heights; it is a post hoc importance and names no unique molecule.`}
      provenance={provenance}
      controls={<>
        {present.length > 1 && <Segmented label="Models" value={set.key} options={present.map((s) => s.key)} optionLabel={(k) => present.find((s) => s.key === k)?.label ?? k} onChange={setSet} />}
        <Segmented label="Shade" value={shade} options={SHADES} optionLabel={(o) => (o === 'bands' ? 'Functional bands' : 'VIP ≥ 1 in both')} onChange={setShade} />
      </>}>
      <SpectrumPlot height={340} yLabel="VIP" window={[d3.max(wn) ?? 4000, d3.min(wn) ?? 500]} includeZero hover="readout"
        markers={[{ at: 1, orient: 'h', label: 'VIP = 1', color: INK.muted }]}
        series={[
          { id: ma, label: label(ma, set.k[0]), wn, y: a, color: colors[ma] ?? INK.accent, role: 'line', width: 1.2 },
          { id: mb, label: label(mb, set.k[1]), wn, y: b, color: colors[mb] ?? INK.negative, role: 'line', width: 1.2 },
        ]}
        background={({ x, y, ih }) => {
          if (shade === 'bands') {
            // the bands drawn here rather than as annotations, so a label too wide for its band (a phone) is left off
            return <g pointerEvents="none">{bands.map((bd) => {
              const x0 = x(bd.hi), x1 = x(bd.lo)
              return <g key={bd.label}>
                <rect x={x0} width={x1 - x0} height={ih} fill={bd.fill} fillOpacity={0.8} />
                {x1 - x0 >= bd.label.length * 6 + 4 && <text x={(x0 + x1) / 2} y={11} textAnchor="middle" fontSize={10} fill={INK.muted}>{bd.label}</text>}
              </g>
            })}</g>
          }
          const area = d3.area<number>().x((i) => x(wn[i])).y0(y(0)).y1((i) => y(a[i]))
          return <g pointerEvents="none">{runs.map(([s, e]) => <path key={s} d={area(d3.range(s, e + 1)) ?? ''} fill={SHARED()} fillOpacity={0.45} />)}</g>
        }} />
      <Key items={[
        { label: label(ma, set.k[0]), color: colors[ma] ?? INK.accent, kind: 'line' },
        { label: label(mb, set.k[1]), color: colors[mb] ?? INK.negative, kind: 'line' },
        { label: 'VIP = 1', color: INK.muted, kind: 'dash' },
        ...(shade === 'bands' ? [] : [{ label: 'VIP ≥ 1 in both', color: SHARED(), kind: 'square' as const }]),
      ]} />
      <div className="legend-note" style={{ marginTop: 6 }}>
        Recomputed in the browser from the drawn profiles: Spearman r {f3(overlap.spearman_r)}, VIP ≥ 1 at {overlap.important_a_n} and {overlap.important_b_n} wavenumbers,
        Jaccard {f3(overlap.important_jaccard)}, {overlap.top_n_overlap} of the top 200 shared, {(100 * overlap.a_vip_mass_on_b_important).toFixed(1)} % of {ma}'s VIP² where {mb} has VIP ≥ 1.
        {shown ? <> The notebook displayed r {f3(shown.spearman_r)}, {String(shown.important_a_n)} and {String(shown.important_b_n)}, Jaccard {f3(shown.important_jaccard)}, {String(shown.top_n_overlap)} shared, {(100 * (num(shown.a_vip_mass_on_b_important) ?? NaN)).toFixed(1)} %{agree ? ': the same' : ''}; band shares agree within {shareGap < 5e-4 ? 'the export’s 4 decimals' : f3(shareGap)}.</> : null}
      </div>
    </ChartFrame>
  )
}
