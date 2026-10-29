import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { robustDomain } from '@/lib/axes'
import { useColourScheme } from '@/lib/colourSchemes'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { useSearchParam } from '@/lib/url'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { seasonColor, type Channel, type Pair, type PairSite } from './data'

const COLOURS = ['Site', 'Season', 'Iron', 'Fe/EC', 'Fe/BC'] as const
type Colour = (typeof COLOURS)[number]
/** ETAD_Aug13 cell 45 asked whether iron explains a change of slope above 10 µg/m³ */
const HIGH = 10

interface Pairing { key: string; title: string; x: (p: Pair) => number | null; y: (p: Pair) => number | null; xLabel: string; yLabel: string }

/**
 * The aethalometer, HIPS and FTIR EC against each other, filter by filter: the three-way method
 * agreement of ETAD_Aug13 §11 (aethalometer vs FTIR EC, vs HIPS, and HIPS vs FTIR EC), plus
 * absorption against absorption (§4: b_ATN against HIPS Fabs). Every site in view on each panel, with
 * the gallery's fits (one per site when several), coloured by site, season or iron (the iron-gradient
 * cross-plots of Multi_Site_Analysis_Fixed §9 and _FollowUp §3–3b).
 */
export function MethodAgreement({ pairs, meta, sites, ch }: { pairs: Pair[]; meta: MetaFile; sites: PairSite[]; ch: Channel }) {
  const { mac } = useMac()
  const hipsLabel = `HIPS BC = Fabs ÷ ${mac} (µg/m³)`
  const [colour, setColour] = useSearchParam<Colour>('af.col', 'Site', COLOURS)
  const ramp = useColourScheme('af-agree', 'sequential')
  const pairings: Pairing[] = [
    { key: 'ec', title: `${ch} BCc against FTIR EC`, x: (p) => p.bc, y: (p) => p.ec, xLabel: `Aethalometer ${ch} BCc (µg/m³)`, yLabel: 'FTIR EC (µg/m³)' },
    { key: 'hips', title: `${ch} BCc against HIPS BC`, x: (p) => p.bc, y: (p) => p.hipsBc, xLabel: `Aethalometer ${ch} BCc (µg/m³)`, yLabel: hipsLabel },
    { key: 'abs', title: `b_ATN ${ch} against HIPS Fabs`, x: (p) => p.batn, y: (p) => p.fabs, xLabel: `b_ATN ${ch} = BCc × σ_ATN (Mm⁻¹)`, yLabel: 'HIPS Fabs (Mm⁻¹)' },
    { key: 'filters', title: 'HIPS BC against FTIR EC', x: (p) => p.hipsBc, y: (p) => p.ec, xLabel: hipsLabel, yLabel: 'FTIR EC (µg/m³)' },
  ]
  const third = (p: Pair): number | null => colour === 'Iron' ? p.iron
    : colour === 'Fe/EC' ? (p.iron !== null && p.ec !== null && p.ec > 0 ? p.iron / p.ec : null)
    : colour === 'Fe/BC' ? (p.iron !== null && p.bc !== null && p.bc > 0.1 ? p.iron / p.bc : null) : null
  const continuous = colour === 'Iron' || colour === 'Fe/EC' || colour === 'Fe/BC'
  const scale = useMemo(() => {
    if (!continuous) return null
    const [lo, hi] = robustDomain(pairs.map(third), 0.02, 0.98).domain
    return d3.scaleSequential(d3.interpolateRgbBasis(ramp.stops.slice(2))).domain([lo, hi === lo ? lo + 1 : hi]).clamp(true)
  }, [pairs, colour, ramp.stops.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const colourOf = (p: Pair) => {
    if (colour === 'Season') return seasonColor(meta, p.season, INK.muted)
    if (scale) { const v = third(p); return v === null ? INK.neutral : scale(v) }
    return p.siteColor
  }
  const pointsOf = (g: Pairing): CrossPoint[] => pairs.flatMap((p) => {
    const x = g.x(p), y = g.y(p)
    if (x === null || y === null) return []
    const t = third(p)
    return [{
      x, y, color: colourOf(p), id: p.id,
      tip: [`${p.id} · ${p.site} · ${p.date}`, `${g.xLabel}: ${fmt(x, 2)}`, `${g.yLabel}: ${fmt(y, 2)}`, ...(continuous ? [`${colour}: ${fmt(t, 3)}`] : []), p.season],
      rec: { id: p.id, source: 'Aethalometer · against the filters', site: p.site, date: p.date,
        fields: [[`${ch} BCc (aethalometer)`, p.bc], ['HIPS Fabs', p.fabs], ['EC (FTIR)', p.ec], ['Iron', p.iron]] },
    }]
  })
  // ETAD_Aug13 cell 45: Fe/BC where either method reads above 10 µg/m³, against the rest
  const threshold = useMemo(() => {
    if (colour !== 'Fe/BC') return null
    const pts = pairs.filter((p) => p.bc !== null && p.ec !== null && third(p) !== null)
    const high = pts.filter((p) => p.bc! > HIGH || p.ec! > HIGH), low = pts.filter((p) => !(p.bc! > HIGH || p.ec! > HIGH))
    return { high: high.length, low: low.length, mHigh: d3.mean(high, third), mLow: d3.mean(low, third) }
  }, [pairs, colour]) // eslint-disable-line react-hooks/exhaustive-deps
  const multi = sites.length > 1
  const panels = pairings.map((g) => ({ g, pts: pointsOf(g) })).filter((x) => x.pts.length >= 3)
  return (
    <ChartFrame id="aeth-filters-agreement" title="Method agreement — the aethalometer, HIPS and FTIR EC filter by filter"
      subtitle={`Each point is one filter: the aethalometer's ${ch} channel averaged over the filter's sampling window (daily bins weighted by their hours inside it) against the same filter's HIPS and FTIR EC. HIPS BC is Fabs ÷ ${mac}, the MAC set under Filters → units (at 10, SPARTAN's public BC). b_ATN = BCc × σ_ATN is an attenuation coefficient that still carries the tape's multiple-scattering enhancement, so its slope against HIPS Fabs is not expected to be 1. Dashed: 1:1. Fits follow the sidebar's estimators${multi ? ', one per site when coloured by site, pooled otherwise' : ''}. Colour by site, season, iron, Fe/EC or Fe/BC (2–98 % of the range; grey: no iron).`}
      provenance="ETAD_Aug13.ipynb §4, §5, §9.5, §11 · ETAD_comprehensive_absorption_analysis.ipynb §4–5 · Multi_Site_Analysis.ipynb §5–6 · Multi_Site_Analysis_Fixed.ipynb §4, §7, §9–10 · Multi_Site_Analysis_FollowUp.ipynb §3–4 · AAARpos.ipynb (method agreement)"
      controls={<Segmented label="Colour" value={colour} options={COLOURS} onChange={setColour} />}>
      {panels.length === 0 ? <Empty>No filter in view has the aethalometer at {ch} with HIPS or FTIR EC.</Empty> : <>
        <FacetGrid items={panels} minCell={300} maxCols={2} render={({ g, pts }, w) => (
          <Crossplot points={pts} title={g.title} xLabel={g.xLabel} yLabel={g.yLabel} maxSide={Math.max(180, w - 80)}
            grouping={multi ? (colour === 'Site' ? 'site' : 'pooled') : undefined} />
        )} />
        {scale ? <Key items={[{ label: `no ${colour === 'Iron' ? 'iron' : colour}`, color: INK.neutral }]}>
          <ColorLegend scale={scale} label={colour === 'Iron' ? 'Iron (µg/m³)' : colour === 'Fe/EC' ? 'Fe/EC' : `Fe / ${ch} BC`} scheme={ramp.scheme} />
        </Key> : <Key items={colour === 'Season'
          ? meta.seasons.filter((s) => pairs.some((p) => p.season === s.name)).map((s) => ({ label: s.name, color: s.color }))
          : sites.map((s) => ({ label: `${s.name} (n=${s.n})`, color: s.color }))} />}
        {threshold && threshold.high > 0 && <Note>
          Mean Fe/BC where the aethalometer or FTIR EC reads above {HIGH} µg/m³: {fmt(threshold.mHigh, 3)} (n={threshold.high});
          elsewhere {fmt(threshold.mLow, 3)} (n={threshold.low}). ETAD_Aug13 read a higher ratio at high loading as support for iron
          interference there; the difference is not a test.
        </Note>}
      </>}
    </ChartFrame>
  )
}
