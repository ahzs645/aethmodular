import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock, type IntervalPanel } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { ExcludedNote } from '@/lib/markedRows'
import { seasonOnly, shortSeason } from '@/lib/seasons'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { MAC_DEFS, macMarker, macOf, seasonColor, seasonsIn, type Channel, type Pair, type PairSite } from './data'

const MIN_N = 3

/** median, quartiles and the ratio of the means (AAARpos Method 2) of one MAC definition over some filters */
function summary(pairs: Pair[], d: (typeof MAC_DEFS)[number]) {
  const both = pairs.filter((p) => macOf(d, p) !== null)
  const v = both.map((p) => macOf(d, p)!).sort(d3.ascending)
  if (v.length < MIN_N) return null
  return {
    n: v.length, mid: d3.quantileSorted(v, 0.5)!, lo: d3.quantileSorted(v, 0.25)!, hi: d3.quantileSorted(v, 0.75)!,
    ratioOfMeans: d3.sum(both, (p) => d.num(p)!) / d3.sum(both, (p) => d.den(p)!),
  }
}

/**
 * The MAC each pairing implies, by site and season: HIPS Fabs over FTIR EC (the filter's own MAC, the
 * AAARpos / ETAD_Aug13 seasonal MAC), HIPS Fabs over the aethalometer's BC, and the aethalometer's
 * b_ATN over FTIR EC (ETAD_comprehensive §11–12's "IR Babs / FTIR EC"). Median and quartiles of the
 * per-filter ratios, with the ratio of the means beside it.
 */
export function ImpliedMac({ pairs, marked = [], meta, sites, ch, sigma, onSite }: {
  pairs: Pair[]; meta: MetaFile; sites: PairSite[]; ch: Channel; sigma: number | null; onSite?: (code: string) => void
  /** Filters → excluded: Mark, the excluded filters paired the same way (lib/markedRows): counted in the note only */
  marked?: Pair[]
}) {
  const one = sites.length === 1
  const nMarked = marked.filter((p) => sites.some((s) => s.name === p.site) && MAC_DEFS.some((d) => macOf(d, p) !== null)).length
  const blocks = useMemo<IntervalBlock[]>(() => sites.map((s) => {
    const own = pairs.filter((p) => p.site === s.name)
    const groups: { key: string; label: string; full: string; color: string; ps: Pair[] }[] = [
      { key: 'all', label: 'All seasons', full: 'All seasons', color: s.color, ps: own },
      ...seasonsIn(own, meta).map((se) => ({ key: se, label: shortSeason(seasonOnly(se)), full: seasonOnly(se), color: seasonColor(meta, se, INK.muted), ps: own.filter((p) => p.season === se) })),
    ]
    return {
      key: s.code, title: one ? undefined : s.name, color: s.color,
      rows: groups.map((g) => ({ g, sums: MAC_DEFS.map((d) => summary(g.ps, d)) })).filter(({ sums }) => sums.some(Boolean)).map(({ g, sums }) => {
        return {
          key: `${s.code}|${g.key}`, label: g.label, strong: g.key === 'all', labelColor: g.key === 'all' ? undefined : g.color,
          note: `n=${d3.max(sums, (x) => x?.n ?? 0) ?? 0}`,
          onClick: !one && onSite ? () => onSite(s.code) : undefined,
          marks: sums.map((x) => (x ? [{ mid: x.mid, lo: x.lo, hi: x.hi, color: g.color }, { mid: x.ratioOfMeans, color: g.color, shape: 'diamond' as const, r: 4 }] : [])),
          tip: [`${s.name} · ${g.full}`, ...MAC_DEFS.map((d, k) => sums[k]
            ? `${d.label}: median ${fmt(sums[k]!.mid, 2)} (IQR ${fmt(sums[k]!.lo, 2)}–${fmt(sums[k]!.hi, 2)}), ratio of means ${fmt(sums[k]!.ratioOfMeans, 2)}, n=${sums[k]!.n}`
            : `${d.label}: fewer than ${MIN_N} filters`), ...(!one && onSite ? ['click to look at this site alone'] : [])],
        }
      }),
    }
  }), [pairs, meta, sites, one, onSite])

  const { mac } = useMac()
  const panels: IntervalPanel[] = MAC_DEFS.map((d) => ({
    heading: d.key === 'aethOverEc' ? `b_ATN ${ch} ÷ FTIR EC` : d.key === 'hipsOverBc' ? `HIPS Fabs ÷ ${ch} BCc` : d.label,
    label: 'm²/g', reference: d.reference(sigma ?? NaN), marker: macMarker(d, mac), format: (v: number) => fmt(v, 1),
  })).map((p) => (Number.isFinite(p.reference) ? p : { ...p, reference: undefined }))

  const any = blocks.some((b) => b.rows.some((r) => r.marks.some((m) => m.length)))
  return (
    <ChartFrame id="aeth-filters-mac" title={`Implied MAC by ${one ? 'season' : 'site and season'} — what each pairing says the mass absorption cross-section is`}
      subtitle={`Per filter, a ratio of absorption to mass: HIPS Fabs (633 nm) ÷ FTIR EC; HIPS Fabs ÷ the aethalometer's ${ch} BCc; and the aethalometer's attenuation coefficient b_ATN = ${ch} BCc × σ_ATN${sigma ? ` (${fmt(sigma, 3)} m²/g)` : ''} ÷ FTIR EC. Dot and bar: median and interquartile range of the per-filter ratios; diamond: ratio of the means (AAARpos Method 2). The line is 10 m²/g, SPARTAN's MAC at 633 nm, for the HIPS ratios (dashed beside it: the MAC set under Filters → units, when it is not 10), and σ_ATN for the last (where the aethalometer's BC equals EC). b_ATN keeps the filter-tape multiple-scattering enhancement, so it is not on HIPS's scale. Seasons follow the gallery's calendar; at least ${MIN_N} filters per row.`}
      provenance="AAARpos.ipynb (MAC methods, seasonal MAC) · ETAD_Aug13.ipynb §9.6 · ETAD_comprehensive_absorption_analysis.ipynb §11–12 (aethalometer MAC = IR Babs / FTIR EC)">
      {any ? <>
        <IntervalStrip blocks={blocks} panels={panels} labelWidth={110} noteWidth={44} minPanelWidth={170} />
        <Key items={[{ label: 'median, interquartile range', color: INK.muted, kind: 'dot' }, { label: 'ratio of the means', color: INK.muted, kind: 'diamond' }]} />
        <ExcludedNote count={nMarked} />
      </> :<Empty>No site in view has at least {MIN_N} filters with both members of any pairing.</Empty>}
    </ChartFrame>
  )
}
