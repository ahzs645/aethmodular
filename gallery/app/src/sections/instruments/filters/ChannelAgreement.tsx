import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock, type IntervalPanel } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { pearson } from '@/lib/stats'
import { INK } from '@/lib/theme'
import type { FilterRow } from '@/lib/types'
import { CHANNELS, FTIR_EC, HIPS_BC, HIPS_FABS, HIPS_NM, NEAREST_TO_HIPS, atnField, bccField, type Channel, type PairSite } from './data'
import { num } from './mac'

const MIN_N = 8

/** r² of two columns over the filters with both, and how many */
function r2(xs: (number | null)[], ys: (number | null)[]) {
  const k = xs.map((_, i) => i).filter((i) => xs[i] !== null && ys[i] !== null)
  if (k.length < MIN_N) return null
  const r = pearson(k.map((i) => xs[i]!), k.map((i) => ys[i]!))
  return r === null ? null : { r2: r * r, n: k.length }
}

/**
 * Which channel agrees best with the filters? For every site and MA350 channel: r² of the channel's
 * BCc against HIPS Fabs (and of its spot attenuation, ATN, which HIPS_vs_Aethalometer asked about),
 * r² against FTIR EC, and the ratio of the channel's BCc to HIPS BC (Fabs ÷ 10). Clicking a channel
 * makes it the page's channel.
 */
export function ChannelAgreement({ rows, sites, extra, wavelengths, ch, onChannel }: {
  rows: FilterRow[]; sites: PairSite[]; extra: Map<string, Record<string, number | null>>
  wavelengths: Record<string, number>; ch: Channel; onChannel: (c: Channel) => void
}) {
  const one = sites.length === 1
  const hasAtn = extra.size > 0
  const blocks = useMemo<IntervalBlock[]>(() => sites.map((s) => {
    const own = rows.filter((r) => r.site === s.name)
    const fabs = own.map((r) => num(r[HIPS_FABS])), ec = own.map((r) => num(r[FTIR_EC]))
    return {
      key: s.code, title: one ? undefined : s.name, color: s.color,
      rows: CHANNELS.map((c) => {
        const bc = own.map((r) => num(r[bccField(c)])), atn = own.map((r) => num(extra.get(r.id)?.[atnField(c)]))
        const vsFabs = r2(bc, fabs), atnFabs = r2(atn, fabs), vsEc = r2(bc, ec)
        const ratio = own.flatMap((r) => { const b = num(r[bccField(c)]), h = num(r[HIPS_BC]); return b !== null && h !== null && h > 0 ? [b / h] : [] }).sort(d3.ascending)
        const q = ratio.length >= MIN_N ? { mid: d3.quantileSorted(ratio, 0.5)!, lo: d3.quantileSorted(ratio, 0.25)!, hi: d3.quantileSorted(ratio, 0.75)! } : null
        const nm = wavelengths[c]
        return {
          key: `${s.code}|${c}`, label: `${c} (${nm} nm)${c === NEAREST_TO_HIPS ? ' ≈ HIPS' : ''}`, selected: c === ch, strong: c === ch,
          note: `n=${vsFabs?.n ?? vsEc?.n ?? 0}`, onClick: () => onChannel(c),
          marks: [
            [...(vsFabs ? [{ mid: vsFabs.r2, color: s.color }] : []), ...(atnFabs ? [{ mid: atnFabs.r2, color: s.color, shape: 'hollow' as const }] : [])],
            vsEc ? [{ mid: vsEc.r2, color: s.color }] : [],
            q ? [{ ...q, color: s.color }] : [],
          ],
          tip: [`${s.name} · ${c} (${nm} nm)`,
            `r² BCc vs HIPS Fabs ${fmt(vsFabs?.r2, 3)} (n=${vsFabs?.n ?? 0})`,
            ...(hasAtn ? [`r² ATN vs HIPS Fabs ${fmt(atnFabs?.r2, 3)} (n=${atnFabs?.n ?? 0})`] : []),
            `r² BCc vs FTIR EC ${fmt(vsEc?.r2, 3)} (n=${vsEc?.n ?? 0})`,
            q ? `BCc ÷ HIPS BC: median ${fmt(q.mid, 2)} (IQR ${fmt(q.lo, 2)}–${fmt(q.hi, 2)})` : 'BCc ÷ HIPS BC: too few filters',
            c === ch ? 'the page\'s channel' : 'click to use this channel on the page'],
        }
      }),
    }
  }), [rows, sites, one, extra, hasAtn, wavelengths, ch, onChannel])

  const panels: IntervalPanel[] = [
    { heading: 'against HIPS Fabs', label: 'r²', domain: [0, 1], format: (v) => fmt(v, 1) },
    { heading: 'against FTIR EC', label: 'r²', domain: [0, 1], format: (v) => fmt(v, 1) },
    { heading: 'BCc ÷ HIPS BC', label: 'ratio (median, IQR)', reference: 1, format: (v) => fmt(v, 1) },
  ]
  const any = blocks.some((b) => b.rows.some((r) => r.marks.some((m) => m.length)))
  return (
    <ChartFrame id="aeth-filters-channels" title="Agreement by channel — which aethalometer wavelength tracks the filters"
      subtitle={`For each MA350 channel, over the filters with both values: r² of its BCc against HIPS Fabs (filled)${hasAtn ? ' and of its spot attenuation ATN1 (hollow)' : ''}, r² against FTIR EC, and its BCc ÷ HIPS BC (Fabs ÷ 10; median and IQR, 1 = agreement). HIPS reads at ${HIPS_NM} nm, so the nearest channel is ${NEAREST_TO_HIPS} (625 nm), not the UV the notebook expected. At least ${MIN_N} filters per value. Click a channel to use it everywhere on this page.`}
      provenance="HIPS_vs_Aethalometer_Optical_Comparison.ipynb §6–9 (ATN vs BCc against HIPS, R² by wavelength) · ETAD_Aug13.ipynb §12 (Red 625 nm vs IR 880 nm against HIPS)">
      {any ? <>
        <IntervalStrip blocks={blocks} panels={panels} labelWidth={130} noteWidth={44} minPanelWidth={160} />
        <Key items={[{ label: 'BCc', color: INK.muted, kind: 'dot' }, ...(hasAtn ? [{ label: 'ATN1 (spot attenuation)', color: INK.muted, kind: 'hollow' as const }] : [])]} />
        <Note>
          The notebook put HIPS at 405 nm and expected UV or Blue to match best; HIPS is a He–Ne reading at 633 nm
          (docs/filter-optics-reference.md). ATN is the loading of the current tape spot: it climbs as the spot darkens
          and drops at every advance, so its window mean is not a concentration and its r² is near zero by construction.
          Values here use the gallery's overlap matching (bins weighted by hours inside each filter's window), not the
          notebook's ±1 day mean, which is why the r² run higher than the notebook's.
        </Note>
      </> : <Empty>No site in view has {MIN_N} filters with an aethalometer channel and HIPS or FTIR EC.</Empty>}
    </ChartFrame>
  )
}
