import { useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { XAxis } from '@/components/Axes'
import { Key } from '@/components/Key'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { f2, pct } from '@/lib/format'
import { seasonOfDate } from '@/lib/seasons'
import { FONT, INK, inkOn } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { ExcludedNote } from '@/lib/markedRows'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { AAE_PAIRS, CLASSES, CLASS_COLOR, classShares, type AaePair, type AaeRegions, type ClassShares as Shares } from './aae'
import { dailyAae, rowAae, type AaeFrom, type Basis } from './AaeValues'
import { NOTEBOOK, provenance } from './common'
import { siteOf, type DailySite } from './data'

interface Bar { key: string; site: string; label: string; color: string; strong: boolean; s: Shares }
const FROMS = ['absorption', 'BCc'] as const

/**
 * addis_01 Task 4: the share of fossil / mixed / biomass AAE by season, as 100 % bars, every site in view
 * under its own calendar (the Filters panel's). Switching to the notebooks' BCc AAE shows how far that
 * choice moves the classes.
 */
export function ClassShares({ rows, meta, daily, pair, basis, regions }: {
  rows: FilterRow[]; meta: MetaFile; daily: DailySite[]; pair: AaePair; basis: Basis; regions: AaeRegions
}) {
  const [from, setFrom] = useSearchParam<AaeFrom>('aes.sharesFrom', 'absorption', FROMS)
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)

  const bars = useMemo<Bar[]>(() => {
    const groups: { code: string; name: string; color: string; items: { season: string | null; v: number | null }[] }[] = basis === 'Daily bins'
      ? daily.map((d) => {
        const site = siteOf(d.code, meta)!.site, seasons = seasonsForSite(meta, site), vals = dailyAae(d.daily, pair, from)
        return { code: d.code, name: d.name, color: d.color, items: d.daily.start.map((st, i) => ({ season: seasonOfDate(st.slice(0, 10), Number(st.slice(5, 7)), seasons)?.name ?? null, v: vals[i] })) }
      })
      : meta.sites.map((s) => ({ code: s.code, name: s.name, color: s.color, items: rows.filter((r) => r.code === s.code).map((r) => ({ season: r.season, v: rowAae(r, pair, from) })) }))
    return groups.flatMap((g) => {
      const all = classShares(g.items.map((i) => i.v), regions)
      if (!all.n) return []
      const seasons = [...new Set(g.items.filter((i) => typeof i.v === 'number').map((i) => i.season).filter((s): s is string => !!s))]
      const order = meta.seasons.map((s) => s.name)
      seasons.sort((a, b) => order.indexOf(a) - order.indexOf(b))
      return [{ key: g.code, site: g.name, label: g.name, color: g.color, strong: true, s: all },
        ...seasons.map((season) => ({ key: `${g.code}|${season}`, site: g.name, label: season.replace(/^[^·]+ · /, ''), color: g.color, strong: false,
          s: classShares(g.items.filter((i) => i.season === season).map((i) => i.v), regions) }))]
    })
  }, [basis, daily, rows, meta, pair, from, regions])

  // on a narrow screen the labels give way (cut short, full name in the tooltip) and the n column goes
  const narrow = width < 560
  const rowH = 22, noteW = narrow ? 0 : 120
  const labelW = Math.min(narrow ? 120 : 240, 16 + 6.4 * Math.max(10, ...bars.map((b) => b.label.length + (b.strong ? 0 : 2))))
  const fitLabel = (t: string) => (t.length * 6.4 + 16 > labelW ? `${t.slice(0, Math.max(4, Math.floor((labelW - 22) / 6.4)))}…` : t)
  const iw = Math.max(120, width - labelW - noteW - 12)
  const x = d3.scaleLinear().domain([0, 1]).range([0, iw])
  // a gap before each site's block
  let yAt = 0
  const placed = bars.map((b, i) => { if (b.strong && i > 0) yAt += 8; const y = yAt; yAt += rowH; return { ...b, y } })
  const h = yAt + 40
  return (
    <ChartFrame id="aeth-sources-class-shares"
      title={`Fossil / mixed / biomass by season — share of ${basis === 'Daily bins' ? 'daily bins' : 'filters'} in each AAE class`}
      subtitle={`AAE ${pair} classed with config.AAE_REGIONS (fossil ≤ ${regions.fossil_max}, biomass ≥ ${regions.biomass_min}). Seasons are each site's own calendar from the Filters panel. "BCc" recomputes the classes from the AAE the notebooks used (taken from BCc), which moves every value down by about 1 and turns most mixed days fossil and most biomass days mixed.`}
      provenance={provenance([NOTEBOOK.addis01, NOTEBOOK.addis01daily, NOTEBOOK.optics], 'Classes by optics.classify_aae; the notebook stacked the same three shares per Ethiopian season at Addis only.')}
      controls={<Segmented label="AAE from" value={from} options={FROMS} onChange={setFrom} optionLabel={(o) => (o === 'BCc' ? 'BCc (notebooks)' : 'Absorption')} />}>
      <div ref={ref} style={{ position: 'relative' }}>
        {!bars.length ? <Empty>No site in view has aethalometer AAE.</Empty> : <>
          <svg width={labelW + iw + noteW} height={h} fontFamily={FONT.family} style={{ display: 'block' }}>
            <g transform={`translate(${labelW},4)`}>
              {placed.map((b) => {
                let at = 0
                return <g key={b.key} transform={`translate(0,${b.y})`}>
                  <text x={-8} y={rowH / 2} dy="0.35em" textAnchor="end" fontSize={11} fontWeight={b.strong ? 600 : 400} fill={b.strong ? b.color : INK.text}>{fitLabel(b.label)}</text>
                  {CLASSES.map((c) => {
                    const w = x(b.s.shares[c]), x0 = at
                    at += w
                    return <g key={c} onMouseMove={(e) => tip.show(e, [`${b.site}${b.strong ? '' : ` · ${b.label}`}`, `${c}: ${pct(b.s.shares[c])} of ${b.s.n}`, `median AAE ${f2(b.s.median)}`])} onMouseLeave={tip.hide}>
                      <rect x={x0} y={3} width={Math.max(0, w)} height={rowH - 6} fill={CLASS_COLOR[c]} fillOpacity={b.strong ? 0.95 : 0.8} />
                      {w > 34 && <text x={x0 + w / 2} y={rowH / 2} dy="0.35em" textAnchor="middle" fontSize={10} fill={inkOn(CLASS_COLOR[c])} pointerEvents="none">{pct(b.s.shares[c])}</text>}
                    </g>
                  })}
                  {!narrow && <text x={iw + 8} y={rowH / 2} dy="0.35em" fontSize={10} fontFamily={FONT.mono} fill={INK.muted}>n={b.s.n} · med {f2(b.s.median)}</text>}
                </g>
              })}
              <XAxis scale={x} y={yAt + 2} tickCount={5} format={(v) => `${Math.round(+v * 100)} %`} />
            </g>
          </svg>
          <Key items={CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c], kind: 'square' as const }))} />
          {/* Filters → excluded: Mark (lib/markedRows): on the filter windows, the excluded filters with an AAE are in no share; the daily bins are not filters */}
          {basis === 'Filter windows' && <ExcludedNote fields={[AAE_PAIRS[pair].field]} />}
        </>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
