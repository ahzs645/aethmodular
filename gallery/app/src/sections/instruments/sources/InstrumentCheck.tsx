import { useMemo } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Crossplot, type CrossPoint } from '@/components/Crossplot'
import { Key } from '@/components/Key'
import { f2 } from '@/lib/format'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { AAE_PAIRS, MA350_NM, aae, type AaePair } from './aae'
import { NOTEBOOK, provenance } from './common'
import type { DailySite, InstrumentFile } from './data'

const FROMS = ['absorption', 'BCc'] as const
const SOURCES = ['hourly', 'daily'] as const
const SOURCE_LABEL = { hourly: 'Hourly export', daily: 'Pickle columns' }

/**
 * The check on the AAE scale: the MA350's own AAE (with its biomass / fossil split) against the gallery's,
 * per daily bin. From absorption the two share a scale; from BCc, as the notebooks took it, every point
 * drops by about 1. The instrument's AAE is the Blue/IR absorption one (gallery/data/export_aeth_sources.py
 * measures that on the hourly files), so the Blue-IR pair is the like-for-like comparison.
 */
export function InstrumentCheck({ daily, pair, instrument }: { daily: DailySite[]; pair: AaePair; instrument: InstrumentFile | null }) {
  const [from, setFrom] = useSearchParam<(typeof FROMS)[number]>('aes.instFrom', 'absorption', FROMS)
  const [source, setSource] = useSearchParam<(typeof SOURCES)[number]>('aes.instSource', 'hourly', SOURCES)
  const p = AAE_PAIRS[pair]
  const { points, sites } = useMemo(() => {
    const out: CrossPoint[] = [], sites: { name: string; color: string; n: number }[] = []
    for (const d of daily) {
      const inst = instrument?.sites[d.code]?.[source]
      if (!inst?.n) continue
      const at = new Map(d.daily.start.map((s, i) => [s, i]))
      let n = 0
      inst.start.forEach((s, k) => {
        const i = at.get(s), own = inst.aae[k]
        if (i === undefined || own == null) return
        const g = from === 'absorption' ? d.daily[p.daily][i] : aae(d.daily[p.short][i], d.daily[p.long][i], MA350_NM[p.short], MA350_NM[p.long])
        if (g == null || !Number.isFinite(g)) return
        n++
        out.push({ x: own, y: g, color: d.color, hollow: from === 'BCc',
          tip: [`${d.name} · bin from ${s}`, `instrument AAE ${f2(own)} · biomass ${f2(inst.bb_pct[k])} %`, `gallery AAE ${pair} from ${from}: ${f2(g)}`],
          rec: { id: `${d.code}:${s}`, source: 'Aethalometer sources · instrument AAE check', site: d.name, date: s,
            fields: [['instrument AAE', own], ['instrument biomass (%)', inst.bb_pct[k]], [`AAE ${pair} (${from})`, +g.toFixed(3)]] } })
      })
      if (n) sites.push({ name: d.name, color: d.color, n })
    }
    return { points: out, sites }
  }, [daily, instrument, from, pair, p, source])
  const named = (src: (typeof SOURCES)[number]) => Object.values(instrument?.sites ?? {}).filter((s) => (s[src]?.n ?? 0) > 0).map((s) => s.site.replace('_', ' '))
  const ends = Object.values(instrument?.sites ?? {}).find((s) => s.daily.n > 0)?.daily
  return (
    <ChartFrame id="aeth-sources-firmware"
      title={`The instrument's own AAE against the gallery's — the MA350's AAE (x) and AAE ${pair} from ${from} (y), per daily bin`}
      subtitle={`The MA350 reports its own AAE and a biomass / fossil-fuel split of BCc (end members AAE ${ends?.aae_fossil ?? 1} fossil, ${ends?.aae_biomass ?? 2} biomass). Hourly export: the MAIA hourly files averaged over each daily bin (${named('hourly').join(', ') || 'none'}; at least 18 hours). Pickle columns: the processed pickles' own daily values (${named('daily').join(', ') || 'none'}). The instrument's AAE is the Blue/IR one from absorption, so choose Blue-IR above for like with like. From BCc (switch) every point drops by about 1, the offset the notebooks carried. Addis has no instrument AAE: its pickles lack the columns and the MAIA "Central" file is a different instrument.`}
      provenance={provenance([NOTEBOOK.optics], 'Reads aethalometer/sources/instrument.json (gallery/data/export_aeth_sources.py) joined to daily/<CODE>.json by bin start.')}
      controls={<>
        <Segmented label="Instrument values" value={source} options={SOURCES} onChange={setSource} optionLabel={(o) => SOURCE_LABEL[o]} />
        <Segmented label="Gallery AAE from" value={from} options={FROMS} onChange={setFrom} optionLabel={(o) => (o === 'BCc' ? 'BCc (notebooks)' : 'Absorption')} />
      </>}>
      {!instrument ? <Empty>Loading the instrument values…</Empty>
        : !points.length ? <Empty>No site in view has the instrument's own AAE ({SOURCE_LABEL[source].toLowerCase()}: {named(source).join(', ') || 'none'}). Pick one of those sites in the Filters panel.</Empty> : <>
          <Crossplot points={points} title={`Instrument AAE vs gallery AAE ${pair}`} xLabel="MA350 AAE (Blue/IR, absorption)" yLabel={`AAE ${pair} from ${from}`} grouping="site" maxSide={340} />
          <Key items={[...sites.map((s) => ({ label: `${s.name} (${s.n} bins)`, color: s.color, kind: from === 'BCc' ? 'hollow' as const : 'dot' as const })), { label: '1:1', color: INK.identity, kind: 'dash' as const }]} />
        </>}
    </ChartFrame>
  )
}
