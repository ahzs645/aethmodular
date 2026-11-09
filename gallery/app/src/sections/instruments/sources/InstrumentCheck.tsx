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
      const hourly = source === 'hourly' ? instrument?.sites[d.code]?.hourly : undefined
      const verifiedCoverage = hourly?.coverage_version === 2
      const at = new Map(d.daily.start.map((s, i) => [s, i]))
      let n = 0
      inst.start.forEach((s, k) => {
        const i = at.get(s), own = inst.aae[k]
        if (i === undefined || own == null || !Number.isFinite(own)) return
        const g = from === 'absorption' ? d.daily[p.daily][i] : aae(d.daily[p.short][i], d.daily[p.long][i], MA350_NM[p.short], MA350_NM[p.long])
        if (g == null || !Number.isFinite(g)) return
        const minHours = hourly?.min_hours ?? 18
        const splitHours = hourly?.split_hours?.[k]
        const biomass = hourly && (!verifiedCoverage || splitHours == null || splitHours < minHours || hourly.bb_status?.[k] !== 'available')
          ? null : inst.bb_pct[k]
        const biomassLabel = biomass == null ? 'modelled biomass fraction unavailable' : `modelled biomass ${f2(biomass)} %`
        const coverageTip = hourly ? verifiedCoverage
          ? [`${hourly.aae_hours?.[k] ?? '?'} AAE hours · ${splitHours ?? '?'} paired source-split hours · ${hourly.bin_hours ?? 24} elapsed hours`,
            `${hourly.duplicate_rows?.[k] ?? 0} repeated rows · ${hourly.conflict_hours?.[k] ?? 0} conflicting hours excluded for affected values`,
            `UTC window ${hourly.start_utc?.[k] ?? '?'} to ${hourly.end_utc?.[k] ?? '?'} · local zone ${hourly.timezone ?? '?'}`]
          : ['Independent hourly coverage was not recorded in this saved result; biomass fraction unavailable.'] : []
        const fields: NonNullable<CrossPoint['rec']>['fields'] = [
          ['instrument AAE', own], ['instrument modelled biomass (%)', biomass], [`AAE ${pair} (${from})`, +g.toFixed(3)],
        ]
        if (hourly) fields.push(['hourly coverage verified', verifiedCoverage ? 'yes' : 'not recorded'],
          ['AAE hours', hourly.aae_hours?.[k]], ['paired source-split hours', splitHours],
          ['repeated rows', hourly.duplicate_rows?.[k]], ['conflicting hours', hourly.conflict_hours?.[k]],
          ['bin start (UTC)', hourly.start_utc?.[k]], ['bin end (UTC)', hourly.end_utc?.[k]],
          ['local timezone', hourly.timezone], ['biomass fraction status', hourly.bb_status?.[k] ?? 'coverage not recorded'])
        n++
        out.push({ x: own, y: g, color: d.color, hollow: from === 'BCc',
          tip: [`${d.name} · bin from ${s}`, `instrument AAE ${f2(own)} · ${biomassLabel}`, `gallery AAE ${pair} from ${from}: ${f2(g)}`, ...coverageTip],
          rec: { id: `${d.code}:${s}`, source: 'Aethalometer sources · instrument AAE check', site: d.name, date: s,
            fields } })
      })
      if (n) sites.push({ name: d.name, color: d.color, n })
    }
    return { points: out, sites }
  }, [daily, instrument, from, pair, p, source])
  const named = (src: (typeof SOURCES)[number]) => Object.values(instrument?.sites ?? {}).filter((s) => (s[src]?.n ?? 0) > 0).map((s) => s.site.replace('_', ' '))
  const ends = Object.values(instrument?.sites ?? {}).find((s) => s.daily.n > 0)?.daily
  const hasLegacyHourly = Object.values(instrument?.sites ?? {}).some((s) => s.hourly?.n && s.hourly.coverage_version !== 2)
  return (
    <ChartFrame id="aeth-sources-firmware"
      title={`The instrument's own AAE against the gallery's — the MA350's AAE (x) and AAE ${pair} from ${from} (y), per daily bin`}
      subtitle={`The MA350 reports its own AAE and a modelled biomass / fossil-fuel split of BCc (end members AAE ${ends?.aae_fossil ?? 1} fossil, ${ends?.aae_biomass ?? 2} biomass). Hourly export: the MAIA hourly files averaged over each daily bin (${named('hourly').join(', ') || 'none'}). AAE and biomass fractions independently need at least 18 distinct UTC hourly intervals in the 24 elapsed hours from the local bin start. Identical repeats count once; conflicting values make that metric's hour unavailable. Hover or open a point for separate coverage counts.${hasLegacyHourly ? ' Older saved results lack these coverage counts: their AAE remains inspectable, but their biomass fraction is unavailable until rebuilt.' : ''} Pickle columns: the processed pickles' own daily values (${named('daily').join(', ') || 'none'}). The instrument's AAE is the Blue/IR one from absorption, so choose Blue-IR above for like with like. From BCc (switch) every point drops by about 1, the offset the notebooks carried. Addis has no instrument AAE: its pickles lack the columns and the MAIA "Central" file is a different instrument.`}
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
