import { useMemo } from 'react'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { f2, pct } from '@/lib/format'
import { useSearchParam } from '@/lib/url'
import { AAE_PAIRS, CLASSES, CLASS_COLOR, MA350_NM, MA350_SIGMA, aaeWithRatio, classShares, ma350Ratio, type AaePair, type AaeRegions } from './aae'
import { dailyAae } from './AaeValues'
import { NOTEBOOK, provenance } from './common'
import type { DailySite } from './data'

/** addis_01's MAC_COMBINATIONS (UV MAC, IR MAC), m²/g */
const ADDIS01_MACS: [number, number][] = [[7.77, 7.77], [10.0, 7.77], [12.5, 7.77], [15.0, 7.77], [12.5, 8.5], [15.0, 9.0]]

/**
 * addis_01 Task 3, "MAC sensitivity", recast: the notebook multiplied UV and IR BCc by a pair of MACs
 * before taking the AAE, so all that mattered was their ratio. Here the classes are recomputed for every
 * ratio from 1 (BCc as it is, the notebooks' AAE) to 3; the MA350's own cross-sections put it at 2.38
 * for UV/IR, which is the AAE the gallery exports. The notebook's six pairs are ringed.
 */
export function CrossSectionSweep({ daily, pair, regions }: { daily: DailySite[]; pair: AaePair; regions: AaeRegions }) {
  const codes = daily.map((d) => d.code)
  // the options follow the sites in view, so the key is validated here rather than registered with a list
  const [code, setCode] = useSearchParam<string>('aes.sigmaSite', 'ETAD')
  const site = daily.find((d) => d.code === code) ?? daily[0]
  const p = AAE_PAIRS[pair], nmS = MA350_NM[p.short], nmL = MA350_NM[p.long], own = ma350Ratio(p.short, p.long)
  const macs = pair === 'UV-IR' ? ADDIS01_MACS.map(([u, i]) => ({ u, i, ratio: u / i })) : []
  const steps = useMemo(() => {
    const grid = Array.from({ length: 101 }, (_, k) => 1 + k * 0.02)
    return [...new Set([...grid, own, ...macs.map((m) => m.ratio)].map((v) => +v.toFixed(4)))].sort((a, b) => a - b)
  }, [own, macs.length]) // eslint-disable-line react-hooks/exhaustive-deps
  const series = useMemo<SweepSeries[]>(() => {
    if (!site) return []
    const bcc = dailyAae(site.daily, pair, 'BCc').filter((v): v is number => v !== null && Number.isFinite(v))
    const at = steps.map((ratio) => ({ ratio, s: classShares(bcc.map((v) => aaeWithRatio(v, ratio, nmS, nmL)), regions) }))
    return CLASSES.map((c) => ({ id: c, label: c, color: CLASS_COLOR[c], marks: false, interval: 'none' as const, width: c === 'Biomass' ? 2.4 : 1.6,
      points: at.map(({ ratio, s }) => {
        const mac = macs.filter((m) => Math.abs(m.ratio - ratio) < 1e-4).map((m) => `addis_01 MAC UV ${m.u} / IR ${m.i}`)
        return { x: ratio, y: 100 * s.shares[c], tip: [`σ ${p.short} / σ ${p.long} = ${f2(ratio)}${Math.abs(ratio - own) < 1e-4 ? ' (MA350)' : ratio === 1 ? ' (BCc as is)' : ''}`, ...mac,
          `${c}: ${pct(s.shares[c])} of ${s.n} bins`, `median AAE ${f2(s.median)}`] }
      }) }))
  }, [site, steps, pair, regions]) // eslint-disable-line react-hooks/exhaustive-deps
  const atOwn = series.find((s) => s.id === 'Biomass')?.points.find((q) => Math.abs((q.x as number) - own) < 1e-4)?.y
  const atOne = series.find((s) => s.id === 'Biomass')?.points.find((q) => q.x === 1)?.y
  return (
    <ChartFrame id="aeth-sources-cross-section"
      title={`What the cross-section does to the classes — share of daily bins per class against the ${p.short}/${p.long} cross-section ratio`}
      subtitle={`Each daily bin's AAE from BCc, turned back into absorption with an assumed σ ${p.short}/σ ${p.long}, then classed. At 1 it is the notebooks' BCc AAE; at ${f2(own)} (the MA350's ${p.short} ${MA350_SIGMA[p.short].toFixed(2)} and ${p.long} ${MA350_SIGMA[p.long].toFixed(2)} m²/g) it is the absorption AAE the gallery uses.${macs.length ? ' Rings: the six UV/IR MAC pairs addis_01 tried, all of which sit below the instrument\'s own ratio.' : ''}`}
      provenance={provenance([NOTEBOOK.addis01, NOTEBOOK.addis01daily, NOTEBOOK.optics], 'Exact: AAE(ratio) = AAE_BCc + ln(ratio) / ln(λ_long/λ_short) for every bin (optics.aae_from_columns).')}
      controls={codes.length > 1 ? <Select label="Site" value={site?.code ?? ''} options={codes} onChange={setCode} optionLabel={(c) => daily.find((d) => d.code === c)?.name ?? c} /> : undefined}>
      {!site ? <Empty>No site in view has aethalometer daily bins.</Empty> : <>
        <SweepPlot series={series} xLabel={`σ ${p.short} / σ ${p.long} (1 = AAE from BCc)`} yLabel="Share of daily bins (%)" yDomain={[0, 100]} height={280}
          xMarkers={[{ at: 1, label: 'BCc (notebooks)' }, { at: own, label: `MA350 ${f2(own)}`, labelSide: 'left' }]}
          chosen={macs.map((m) => ({ series: 'Biomass', x: +m.ratio.toFixed(4) }))} />
        <Key items={[...CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c], kind: 'line' as const })), ...(macs.length ? [{ label: 'addis_01 MAC pairs', color: CLASS_COLOR.Biomass, kind: 'hollow' as const }] : [])]} />
        {atOwn != null && atOne != null && <p className="chart-note">{site.name}: biomass {Math.round(atOne)} % of bins from BCc against {Math.round(atOwn)} % from absorption.</p>}
      </>}
    </ChartFrame>
  )
}
