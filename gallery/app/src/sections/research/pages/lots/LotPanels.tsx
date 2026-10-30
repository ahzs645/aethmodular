/**
 * char_12 §4's spectra by PTFE lot (04_lot_effect left, 04b), for the filters in view: each lot's centre and band
 * per wavenumber, recomputed in the browser from the exported per-filter spectra, over all filters (lot and season
 * confounded) and within one season (season held fixed), pooled over the sites in view or one row per site.
 */
import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { SpectrumPlot, type SpectrumBand, type SpectrumSeries } from '@/components/SpectrumPlot'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { columnsIn, defaultSeason, lotsOf, profile, seasonAt, VIEW_KEYS, type LotExport, type Member, type ViewKey } from '../../common/lotSpectra'
import { Grid, shortSeason } from '../../common/ui'
import { EXPORTER, FOLDER, LotNote, lotColor, NB, useView, type Lots } from './data'

type Region = 'ch' | 'co' | 'full'
type BandKey = '5-95' | '10-90' | '25-75' | 'none'
type Layout = 'pooled' | 'site'
const BAND_Q: Record<BandKey, [number, number] | null> = { '5-95': [0.05, 0.95], '10-90': [0.1, 0.9], '25-75': [0.25, 0.75], none: null }
const BAND_LABEL: Record<BandKey, string> = { '5-95': '5–95 %', '10-90': '10–90 %', '25-75': '25–75 %', none: 'None' }
const VIEW_LABEL: Record<ViewKey, string> = { raw: 'Raw', deriv: '2nd derivative', airspec: 'AIRSpec' }
const REGION_LABEL: Record<Region, string> = { ch: 'C–H 2800–3000', co: 'C=O 1450–1850', full: 'Full range' }

export interface PanelDefaults { view: ViewKey; region: Region; band: BandKey }

/** an instrument filter: a member scanned (in any of its analyses) on that instrument */
export const onInstrument = (m: Member, inst: string) => inst === 'all' || (m.instrument ?? '').split('/').includes(inst)
export const instrumentsOf = (members: readonly Member[]) => [...new Set(members.flatMap((m) => (m.instrument ? m.instrument.split('/') : [])))].sort()

/** median over columns of |centre A − centre B| relative to the lots' mean band width: how far apart two lots sit */
function gapOf(a: { mid: Float64Array; lo: Float64Array; hi: Float64Array }, b: typeof a): number {
  const r: number[] = []
  for (let j = 0; j < a.mid.length; j++) {
    const w = (a.hi[j] - a.lo[j] + b.hi[j] - b.lo[j]) / 2
    if (w > 0) r.push(Math.abs(a.mid[j] - b.mid[j]) / w)
  }
  r.sort((x, y) => x - y)
  return r.length ? r[Math.floor(r.length / 2)] : NaN
}

export function LotPanels({ exp, lots, id, prefix, defaults, figure, setPick }: {
  exp: LotExport; lots: Lots; id: string; prefix: string; defaults: PanelDefaults
  /** which char_12 figure this instance stands in for, for the provenance line */
  figure: string
  setPick: React.ReactNode
}) {
  const [view, setView] = useSearchParam<ViewKey>(`${prefix}.view`, defaults.view, VIEW_KEYS)
  const [regionPick, setRegion] = useSearchParam<Region>(`${prefix}.region`, defaults.region, ['ch', 'co', 'full'])
  const [centre, setCentre] = useSearchParam<'mean' | 'median'>(`${prefix}.centre`, 'mean', ['mean', 'median'])
  const [band, setBand] = useSearchParam<BandKey>(`${prefix}.band`, defaults.band, ['5-95', '10-90', '25-75', 'none'])
  const [layout, setLayout] = useSearchParam<Layout>(`${prefix}.layout`, 'pooled', ['pooled', 'site'])
  const [improve, setImprove] = useSearchParam<'off' | 'on'>(`${prefix}.improve`, 'off', ['off', 'on'])
  const seasonNames = lots.seasons.map((s) => s.name)
  const [seasonPick, setSeason] = useSearchParam<string>(`${prefix}.season`, '', seasonNames)
  const season = seasonNames.includes(seasonPick) ? seasonPick : defaultSeason(seasonNames)
  const instruments = useMemo(() => instrumentsOf(lots.members), [lots.members])
  const [instPick, setInst] = useSearchParam<string>(`${prefix}.inst`, 'all', ['all', ...instruments])
  const inst = instPick === 'all' || instruments.includes(instPick) ? instPick : 'all'
  const { values, error } = useView(exp, view)
  // the derivative ships only char_12's two pertinent regions
  const region: Region = view === 'deriv' && regionPick === 'full' ? 'ch' : regionPick
  const regions: Region[] = view === 'deriv' ? ['ch', 'co'] : ['ch', 'co', 'full']
  const v = exp.views[view]
  const [from, to] = region === 'full' ? [0, v.wn.length] : columnsIn(v.wn, ...exp.regions[region])
  const wn = useMemo(() => v.wn.slice(from, to), [v.wn, from, to])
  const q = BAND_Q[band]

  const shown = useMemo(() => lots.members.filter((m) => m.lot !== null && onInstrument(m, inst)), [lots.members, inst])
  const noLot = lots.members.filter((m) => m.lot === null && onInstrument(m, inst)).length
  const lotOrder = useMemo(() => lotsOf(shown).sort((a, b) => a - b), [shown])
  const groups = useMemo(() => {
    if (layout === 'pooled') return [{ key: 'all', label: [...new Set(shown.map((m) => lots.siteName(m.code)))].join(', '), members: shown }]
    return [...new Set(shown.map((m) => m.code))].map((code) => ({ key: code, label: lots.siteName(code), members: shown.filter((m) => m.code === code) }))
  }, [layout, shown, lots])

  const panels = useMemo(() => {
    if (!values) return []
    return groups.map((g) => {
      const held = layout === 'site' ? seasonAt(g.members, season) : season
      return { ...g, held, sub: [
      { key: 'all', title: 'All filters (lot and season confounded)', members: g.members },
      { key: 'season', title: `${shortSeason(held)} only (season held fixed)`, members: g.members.filter((m) => m.season === held) },
    ].map((s) => ({ ...s, lots: lotsOf(s.members).sort((a, b) => a - b).map((lot) => {
      const idx = s.members.filter((m) => m.lot === lot).map((m) => m.i)
      return { lot, n: idx.length, ...profile(values, v.points, idx, from, to, centre, idx.length >= 2 ? (q ?? [0.1, 0.9]) : null) }
    }) })) }
    })
  }, [values, groups, season, layout, v.points, from, to, centre, q])

  // the headline: in the first group, how far the two largest lots sit apart, over all filters and within the season
  const head = panels[0]
  const pair = head ? lotsOf(head.members).slice(0, 2).sort((a, b) => a - b) : []
  const gapIn = (k: 'all' | 'season') => {
    const s = head?.sub.find((x) => x.key === k)
    const a = s?.lots.find((l) => l.lot === pair[0]), b = s?.lots.find((l) => l.lot === pair[1])
    return a && b && a.n >= 2 && b.n >= 2 ? gapOf(a, b) : NaN
  }
  const gAll = gapIn('all'), gSeason = gapIn('season')
  const pctText = (x: number) => `${Math.round(100 * x)} %`
  const bandText = q ? BAND_LABEL[band] : '10–90 %'
  const title = pair.length === 2 && Number.isFinite(gAll) && Number.isFinite(gSeason)
    ? `${v.label} by PTFE lot — lots ${pair[0]} and ${pair[1]} sit a median ${pctText(gAll)} of their ${bandText} spread apart over all filters, ${pctText(gSeason)} within ${shortSeason(head.held)}${panels.length > 1 ? ` (${head.label})` : ''}`
    : `${v.label} by PTFE lot`

  const improveBand = (s: [number, number]): { band: SpectrumBand; line: SpectrumSeries } => ({
    band: { id: 'improve', wn, lo: exp.improve.deriv_p05.slice(...s), hi: exp.improve.deriv_p95.slice(...s), color: INK.muted, opacity: 0.12 },
    line: { id: 'improve-mean', wn, y: exp.improve.deriv_mean.slice(...s), color: INK.muted, label: `IMPROVE mean (n=${exp.improve.n})`, dash: '4 3', width: 1.2 },
  })
  const ref = view === 'deriv' && improve === 'on' ? improveBand([from, to]) : null
  const window: [number, number] | undefined = region === 'full' ? undefined : [exp.regions[region][1], exp.regions[region][0]]

  const controls = <>
    {setPick}
    <Segmented label="Spectra" value={view} options={VIEW_KEYS} onChange={setView} optionLabel={(o) => VIEW_LABEL[o]} title={v.binning} />
    <Segmented label="Region" value={region} options={regions} onChange={setRegion} optionLabel={(o) => REGION_LABEL[o]} />
    {seasonNames.length > 1 && <Select label="Hold fixed" value={season} options={seasonNames} onChange={setSeason} optionLabel={shortSeason} title="The season the right-hand panel keeps (char_12: the dry season)" />}
    <Segmented label="Centre" value={centre} options={['mean', 'median'] as const} onChange={setCentre} />
    <Segmented label="Band" value={band} options={['5-95', '10-90', '25-75', 'none'] as const} onChange={setBand} optionLabel={(o) => BAND_LABEL[o]} />
    <Segmented label="Layout" value={layout} options={['pooled', 'site'] as const} onChange={setLayout} optionLabel={(o) => (o === 'pooled' ? 'Sites pooled' : 'Per site')} />
    {instruments.length > 1 && <Select label="FTIR" value={inst} options={['all', ...instruments]} onChange={setInst} optionLabel={(o) => (o === 'all' ? 'Every instrument' : `Instrument ${o}`)} title="SPARTAN scans carry the spectrometer's serial number (Addis's export does not, so it drops out of a single instrument)" />}
    {view === 'deriv' && <Segmented label="IMPROVE" value={improve} options={['off', 'on'] as const} onChange={setImprove} optionLabel={(o) => (o === 'on' ? '5–95 % shown' : 'Hidden')} />}
  </>

  return (
    <ChartFrame id={id} title={title} controls={controls}
      subtitle={`${v.label} (${v.unit}), ${centre} of each PTFE lot (line) and ${q ? `the ${BAND_LABEL[band]} of its filters (shaded)` : 'no band'}, per wavenumber, over the filters shown. Left: all of them, where lot and season are confounded (at Addis lot 248 is mostly dry-season); right: ${shortSeason(season)} only, the controlled contrast. If the substrate mattered, the curves on the right would separate. char_12 draws the mean, with 5–95 % (04) or 10–90 % (04b).`}
      tip={<>The headline gap is the median, over the wavenumbers shown, of |centre A − centre B| divided by the lots' mean band width (10–90 % when the band is off). {v.binning}. Lots with one filter draw a line and no band. Per site, a site whose calendar has no season of that name holds its own dry-named season (or its largest) instead; pooled, the season applies as named, so under per-site calendars it may cover one site only. Instruments: SPARTAN lot 248 was scanned on instruments 823 and 2837 only, never on 151, so a pooled lot contrast at Beijing or JPL is partly an instrument contrast; pick one instrument to remove that.</>}
      provenance={<>{NB} §4, {figure}. Spectra: <code>{EXPORTER}</code> → <code>{FOLDER}/{v.file}</code>; centres and bands recomputed here (common/lotSpectra.ts).</>}>
      {error ? <Empty>Could not load the {v.label.toLowerCase()} spectra: {error}</Empty>
        : !values ? <Empty>Loading the spectra…</Empty>
        : !shown.length ? <Empty>No exported spectrum with a known lot among the filters shown{lots.set === 'view' && lots.unmatched ? ` (${lots.unmatched} spectra of these sites have no gallery row: pick “In view + no gallery row”)` : ''}.</Empty>
        : panels.map((g) => <div key={g.key}>
          {layout === 'site' && <p className="subtitle" style={{ margin: '8px 0 2px', fontWeight: 600 }}>{g.label}</p>}
          <Grid min={320}>
            {g.sub.map((s) => {
              const series: SpectrumSeries[] = s.lots.map((l) => ({ id: `${g.key}-${s.key}-${l.lot}`, wn, y: l.mid, color: lotColor(l.lot, lotOrder.indexOf(l.lot)), label: `lot ${l.lot} (n=${l.n})`, width: 2.2 }))
              const bands: SpectrumBand[] = q ? s.lots.filter((l) => l.n >= 2).map((l) => ({ id: `${g.key}-${s.key}-${l.lot}-b`, wn, lo: l.lo, hi: l.hi, color: lotColor(l.lot, lotOrder.indexOf(l.lot)), opacity: 0.15 })) : []
              return <SpectrumPlot key={s.key} title={<>{s.title}: {s.lots.length ? s.lots.map((l) => `lot ${l.lot} n=${l.n}`).join(', ') : 'no filters'}</>}
                series={ref ? [ref.line, ...series] : series} bands={ref ? [ref.band, ...bands] : bands} window={window}
                yLabel={v.unit} hover="readout" zeroLine={view === 'deriv'} height={290} />
            })}
          </Grid>
        </div>)}
      <Key items={[...lotOrder.map((l, i) => ({ label: `lot ${l}`, color: lotColor(l, i), kind: 'line' as const })),
        ...(q ? [{ label: `${BAND_LABEL[band]} of filters`, color: INK.muted, kind: 'faded' as const }] : []),
        ...(ref ? [{ label: `IMPROVE mean and 5–95 % (n=${exp.improve.n})`, color: INK.muted, kind: 'dash' as const }] : [])]} />
      <LotNote lots={lots} of={(m) => m.lot !== null && onInstrument(m, inst)} />
      {(noLot > 0 || lots.unmatched > 0) && <Note>
        {noLot > 0 && <>{noLot} filter{noLot === 1 ? '' : 's'} shown {noLot === 1 ? 'has' : 'have'} no lot in the source (Addis spectra the lot map does not cover) and {noLot === 1 ? 'is' : 'are'} left out. </>}
        {lots.set === 'view' && lots.unmatched > 0 && <>{lots.unmatched} exported spectra of these sites have no gallery row; “In view + no gallery row” adds them.</>}
      </Note>}
    </ChartFrame>
  )
}
