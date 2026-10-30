/**
 * char_12 §4b's band table (lot_bands_within_dry.csv), for the filters in view: each filter's mean
 * area-normalised AIRSpec absorbance in each charcoal_spectra band, the median per lot within one season, and
 * the Mann–Whitney test between the two largest lots.
 */
import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Select } from '@/components/ChartFrame'
import { fmt } from '@/lib/format'
import { useSearchParam } from '@/lib/url'
import { defaultSeason, lotContrast, lotsOf, type LotExport } from '../../common/lotSpectra'
import { DataTable, pText, shortSeason } from '../../common/ui'
import { EXPORTER, FOLDER, LotNote, lotColor, NB, type Lots } from './data'

export function BandMeans({ exp, lots, setPick }: { exp: LotExport; lots: Lots; setPick: React.ReactNode }) {
  const seasonNames = lots.seasons.map((s) => s.name)
  const [seasonPick, setSeason] = useSearchParam<string>('ls.b.season', '', seasonNames)
  const season = seasonNames.includes(seasonPick) ? seasonPick : defaultSeason(seasonNames)
  const codes = useMemo(() => [...new Set(lots.members.map((m) => m.code))], [lots.members])
  const [sitePick, setSite] = useSearchParam<string>('ls.b.site', 'all', ['all', ...codes])
  const site = sitePick === 'all' || codes.includes(sitePick) ? sitePick : 'all'
  const ms = lots.members.filter((m) => m.lot !== null && m.season === season && (site === 'all' || m.code === site))
  const pair = lotsOf(ms).slice(0, 2).sort((a, b) => a - b)
  const saved = new Map((exp.check.saved.lot_bands ?? []).map((r) => [r.band, r]))
  const rows = Object.entries(exp.bands).map(([band, w]) => {
    const v = exp.band_means[band]
    const c = pair.length === 2 ? lotContrast(ms.map((m) => ({ lot: m.lot, escape: v[m.i] })), pair[0], pair[1]) : null
    return { band, w, c, nb: saved.get(band) }
  })
  const sig = rows.filter((r) => r.c?.test && r.c.test.p < 0.05)
  const title = pair.length < 2 ? 'Band means by lot'
    : sig.length ? `Within ${shortSeason(season)}, lots ${pair[0]} and ${pair[1]} differ in ${sig.length} of ${rows.length} bands (${sig.map((r) => r.band).join(', ')}; p < 0.05)`
    : `Within ${shortSeason(season)}, lots ${pair[0]} and ${pair[1]} do not differ in any band (p ≥ 0.05)`
  return (
    <ChartFrame id="ls-bands" title={title}
      controls={<>{setPick}
        {seasonNames.length > 1 && <Select label="Season" value={season} options={seasonNames} onChange={setSeason} optionLabel={shortSeason} />}
        {codes.length > 1 && <Select label="Site" value={site} options={['all', ...codes]} onChange={setSite} optionLabel={(o) => (o === 'all' ? 'Sites pooled' : lots.siteName(o))} />}
      </>}
      subtitle="The AIRSpec-baselined, clipped, unit-area spectra (as 04b) reduced to each filter's mean value in char_10's bands (×10⁻⁴; charcoal_spectra.band_area on the unbinned spectrum), median per lot within one season, Mann–Whitney between the two largest lots. char_12 finds two bands separating mildly within the Addis dry season (lot 248 ~14 % higher in O–H/N–H, ~7 % lower in aliphatic C–H), an order of magnitude below the seasonal contrasts, and notes the two lots were not deployed over the same dates."
      provenance={<>{NB} §4b, lot_bands_within_dry.csv. Band means: <code>{EXPORTER}</code> → <code>{FOLDER}/lot_spectra.json</code> (band_means); medians and tests recomputed here.</>}>
      {pair.length < 2 ? <Empty>Fewer than two lots among the {shortSeason(season)} filters shown.</Empty> : <>
        <DataTable rows={rows} rowKey={(r) => r.band} columns={[
          { key: 'b', label: 'Band', value: (r) => r.band },
          { key: 'w', label: 'Window (cm⁻¹)', value: (r) => `${r.w[0]}–${r.w[1]}` },
          { key: 'a', label: <span style={{ color: lotColor(pair[0]) }}>Lot {pair[0]} median (n {rows[0].c?.na ?? 0})</span>, num: true, value: (r) => fmt(r.c?.medianA, 2) },
          { key: 'c', label: <span style={{ color: lotColor(pair[1]) }}>Lot {pair[1]} median (n {rows[0].c?.nb ?? 0})</span>, num: true, value: (r) => fmt(r.c?.medianB, 2) },
          { key: 'p', label: 'Mann–Whitney p', num: true, value: (r) => pText(r.c?.test?.p ?? NaN) },
          { key: 'nb', label: 'char_12 (248 / 251, p)', num: true, value: (r) => (r.nb ? `${fmt(r.nb.median248, 2)} / ${fmt(r.nb.median251, 2)}, ${pText(r.nb.p)}` : '—') },
        ]} />
        <LotNote lots={lots} of={(m) => m.lot !== null && m.season === season && (site === 'all' || m.code === site)} />
        <Note>The last column is char_12's table (its Addis dry-season filters: 27 of lot 248, 46 of lot 251); “{"char_12's 239 (Addis)"}” under Filters, with the dry season, reproduces it.</Note>
      </>}
    </ChartFrame>
  )
}
