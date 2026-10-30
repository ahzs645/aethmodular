/**
 * char_12 §4's escape score by lot × season (04_lot_effect, right; lot_effect.csv), for the filters in view: box
 * statistics per lot and season, the lot contrast over all filters and within one season with its Mann–Whitney
 * test, and the lot × season × instrument counts that say how confounded the contrast is.
 */
import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import { boxStats, defaultSeason, lotContrast, lotsOf, seasonAt, type LotExport, type Member } from '../../common/lotSpectra'
import { DataTable, pText, shortSeason } from '../../common/ui'
import { EXPORTER, FOLDER, lotColor, NB, type Lots } from './data'

type Split = 'held' | 'each'

export function EscapeByLot({ exp, lots, setPick }: { exp: LotExport; lots: Lots; setPick: React.ReactNode }) {
  const [split, setSplit] = useSearchParam<Split>('ls.e.split', 'held', ['held', 'each'])
  const [layout, setLayout] = useSearchParam<'pooled' | 'site'>('ls.e.layout', 'site', ['pooled', 'site'])
  const seasonNames = lots.seasons.map((s) => s.name)
  const [seasonPick, setSeason] = useSearchParam<string>('ls.e.season', '', seasonNames)
  const season = seasonNames.includes(seasonPick) ? seasonPick : defaultSeason(seasonNames)
  const shown = useMemo(() => lots.members.filter((m) => m.lot !== null), [lots.members])
  const lotOrder = useMemo(() => lotsOf(shown).sort((a, b) => a - b), [shown])
  const groups = useMemo(() => layout === 'pooled'
    ? [{ key: 'all', label: [...new Set(shown.map((m) => lots.siteName(m.code)))].join(', '), members: shown }]
    : [...new Set(shown.map((m) => m.code))].map((code) => ({ key: code, label: lots.siteName(code), members: shown.filter((m) => m.code === code) })), [layout, shown, lots])

  // per site, a site whose calendar lacks the picked season holds its own (common/lotSpectra seasonAt)
  const heldAt = (ms: Member[]) => (layout === 'site' ? seasonAt(ms, season) : season)
  const seasonGroups = (ms: Member[], held: string) => split === 'each'
    ? lots.seasons.map((s) => ({ key: s.name, label: shortSeason(s.name), members: ms.filter((m) => m.season === s.name) }))
    : [{ key: 'in', label: shortSeason(held), members: ms.filter((m) => m.season === held) },
      { key: 'out', label: `not ${shortSeason(held)}`, members: ms.filter((m) => m.season !== held) }]

  const blocks: IntervalBlock[] = groups.map((g) => ({ key: g.key, title: g.label, color: layout === 'site' ? lots.siteColor(g.key) : undefined,
    rows: lotsOf(g.members).sort((a, b) => a - b).flatMap((lot) => seasonGroups(g.members.filter((m) => m.lot === lot), heldAt(g.members)).filter((s) => s.members.length).map((s) => {
      const b = boxStats(s.members.map((m) => m.escape)), color = lotColor(lot, lotOrder.indexOf(lot))
      return { key: `${g.key}-${lot}-${s.key}`, label: `lot ${lot} · ${s.label}`, labelColor: color, note: `n=${b.n}`,
        tip: [`${g.label} · lot ${lot} · ${s.label}`, `median ${fmt(b.median, 3)} (IQR ${fmt(b.q1, 3)}–${fmt(b.q3, 3)})`, `whiskers ${fmt(b.wlo, 3)}–${fmt(b.whi, 3)}${b.fliers.length ? `, ${b.fliers.length} beyond` : ''}`, `n ${b.n}`],
        marks: [[{ mid: b.median, lo: b.wlo, hi: b.whi, color, faint: true, r: 0 }, { mid: b.median, lo: b.q1, hi: b.q3, color, shape: 'square' as const, r: 4 }]] }
    })) })).filter((b) => b.rows.length)

  // the lot contrast, char_12's lot_effect table: the two largest lots, over all filters and within the season
  const table = groups.flatMap((g) => {
    const [a, b] = lotsOf(g.members).slice(0, 2).sort((x, y) => x - y)
    if (b === undefined) return []
    const held = heldAt(g.members)
    return [{ k: 'all', subset: 'All filters', ms: g.members }, { k: 'in', subset: `${shortSeason(held)} only`, ms: g.members.filter((m) => m.season === held) }]
      .map((s) => ({ key: `${g.key}-${s.k}`, site: g.label, subset: s.subset, a, b, ...lotContrast(s.ms, a, b) }))
      .filter((r) => r.na + r.nb > 0)
  })
  const head = table.find((r) => r.a === 248 && r.b === 251 && r.subset === 'All filters') ?? table[0]
  const headIn = head && table.find((r) => r.site === head.site && r.subset !== 'All filters')
  const pa = head?.test?.p ?? NaN, pi = headIn?.test?.p ?? NaN
  const title = head && headIn
    ? pa < 0.05 && pi >= 0.05
      ? `Season, not lot, moves the escape score${groups.length > 1 ? ` at ${head.site}` : ''}: lots ${head.a} and ${head.b} differ over all filters (p ${pText(pa)}) but not within ${headIn.subset.replace(/ only$/, '')} (p ${pText(pi)})`
      : `Escape score by lot × season${groups.length > 1 ? ` — ${head.site}` : ''}: lots ${head.a} vs ${head.b}, p ${pText(pa)} over all filters, p ${pText(pi)} within ${headIn.subset.replace(/ only$/, '')}`
    : 'Escape score by lot × season'
  const saved = exp.check.saved.lot_effect ?? []

  // how confounded: filters per lot, season and instrument
  const cross = groups.flatMap((g) => lotsOf(g.members).sort((a, b) => a - b).map((lot) => {
    const ms = g.members.filter((m) => m.lot === lot)
    const inst = new Map<string, number>()
    for (const m of ms) for (const i of (m.instrument ?? '').split('/').filter(Boolean)) inst.set(i, (inst.get(i) ?? 0) + 1)
    return { key: `${g.key}-${lot}`, site: g.label, lot, n: ms.length, bySeason: lots.seasons.map((s) => ({ ...s, n: ms.filter((m) => m.season === s.name).length })).filter((s) => s.n),
      inst: [...inst.entries()].sort().map(([i, n]) => `${i}: ${n}`).join(', ') || '—' }
  }))

  return (
    <ChartFrame id="ls-escape" title={title}
      controls={<>{setPick}
        {seasonNames.length > 1 && <Select label="Season" value={season} options={seasonNames} onChange={setSeason} optionLabel={shortSeason} title="The season held fixed (char_12: the dry season)" />}
        <Segmented label="Split" value={split} options={['held', 'each'] as const} onChange={setSplit} optionLabel={(o) => (o === 'held' ? `${shortSeason(season)} vs the rest` : 'Each season')} />
        <Segmented label="Layout" value={layout} options={['site', 'pooled'] as const} onChange={setLayout} optionLabel={(o) => (o === 'pooled' ? 'Sites pooled' : 'Per site')} />
      </>}
      subtitle={`char_12's envelope-escape score: the share of the C–H (2800–3000) and C=O / aromatic (1450–1850 cm⁻¹) wavenumbers where a filter's second-derivative spectrum leaves the 5–95 % envelope of ${exp.improve.n} IMPROVE spectra; an IMPROVE filter escapes about 10 % of the time (dashed line; the IMPROVE median is ${fmt(exp.improve.escape_median, 3)}). Per PTFE lot and season: square at the median, bar the interquartile range, thin line the 1.5 × IQR whiskers (char_12 draws box plots). char_12 scored Addis only; the other sites' scores are this export's, with the same processing and envelope.`}
      tip={<>The scores are recomputed by the exporter from the raw spectra exactly as char_12 §3 does (char_12 saves no per-filter table); on char_12's 239 Addis filters they reproduce its lot_effect.csv and escape_by_group.csv to the last digit. SPARTAN sites are scanned on three spectrometers and lot 248 was never scanned on instrument 151, so at Beijing and JPL a lot contrast is partly an instrument contrast (the counts table below). Mann–Whitney is two-sided, scipy's normal approximation.</>}
      provenance={<>{NB} §3–4, figure 04 (right) and lot_effect.csv. Scores: <code>{EXPORTER}</code> → <code>{FOLDER}/lot_spectra.json</code>; boxes and tests recomputed here (common/lotSpectra.ts).</>}>
      {!blocks.length ? <Empty>No filter with a known lot among the filters shown.</Empty> : <>
        <IntervalStrip blocks={blocks} labelWidth={150} noteWidth={50} rowHeight={22}
          panels={[{ label: 'Envelope-escape score (share of pertinent wavenumbers)', reference: 0.1, marker: { value: exp.improve.escape_median, label: 'IMPROVE median' } }]} />
        <Key items={[...lotOrder.map((l, i) => ({ label: `lot ${l}`, color: lotColor(l, i), kind: 'square' as const })), { label: 'null for a calibration filter (0.10)', color: INK.axis, kind: 'line' }]} />
        <DataTable rows={table} rowKey={(r) => r.key} columns={[
          ...(groups.length > 1 || layout === 'site' ? [{ key: 's', label: 'Site', value: (r: (typeof table)[number]) => r.site }] : []),
          { key: 'u', label: 'Filters', value: (r) => r.subset },
          { key: 'n', label: 'n (lot A / B)', num: true, value: (r) => `${r.na} / ${r.nb}` },
          { key: 'ma', label: 'Median escape, lot A', num: true, value: (r) => <><span style={{ color: lotColor(r.a, lotOrder.indexOf(r.a)) }}>{r.a}</span> {fmt(r.medianA, 3)}</> },
          { key: 'mb', label: 'Median escape, lot B', num: true, value: (r) => <><span style={{ color: lotColor(r.b, lotOrder.indexOf(r.b)) }}>{r.b}</span> {fmt(r.medianB, 3)}</> },
          { key: 'p', label: 'Mann–Whitney p', num: true, value: (r) => pText(r.test?.p ?? NaN) },
        ]} />
        {saved.length > 0 && <Note>char_12 (its 239 Addis filters, Dry Oct–Feb): {saved.map((r) => `${r.subset} — lot 248 ${fmt(r.median248, 3)} (n ${r.n248}) vs 251 ${fmt(r.median251, 3)} (n ${r.n251}), p ${pText(r.p ?? NaN)}`).join('; ')}. “{"char_12's 239 (Addis)"}” under Filters reproduces these.</Note>}
        <p className="subtitle" style={{ margin: '10px 0 4px' }}>Filters per lot and season, and the spectrometers they were scanned on (Addis's export names none):</p>
        <DataTable rows={cross} rowKey={(r) => r.key} columns={[
          ...(groups.length > 1 || layout === 'site' ? [{ key: 's', label: 'Site', value: (r: (typeof cross)[number]) => r.site }] : []),
          { key: 'l', label: 'Lot', value: (r) => <span style={{ color: lotColor(r.lot, lotOrder.indexOf(r.lot)) }}>{r.lot}</span> },
          { key: 'n', label: 'n', num: true, value: (r) => r.n },
          { key: 'se', label: 'By season', value: (r) => r.bySeason.map((s, i) => <span key={s.name}>{i ? ', ' : ''}<span style={{ color: s.color }}>{shortSeason(s.name)}</span> {s.n}</span>) },
          { key: 'i', label: 'Instrument: n', value: (r) => r.inst },
        ]} />
      </>}
    </ChartFrame>
  )
}
