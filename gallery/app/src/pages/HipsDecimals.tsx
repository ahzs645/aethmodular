import { useMemo, type ReactNode } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Segmented } from '@/components/Controls'
import { LoadState } from '@/components/ExportState'
import type { MeetingBundle } from '@/components/Figure'
import { Key } from '@/components/Key'
import { Marker } from '@/components/RefLines'
import { ScatterPlot, type ScatterPoint, type ScatterScales } from '@/components/ScatterPlot'
import { ScrubField } from '@/components/ScrubField'
import { claimOf, digitSplit, placeName, type Claim } from '@/lib/claimDigits'
import { useExport } from '@/lib/data'
import { placePointLabels } from '@/lib/fitLabel'
import { fmt, pct } from '@/lib/format'
import { correction, noiseGain } from '@/lib/hipsLinearise'
import { FixedSetNote } from '@/lib/markedRows'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { rowsOf } from '@/lib/table'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { Ceil, Linearise } from '@/pages/HipsLinearisePage'
import { DataTable, LINEAR, Readouts, STAT_NAME, useStat } from '@/pages/hipsLineariseParts'

/**
 * How many decimal places a linearised HIPS BC can carry (the "Linearising HIPS at Addis" page). Two charts on one
 * idea: the last digit worth writing is the one the uncertainty starts in (lib/claimDigits), so an uncertainty axis
 * on a log scale is a ladder of decimal places.
 *
 *  - SiteDecimals: every SPARTAN site's median (or mean) BC, with the range the IMPROVE ceiling's 90 % interval puts
 *    on it, from the 11 Nov procedure's saved site statistics
 *  - FilterDecimals: every Addis filter, the ceiling's range beside the reading noise the correction multiplies,
 *    recomputed in the browser at the page's ceiling
 */
const PROCEDURE = 'meetings/hips-procedure-20261111.json'
const PROCEDURE_COMMAND = 'uv run python gallery/data/export_meeting_20261111_procedure.py'
const PROCEDURE_SOURCE = <>gallery/data/export_meeting_20261111_procedure.py (research/ftir_hips_chem/scripts/hips_loading.py); tables <code>traced</code> and <code>found</code> in <code>{PROCEDURE}</code>. Rounding rule: <code>gallery/app/src/lib/claimDigits.ts</code>.</>
const FILTER_SOURCE = <>research/ftir_hips_chem/workflows/run_addis_reference_closure_20261110.py; per-filter optical depths in <code>hips_linearise/linearise.json</code>; detection limit from <code>docs/filter-optics-reference.md</code>. Rounding rule: <code>gallery/app/src/lib/claimDigits.ts</code>.</>

/** HIPS BC is Fabs ÷ 10 with Fabs given to 0.1 Mm⁻¹, so two decimals is the finest it is reported to */
const FINEST = -2
const NOISE = RESEARCH_SERIES[2]
/** the laboratory's median HIPS detection limit, Mm⁻¹ (docs/filter-optics-reference.md) */
const DETECTION_LIMIT = 1.6

const SITE_VIEWS = ['lost', 'all'] as const
/** sites named on the chart whatever band they fall in: Addis, Delhi and the lightest large site */
const ALWAYS_NAMED = new Set(['ETAD', 'INDH', 'USPA'])

interface Traced { key: string; reported: number | null; linearised: number | null; lo: number | null; hi: number | null; code: string; site: string }
interface Found { code: string; filters: number; median_tau: number; expected_under_read: number }
interface SiteRow extends Claim { code: string; site: string; name: string; tau: number; filters: number; underRead: number; reported: number; linearised: number; lo: number; hi: number }

/** "A", "A and B", "A, B and C" */
const listed = (names: string[]) => (names.length < 2 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`)

/** a computed value with the digits its range does not support in faint ink */
function Digits({ value, place }: { value: number; place: number }) {
  const [kept, rest] = digitSplit(value, place, -FINEST)
  return <span style={{ whiteSpace: 'nowrap' }}>{kept}<span style={{ color: INK.muted, opacity: 0.75, fontWeight: 400 }}>{rest}</span></span>
}

/**
 * The decimal places a log uncertainty axis stands for: a dashed line at each power of ten, the band above it named
 * for the last digit a value that uncertain is written to, and the bands that leave no decimal at all shaded.
 */
function PlaceBands({ y, iw }: Pick<ScatterScales, 'y' | 'iw'>) {
  const [lo, hi] = y.domain()
  const top = Math.ceil(Math.log10(hi) - 1e-9) - 1
  const places = Array.from({ length: Math.max(0, top - FINEST + 1) }, (_, i) => top - i)
  // uncertainty grows with optical depth on both charts: dots leave the top-left corner of a band free, and the whole
  // right of the lowest band, which the light sites cross on the left
  const narrow = iw < 480
  return <g pointerEvents="none">
    {places.map((place) => {
      const upper = Math.min(hi, 10 ** (place + 1)), lower = place === FINEST ? lo : Math.max(lo, 10 ** place)
      if (!(upper > lower)) return null
      const y0 = y(upper), y1 = y(lower)
      return <g key={place}>
        {place >= 0 && <rect x={0} y={y0} width={iw} height={y1 - y0} fill={INK.shade} fillOpacity={0.75} />}
        {upper < hi && <Marker orient="h" at={y0} from={0} to={iw} color={INK.axis} opacity={0.6} />}
        <text x={place === FINEST ? iw - 8 : 8} y={y0 + 14} textAnchor={place === FINEST ? 'end' : 'start'} fontSize={10.5} fontFamily={FONT.mono} fill={INK.muted}>
          {placeName(place)}{place === FINEST && !narrow ? ', as reported' : ''}</text>
      </g>
    })}
  </g>
}

// ------------------------------------------------------------------ every site: what its BC can be written to
export function SiteDecimals() {
  const { data, error } = useExport<MeetingBundle>(PROCEDURE, (b) => (b.tables?.traced && b.tables?.found ? null : 'no "traced" or "found" table'))
  const [stat] = useStat()
  const [view, setView] = useSearchParam<(typeof SITE_VIEWS)[number]>('hl.digits', 'lost', SITE_VIEWS)
  const sites = useMemo<SiteRow[]>(() => {
    if (!data) return []
    const found = new Map(rowsOf<Found>(data.tables.found).map((f) => [f.code, f]))
    return rowsOf<Traced>(data.tables.traced).flatMap((r) => {
      const f = found.get(r.code)
      if (r.key !== stat || !f || r.reported == null || r.linearised == null || r.lo == null || r.hi == null) return []
      return [{ code: r.code, site: r.site, name: r.site.split(',')[0], tau: f.median_tau, filters: f.filters, underRead: f.expected_under_read,
        reported: r.reported, linearised: r.linearised, lo: r.lo, hi: r.hi, ...claimOf(r.linearised, r.lo, r.hi, FINEST) }]
    }).sort((a, b) => b.uncertainty - a.uncertainty)
  }, [data, stat])
  const points = useMemo<ScatterPoint[]>(() => sites.map((s) => ({
    key: s.code, x: s.tau, y: s.uncertainty, color: LINEAR, r: s.code === 'ETAD' ? 6.5 : 4.5, opacity: s.code === 'ETAD' ? 1 : 0.8,
    tip: [s.site, `median optical depth ${fmt(s.tau, 2)} · ${s.filters} filters`, `as reported ${fmt(s.reported, 2)} µg/m³`,
      `linearised ${fmt(s.linearised, 2)}, range ${fmt(s.lo, 2)} to ${fmt(s.hi, 2)} (± ${fmt(s.uncertainty, 2)})`, `write it as ${s.text}`],
  })), [sites])
  if (error || !data) return (
    <ChartFrame id="hl-decimals" title="How many decimal places a linearised HIPS BC can carry" provenance={PROCEDURE_SOURCE}>
      <LoadState what="the 11 Nov procedure export" error={error} command={PROCEDURE_COMMAND} />
    </ChartFrame>
  )
  const name = STAT_NAME[stat]
  const whole = sites.filter((s) => s.place >= 0), tenths = sites.filter((s) => s.place === -1)
  const lost = sites.filter((s) => s.place > FINEST)
  const addis = sites.find((s) => s.code === 'ETAD')
  const named = sites.filter((s) => s.place >= 0 || ALWAYS_NAMED.has(s.code)).slice(0, 8)
  // where a site's dot is drawn: one below the axis is pinned to its foot
  const at = (x: ScatterScales['x'], y: ScatterScales['y']) => (s: SiteRow) => ({ key: s.code, text: s.name, x: x(s.tau), y: y(Math.max(y.domain()[0], s.uncertainty)) })
  const labels = ({ x, y, iw, ih }: ScatterScales) => <g pointerEvents="none">
    {placePointLabels(named.map(at(x, y)), iw, ih, { weight: 400, radius: 7, avoid: sites.filter((s) => !named.includes(s)).map(at(x, y)) }).map((l) => <g key={l.key}>
      {l.connector && <line {...l.connector} stroke={INK.muted} strokeWidth={1} />}
      <text x={l.x} y={l.y} textAnchor={l.anchor} fontSize={10.5} fontFamily={FONT.mono} fill={INK.text}>{l.text}</text>
    </g>)}
  </g>
  const title = whole.length
    ? `${name} BC: ${whole.length > 5 ? `${whole.length} sites` : listed(whole.map((s) => s.name))} can be written to whole numbers only`
    : `${name} BC: every site keeps at least one decimal`
  const shown = view === 'all' ? sites : lost
  return (
    <ChartFrame id="hl-decimals" title={`${title} — uncertainty of each site's linearised HIPS BC against its optical depth`}
      subtitle={`Each dot is one SPARTAN site. Its height is half the range that the IMPROVE ceiling's 90 % interval (1.82 to 2.70) puts on the site's linearised ${name.toLowerCase()} BC. The last digit worth writing is the one that uncertainty starts in: 1 µg/m³ or more leaves whole numbers, 0.1 to 1 leaves one decimal. HIPS BC is reported to two decimals, so anything above the lowest band has lost a digit.`}
      provenance={PROCEDURE_SOURCE}
      controls={<>
        <Segmented label="Table" value={view} options={SITE_VIEWS} onChange={setView}
          optionLabel={(o) => (o === 'lost' ? `Sites that lose a digit (${lost.length})` : `All (${sites.length})`)} />
      </>}>
      <ScatterPlot points={points} xLabel="Site median HIPS optical depth" yLabel="± on linearised BC (µg/m³)" yScale="log"
        xDomain={[0, 1.05]} yDomain={[0.001, 10]} nice={false} height={340} outside="pin" note={false}
        under={({ y, iw }) => <PlaceBands y={y} iw={iw} />}>{labels}</ScatterPlot>
      {addis && <Readouts items={[
        { label: 'Addis, as reported', value: fmt(addis.reported, 2), sub: `µg/m³ · filters read ${pct(addis.underRead)} low` },
        { label: 'Linearised', value: <Digits value={addis.linearised} place={addis.place} />, sub: 'faint = inside the range' },
        { label: 'Range', value: `${fmt(addis.lo, 2)} to ${fmt(addis.hi, 2)}`, sub: `± ${fmt(addis.uncertainty, 2)} µg/m³` },
        { label: 'Write it as', value: addis.text, sub: placeName(addis.place) },
      ]} />}
      <DataTable rows={shown} rowKey={(s) => s.code} highlight={(s) => s.code === 'ETAD'} columns={[
        { key: 's', label: 'Site', value: (s) => s.site },
        { key: 'r', label: 'As reported', num: true, value: (s) => fmt(s.reported, 2) },
        { key: 'w', label: 'Write it as', num: true, value: (s) => <strong>{s.text}</strong> },
        { key: 'l', label: 'As computed', num: true, value: (s) => <Digits value={s.linearised} place={s.place} /> },
        { key: 'g', label: 'Range', num: true, value: (s) => `${fmt(s.lo, 2)} to ${fmt(s.hi, 2)}` },
        { key: 'u', label: '±', num: true, value: (s) => fmt(s.uncertainty, 2) },
        { key: 't', label: 'Optical depth', num: true, value: (s) => fmt(s.tau, 2) },
      ]} />
      <Note>All in µg/m³. {name} BC, linearised: {whole.length ? `${listed(whole.map((s) => s.name))} to whole numbers` : 'no site to whole numbers only'}, {tenths.length} sites to one decimal, and {sites.length - lost.length} keep both decimals. This is the least each number loses. The range covers the ceiling's 90 % interval and nothing else: not the 2015–2025 ceiling (1.71, below that interval), not the MAC, and not whether IMPROVE's curve holds on SPARTAN filters. As reported, every site keeps both decimals, and the dark sites are precisely low.</Note>
      <FixedSetNote what="the 11 Nov procedure's saved site statistics (every SPARTAN filter with HIPS and a sampled volume)" />
    </ChartFrame>
  )
}

// ------------------------------------------------------------------ one Addis filter: the ceiling beside the noise
export function FilterDecimals({ data, ceil }: { data: Linearise; ceil: Ceil }) {
  const [noiseText, setNoise] = useSearchParam<string>('hl.noise', String(DETECTION_LIMIT))
  const noise = Math.min(10, Math.max(0.1, Number(noiseText) || DETECTION_LIMIT))
  const c = ceil.ceiling
  const lo = data.ceilings.find((k) => k.key === 'lo')!.tau_inf, hi = data.ceilings.find((k) => k.key === 'hi')!.tau_inf
  const rows = useMemo(() => [...data.filters].sort((p, q) => p.tau - q.tau).flatMap((f) => {
    const value = f.hips_bc * correction(f.tau, c)
    // the ceiling's interval is lopsided (a lower ceiling raises the value more than a higher one lowers it)
    const top = f.hips_bc * correction(f.tau, lo), foot = f.hips_bc * correction(f.tau, hi)
    const fromCeiling = (top - foot) / 2
    const fromNoise = (noise / data.mac) * noiseGain(f.tau, c)
    // each side of the written range: the ceiling's reach on that side and the noise, in quadrature
    const down = Math.hypot(Math.max(0, value - foot), fromNoise), up = Math.hypot(Math.max(0, top - value), fromNoise)
    return [value, fromCeiling, fromNoise].every(Number.isFinite) ? [{ f, value, fromCeiling, fromNoise, ...claimOf(value, value - down, value + up, FINEST) }] : []
  }), [data, c, lo, hi, noise])
  const points = useMemo<ScatterPoint[]>(() => rows.flatMap((r) => {
    const tip = [`${r.f.id} · ${r.f.season}`, `optical depth ${fmt(r.f.tau, 2)} · as reported ${fmt(r.f.hips_bc, 2)} µg/m³`, `linearised ${fmt(r.value, 2)} µg/m³`,
      `± ${fmt(r.fromCeiling, 2)} from the ceiling · ± ${fmt(r.fromNoise, 2)} from noise`, `write it as ${r.text}`]
    return [
      { id: r.f.id, key: `${r.f.id}-noise`, x: r.f.tau, y: r.fromNoise, color: NOISE, hollow: true, tip },
      { id: r.f.id, key: `${r.f.id}-ceiling`, x: r.f.tau, y: r.fromCeiling, color: LINEAR, tip },
    ]
  }), [rows])
  if (!rows.length) return null
  const mid = rows[Math.floor(rows.length / 2)], dark = rows[rows.length - 1]
  const ceilingLarger = rows.filter((r) => r.fromCeiling > r.fromNoise).length
  const whole = rows.filter((r) => r.place >= 0).length
  const title = ceilingLarger > rows.length / 2
    ? 'On one Addis filter the borrowed ceiling costs more digits than the reading noise'
    : 'At this reading noise, noise costs one Addis filter more digits than the ceiling'
  const value = (v: number): ReactNode => `± ${fmt(v, 2)}`
  return (
    <ChartFrame id="hl-decimals-filters" title={`${title} — uncertainty of each filter's linearised HIPS BC against its optical depth`}
      subtitle={`Every Addis filter twice: the half-range the ceiling's 90 % interval gives its linearised BC (filled), and the reading noise after the correction has multiplied it by 1 / (1 − τ/τ∞)² (hollow). The bands are the same ladder as above. Reading noise is yours to set: the default, ± ${DETECTION_LIMIT} Mm⁻¹, is the laboratory's median detection limit, a generous stand-in for the noise of one reading. The ceiling's range is a shared error and does not shrink in an average; the noise does.`}
      provenance={FILTER_SOURCE}
      controls={<>
        <span className="control" title="The noise of one HIPS reading before linearising, in absorption units: drag sideways or click to type. ÷ 10 gives µg/m³.">Reading noise ± <ScrubField label="Reading noise" value={noise} min={0.1} max={10} step={0.1} width={96} unit="Mm⁻¹" format={(v) => fmt(v, 1)} onChange={(v) => setNoise(String(+v.toFixed(1)))} /></span>
      </>}>
      <ScatterPlot points={points} xLabel="Measured HIPS optical depth τ" yLabel="± on linearised BC (µg/m³)" yScale="log"
        xDomain={[0.4, 1.6]} yDomain={[0.05, 50]} nice={false} height={340} outside="pin" note={false}
        under={({ x, y, iw, ih }) => <>
          <PlaceBands y={y} iw={iw} />
          <Marker orient="v" at={x(mid.f.tau)} from={0} to={ih} label="median filter" labelSide="left" labelDy={ih - 24} />
        </>} />
      <Key items={[{ label: "from the ceiling's 90 % interval", color: LINEAR, kind: 'dot' }, { label: `from reading noise of ± ${fmt(noise, 1)} Mm⁻¹`, color: NOISE, kind: 'hollow' }]} />
      <Readouts items={[
        { label: 'Median filter', value: fmt(mid.f.hips_bc, 2), sub: `as reported · ${mid.f.id}` },
        { label: 'Linearised', value: <Digits value={mid.value} place={mid.place} />, sub: `× ${fmt(correction(mid.f.tau, c), 2)} · faint = in range` },
        { label: 'From the ceiling', value: value(mid.fromCeiling), sub: 'shared by every filter' },
        { label: 'From noise', value: value(mid.fromNoise), sub: `± ${fmt(noise / data.mac, 2)} × ${fmt(noiseGain(mid.f.tau, c), 1)}` },
        { label: 'Write it as', value: mid.text, sub: c < lo || c > hi ? 'this ceiling is outside the interval' : placeName(mid.place) },
      ]} />
      <Note>The ceiling's range is the larger of the two on {ceilingLarger} of {rows.length} filters, and {whole} of {rows.length} can be written to whole numbers at best. The darkest filter ({dark.f.id}, τ {fmt(dark.f.tau, 2)}) is {fmt(dark.f.hips_bc, 2)} µg/m³ as reported and {fmt(dark.value, 1)} linearised, ± {fmt(dark.fromCeiling, 1)} from the ceiling and ± {fmt(dark.fromNoise, 1)} from noise: {dark.text}. "Write it as" combines the two in quadrature on each side of the value, so its range is lopsided like the ceiling's. Ceiling {ceil.label} sets the value and the noise factor; the range is always the 90 % interval.</Note>
      <FixedSetNote what="the closure workflow's counted Addis filters" />
    </ChartFrame>
  )
}
