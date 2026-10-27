import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Crossplot } from '@/components/Crossplot'
import { FacetGrid } from '@/components/FacetGrid'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import type { SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SeasonMeta, SiteMeta } from '@/lib/types'
import { bcc, CHANNEL_COLOUR, CHANNELS, dates, HOURS, hourRuns, MIN_MINUTES, seasonsOf, type Channel, type SurfaceFile } from './data'
import { allYear, cycleSeries, dayGroups, HourPlot, inGroup, type DayGroup } from './HourCycle'

const WAVELENGTH_NB = 'research/ftir_hips_chem/addis_05_diurnal_wavelength_analysis.ipynb, multisite_diurnal_wavelength_analysis.ipynb, notebooks/analysis/diurnal_pub_figures.ipynb'
const AERONET_NB = 'notebooks/analysis/meteorology/aeronet_addis_diagnostics.ipynb and aeronet_addis_deep_dive.ipynb'
const UG = 'µg/m³'

interface Props { file: SurfaceFile; site: SiteMeta; seasons: readonly SeasonMeta[]; split: boolean; shade: readonly number[] }

/**
 * The hourly table's dates and seasons; which day-hours count (minute records: at least MIN_MINUTES
 * valid IR minutes; hourly-means files: every hour); the minutes each stands for; and a channel in µg/m³.
 */
function useHourly(file: SurfaceFile, seasons: readonly SeasonMeta[]) {
  return useMemo(() => {
    const days = dates(file.first, file.hourly)
    const n = file.hourly.n
    return {
      days, season: seasonsOf(days, seasons),
      full: (i: number) => (n ? (n[i] ?? 0) >= MIN_MINUTES : true),
      minutes: (i: number) => (n ? n[i] ?? 0 : 60),
      ug: (c: Channel) => bcc(file, c),
    }
  }, [file, seasons])
}

/** how the hourly table was made, for subtitles */
const madeFrom = (file: SurfaceFile) => file.resolution === 'hourly'
  ? `the hourly means in ${file.file} (negatives dropped)`
  : `hourly means of the minute record ${file.file} (negatives and values above mean + 3 SD dropped), kept where the hour has at least ${MIN_MINUTES} valid IR minutes`

/** facets when a chart has several lines per group and the seasons are split */
function facetBy<T>(groups: DayGroup[], render: (g: DayGroup, w?: number) => T) {
  return groups.length > 1
    ? <FacetGrid items={groups} minCell={340} maxCols={3} render={(g, w) => render(g, w) as never} />
    : render(groups[0]) as never
}

type ChannelMode = 'bcc' | 'ratio'
/** BCc at the five wavelengths by local hour (median, IQR), or each over IR: the spectral shape through the day. */
export function ChannelCycle({ file, site, seasons, split, shade }: Props) {
  const [mode, setMode] = useSearchParam<ChannelMode>('dn.ch', 'bcc', ['bcc', 'ratio'])
  const { season, full, ug } = useHourly(file, seasons)
  const groups = dayGroups(split, seasons, season)
  const t = file.hourly
  const ir = ug('IR')
  const series = (g: DayGroup): SweepSeries[] => CHANNELS.filter((c) => t[c] && (mode === 'bcc' || c !== 'IR')).flatMap((c) => {
    const v = ug(c)
    const values = mode === 'bcc' ? v : v.map((x, i) => (x != null && ir[i] != null && ir[i]! > 0.1 ? x / ir[i]! : null))
    return cycleSeries({ id: c, label: mode === 'bcc' ? c : `${c}/IR`, hours: t.h, values, seasons: season, groups: [g], keep: full, color: CHANNEL_COLOUR[c], digits: mode === 'bcc' ? 2 : 3 })
  })
  return (
    <ChartFrame id="diurnal-channels" title={`${site.name} · black carbon through the day — five wavelengths by local hour`}
      subtitle={`MA350 BCc as ${madeFrom(file)}; median over days and interquartile band. "÷ IR" divides each channel by IR in the same hour (IR above 0.1 ${UG}): a flat line at 1 is a wavelength-independent BCc. Shaded: AERONET's inversion hours.`}
      provenance={WAVELENGTH_NB}
      controls={<Segmented label="Show" value={mode} options={['bcc', 'ratio'] as ChannelMode[]} onChange={setMode} optionLabel={(k) => (k === 'bcc' ? 'BCc' : '÷ IR')} />}>
      {facetBy(groups, (g, w) => <HourPlot key={g.name} title={groups.length > 1 ? g.name : undefined} width={w} series={series(g)} offset={file.offset} shade={shade}
        yLabel={mode === 'bcc' ? `BCc (${UG})` : 'BCc / IR BCc'} digits={mode === 'bcc' ? 2 : 3} yMarkers={mode === 'ratio' ? [{ at: 1, color: INK.identity }] : undefined} />)}
      <Key items={CHANNELS.filter((c) => mode === 'bcc' || c !== 'IR').map((c) => ({ label: c, color: CHANNEL_COLOUR[c], kind: 'line' as const }))} />
    </ChartFrame>
  )
}

/** AAE by hour from absorption (BCc × the MA350 cross-section), UV/IR and Blue/IR. */
export function AaeCycle({ file, site, seasons, split, shade }: Props) {
  const { season, full } = useHourly(file, seasons)
  const groups = dayGroups(split, seasons, season)
  const t = file.hourly
  const pairs = [['aae_uv_ir', 'AAE UV/IR', CHANNEL_COLOUR.UV], ['aae_blue_ir', 'AAE Blue/IR', CHANNEL_COLOUR.Blue]] as const
  const series = (g: DayGroup) => pairs.filter(([k]) => t[k]).flatMap(([k, label, color]) =>
    cycleSeries({ id: k, label, hours: t.h, values: t[k]!, seasons: season, groups: [g], keep: full, color, digits: 2 }))
  return (
    <ChartFrame id="diurnal-aae" title={`${site.name} · absorption Ångström exponent by local hour — UV/IR and Blue/IR`}
      subtitle="AAE of each day-hour's mean absorption, b = BCc × σ_ATN (UV 24.069, Blue 19.070, IR 10.120 m²/g), so the firmware's wavelength-dependent cross-section is undone (optics.aae; a BCc ratio reads about 1 lower). Median over days and interquartile band; 1 is the black-carbon reference, 2 a biomass-smoke one. At Addis the UV channel is out of range on a third of days (the brown-carbon test below)."
      provenance={`${WAVELENGTH_NB}; research/ftir_hips_chem/scripts/optics.py`}>
      {facetBy(groups, (g, w) => <HourPlot key={g.name} title={groups.length > 1 ? g.name : undefined} width={w} series={series(g)} offset={file.offset} shade={shade}
        yLabel="AAE (absorption)" yMarkers={[{ at: 1, label: 'BC ≈ 1', color: INK.identity }, { at: 2, label: 'smoke ≈ 2', color: INK.warn }]} />)}
      <Key items={pairs.map(([, label, color]) => ({ label, color, kind: 'line' as const }))} />
    </ChartFrame>
  )
}

const WINDOWS: { key: string; label: string; hours: number[] }[] = [
  { key: 'm', label: '06–10', hours: [6, 7, 8, 9] }, { key: 'l', label: '10–14', hours: [10, 11, 12, 13] },
  { key: 'a', label: '14–18', hours: [14, 15, 16, 17] }, { key: 'e', label: '18–22', hours: [18, 19, 20, 21] },
  { key: 'n', label: '22–06', hours: [22, 23, 0, 1, 2, 3, 4, 5] },
]

/** How much of a 24 h filter's black-carbon mass each part of the day puts there (pw_mac_calibration §3). */
export function MorningShare({ file, site, seasons, split }: Omit<Props, 'shade'>) {
  const { days, season, minutes, ug } = useHourly(file, seasons)
  const perDay = useMemo(() => {
    const t = file.hourly, ir = ug('IR')
    const acc = new Map<string, { season: string; minutes: number; mass: number[] }>()
    t.h.forEach((h, i) => {
      const v = ir[i]
      if (v == null) return
      const d = acc.get(days[i]) ?? { season: season[i], minutes: 0, mass: HOURS.map(() => 0) }
      const w = minutes(i)
      d.minutes += w; d.mass[h] += v * w
      acc.set(days[i], d)
    })
    return [...acc.values()].filter((d) => d.minutes >= 18 * 60 && d3.sum(d.mass) > 0)
      .map((d) => { const total = d3.sum(d.mass); return { season: d.season, share: WINDOWS.map((w) => d3.sum(w.hours, (h) => d.mass[h]) / total) } })
  }, [file, days, season, minutes, ug])
  const groups = split ? dayGroups(true, seasons, perDay.map((d) => d.season)) : [allYear()]
  const blocks: IntervalBlock[] = groups.map((g) => {
    const ds = perDay.filter((d) => inGroup(g, d.season))
    return { key: g.name, title: groups.length > 1 ? `${g.name} (${ds.length} days)` : `${ds.length} days`, color: g.color,
      rows: WINDOWS.map((w, k) => {
        const v = ds.map((d) => d.share[k]).sort(d3.ascending), f = w.hours.length / 24
        const q = (p: number) => (v.length ? d3.quantileSorted(v, p)! : null)
        return { key: `${g.name}-${w.key}`, label: `${w.label} (${w.hours.length} h)`, strong: w.key === 'm',
          marks: [[{ mid: q(0.5) == null ? null : 100 * q(0.5)!, lo: q(0.25) == null ? null : 100 * q(0.25)!, hi: q(0.75) == null ? null : 100 * q(0.75)!, color: g.color }],
            [{ mid: q(0.5) == null ? null : q(0.5)! / f, lo: q(0.25) == null ? null : q(0.25)! / f, hi: q(0.75) == null ? null : q(0.75)! / f, color: g.color }]],
          tip: [`${g.name} · ${w.label}`, `median ${fmt(100 * (q(0.5) ?? NaN), 1)} % of the day's mass (IQR ${fmt(100 * (q(0.25) ?? NaN), 1)}–${fmt(100 * (q(0.75) ?? NaN), 1)})`,
            `${fmt(100 * f, 1)} % of the hours`, `${v.length} days with at least 18 h`] }
      }),
    }
  })
  const morning = d3.median(perDay, (d) => d.share[0])
  return (
    <ChartFrame id="diurnal-morning-share" title={`${site.name} · where a 24 h filter's black carbon comes from — share of the day's IR BCc by time of day`}
      subtitle={`For each local day with at least 18 h of valid IR ${file.resolution === 'hourly' ? 'hourly means' : 'minutes'}, the share of the day's summed IR BCc (concentration × minutes, which a constant-flow filter integrates) falling in each window; median and interquartile range over days. Right: the share divided by the window's share of the day, 1 = no weighting. Over all days the 06–10 window carries a median ${fmt(100 * (morning ?? NaN), 1)} % against the 16.7 % its four hours would.`}
      provenance="notebooks/analysis/meteorology/pw_mac_calibration.ipynb (§3, morning-weighting bias on 24 h filters)">
      {perDay.length ? <IntervalStrip blocks={blocks} labelWidth={150} panels={[{ label: "Share of the day's IR BCc (%)", format: (v) => fmt(v, 0) },
        { label: 'Share ÷ share of hours', reference: 1, format: (v) => fmt(v, 1) }]} />
        : <Empty>No day has 18 h of valid IR minutes.</Empty>}
    </ChartFrame>
  )
}

/**
 * ma350_diurnal_coverage.py: the photometer averages over its hours, the filter over 24. Red BCc
 * (625 nm, next to the HIPS 633 nm) by hour with the AERONET hours shaded, and per day the ratio of
 * the 24 h mean to the mean over those hours.
 */
export function CoverageCheck({ file, site, seasons, split }: Omit<Props, 'shade'>) {
  const cov = file.coverage
  const { season: hourlySeason, full, ug } = useHourly(file, seasons)
  const red = file.hourly.Red ? ug('Red') : null
  const covDays = useMemo(() => (cov ? dates(file.first, cov) : []), [file, cov])
  const covSeason = useMemo(() => seasonsOf(covDays, seasons), [covDays, seasons])
  if (!cov || !red) return null
  const hours = cov.hours
  const cycle = cycleSeries({ id: 'red', label: 'Red BCc', hours: file.hourly.h, values: red, seasons: hourlySeason, groups: [allYear()], keep: full, color: CHANNEL_COLOUR.Red })
  const dayMean = d3.mean(red.filter((v, i) => v != null && full(i)) as number[])
  const groups = split ? dayGroups(true, seasons, covSeason) : [allYear()]
  const blocks: IntervalBlock[] = groups.map((g) => ({ key: g.name, title: groups.length > 1 ? g.name : undefined, color: g.color, rows: (['ratio_mean', 'ratio_median'] as const).map((k) => {
    const v = cov[k].filter((x, i) => x != null && inGroup(g, covSeason[i])).map(Number).sort(d3.ascending)
    const q = (p: number) => (v.length ? d3.quantileSorted(v, p)! : null)
    return { key: `${g.name}-${k}`, label: k === 'ratio_mean' ? 'from daily means' : 'from daily medians',
      note: `n=${v.length}`, marks: [[{ mid: q(0.5), lo: q(0.25), hi: q(0.75), color: g.color }]],
      tip: [`${g.name}, ${k === 'ratio_mean' ? 'mean' : 'median'} of the day / of its AERONET hours`, `median ${fmt(q(0.5), 3)} (IQR ${fmt(q(0.25), 3)}–${fmt(q(0.75), 3)})`, `${v.length} days`] }
  }) }))
  return (
    <ChartFrame id="diurnal-coverage" title={`${site.name} · the 24 h filter against AERONET's hours — Red BCc, 24 h over the inversion hours`}
      subtitle={`Left: hourly mean Red BCc (625 nm), median over days with the interquartile band; shaded the inversion hours (${hourRuns(hours)}), dashed the mean over all hours. Right: per day, the 24 h mean (or median) of raw Red BCc over the same over those hours, on days with more than 1000 minutes and more than 300 inside them${file.resolution === 'hourly' ? ' (here hourly means: 17 hours and 6 inside, the median then of hours)' : ''}. A ratio of 1 means the photometer's hours stand for the filter's day; above 1 they under-sample it, so a column-to-surface ratio such as the absorption scale height should be multiplied by it.`}
      provenance="research/ftir_ec_phase3/scripts/ma350_diurnal_coverage.py; AERONET_SPARTAN_2026-08-22.md (Addis 0.92–1.05, Pasadena 1.07–1.09)">
      <FacetGrid items={['cycle', 'ratio']} minCell={380} maxCols={2} render={(k, w) => k === 'cycle'
        ? <HourPlot series={cycle} width={w} yLabel={`Red BCc (${UG})`} offset={file.offset} shade={hours} include={[0]}
            yMarkers={dayMean != null ? [{ at: dayMean, label: '24 h mean', color: INK.muted }] : undefined} />
        : <IntervalStrip blocks={blocks} labelWidth={140} noteWidth={50} minPanelWidth={Math.max(160, (w ?? 380) - 240)}
            panels={[{ label: 'Ratio, 24 h / AERONET hours', reference: 1, format: (v) => fmt(v, 2) }]} />} />
    </ChartFrame>
  )
}

/** the matched table's dates and seasons */
function useMatched(file: SurfaceFile, seasons: readonly SeasonMeta[]) {
  return useMemo(() => {
    const days = file.matched ? dates(file.first, file.matched) : []
    return { days, season: seasonsOf(days, seasons) }
  }, [file, seasons])
}

/** Surface BC and column AOD at the same moments, by hour: the decoupling the deep-dive notebook found at Addis. */
export function ColumnCoupling({ file, site, seasons, split, shade }: Props) {
  const m = file.matched
  const { season } = useMatched(file, seasons)
  if (!m) return null
  const groups = dayGroups(split, seasons, season)
  const panels = [
    { key: 'bc_ir', title: 'Surface IR BCc', axis: `IR BCc (${UG})`, values: m.bc_ir, digits: 2 },
    { key: 'aod500', title: 'Column AOD 500', axis: 'AOD 500 nm', values: m.aod500, digits: 3 },
    { key: 'bc_aod', title: 'BC / AOD', axis: `IR BCc / AOD 500 (${UG})`, values: m.bc_aod, digits: 1 },
  ]
  return (
    <ChartFrame id="diurnal-coupling" title={`${site.name} · surface against column through the day — BC, AOD and BC/AOD at the photometer's moments`}
      subtitle={`Each AERONET direct-sun retrieval paired with the mean IR BCc of the ±15 min around it (at least 5 valid minutes); retrievals averaged per day-hour, then the median by hour and interquartile band. A shallow morning boundary layer shows as BC high and AOD flat, so BC/AOD (how much of the column sits at the surface) falls through the morning.`}
      provenance={AERONET_NB}>
      <FacetGrid items={panels} minCell={300} maxCols={3} render={(p, w) => <HourPlot title={p.title} width={w} height={230} offset={file.offset} shade={shade} yLabel={p.axis} digits={p.digits}
        series={cycleSeries({ id: p.key, label: p.title, hours: m.h, values: p.values, seasons: season, groups, digits: p.digits })} />} />
      {split && <Key items={groups.map((g) => ({ label: g.name, color: g.color, kind: 'line' as const }))} />}
    </ChartFrame>
  )
}

/** Surface absorption AAE against the column's extinction Ångström exponent, hour by hour. */
export function SurfaceVsColumnAae({ file, site, seasons, shade }: Omit<Props, 'split'>) {
  const m = file.matched
  const { season } = useMatched(file, seasons)
  if (!m) return null
  const series = [
    ...cycleSeries({ id: 'aae', label: 'Surface AAE UV/IR (absorption)', hours: m.h, values: m.aae, seasons: season, groups: [allYear()], color: CHANNEL_COLOUR.UV }),
    ...cycleSeries({ id: 'ae', label: 'Column Ångström exponent 440–870', hours: m.h, values: m.ae, seasons: season, groups: [allYear()], color: INK.accent }),
  ]
  return (
    <ChartFrame id="diurnal-aae-ae" title={`${site.name} · surface AAE against the column's Ångström exponent — by local hour`}
      subtitle="At the photometer's moments: the MA350 AAE(UV, IR) of the absorption in the ±15 min around each retrieval, and AERONET's 440–870 nm extinction Ångström exponent. They measure different things (absorption's wavelength dependence at the surface, particle size in the column); both near 1–1.5 says combustion aerosol top to bottom, a low column exponent with a high surface AAE says dust aloft over smoke below."
      provenance={`${AERONET_NB} (experiment 4); the notebook took AAE from the BCc ratio, about 1 lower`}>
      <HourPlot series={series} offset={file.offset} shade={shade} yLabel="Exponent" />
      <Key items={series.map((s) => ({ label: s.label, color: s.color, kind: 'line' as const }))} />
    </ChartFrame>
  )
}

/** Within-day variability, surface against column: each day's CV of matched BC and of AOD. */
export function IntradayCv({ file, site, seasons }: { file: SurfaceFile; site: SiteMeta; seasons: readonly SeasonMeta[] }) {
  const t = file.intraday
  const days = useMemo(() => (t ? dates(file.first, t) : []), [file, t])
  const season = useMemo(() => seasonsOf(days, seasons), [days, seasons])
  if (!t) return null
  const colourOf = new Map(seasons.map((s) => [s.name, s.color]))
  const points = days.flatMap((d, i) => {
    const x = t.aod_cv[i], y = t.bc_cv[i]
    return x == null || y == null ? [] : [{ x, y, color: colourOf.get(season[i]) ?? INK.muted,
      tip: [d, season[i] || 'no season', `AOD CV ${fmt(x, 2)}`, `BC CV ${fmt(y, 2)}`, `${t.n[i]} matched retrievals`] }]
  })
  const ratio = d3.median(points, (p) => p.y / p.x)
  return (
    <ChartFrame id="diurnal-intraday-cv" title={`${site.name} · within-day variability — surface BC against column AOD`}
      subtitle={`One point per day with at least 3 retrievals matched to the MA350: the coefficient of variation (SD / mean) of the ±15 min IR BCc across that day's retrievals, against the CV of their AOD 500. Above the 1:1 line the surface swings more than the column; the median day's BC CV is ${fmt(ratio, 1)} × its AOD CV.`}
      provenance={`${AERONET_NB} (experiment 6, temporal variability)`}>
      {points.length ? <Crossplot points={points} title="" xLabel="CV of AOD 500 within the day" yLabel="CV of surface IR BCc within the day" fit={false} grouping="none" />
        : <Empty>No day has three matched retrievals.</Empty>}
      <Key items={seasons.filter((s) => season.includes(s.name)).map((s) => ({ label: s.name, color: s.color }))} />
    </ChartFrame>
  )
}

