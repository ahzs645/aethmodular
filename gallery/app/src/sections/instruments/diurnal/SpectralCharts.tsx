import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { ColorLegend } from '@/components/ColorLegend'
import { FacetGrid } from '@/components/FacetGrid'
import { HeatCell } from '@/components/HeatCell'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useTooltip } from '@/hooks/useTooltip'
import { useColourScheme } from '@/lib/colourSchemes'
import { loadJson } from '@/lib/data'
import { seasonOnly } from '@/lib/seasons'
import { fmt, signed } from '@/lib/format'
import { FONT, INK } from '@/lib/theme'
import { useSearchParam } from '@/lib/url'
import type { SeasonMeta, SiteMeta } from '@/lib/types'
import { bcc, CHANNEL_COLOUR, CHANNELS, dates, DIURNAL_DIR, HOURS, hourLabel, MIN_DAYS, MIN_MINUTES, seasonsOf, utcLabel, type Channel, type SurfaceFile } from './data'
import { ALL_YEAR, allYear, cycleSeries, daySplit, dayGroups, HourPlot, inGroup, splitLabels, type DayGroup, type DaySplit } from './HourCycle'
import { blockSummary, dailyCv, derive, hourMedians, median, normalisedProfile, representativeDays, WAVELENGTH_NM, type BlockRow, type Derived } from './spectral'

const A05 = 'research/ftir_hips_chem/addis_05_diurnal_wavelength_analysis.ipynb'
const MS = 'research/ftir_hips_chem/multisite_diurnal_wavelength_analysis.ipynb'
const UG = 'µg/m³'

/**
 * What the notebooks ran on against what this page has: they took every quantity minute by minute
 * (the multisite one after a 10-min rolling median and a 1st–99th percentile fence), then the median
 * by hour; here each quantity is taken on the day-hour means the exporter wrote.
 */
const BASIS = (file: SurfaceFile) => file.resolution === 'hourly'
  ? `taken on the hourly means in ${file.file} (negatives dropped)`
  : `taken on hourly means of ${file.file} (negatives and values above mean + 3 SD dropped, hours with at least ${MIN_MINUTES} valid IR minutes)`
const WINDOW_NOTE = 'The notebooks took each quantity minute by minute (multisite after a 10-min rolling median) and then the hour\'s median; here it is each day-hour\'s mean, so the bands are narrower and a ratio or CV of means is not the mean of minute ratios.'

/** one site's day-hour table with every derived column, its dates and seasons, and the notebooks' keep rule */
export interface Spectral {
  file: SurfaceFile; days: string[]; season: string[]; hours: number[]
  ch: Record<Channel, (number | null)[]>; d: Derived
  /** minute records: at least MIN_MINUTES valid IR minutes in the day-hour; hourly means: all */
  keep: (i: number) => boolean
  /** the minutes a day-hour stands for */
  minutes: (i: number) => number
}
export function spectralOf(file: SurfaceFile, seasons: readonly SeasonMeta[]): Spectral {
  const t = file.hourly, days = dates(file.first, t), n = t.n
  const ch = Object.fromEntries(CHANNELS.map((c) => [c, bcc(file, c)])) as Record<Channel, (number | null)[]>
  return {
    file, days, season: seasonsOf(days, seasons), hours: t.h, ch, d: derive(ch),
    keep: (i) => (n ? (n[i] ?? 0) >= MIN_MINUTES : true),
    minutes: (i) => (n ? n[i] ?? 0 : 60),
  }
}

/** several sites' surface files at once (the cross-site charts); each appears as it arrives */
export function useSurfaceFiles(codes: readonly string[]): Record<string, SurfaceFile> {
  const key = codes.join(',')
  const [files, setFiles] = useState<Record<string, SurfaceFile>>({})
  useEffect(() => {
    let live = true
    for (const code of key ? key.split(',') : []) {
      loadJson<SurfaceFile>(`${DIURNAL_DIR}/aeth/${code}.json`)
        .then((f) => { if (live) setFiles((s) => (s[code] ? s : { ...s, [code]: f })) })
        .catch(() => { /* none for this site */ })
    }
    return () => { live = false }
  }, [key])
  return files
}

interface SiteProps { spectral: Spectral; site: SiteMeta; split: DaySplit | null; shade: readonly number[] }

/** addis_05 §4–5 and multisite §5–6: Delta-C (UV − IR) and the spread of the five channels, by local hour. */
export function DeltaCAndCv({ spectral: s, site, split, shade }: SiteProps) {
  // each day-hour's group under the page's split (s.season stays the calendar's, for the seasonal charts)
  const labels = useMemo(() => splitLabels(s.days, split), [s.days, split])
  const groups = dayGroups(split, labels)
  const one = groups.length === 1
  const panels = [
    { key: 'dc', title: 'Delta-C, UV − IR', axis: `UV BCc − IR BCc (${UG})`, values: s.d.deltaC, digits: 2, ref: 0, color: '#8B4513' },
    { key: 'cv', title: 'CV across the five channels', axis: 'SD / mean of the five BCc', values: s.d.cv, digits: 3, ref: null, color: INK.accent },
  ]
  const drawn = panels.map((p) => ({ ...p, series: cycleSeries({ id: p.key, label: p.title, hours: s.hours, values: p.values, seasons: labels, groups, keep: s.keep, digits: p.digits, color: one ? p.color : undefined }) }))
  const all = (v: (number | null)[]) => median(v.filter((_, i) => s.keep(i)))
  return (
    <ChartFrame id="diurnal-deltac-cv" title={`${site.name} · how far apart the wavelengths sit through the day — Delta-C and the channels' CV by local hour`}
      subtitle={`Left: Delta-C = UV BCc − IR BCc, the notebooks' biomass-burning tracer (positive: more UV absorption than IR). Right: the coefficient of variation (SD with n − 1, over mean) across the five channels' BCc, where their mean is above 0.1 ${UG}: high when the wavelengths read differently, near 0 when they collapse onto one value. Both ${BASIS(s.file)}; median over days and interquartile band, at hours with at least ${MIN_DAYS} day-hours. ${WINDOW_NOTE} Over all hours the median Delta-C is ${signed(all(s.d.deltaC), 2)} ${UG} and the median CV ${fmt(all(s.d.cv), 3)}. Shaded: AERONET's inversion hours.`}
      provenance={`${A05} (§4 Delta-C, §5 wavelength spread, with the season panels); ${MS} (§5, §6)`}>
      <FacetGrid items={drawn} minCell={320} maxCols={2} render={(p, w) => (
        <HourPlot title={p.title} width={w} height={240} series={p.series} offset={s.file.offset} shade={shade} yLabel={p.axis} digits={p.digits}
          yMarkers={p.ref != null ? [{ at: p.ref, color: INK.identity }] : undefined} />
      )} />
      {!one && <Key items={groups.map((g) => ({ label: g.name, color: g.color, kind: 'line' as const }))} />}
    </ChartFrame>
  )
}

/** a channel's profile against its daily mean, per group of day-hours */
interface Profile { key: string; title: string; color?: string; offset: number; rows: { c: Channel; ratio: (number | null)[]; median: (number | null)[]; n: number[]; dailyMean: number | null }[] }
function profileOf(s: Spectral, key: string, title: string, keep: (i: number) => boolean, color?: string): Profile {
  return {
    key, title, color, offset: s.file.offset,
    rows: CHANNELS.filter((c) => s.file.hourly[c]).map((c) => {
      const { median: med, n } = hourMedians(s.hours, s.ch[c], (i) => s.keep(i) && keep(i), MIN_DAYS)
      const { ratio, dailyMean } = normalisedProfile(med)
      return { c, ratio, median: med, n, dailyMean }
    }),
  }
}

type Across = 'sites' | 'seasons'
type Draw = 'cells' | 'lines'
const PM = { top: 22, right: 6, bottom: 22, left: 84 }

/** one profile as a heat map: channel rows, hour columns */
function ProfileGrid({ p, width, fill, tip }: { p: Profile; width: number; fill: (v: number | null) => string | null; tip: ReturnType<typeof useTooltip> }) {
  const iw = width - PM.left - PM.right, cw = iw / 24, ch = 18
  return (
    <svg width={width} height={PM.top + p.rows.length * ch + PM.bottom} fontFamily={FONT.family}>
      <text x={PM.left} y={13} fontSize={12} fontWeight={600} fill={p.color ?? INK.text}>{p.title}</text>
      <g transform={`translate(${PM.left},${PM.top})`}>
        {p.rows.map((r, i) => <g key={r.c}>
          <text x={-6} y={i * ch + ch / 2} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.muted}>{r.c} ({WAVELENGTH_NM[r.c]})</text>
          {HOURS.map((h) => <HeatCell key={h} x={h * cw} y={i * ch} width={cw - 0.5} height={ch - 0.5} fill={fill(r.ratio[h])}
            onEnter={(e) => tip.show(e, [`${p.title} · ${r.c} · ${hourLabel(h)} (${utcLabel(p.offset)})`,
              r.ratio[h] == null ? `${r.n[h]} day-hours (too few)` : `${fmt(r.ratio[h], 2)} × the channel's daily mean`,
              `median ${fmt(r.median[h], 2)} ${UG} over ${r.n[h]} day-hours`, `daily mean of the hourly medians ${fmt(r.dailyMean, 2)} ${UG}`])} onLeave={tip.hide} />)}
        </g>)}
        {[0, 6, 12, 18].map((h) => <text key={h} x={h * cw} y={p.rows.length * ch + 13} fontSize={10} fill={INK.muted}>{String(h).padStart(2, '0')}</text>)}
      </g>
    </svg>
  )
}

/**
 * addis_05 §8 and multisite §10: each channel's hour-of-day median over the mean of its 24 hourly
 * medians, so every row reads 1 at the daily average. Rows that move together are wavelengths that
 * see the same aerosol; a row that departs is a channel with its own diurnal story.
 */
export function NormalisedHeatmap({ focus, focusName, spectra, sites, seasons }: {
  focus: Spectral | null; focusName: string; spectra: Record<string, Spectral>; sites: SiteMeta[]; seasons: readonly SeasonMeta[]
}) {
  const ref = useRef<HTMLDivElement>(null)
  const tip = useTooltip(ref)
  const [across, setAcross] = useSearchParam<Across>('dn.hm', 'sites', ['sites', 'seasons'])
  const [draw, setDraw] = useSearchParam<Draw>('dn.hmd', 'cells', ['cells', 'lines'])
  const colours = useColourScheme('diurnal-heatmap', 'diverging', 'Red–yellow–blue')
  const scale = d3.scaleDiverging<string>(d3.interpolateRgbBasis(colours.stops)).domain([-1, 0, 1]).clamp(true)
  const fill = (v: number | null) => (v == null || !(v > 0) ? null : scale(Math.log2(v)))
  const panels = useMemo<Profile[]>(() => {
    if (across === 'sites') return sites.flatMap((s) => (spectra[s.code] ? [profileOf(spectra[s.code], s.code, s.name, () => true, s.color)] : []))
    if (!focus) return []
    const groups = dayGroups(daySplit('seasons', seasons), focus.season)
    return [...groups.map((g) => profileOf(focus, g.name, seasonOnly(g.name), (i) => inGroup(g, focus.season[i]), g.color)), profileOf(focus, ALL_YEAR, ALL_YEAR, () => true)]
  }, [across, sites, spectra, focus, seasons])
  return (
    <ChartFrame id="diurnal-heatmap" title="The day's spectral fingerprint — each channel's hourly median over its own daily mean"
      subtitle={`For each channel, the median BCc in each local standard hour divided by the mean of its 24 hourly medians (1 = the channel's daily average), ${across === 'sites' ? 'one panel per site in view with an hourly MA350 export, every day' : `${focusName}'s seasons and the whole year`}; hours with fewer than ${MIN_DAYS} day-hours are left blank and out of the mean. Coloured on a log scale from ½× to 2× (the notebooks drew 0.4–1.8 and 0.5–2.0 on a linear one). ${WINDOW_NOTE} Curves: the same numbers as lines (addis_05's right panel).`}
      provenance={`${A05} (§8); ${MS} (§10, heatmap_all_sites)`}
      controls={<>
        <Segmented label="Panels" value={across} options={['sites', 'seasons'] as Across[]} onChange={setAcross} optionLabel={(k) => (k === 'sites' ? 'Sites' : focus ? `${focusName} by season` : 'Seasons')} />
        <Segmented label="Draw" value={draw} options={['cells', 'lines'] as Draw[]} onChange={setDraw} optionLabel={(k) => (k === 'cells' ? 'Heat map' : 'Curves')} />
      </>}>
      <div ref={ref} className="chart-wrap">
        {!panels.length ? <Empty>No site in view has an hourly MA350 export.</Empty>
          : draw === 'cells'
            ? <FacetGrid items={panels} minCell={340} maxCols={2} render={(p, w) => <ProfileGrid p={p} width={w} fill={fill} tip={tip} />} />
            : <FacetGrid items={panels} minCell={320} maxCols={2} render={(p, w) => (
                <HourPlot title={p.title} width={w} height={220} offset={p.offset} yLabel="× daily mean" hours={[0, 23]}
                  yMarkers={[{ at: 1, color: INK.identity }]}
                  series={p.rows.map((r) => ({ id: r.c, label: r.c, color: CHANNEL_COLOUR[r.c], curve: 'monotone' as const,
                    points: HOURS.map((h) => ({ x: h, y: r.ratio[h], tip: [`${p.title} · ${r.c}`, hourLabel(h), `${fmt(r.ratio[h], 2)} × daily mean`, `${r.n[h]} day-hours`] })) }))} />
              )} />}
        {draw === 'cells' ? panels.length > 0 && <div className="legend">
          <ColorLegend scale={scale} scheme={colours.scheme} width={160} label="× the channel's daily mean" format={(v) => `${fmt(2 ** v, v === 0 ? 0 : 1)}×`} />
        </div> : <Key items={CHANNELS.map((c) => ({ label: c, color: CHANNEL_COLOUR[c], kind: 'line' as const }))} />}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/**
 * addis_05 §7 and multisite §9: three single days, picked by their median wavelength CV, drawn hour
 * by hour at all five channels: how individual days look against the medians above.
 */
export function RepresentativeDays({ spectral: s, site, shade }: { spectral: Spectral; site: SiteMeta; shade: readonly number[] }) {
  const picks = useMemo(() => {
    const cv = s.d.cv.map((v, i) => (s.keep(i) ? v : null))
    const daily = dailyCv(s.days, cv, s.ch.IR, s.minutes)
    const rep = representativeDays(daily)
    return { n: daily.length, rep }
  }, [s])
  const panels = useMemo(() => (picks.rep ?? []).map(({ kind, day }) => {
    const idx = s.days.flatMap((d, i) => (d === day.day ? [i] : []))
    const series: SweepSeries[] = CHANNELS.filter((c) => s.file.hourly[c]).map((c) => ({
      id: c, label: c, color: CHANNEL_COLOUR[c], curve: 'linear' as const, gaps: true,
      points: HOURS.map((h) => {
        const i = idx.find((j) => s.hours[j] === h)
        const v = i == null ? null : s.ch[c][i]
        return { x: h, y: v, tip: [`${day.day} · ${c}`, hourLabel(h), `${fmt(v, 2)} ${UG}`, ...(i != null && s.file.hourly.n ? [`${s.file.hourly.n[i]} valid IR minutes`] : [])] }
      }),
    }))
    return { kind, day, season: s.season[idx[0]] ? seasonOnly(s.season[idx[0]]) : 'no season', series }
  }), [picks, s])
  const days = s.file.resolution === 'hourly' ? '20 hours' : '1200 valid IR minutes'
  return (
    <ChartFrame id="diurnal-representative-days" title={`${site.name} · three single days — high, typical and low spread between the wavelengths`}
      subtitle={`Of the ${picks.n} local days with at least ${days} (the notebooks' 1200 minutes), each day's median of its day-hours' five-channel CV ranks it; drawn are the sixth highest and sixth lowest (skipping the extremes) and the middle day by rank, as multisite's find_representative_days picks them (addis_05 took the day closest to the median CV). The notebooks drew each minute (addis_05 the instrument's smoothed BCc, multisite a 10-min rolling median); here each line is the day's hourly means, ${BASIS(s.file)}.`}
      provenance={`${A05} (§7); ${MS} (§9, representative_days)`}>
      {!picks.rep ? <Empty>Fewer than ten days at {site.name} have {days}.</Empty> : <>
        <FacetGrid items={panels} minCell={300} maxCols={3} render={(p, w) => (
          <HourPlot title={`${p.kind} · ${p.day.day} · CV ${fmt(p.day.cv, 3)} · ${p.season.replace(/ \(.*\)$/, '')}`} width={w} height={230} series={p.series} offset={s.file.offset}
            shade={shade} yLabel={`BCc (${UG})`} include={[0]} hours={[0, 23]} />
        )} />
        <Key items={CHANNELS.map((c) => ({ label: c, color: CHANNEL_COLOUR[c], kind: 'line' as const }))} />
      </>}
    </ChartFrame>
  )
}

type CrossKey = 'uvIr' | 'blueIr' | 'greenIr' | 'aae' | 'dc'
type AaeFrom = 'bcc' | 'abs'
const CROSS: { key: CrossKey; title: string; axis: string; ref: number; digits: number; extra?: number }[] = [
  { key: 'uvIr', title: 'UV / IR', axis: 'UV BCc / IR BCc', ref: 1, digits: 3 },
  { key: 'blueIr', title: 'Blue / IR', axis: 'Blue BCc / IR BCc', ref: 1, digits: 3 },
  { key: 'greenIr', title: 'Green / IR', axis: 'Green BCc / IR BCc', ref: 1, digits: 3 },
  { key: 'aae', title: 'AAE UV/IR (BCc ratio)', axis: 'AAE from the BCc ratio', ref: 1, digits: 2, extra: 1.5 },
  { key: 'dc', title: 'Delta-C ÷ the site’s median IR', axis: 'Delta-C / median IR BCc', ref: 0, digits: 3 },
]
function crossValues(s: Spectral, key: CrossKey, aae: AaeFrom): (number | null)[] {
  if (key === 'uvIr') return s.d.uvIr
  if (key === 'blueIr') return s.d.blueIr
  if (key === 'greenIr') return s.d.greenIr
  if (key === 'aae') return aae === 'bcc' ? s.d.aaeUvIr : s.file.hourly.aae_uv_ir ?? []
  const m = median(s.ch.IR.filter((_, i) => s.keep(i)))
  return s.d.deltaC.map((v) => (v == null || !m ? null : v / m))
}

/**
 * multisite §8: the sites on one axis each, every day. If one instrument's wavelengths collapse onto
 * IR and the others' do not, the collapse is the instrument's, not the air's.
 */
export function CrossSiteOverlay({ spectra, sites }: { spectra: Record<string, Spectral>; sites: SiteMeta[] }) {
  const [aae, setAae] = useSearchParam<AaeFrom>('dn.xaae', 'bcc', ['bcc', 'abs'])
  const have = sites.filter((s) => spectra[s.code])
  const panels = useMemo(() => CROSS.map((p) => ({
    ...p, ...(p.key === 'aae' && aae === 'abs' ? { title: 'AAE UV/IR (absorption)', axis: 'AAE from b = BCc × σ_ATN' } : {}),
    series: have.map((site): SweepSeries => {
      const s = spectra[site.code]
      const { median: med, n } = hourMedians(s.hours, crossValues(s, p.key, aae), s.keep, MIN_DAYS)
      return { id: site.code, label: site.name, color: site.color, curve: 'monotone' as const,
        points: HOURS.map((h) => ({ x: h, y: med[h], tip: [`${site.name} · ${p.title}`, `${hourLabel(h)} (${utcLabel(s.file.offset)})`, med[h] == null ? `${n[h]} day-hours (too few)` : `median ${fmt(med[h], p.digits)}`, `${n[h]} day-hours`] })) }
    }),
  })), [have, spectra, aae]) // eslint-disable-line react-hooks/exhaustive-deps
  if (have.length < 2) return null
  return (
    <ChartFrame id="diurnal-cross-site" title="The sites on one axis — is the collapse of the wavelengths onto IR one instrument's or every site's?"
      subtitle={`Every site in view with an hourly MA350 export (${have.map((s) => `${s.name}: ${spectra[s.code].file.resolution === 'hourly' ? 'hourly means' : 'minute record'}`).join('; ')}), all days, each in its own local standard time: the median by hour of each channel over IR (IR above 0.1 ${UG}), of the AAE (by default from the UV/IR BCc ratio, −ln(UV/IR) / ln(375/880), the notebook's definition, which it labelled 370 nm; "absorption" switches to the exporter's AAE of b = BCc × σ_ATN, as in the AAE chart above, about 1 higher, which is what the notebook's 1 and 1.5 references are drawn for), and of Delta-C over the site's median IR BCc, at hours with at least ${MIN_DAYS} day-hours. ${WINDOW_NOTE} Dotted at 1.5 on the AAE panel: the notebook's biomass reference. Green/IR is multisite's separate figure for Addis's offset green.`}
      provenance={`${MS} (§8, cross_site_comparison and green_ir_ratio_cross_site)`}
      controls={<Segmented label="AAE from" value={aae} options={['bcc', 'abs'] as AaeFrom[]} onChange={setAae} optionLabel={(k) => (k === 'bcc' ? 'BCc ratio (notebook)' : 'Absorption')} />}>
      <FacetGrid items={panels} minCell={280} maxCols={3} render={(p, w) => (
        <SweepPlot title={p.title} width={w} height={220} series={p.series} xScale="linear" xDomain={[0, 23]} xLabel="Hour of day (each site's local standard time)"
          yLabel={p.axis} xFormat={(v) => hourLabel(+v)} yFormat={(v) => fmt(v, p.digits)}
          yMarkers={[{ at: p.ref, color: INK.identity }, ...(p.extra != null ? [{ at: p.extra, label: 'biomass 1.5', color: INK.warn }] : [])]} />
      )} />
      <Key items={have.map((s) => ({ label: s.name, color: s.color, kind: 'line' as const }))} />
    </ChartFrame>
  )
}

type SummaryRows = 'site' | 'sites'
/** addis_05 §9 and multisite's summary_stats table: medians of each diagnostic by time block. */
export function SpectralSummary({ focus, site, spectra, sites, seasons }: {
  focus: Spectral | null; site: SiteMeta | null; spectra: Record<string, Spectral>; sites: SiteMeta[]; seasons: readonly SeasonMeta[]
}) {
  const [rows, setRows] = useSearchParam<SummaryRows>('dn.sum', 'site', ['site', 'sites'])
  const blocks = useMemo<{ key: string; title: string; color?: string; minuteFile: boolean; rows: BlockRow[] }[]>(() => {
    const of = (s: Spectral, g: DayGroup) => blockSummary(s.hours, s.ch.IR, s.d, (i) => s.keep(i) && inGroup(g, s.season[i]), s.minutes)
    if (rows === 'sites') return sites.flatMap((x) => (spectra[x.code] ? [{ key: x.code, title: x.name, color: x.color, minuteFile: spectra[x.code].file.resolution === 'minute', rows: of(spectra[x.code], allYear()) }] : []))
    if (!focus) return []
    const groups = [...dayGroups(daySplit('seasons', seasons), focus.season), allYear()]
    return groups.map((g) => ({ key: g.name, title: g.name === ALL_YEAR ? 'All' : seasonOnly(g.name), color: g.name === ALL_YEAR ? undefined : g.color, minuteFile: focus.file.resolution === 'minute', rows: of(focus, g) }))
  }, [rows, sites, spectra, focus, seasons])
  const num = { fontFamily: FONT.mono, textAlign: 'right' as const }
  return (
    <ChartFrame id="diurnal-spectral-summary" title={`The diagnostics by time of day — ${rows === 'site' && site ? `${site.name} by season` : 'every site, all year'}`} exportable={false}
      subtitle={`Medians over the day-hours in each of the multisite notebook's time blocks (addis_05 split the day 00–05, 05–09, 09–13, 13–17, 17–21, 21–24): IR BCc, UV/IR, the AAE from the BCc ratios (UV/IR and Blue/IR, the notebooks' definition), Delta-C and the five-channel CV. n counts day-hours with valid IR and, for minute records, the minutes behind them (the notebooks' n_minutes counted minutes). ${WINDOW_NOTE}`}
      provenance={`${A05} (§9, diurnal_wavelength_summary.csv); ${MS} (§10, multisite_summary_stats.csv)`}
      controls={<Segmented label="Rows" value={rows} options={['site', 'sites'] as SummaryRows[]} onChange={setRows} optionLabel={(k) => (k === 'site' ? (site ? `${site.name} by season` : 'This site') : 'Every site')} />}>
      {!blocks.length ? <Empty>No site in view has an hourly MA350 export.</Empty> : (
        <div style={{ overflowX: 'auto' }}>
          <table className="census-table">
            <thead><tr><th>Time block</th><th style={num}>n (day-hours)</th><th style={num}>IR BCc ({UG})</th><th style={num}>UV/IR</th><th style={num}>AAE UV/IR</th><th style={num}>AAE Blue/IR</th><th style={num}>Delta-C ({UG})</th><th style={num}>CV</th></tr></thead>
            <tbody>{blocks.flatMap((b) => [
              <tr key={b.key}><td colSpan={8} style={{ fontWeight: 650, color: b.color ?? INK.text, paddingTop: 12 }}>{b.title}</td></tr>,
              ...b.rows.map((r) => (
                <tr key={`${b.key}-${r.block}`}>
                  <td>{r.block}</td>
                  <td style={num}>{r.n.toLocaleString()}{b.minuteFile && r.n ? <span style={{ color: INK.muted }}> ({r.minutes.toLocaleString()} min)</span> : null}</td>
                  <td style={num}>{fmt(r.ir, 2)}</td><td style={num}>{fmt(r.uvIr, 3)}</td><td style={num}>{fmt(r.aaeUvIr, 2)}</td><td style={num}>{fmt(r.aaeBlueIr, 2)}</td>
                  <td style={num}>{signed(r.deltaC, 2)}</td><td style={num}>{fmt(r.cv, 3)}</td>
                </tr>
              )),
            ])}</tbody>
          </table>
        </div>
      )}
    </ChartFrame>
  )
}
