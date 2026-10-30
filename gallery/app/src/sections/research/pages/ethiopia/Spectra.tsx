import { useMemo } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'
import { Figure } from '@/components/Figure'
import { Legend, useLegend } from '@/components/Legend'
import { SpectrumPlot, type SpectrumAnnotation, type SpectrumBand, type SpectrumSeries } from '@/components/SpectrumPlot'
import { useChartOptions } from '@/lib/chartOptions'
import { useColourScheme } from '@/lib/colourSchemes'
import { f2 } from '@/lib/format'
import { FixedSetNote } from '@/lib/markedRows'
import { categoricalScale } from '@/lib/palettes'
import { num, type Row } from '@/lib/table'
import { INK } from '@/lib/theme'
import { Grid } from '../../common/ui'
import { provenance, rows as tableRows, type SpectralExport } from '../spectralCommon'
import { ALL, BAND_LABEL, BANDS, bandArea, flatten, groupOf, MONTH_SPLITS, OTHER, pool, profileStats, rankDesc, savgol, UNDATED } from './spectra'

/**
 * June 2026 notebooks 01 (the Adama PTFE spectra), 04 (the ETAD spectra by month), 06 [21] (mean raw vs
 * 2nd-derivative spectrum) and 13 [9] (the organic-band profile): spectra, not models, so everything is drawn in
 * the browser from the export's spectra (absorbance in mAbs; spectra.ts holds the maths). None of it depends on
 * the MAC. The export only has these tables where the Drive resolved (export_research_spectral.june_spectra);
 * without them the figures say so.
 */

const pct1 = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(1)} %`)
const mAbs = (v: unknown) => (num(v as never) ?? NaN) / 1000
const col = (rs: Row[], key: string) => rs.map((r) => mAbs(r[key]))
const ARTIFACT = 'rgba(214, 39, 40, 0.10)', ORGANIC = 'rgba(44, 160, 44, 0.08)'
const BANDS01: SpectrumAnnotation[] = [
  { lo: 1150, hi: 1250, label: 'CF (PTFE)', fill: ARTIFACT }, { lo: 2800, hi: 3000, label: 'CH', fill: ORGANIC },
  { lo: 1650, hi: 1750, label: 'C=O', fill: ORGANIC }, { lo: 3100, hi: 3500, label: 'OH', fill: ORGANIC },
]
const WINDOWS = [
  { value: 'full', label: 'Full range', window: undefined },
  { value: 'ch', label: '3600–2600 cm⁻¹', window: [3600, 2600] as [number, number] },
  { value: 'fingerprint', label: '1900–1000 cm⁻¹', window: [1900, 1000] as [number, number] },
]
const windowOf = (v: unknown) => WINDOWS.find((w) => w.value === v)?.window

export function SpectraMissing({ data }: { data: SpectralExport }) {
  return <ChartFrame id="ethiopia-spectra" title="The ETAD, Adama and training spectra (June 01, 04, 06, 13)" exportable={false}
    provenance={provenance(data, ['adama_peak_vs_ec'])}>
    <Note>This export was built without the Drive's spectra, so the spectra figures are not drawn; the intro lists what re-running the exporter needs.</Note>
  </ChartFrame>
}

// ---------------------------------------------------------------- 01: the five Adama PTFE spectra

export function AdamaSpectra({ data }: { data: SpectralExport }) {
  const spec = tableRows(data, 'adama_spectra'), saved = tableRows(data, 'adama_peak_vs_ec')
  const o = useChartOptions('ethiopia-adama-spectra', [
    { key: 'view', label: 'Spectra', default: 'raw', choices: [{ value: 'raw', label: 'Raw' }, { value: 'flat', label: "Flattened (01's end-point line)" }] },
    { key: 'win', label: 'Range', default: 'full', choices: WINDOWS.map(({ value, label }) => ({ value, label })) },
  ])
  const colours = useColourScheme('ethiopia-adama-spectra', 'categorical', 'Tableau 10')
  const lg = useLegend()
  const wn = useMemo(() => spec.map((r) => num(r.wavenumber) ?? NaN), [spec])
  const ids = saved.map((r) => String(r.spectrum))
  const label = (r: Row) => `${r.spectrum} (≈ ${r.FilterId_assumed}, general EC ${f2(r.EC_ftir_ug_m3)} µg/m³)`
  const colour = categoricalScale(colours.name, saved.map(label), colours.reverse)

  // 01 [9]: the flattened CH and C=O areas, recomputed here; the export checks the same arithmetic against 01's table
  const areas = useMemo(() => {
    const out = saved.map((r): Row => {
      const flat = flatten(col(spec, String(r.spectrum)), wn)
      return { spectrum: String(r.spectrum), filter: String(r.FilterId_assumed), EC: num(r.EC_ftir_ug_m3), CH: bandArea(flat, wn, 2800, 3000), CO: bandArea(flat, wn, 1650, 1750) }
    })
    const chRank = rankDesc(out.map((r) => r.CH as number)), coRank = rankDesc(out.map((r) => r.CO as number)), ecRank = rankDesc(out.map((r) => r.EC as number))
    return out.map((r, i): Row => ({ ...r, 'CH rank': chRank[i], 'C=O rank': coRank[i], 'EC rank': ecRank[i], label: `${r.filter} (${r.spectrum})` }))
  }, [saved, spec, wn])
  const top = areas.find((r) => r['EC rank'] === 1)

  const series: SpectrumSeries[] = saved.filter((r) => lg.show(label(r))).map((r) => {
    const y = col(spec, String(r.spectrum))
    return { id: String(r.spectrum), label: label(r), group: label(r), wn, y: o.values.view === 'flat' ? flatten(y, wn) : y, color: colour(label(r)), role: 'line', width: 1.1 }
  })
  return <>
    <ChartFrame id="ethiopia-adama-spectra" title="The five Adama PTFE spectra — the Teflon CF doublet and a sloping, loading-dependent baseline dominate"
      subtitle={`June 01: the raw FTIR spectra of the five Adama PTFE filters (batch 54, sample rows ${ids.join(', ')}), with the PTFE carbon–fluorine doublet (about 1150–1250 cm⁻¹) and the aerosol bands 01 marks (CH, C=O, OH). The non-zero absorbance at 4000 cm⁻¹ is Teflon scattering, which grows with loading: a confounder, not signal. 01's map from spectrum row to filter is assumed (sorted rows to sorted filter ids) until a barcode crosswalk confirms it. "Flattened" subtracts 01's crude straight line through each spectrum's two end points, the correction behind the band areas below.`}
      controls={o.controls()} provenance={provenance(data, ['adama_spectra', 'adama_peak_vs_ec'], 'research/ftir_ec_calibration_2026_06_25/01_adama_spectra.ipynb fig01.')}>
      <SpectrumPlot series={series} window={windowOf(o.values.win)} annotations={BANDS01} hover="readout" dim={lg.dim} height={320} zeroLine={o.values.view === 'flat'}
        yLabel={o.values.view === 'flat' ? 'Absorbance − end-point line' : 'Absorbance'} />
      <Legend {...lg.props} items={saved.map((r) => ({ label: label(r), color: colour(label(r)), shape: 'line' as const }))} />
    </ChartFrame>

    <Figure kind="crossplot" id="ethiopia-adama-peaks" data={areas} x="EC" y="CH" idColumn="label" tipColumns={['CH rank', 'C=O rank', 'EC rank']} fit={false} sameScale={false} maxSide={380}
      xLabel="General FTIR EC (µg/m³)" yLabel="CH band area, 2800–3000 cm⁻¹ (flattened)"
      variants={[{ key: 'band', label: 'Band', choices: [{ value: 'CH', label: 'CH (2800–3000)' }, { value: 'CO', label: 'C=O (1650–1750)', patch: { y: 'CO', yLabel: 'C=O band area, 1650–1750 cm⁻¹ (flattened)' } }] }]}
      title={`Does the highest-EC Adama filter have the biggest CH peak? No: ${top ? `${top.filter} is first in EC and ${top['CH rank']}${top['CH rank'] === 2 ? 'nd' : top['CH rank'] === 3 ? 'rd' : 'th'} in CH area` : 'the ranks differ'}`}
      subtitle="June 01's hook for the meeting: if the calibration misses char, peak size should not track EC. Band areas of the flattened spectra (negative: the end-point line runs above these bands), recomputed here from the spectra; they equal 01's saved table (tables/adama_spectra_peak_vs_ec.csv) to 4 decimals. The EC is the deployed general calibration's, on 01's assumed spectrum-to-filter map. Five filters and a crude baseline: a motivation, not a result."
      provenance={provenance(data, ['adama_spectra', 'adama_peak_vs_ec'], 'research/ftir_ec_calibration_2026_06_25/01_adama_spectra.ipynb fig01b.')} />
  </>
}

// ---------------------------------------------------------------- 04: the ETAD spectra, all and by month split

export function EtadSpectra({ data }: { data: SpectralExport }) {
  const dist = tableRows(data, 'etad_spectra_distribution'), mom = tableRows(data, 'etad_month_moments'), meta = tableRows(data, 'etad_spectra_meta')
  const o = useChartOptions('ethiopia-etad-spectra', [
    { key: 'split', label: 'Group', style: 'select', default: 'all', choices: MONTH_SPLITS.map((s) => ({ value: s.key, label: s.label })) },
    { key: 'win', label: 'Range', default: 'full', choices: WINDOWS.map(({ value, label }) => ({ value, label })) },
  ])
  const colours = useColourScheme('ethiopia-etad-spectra', 'categorical', 'Tableau 10')
  const lg = useLegend()
  const split = MONTH_SPLITS.find((s) => s.key === o.values.split) ?? MONTH_SPLITS[0]
  const wn = useMemo(() => dist.map((r) => num(r.wavenumber) ?? NaN), [dist])

  const groups = useMemo(() => {
    if (split.key === 'all') return []
    const names = [...split.groups.map(([g]) => g), OTHER, UNDATED]
    const monthOf = (r: Row) => (num(r.month) == null ? 'none' : String(num(r.month)))
    return names.flatMap((g) => {
      const months = [...new Set(meta.filter((r) => groupOf(num(r.month), split) === g).map(monthOf))]
      const parts = months.map((m): [number, number[], number[]] => [meta.filter((r) => monthOf(r) === m).length, col(mom, `mean ${m}`), col(mom, `sd ${m}`)])
      const p = pool(parts)
      return p.n ? [{ name: `${g} (n = ${p.n})`, ...p }] : []
    })
  }, [split, meta, mom])
  const colour = categoricalScale(colours.name, groups.map((g) => g.name), colours.reverse)

  let series: SpectrumSeries[], bands: SpectrumBand[], items: { label: string; color: string; shape: 'line' | 'square' }[]
  if (split.key === 'all') {
    const outer = '5–95 % of spectra', inner = 'interquartile range', med = 'median', mean = 'mean spectrum (04)'
    bands = [
      ...(lg.show(outer) ? [{ id: outer, wn, lo: col(dist, 'q5'), hi: col(dist, 'q95'), color: '#4e79a7', opacity: 0.14, group: outer }] : []),
      ...(lg.show(inner) ? [{ id: inner, wn, lo: col(dist, 'q25'), hi: col(dist, 'q75'), color: '#4e79a7', opacity: 0.3, group: inner }] : []),
    ]
    series = [
      ...(lg.show(med) ? [{ id: med, label: med, group: med, wn, y: col(dist, 'q50'), color: '#4e79a7', role: 'line' as const, width: 1.2 }] : []),
      ...(lg.show(mean) ? [{ id: mean, label: mean, group: mean, wn, y: col(dist, 'mean'), color: INK.text, role: 'line' as const, width: 1.4 }] : []),
    ]
    items = [{ label: outer, color: '#4e79a7', shape: 'square' }, { label: inner, color: '#4e79a7', shape: 'square' }, { label: med, color: '#4e79a7', shape: 'line' }, { label: mean, color: INK.text, shape: 'line' }]
  } else {
    const shown = groups.filter((g) => lg.show(g.name))
    series = shown.map((g) => ({ id: g.name, label: g.name, group: g.name, wn, y: g.mean, color: colour(g.name), role: 'line', width: 1.3 }))
    bands = shown.map((g) => ({ id: g.name, group: g.name, wn, lo: g.mean.map((v, j) => v - g.sd[j]), hi: g.mean.map((v, j) => v + g.sd[j]), color: colour(g.name), opacity: 0.12 }))
    items = groups.map((g) => ({ label: g.name, color: colour(g.name), shape: 'line' }))
  }
  const dated = meta.filter((r) => num(r.month) != null).length

  return <ChartFrame id="ethiopia-etad-spectra" title="All 319 ETAD spectra — a loading-driven baseline sets most of the spread; the month split moves the level more than the band shapes"
    subtitle={`June 04: the raw FTIR spectra of every ETAD (Addis) spectrum (19 filters scanned twice). 04 overlaid all 319 with their mean (fig 1), then split them by a placeholder "high-biomass (Oct–Feb) vs diesel (Jun–Aug)" month list that was meant to be replaced by Navid's months (fig 2). Here the spread is drawn as quantile bands (every 4th channel, about 5 cm⁻¹), and any month split is pooled exactly from per-month means and standard deviations: line = group mean, band = ±1 SD. ${dated} of 319 spectra have a sampling date in the ETAD metadata (04's join); the other ${319 - dated} are "${UNDATED}". The two Ethiopian calendars differ only in February. Absorbance is raw: the Teflon CF doublet near 1200 cm⁻¹ is substrate.`}
    controls={o.controls()} provenance={provenance(data, ['etad_spectra_distribution', 'etad_month_moments', 'etad_spectra_meta'], 'research/ftir_ec_calibration_2026_06_25/04_ethiopia_etad_spectra.ipynb figs 04 (all, by season).')}>
    <SpectrumPlot series={series} bands={bands} window={windowOf(o.values.win)} annotations={[BANDS01[0]]} hover="readout" dim={lg.dim} height={330} />
    <Legend {...lg.props} items={items} note={split.key === 'all' ? undefined : 'line: group mean · band: ±1 SD over its spectra'} />
    <FixedSetNote what="the notebook's 319 ETAD spectra, summarised by the exporter" />
  </ChartFrame>
}

// ---------------------------------------------------------------- 13 [9]: the organic-band profile

export function BandProfile({ data }: { data: SpectralExport }) {
  const meta = tableRows(data, 'etad_spectra_meta'), printed = tableRows(data, 'band_profile_printed')
  const improve = printed.find((r) => String(r.set).startsWith('IMPROVE'))
  const IMPROVE = "IMPROVE lot 248+251 (13's printed median)"
  const long = useMemo(() => {
    const out: Row[] = []
    for (const r of meta) for (const b of BANDS) {
      const row: Row = { band: BAND_LABEL[b], fraction: num(r[b]), id: String(r.ExternalFilterId) }
      for (const s of MONTH_SPLITS) row[`group ${s.key}`] = groupOf(num(r.month), s)
      out.push(row)
    }
    if (improve) for (const b of BANDS) out.push({ band: BAND_LABEL[b], fraction: num(improve[b]), ...Object.fromEntries(MONTH_SPLITS.map((s) => [`group ${s.key}`, IMPROVE])) })
    return out
  }, [meta, improve])
  const all = profileStats(meta)
  const at = (b: string) => all.find((s) => s.band === b)
  return <Figure kind="intervals" id="ethiopia-band-profile" data={long} row="band" series="group all" labelWidth={210}
    panels={[{ value: 'fraction', label: 'Share of the four organic bands\' area (median and IQR over spectra)' }]} aggregate={{ stat: 'median', interval: 'iqr' }}
    variants={[{ key: 'split', label: 'ETAD grouped by', choices: MONTH_SPLITS.map((s) => ({ value: s.key, label: s.label.replace(/ \((04 )?fig \d\)$/, ''), patch: { series: `group ${s.key}` } })) }]}
    colors={{ [IMPROVE]: INK.text, [ALL]: '#d62728' }}
    title={`Addis's organic bands are OH-dominated (median ${pct1(at('OH')?.median)}) with almost no aromatic signal (${pct1(at('aromatic')?.median)}), and the lot-248+251 training set looks the same`}
    subtitle={`June 13's peak-ratio "biomass classifier": offset each spectrum by its 3900–4000 cm⁻¹ mean, integrate CH, C=O, OH and the aromatic C=C band above each band's own minimum, and take each as a share of the four. Per ETAD spectrum here (recomputed from the Drive's spectra; the medians equal 13's printed profile), grouped by any month split; IMPROVE is 13's printed median over the 13,012 training filters (median ${improve ? BANDS.map((b) => `${b} ${pct1(num(improve[b]))}`).join(', ') : '—'}), as its per-filter fractions were not saved. 13 read it as oxygenated smouldering-wood smoke rather than aromatic char, and since the training set already matches, selecting "biomass-like" IMPROVE filters by this profile did not move the HIPS fit (the ML page's frontier). Band areas, no model: no MAC.`}
    provenance={provenance(data, ['etad_spectra_meta', 'band_profile_printed'], 'research/ftir_ec_calibration_2026_06_25/13_chase_origin_and_biomass.ipynb fig13_organic_band_profile.')} />
}

// ---------------------------------------------------------------- 06 [21]: mean raw vs mean 2nd derivative

export function DerivativeSpectra({ data }: { data: SpectralExport }) {
  const means = tableRows(data, 'spectra_means')
  const o = useChartOptions('ethiopia-derivative', [
    { key: 'w', label: 'Savitzky–Golay window', default: '11', choices: ['5', '7', '11', '15', '21'].map((v) => ({ value: v, label: v === '11' ? '11 (06, Weakley)' : v })) },
    { key: 'win', label: 'Range', default: 'full', choices: WINDOWS.map(({ value, label }) => ({ value, label })) },
  ])
  const lg = useLegend()
  const wn = useMemo(() => means.map((r) => num(r.wavenumber) ?? NaN), [means])
  const sets = [{ key: 'training 906', label: '906-filter IMPROVE training set (CAL-0)', color: '#1f77b4' }, { key: 'ETAD 319', label: 'ETAD (319 spectra)', color: '#d62728' }]
  const w = Number(o.values.w)
  const curves = useMemo(() => sets.map((s) => { const raw = col(means, s.key); return { ...s, raw, d2: savgol(raw, w, 2, 2) } }), [means, w]) // eslint-disable-line react-hooks/exhaustive-deps
  const shown = curves.filter((c) => lg.show(c.label))
  const line = (c: (typeof curves)[number], y: number[]): SpectrumSeries => ({ id: c.key, label: c.label, group: c.label, wn, y, color: c.color, role: 'line', width: 1 })
  const annotations = [...BANDS01.slice(1), BANDS01[0]]
  return <ChartFrame id="ethiopia-derivative" title="The 2nd derivative strips the sloping PTFE baseline and sharpens the bands — why it needs fewer components"
    subtitle={`June 06 [21] (Weakley's preprocessing): the mean raw spectrum of the 906-filter calibration set and its mean Savitzky–Golay 2nd derivative (polynomial order 2, window ${w} channels of about 1.29 cm⁻¹). The derivative is linear, so the mean of the derivatives is the derivative of the mean, taken here in the browser exactly as scipy does (edges by polynomial fit; per channel², as 06 left delta at 1). The ETAD mean (not in 06) is drawn on the same grid: the raw level differs by loading and lot, while the derivative bands line up. The RMSECV comparison of raw and 2nd-derivative calibrations (06 [19], 09 [9]) needs the curves only the notebooks drew; their component picks are on Machine learning → June 2026 variants.`}
    controls={o.controls()} provenance={provenance(data, ['spectra_means'], 'research/ftir_ec_calibration_2026_06_25/06_calibration_variants_components.ipynb fig06_second_derivative_spectra.')}>
    <Grid min={420}>
      <SpectrumPlot title="Mean raw spectrum" series={shown.map((c) => line(c, c.raw))} window={windowOf(o.values.win)} annotations={annotations} hover="readout" dim={lg.dim} height={260} />
      <SpectrumPlot title={`Mean 2nd derivative (window ${w})`} series={shown.map((c) => line(c, c.d2))} window={windowOf(o.values.win)} annotations={annotations} hover="readout" dim={lg.dim} height={260}
        zeroLine yLabel="2nd derivative (per channel²)" yClip={[0.005, 0.995]} yFormat={(v) => (v === 0 ? '0' : v.toExponential(0))} readoutFormat={(v) => v.toExponential(2)} />
    </Grid>
    <Legend {...lg.props} items={sets.map((s) => ({ label: s.label, color: s.color, shape: 'line' as const }))} />
    <FixedSetNote what="the mean spectra of the notebook's 906 training filters and 319 ETAD spectra" />
  </ChartFrame>
}
