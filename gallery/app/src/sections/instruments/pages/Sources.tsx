import { useMemo } from 'react'
import { Segmented, Toggle } from '@/components/ChartFrame'
import { f2 } from '@/lib/format'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { useAethSites } from '../lib'
import { AAE_PAIRS, AAE_PAIR_KEYS, MA350_NM, bccOffset, ma350Ratio, regionsOf, type AaePair } from '../sources/aae'
import { BASES, type Basis } from '../sources/AaeValues'
import { AaeDistribution } from '../sources/AaeDistribution'
import { AaeTimeline } from '../sources/AaeTimeline'
import { ClassShares } from '../sources/ClassShares'
import { CrossSectionSweep } from '../sources/CrossSectionSweep'
import { InstrumentCheck } from '../sources/InstrumentCheck'
import { MethodBySource } from '../sources/MethodBySource'
import { AaeVsBurning, DominanceRules, PmfVsAae } from '../sources/PmfAgreement'
import { ResidualVsDominance, ThresholdSweep } from '../sources/ThresholdSweep'
import { JACROS, useDailySites, useInstrument, usePmf } from '../sources/data'

/**
 * Sub-page 'sources' of the Aethalometer tab: the absorption Ångström exponent, the fossil / mixed /
 * biomass classes it implies, and how those line up with the ETAD PMF factors and with the method
 * comparisons split by source. Ports addis_01_source_apportionment (+ _daily), dominant_source_comparison,
 * dominant_source_threshold_analysis, threshold_by_source_analysis, source_apportionment_regression and
 * addis_ababa_source_analysis; the charts live in ../sources.
 *
 * Every AAE here is from absorption (BCc × the MA350 cross-section). The notebooks took it from BCc,
 * which sits about 1 lower for every sample; the page says where that moves their numbers.
 */
export function Sources({ rows, meta }: SectionPageProps) {
  const [pair, setPair] = useSearchParam<AaePair>('aes.pair', 'UV-IR', AAE_PAIR_KEYS)
  const [basis, setBasis] = useSearchParam<Basis>('aes.basis', 'Daily bins', BASES)
  const [jacros, setJacros] = useSearchParam<'0' | '1'>('aes.jacros', '0', ['0', '1'])
  const regions = regionsOf(meta as typeof meta & { aae_regions?: { fossil_max: number; biomass_min: number } })
  const aethSites = useAethSites()
  const pmf = usePmf() ?? null
  const instrument = useInstrument() ?? null

  // the sites in view with an aethalometer record, in the gallery's order; Jacros beside Addis on request
  const codes = useMemo(() => {
    const inView = new Set(rows.map((r) => r.code))
    const have = new Set(Object.keys(aethSites?.sites ?? {}))
    const own = meta.sites.map((s) => s.code).filter((c) => inView.has(c) && have.has(c))
    // the Jacros file is the same instrument over a longer span: it replaces the filter-site file, not joins it
    return jacros === '1' && have.has(JACROS) ? own.map((c) => (c === 'ETAD' ? JACROS : c)) : own
  }, [rows, meta.sites, aethSites, jacros])
  const daily = useDailySites(codes, meta)
  const p = AAE_PAIRS[pair]
  const offset = bccOffset(ma350Ratio(p.short, p.long), MA350_NM[p.short], MA350_NM[p.long])
  const pmfInView = !!pmf && rows.some((r) => r.code === pmf.site)

  return (
    <>
      <div className="panel">
        <div className="weather-head-controls" style={{ flexWrap: 'wrap', gap: 12 }}>
          <Segmented label="AAE pair" value={pair} options={AAE_PAIR_KEYS} onChange={setPair} title="Which two channels the AAE is taken between" />
          <Segmented label="Basis" value={basis} options={BASES} onChange={setBasis} title="Every daily bin of the instrument record, or each filter's sampling window (follows the Filters panel)" />
          {(codes.includes('ETAD') || codes.includes(JACROS)) && aethSites?.sites[JACROS] && <Toggle label="Addis: whole record (Jacros file)" checked={jacros === '1'} onChange={(v) => setJacros(v ? '1' : '0')}
            title={`Daily charts: Addis's MA350-0238 from ${aethSites.sites[JACROS].first.slice(0, 7)} to ${aethSites.sites[JACROS].last.slice(0, 7)} (the Jacros file) instead of the filter-site file's ${aethSites.sites.ETAD?.first.slice(0, 7)} to ${aethSites.sites.ETAD?.last.slice(0, 7)}. Same instrument: identical where both have a bin.`} />}
        </div>
        <p className="subtitle" style={{ margin: '10px 0 4px' }}>
          AAE {pair} = −ln(b<sub>{p.short}</sub> / b<sub>{p.long}</sub>) / ln({MA350_NM[p.short]} / {MA350_NM[p.long]}) from <b>absorption</b>, b = BCc × the MA350 cross-section
          (research/ftir_hips_chem/scripts/optics.py). Classes follow config.AAE_REGIONS: fossil fuel ≤ {regions.fossil_max}, mixed, biomass ≥ {regions.biomass_min}.
          An AAE taken from BCc, as every source notebook did, is the same number {f2(offset)}: the instrument has already divided each channel by a cross-section that falls with wavelength.
        </p>
        <details>
          <summary className="subtitle" style={{ cursor: 'pointer' }}>Where this page's numbers differ from the notebooks', and why</summary>
          <ul className="subtitle" style={{ margin: '6px 0 0', paddingLeft: 18 }}>
            <li><b>AAE scale (addis_01, addis_01_daily).</b> The notebooks' AAE was from BCc, so about 1 too low: at Addis it classes roughly a fifth of days as biomass where absorption classes about half, and most mixed days as fossil (optics.py: 52.4 % biomass from absorption against 19.7 % from BCc). Their "MAC sensitivity" pairs all sat below the instrument's own UV/IR ratio of {f2(ma350Ratio('UV', 'IR'))}; the cross-section chart below shows the whole range. The instrument's own AAE (Beijing, Delhi and JPL; the Blue/IR AAE from absorption) sits on the gallery's scale, not on the notebooks'.</li>
            <li><b>Record and filtering (addis_01).</b> addis_01 read the Jacros 60-s record from Google Drive, which is not in the repo, and clipped AAE to −1…3 after dropping UV/IR ratios outside 0.1–10. Here: the daily bins, nothing clipped from the statistics. The Jacros file is the same MA350-0238 as the Addis filter-site file (serial in both pickles; every channel identical on the 515 bins they share), so the toggle above swaps in its longer span rather than adding a second instrument.</li>
            <li><b>Matching (the PMF and regression notebooks).</b> They joined the aethalometer by a legacy 9 AM date match (97 of 102 PMF days had IR BCc; HIPS × MAC as "Fabs/MAC"). Here each filter carries the overlap-weighted mean of the daily bins over its own sampling window, kept where they cover 12 h (export_aethalometer.py), which fits HIPS far better (IR BCc vs HIPS R² about 0.78 at Addis against their 0.60) and leaves fewer PMF filters with an AAE.</li>
            <li><b>Fits and thresholds.</b> The notebooks reported OLS only and five dominance thresholds (30–70 %); here the sidebar's estimators (Deming by default, as AGENTS.md asks for a 1:1 comparison) and a sweep through every threshold.</li>
            <li><b>Not repeated here.</b> The daily PMF contribution bars and seasonal source shares (source_apportionment_regression Task 5, addis_ababa_source_analysis Task 4) are the PMF sources tab.</li>
          </ul>
        </details>
      </div>
      {daily === null ? <div className="panel"><p className="frame-empty">Loading the aethalometer daily bins…</p></div> : <>
        <AaeDistribution rows={rows} meta={meta} daily={daily} pair={pair} basis={basis} regions={regions} />
        <ClassShares rows={rows} meta={meta} daily={daily} pair={pair} basis={basis} regions={regions} />
        <AaeTimeline daily={daily} pair={pair} regions={regions} />
        <CrossSectionSweep daily={daily} pair={pair} regions={regions} />
        <InstrumentCheck daily={daily} pair={pair} instrument={instrument} />
      </>}
      {pmf && pmfInView && <>
        <PmfVsAae rows={rows} meta={meta} pmf={pmf} pair={pair} regions={regions} />
        <AaeVsBurning rows={rows} meta={meta} pmf={pmf} pair={pair} regions={regions} />
        <DominanceRules pmf={pmf} />
      </>}
      {pmf && !pmfInView && <div className="panel"><p className="frame-empty">The PMF charts need {pmf.site_name}: the PMF solution covers {pmf.n} of its filters only. Add it in the Filters panel to compare the AAE classes with the PMF factors.</p></div>}
      <MethodBySource rows={rows} meta={meta} pmf={pmf} pair={pair} regions={regions} />
      <ThresholdSweep rows={rows} meta={meta} pmf={pmf} pair={pair} regions={regions} />
      <ResidualVsDominance rows={rows} meta={meta} pmf={pmf} pair={pair} regions={regions} />
    </>
  )
}
