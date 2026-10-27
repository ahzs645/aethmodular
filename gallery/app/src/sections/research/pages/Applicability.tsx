import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { num, type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { PageIntro, presetName, provenance, rows, useCalibRuns, useSpectral, Waiting } from './spectralCommon'

const CONFIG: Record<string, string> = {
  addis_winner: 'Addis winner (OC/EC 440, spline)',
  common_candidate: 'Common (analogs 440, 2nd deriv.)',
  delhi_winner: 'Delhi winner (analogs 530, 2nd deriv.)',
}
const TARGET: Record<string, string> = { addis_holdout: 'Addis holdout (14)', delhi_holdout: 'Delhi holdout (26)' }
const VARIANT: Record<string, string> = {
  ordinary_pls: 'ordinary PLS', target_centered_lambda0: 'target-centred, λ = 0', dipls_target_centered_heuristic: 'di-PLS target-centred, heuristic λ',
  dipls_target_centered_selected: 'di-PLS target-centred, selected λ', dipls_source_centered_heuristic: 'di-PLS source-centred, heuristic λ',
  dipls_source_centered_selected: 'di-PLS source-centred, selected λ',
}
const at = (map: Record<string, string>, v: unknown) => map[String(v)] ?? String(v)

/**
 * ftir_36, ftir_39, ftir_50 and char_12: is Addis inside the domain the calibration was built on, and
 * does that make the calibration valid there? The explorer's per-run domain flags (calibration_runs.json)
 * cover every gallery calibration; the notebooks' leakage-safe audits come from their saved outputs.
 */
export function Applicability(_props: SectionPageProps) {
  const { data, error } = useSpectral('applicability')
  const { runs } = useCalibRuns()

  const flags = useMemo(() => {
    if (!runs) return []
    const label = new Map(runs.presets.map((p) => [p.key, presetName(p)]))
    return runs.runs.flatMap((r): Row[] => {
      const base = { calibration: label.get(r.preset) ?? r.preset, site: runs.targets[r.target]?.label.split(':')[0] ?? r.target }
      return [{ ...base, measure: 'score envelope (Mahalanobis D², training p95)', pct: r.extrap_pct }, { ...base, measure: 'Q residual (training p95)', pct: r.q_residual_pct }]
    })
  }, [runs])

  if (!data) return <Waiting error={error} />
  const pair = (r: Row) => `${at(CONFIG, r.config)} → ${at(TARGET, r.target)}`
  const domain = rows(data, 'domain_summary').map((r): Row => ({ ...r, pair: pair(r) }))
  const change = rows(data, 'dipls_vs_pls').map((r): Row => ({ ...r, pair: pair(r), centering: `${r.centering}-centred` }))
  const variants = [
    ...rows(data, 'target_summary').map((r): Row => ({ evaluation: at(TARGET, r.target), config: at(CONFIG, r.config), variant: at(VARIANT, r.variant), RMSE: num(r.RMSE_1to1) })),
    ...rows(data, 'source_validation').map((r): Row => ({ evaluation: 'untouched IMPROVE sites (outer 20 %)', config: at(CONFIG, r.config), variant: at(VARIANT, r.variant), RMSE: num(r.RMSE_1to1) })),
  ]

  return <>
    <PageIntro id="applicability-intro" title="Applicability domain — in-domain is a warning system, not a validation" data={data}
      notebooks="research/ftir_ec_phase3/ftir_36_domain_invariant_pls, ftir_39_pls_applicability_domain, ftir_50_spectral_comparison_methods §2–3, §5–6; research/ftir_etad_char/char_12_weakley_transferability">
      <p>The Baselines tab's research diagnostics show the spectral distance of each correction's spectra from its training set. This page asks the calibration question: does being inside the domain make a calibration right at Addis?</p>
    </PageIntro>

    {flags.length > 0 && <Figure kind="heatmap" id="applicability-flags" data={flags} row="site" column="calibration" value="pct" ramp="sequential" domain={[0, 100]} format="f2"
      choose={[{ column: 'measure', label: 'Distance' }]} valueLabel="% of target filters outside" cell={[78, 28]}
      title="Every calibration at every site — the share of target filters outside the training envelope"
      subtitle="The explorer's per-run flags: share of each site's filters beyond the 95th percentile of the calibration's own training distribution, in the model's score space (D²) or off it (Q residual). The corrected Lowest-OC/EC models put almost no Addis filter outside; the raw Ethiopia-shaped and analog cohorts put most outside."
      provenance={<>Extends ftir_30 §5 and ftir_39 to every gallery preset. <code>calibration_runs.json</code> runs[].extrap_pct, q_residual_pct.</>} />}

    <Figure kind="intervals" id="applicability-ftir39" data={domain} row="pair"
      panels={[{ value: 'outside_source_95_pct', label: 'Outside source 95 % (T² or Q), %', reference: 0 },
        { value: 'RMSE_inside', label: 'RMSE, in-domain filters (µg/m³)' }, { value: 'RMSE_outside', label: 'RMSE, out-of-domain filters (µg/m³)' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={360}
      title="ftir_39 — the Addis winner is fully in-domain and still misses Addis"
      subtitle="Leakage-safe audit: a scaler and PCA fitted on source training spectra only, T² and Q thresholds from untouched IMPROVE sites, target labels joined afterwards. Reference: |PLS − HIPS/10|. Derivative-space candidates put 38–64 % of their targets outside and need an explicit flag; the winner puts none outside and keeps a large response bias — conditional shift, not covariate shift."
      provenance={provenance(data, ['domain_summary'])} />

    <Figure kind="intervals" id="applicability-dipls" data={change} row="pair" choose={[{ column: 'centering', label: 'Centering', default: 'source-centred' }]}
      panels={[{ value: 'RMSE_change_pct', lo: 'RMSE_change_ci_low', hi: 'RMSE_change_ci_high', label: 'RMSE change vs ordinary PLS on the locked holdout (%; paired-bootstrap 95 %)', reference: 0 }]} labelWidth={360}
      title="ftir_36 — domain-invariant PLS does not rescue a universal calibration"
      subtitle="Components and the penalty were chosen within IMPROVE before any target label was read. Target-centring (the published default) forces the target's mean predicted loading to the IMPROVE mean: switch Centering to see Delhi RMSE inflate by 940–2190 %. Source-centred, the one gain (common candidate at Delhi, −9 %) barely excludes zero, and the same model stays flat at Addis."
      provenance={provenance(data, ['dipls_vs_pls'])} />

    <Figure kind="heatmap" id="applicability-dipls-rmse" data={variants} row="config" column="variant" value="RMSE" ramp="sequential" format="f2" valueLabel="RMSE (µg/m³ on targets, µg/filter on IMPROVE)"
      choose={[{ column: 'evaluation', label: 'Evaluated on' }]} cell={[96, 28]}
      title="ftir_36 — RMSE of every variant, on unseen IMPROVE sites and on the two locked holdouts"
      subtitle="On untouched IMPROVE sites every variant is within a few per cent of ordinary PLS; the failure appears only at the SPARTAN targets, where the site difference includes a real loading and response change."
      provenance={provenance(data, ['target_summary', 'source_validation'])} />

    <Figure kind="intervals" id="applicability-t2q" data={rows(data, 't2q')} row="target"
      panels={[{ value: 'pct_T2_over_limit', label: 'T² over library 95th pct (%)', reference: 5 }, { value: 'pct_Q_over_limit', label: 'Q over library 95th pct (%)', reference: 5 },
        { value: 'pct_both', label: 'Flagged by both (%)' }]} aggregate={{ stat: 'median', interval: 'none' }} labelWidth={150}
      title="ftir_50 — “Addis is in-domain” splits into two distances that flag different filters"
      subtitle="A 10-component PCA of the 13,634-spectrum IMPROVE library (spline baseline). Hotelling T² is distance inside the model plane; Q is what the plane cannot reconstruct. At Addis the two flags overlap on a third of the flagged filters, so a single Mahalanobis number cannot settle extrapolation."
      provenance={provenance(data, ['t2q'])} />

    <Figure kind="intervals" id="applicability-hubness" data={rows(data, 'hubness')} row="target"
      panels={[{ value: 'unique_ratio', label: 'Distinct neighbours per neighbour slot (top 50)' }, { value: 'pct_no_mutual', label: 'Filters with no mutual analog (%)' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={150}
      title="ftir_50 — Addis analog neighbourhoods are the most redundant of the five sites"
      subtitle="Hubness: a top-50 analog list reuses the same library spectra. Addis draws the fewest distinct neighbours per slot and has the most filters without a selectivity-matched mutual analog: a top-N analog cohort at Addis is not N independent analogs. (The notebook's tl;dr quotes 0.19 and 36 %; its executed table, drawn here, gives these values.)"
      provenance={provenance(data, ['hubness'])} />

    <Figure kind="table" id="applicability-neighbours" data={rows(data, 'neighbours')}
      columns={[{ key: 'target', label: 'Target' }, { key: 'median_top1_r', label: 'Median top-1 r', format: 'f3' }, { key: 'median_top50_mean_r', label: 'Median top-50 mean r', format: 'f3' },
        { key: 'worst_filter_top1_r', label: 'Worst filter top-1 r', format: 'f3' }, { key: 'distinct_sites_in_neighbourhoods', label: 'Sites in neighbourhoods', format: 'int' }, { key: 'dominant_neighbour_site', label: 'Commonest neighbour site' }]}
      title="ftir_50 — nearest analogs per filter, not per cohort"
      subtitle="The per-filter neighbour similarity is an applicability statistic that can travel with each prediction. Addis has the weakest worst-filter match."
      provenance={provenance(data, ['neighbours'])} />

    <Figure kind="heatmap" id="applicability-agreement" data={rows(data, 'metric_agreement')} row="metric" column="against" value="rho" domain={[-1, 1]} format="f2" valueLabel="Spearman ρ between library rankings" cell={[110, 26]}
      title="ftir_50 — similarity metrics disagree on which library spectra are most Addis-like"
      subtitle="Rank agreement between five ways of ranking the library by Addis-likeness (Pearson, spectral angle, information divergence to the Addis median; nearest Pearson to any Addis filter; T² distance to the Addis centre). Any single top-N cohort is one choice among several."
      provenance={provenance(data, ['metric_agreement'])} />

    <Figure kind="intervals" id="applicability-escape" data={rows(data, 'escape')} row="group"
      panels={[{ value: 'median escape', label: 'Median envelope escape (share of pertinent wavenumbers)', reference: 0.1 }, { value: 'share > 2× null', label: 'Share of filters above 2 × null' }]}
      aggregate={{ stat: 'median', interval: 'none' }} labelWidth={200}
      title="char_12 — the deployed calibration is extrapolated at Addis, in every season"
      subtitle="Weakley et al. (2018)'s transferability test: share of the C–H (2800–3000) and C=O/aromatic (1450–1850 cm⁻¹) wavenumbers where a filter's second-derivative spectrum leaves the IMPROVE 5–95 % envelope (916 IMPROVE spectra). An IMPROVE spectrum escapes its own envelope about 10 % of the time (reference). Not a dry-season or char-anomaly effect: Kiremt is the mildest."
      provenance={provenance(data, ['escape'])} />

    <Figure kind="table" id="applicability-lots" data={rows(data, 'lot_effect')}
      title="char_12 — the filter-lot difference is season in disguise"
      subtitle="Escape differs by PTFE lot over all filters but not within the dry season (lot 248 is dry-season heavy)."
      provenance={provenance(data, ['lot_effect'])} />

    <Figure kind="table" id="applicability-scale" data={rows(data, 'raw_scale')}
      title="char_12 — same substrate, heavier deposits"
      subtitle="Median raw absorbance: the PTFE C–F peak matches (same filter material) while the featureless 2000 cm⁻¹ background is twice as high at Addis."
      provenance={provenance(data, ['raw_scale'])} />
  </>
}
