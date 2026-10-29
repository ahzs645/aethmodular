import { Figure } from '@/components/Figure'
import { type Row } from '@/lib/table'
import type { SectionPageProps } from '../../types'
import { SiteSimilarityMatrix } from './SiteSimilarityMatrix'
import { PageIntro, provenance, rows, useSpectral, Waiting } from './spectralCommon'

const tag = (list: string, rs: Row[]) => rs.map((r): Row => ({ list, ...r }))

/**
 * ftir_52: the IMPROVE network as a spectral map. Ward classes over the 13,634-spectrum library with
 * the five SPARTAN targets, site similarity, a validated match threshold, and which IMPROVE spectra
 * each Ethiopian season draws. All tables are the executed notebook's; the site × site matrix is
 * clustered here (lib/cluster) for the dendrogram.
 */
export function NetworkMap(_props: SectionPageProps) {
  const { data, error } = useSpectral('network_map')
  if (!data) return <Waiting error={error} />
  const k = rows(data, 'silhouette').reduce((best, r) => (Number(r.silhouette) > Number(best.silhouette) ? r : best), { k: null, silhouette: -1 } as Row)

  return <>
    <PageIntro id="network-intro" title="The IMPROVE network as a spectral map — classes, neighbours, and why a near-identical spectrum is not a calibration match" data={data}
      notebooks="research/ftir_ec_phase3/ftir_52_network_spectral_map">
      <p>The Spectral similarity tab ranks sites by mean filter-pair r; this page adds the population view: Ward classes, the smoke-class test, and a match threshold validated on EC. The notebook's written tl;dr predates its final run (it names class 2 and 3.3×); every figure here is its executed tables.</p>
    </PageIntro>

    <Figure kind="heatmap" id="network-share" data={rows(data, 'class_share')} row="target" column="Ward class" value="share" ramp="sequential" domain={[0, 1]} format="pct" valueLabel="share of the site's filters"
      title="Where the SPARTAN sites fall — Delhi and Beijing in the smoke-associated class, Addis and Bishoftu not"
      subtitle="Ward linkage on 30 PCA scores (PCA fitted on the library only) of standardised spline-baseline spectra, library and targets in one tree, cut at the silhouette-best k. Classes 3 and 4 are low-deposit spectra; read class 1 vs 2 as the chemistry."
      provenance={provenance(data, ['class_share'])} />

    <Figure kind="table" id="network-classes" data={rows(data, 'classes')}
      columns={[{ key: 'cluster', label: 'Class', format: 'int' }, { key: 'n', label: 'Library spectra', format: 'int' }, { key: 'n_sites', label: 'Sites', format: 'int' }, { key: 'top_sites', label: 'Top sites' },
        { key: 'pct_Jul_Oct', label: 'Jul–Oct (%)', format: 'int' }, { key: 'pct_Dec_Feb', label: 'Dec–Feb (%)', format: 'int' }, { key: 'OC_EC_median', label: 'TOR OC/EC', format: 'f2' },
        { key: 'EC_ug_median', label: 'TOR EC (µg)', format: 'f2' }, { key: 'peak_abs_median', label: 'Peak absorbance', format: 'f3' },
        { key: 'pct_smoke_lineage', label: 'Smoke lineage (%)', format: 'f2' }, { key: 'pct_ocec800', label: 'Lowest-OC/EC 800 (%)', format: 'f2' }]}
      title="What each class is — the top split is deposit and noise, not composition"
      subtitle="Medians per class. Two classes have peak absorbance of 0.003–0.004 against 0.02–0.04: low-loading spectra whose standardised shape is noise. Any class read as chemistry has to be loading-conditioned first."
      provenance={provenance(data, ['classes'])} />

    <Figure kind="sweep" id="network-silhouette" data={rows(data, 'silhouette')} x="k" y="silhouette" series="series" xLabel="Number of Ward classes (k)" yLabel="Silhouette"
      chosen={k.k != null ? [{ series: String(rows(data, 'silhouette')[0]?.series), x: k.k }] : undefined} height={240}
      title="Silhouette by k — four classes, and a flat choice beyond"
      subtitle="On a 4,000-spectrum library subsample (the full silhouette is quadratic in n). The ringed point is the chosen cut."
      provenance={provenance(data, ['silhouette'])} />

    <Figure kind="sweep" id="network-smoke" data={rows(data, 'smoke_by_loading').map((r): Row => ({ ...r, series: 'smoke-lineage enrichment in class 1' }))} x="EC_bin" y="enrichment" series="series"
      xScale="point" reference={1} xLabel="TOR EC loading bin (µg per filter, sextiles with ≥ 10 smoke filters)" yLabel="Enrichment (× the bin's base rate)" height={260}
      title="Is the smoke class just a loading class? — no: inside each loading bin smoke filters still concentrate in it"
      subtitle="There is no wildfire label in the database; smoke-likeness is the deployed EC model's 906-filter smoke lineage. Within narrow EC-loading bins, smoke-lineage filters are 1.1–19 × more likely than other filters at the same loading to sit in class 1 (1 = no enrichment)."
      provenance={provenance(data, ['smoke_by_loading'])} />

    <Figure kind="table" id="network-nearest" data={rows(data, 'nearest_sites')} choose={[{ column: 'measure', label: 'Ranking' }, { column: 'target', label: 'Target' }]}
      columns={[{ key: 'rank', label: 'Rank', format: 'int' }, { key: 'site', label: 'IMPROVE site' }, { key: 'value', label: 'r (or JS distance)', format: 'f3' }]}
      title="Which IMPROVE sites look like each SPARTAN site"
      subtitle="Per-site median spline-baseline spectra (IMPROVE sites with ≥ 20 spectra), Pearson r; and, as a check, the Jensen–Shannon distance between sites' Ward-class mixes. Every target's least-like site is TOOL1 (Arctic Alaska). The notebook displayed the first entries only."
      provenance={provenance(data, ['nearest_sites'])} />

    <SiteSimilarityMatrix table={data.tables.site_correlation} provenance={provenance(data, ['site_correlation'], 'Ordering and tree: average linkage on 1 − r, lib/cluster.ts.')} />

    <Figure kind="table" id="network-threshold" data={rows(data, 'threshold_precision')}
      columns={[{ key: 'criterion', label: 'A leave-one-out nearest neighbour…' }, { key: 'base rate', label: 'at any r', format: 'pct' }, { key: 'strictest cutoff', label: 'at the strictest r cutoff', format: 'pct' }]}
      title="A validated match threshold fails, informatively — spectral near-identity is not calibration equivalence"
      subtitle="For every library spectrum, its nearest neighbour by r. Sharing the Ward class is near-certain at any r (one class holds most of the library) and so calibrates nothing; carrying the same EC loading within 25 % never reaches 75 % precision at any cutoff (the best, at r ≥ 0.9999, keeps 0.5 % of the library)."
      provenance={provenance(data, ['threshold_precision'])} />

    <Figure kind="table" id="network-coverage" data={rows(data, 'threshold')}
      columns={[{ key: 'target', label: 'Target' }, { key: 'median_best_r', label: 'Median best library r', format: 'f3' }, { key: 'pct_above_threshold', label: 'Above the r ≥ 0.9999 operating point (%)', format: 'f2' }, { key: 'pct_below_0.99', label: 'Best r below 0.99 (%)', format: 'f2' }]}
      title="No SPARTAN filter reaches that operating point; Addis and JPL have the most filters below r 0.99"
      provenance={provenance(data, ['threshold'])} />

    <Figure kind="table" id="network-pairs" data={[...tag('most similar pairs', rows(data, 'top_pairs')), ...tag('least similar pairs', rows(data, 'least_pairs'))]} choose={[{ column: 'list', label: 'List' }]}
      columns={[{ key: 'site_A', label: 'Site' }, { key: 'site_B', label: 'Site' }, { key: 'r', label: 'r of median spectra', format: 'text' }]}
      title="The network against itself — the closest and furthest IMPROVE site pairs"
      provenance={provenance(data, ['top_pairs', 'least_pairs'])} />

    <Figure kind="table" id="network-isolation" data={[...tag('most isolated', rows(data, 'isolated')), ...tag('most typical', rows(data, 'typical'))]} choose={[{ column: 'list', label: 'List' }]}
      columns={[{ key: 'site', label: 'Site' }, { key: 'best_r_to_any_site', label: 'Best r to any site', format: 'f3' }, { key: 'median_r_to_network', label: 'Median r to the network', format: 'f3' },
        { key: 'nearest_site', label: 'Nearest site' }, { key: 'n_spectra', label: 'Spectra', format: 'int' }]}
      title="Isolated and typical IMPROVE sites"
      provenance={provenance(data, ['isolated', 'typical'])} />

    <Figure kind="table" id="network-raw" data={rows(data, 'raw_vs_baselined')} columns={Object.keys(rows(data, 'raw_vs_baselined')[0] ?? {}).map((key) => ({ key, format: 'text' as const }))}
      title="Baselining is what makes the network comparable at all"
      subtitle="Raw spectra restricted to the corrected window: every site pair sits at r ≈ 0.999 (one spectrum plus background); after the spline baseline the pairs spread and most sites change which site is their nearest. Raw-space similarity rankings are background rankings."
      provenance={provenance(data, ['raw_vs_baselined'])} />

    <Figure kind="table" id="network-seasons" data={[...tag('all library filters', rows(data, 'season_analogs')), ...tag('EC 3–12 µg band (loading-matched)', rows(data, 'season_analogs_matched'))]}
      choose={[{ column: 'list', label: 'Search' }]}
      title="Each Ethiopian season draws different IMPROVE analogs — even at matched loading"
      subtitle="Top 200 library spectra by r to each Addis season's median spline-baseline spectrum. Dry draws low-loading, higher OC/EC analogs from many sites; Belg and Kiremt draw heavier filters from PUSO1, LTCC1 and the Phoenix sites. One pooled Addis cohort is the wrong object."
      provenance={provenance(data, ['season_analogs', 'season_analogs_matched'])} />

    <Figure kind="table" id="network-overlap" data={rows(data, 'season_overlap')} choose={[{ column: 'search', label: 'Search' }]}
      columns={[{ key: 'pair', label: 'Seasons' }, { key: 'overlap', label: 'Shared top-200 analogs', format: 'pct' }]}
      title="The seasons' analog sets barely overlap"
      provenance={provenance(data, ['season_overlap'])} />

    <Figure kind="table" id="network-season-loading" data={rows(data, 'season_loading')}
      columns={[{ key: 'season', label: 'Addis season' }, { key: 'n', label: 'Filters', format: 'int' }, { key: 'median_Fabs', label: 'Median Fabs (Mm⁻¹)', format: 'f2' },
        { key: 'median_peak_abs', label: 'Median peak absorbance', format: 'f3' }, { key: 'analog_median_EC_ug', label: 'Analogs’ median EC (µg)', format: 'f2' }]}
      title="Part of the seasonal difference is loading — the analogs' EC follows the season's deposit"
      provenance={provenance(data, ['season_loading'])} />
  </>
}
