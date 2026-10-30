/**
 * Charcoal archive C: analogue matching and classifiers (char_02 nearest-analogue matching and its implied
 * temperature, char_03's leave-one-feedstock-out burned/unburned classifier, char_06's Gaussian mixture, char_11's
 * conclusion by transform). Static: the notebooks' 239 filters with their own seasons.
 */
import { useMemo } from 'react'
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { num, type Row } from '@/lib/table'
import type { MetaFile } from '@/lib/types'
import type { SpectralExport } from '../../../research/pages/spectralCommon'
import { provenance, rows } from '../../common'

const NB = {
  c02: 'research/ftir_etad_char/char_02_addis_vs_charcoal.ipynb',
  c03: 'research/ftir_etad_char/char_03_burned_vs_unburned.ipynb',
  c06: 'research/ftir_etad_char/char_06_dry_season_anomaly.ipynb',
  c11: 'research/ftir_etad_char/char_11_normalization_alternatives.ipynb',
}
const SEASONS = ['Dry Season', 'Belg Rainy Season', 'Kiremt Rainy Season']

export function Matching({ data, meta }: { data: SpectralExport; meta: MetaFile }) {
  // the notebooks' seasons take the dry_feb calendar's colours (the same month bins)
  const colors = useMemo(() => {
    const dryFeb = meta.season_conventions?.dry_feb ?? meta.seasons
    const out: Record<string, string> = {}
    for (const name of SEASONS) { const c = dryFeb.find((s) => s.name.toLowerCase().startsWith(name.split(' ')[0].toLowerCase()))?.color; if (c) out[name] = c }
    return { ...out, 'low-similarity': '#b2182b', normal: '#7b8794', 'unburned references (LOO)': '#59a14f', 'burned references (LOO)': '#e15759', 'Addis filters (final model)': '#7A4FA3' }
  }, [meta])
  const ref = rows(data, 'analogue_reference')
  const addisRef = ref[0], heldOut = ref[1]
  const analogue = rows(data, 'analogue')
  const implied = useMemo(() => {
    const out: Row[] = []
    for (const season of SEASONS) {
      const rs = analogue.filter((r) => r.season === season)
      const ts = [...new Set(analogue.map((r) => num(r.median_top_temp)).filter((v): v is number => v != null))].sort((a, b) => a - b)
      for (const t of ts) out.push({ season: `${season.split(' ')[0]} (n ${rs.length})`, T: `${t} °C`, share: rs.length ? rs.filter((r) => num(r.median_top_temp) === t).length / rs.length : null })
    }
    return out
  }, [analogue])
  const loo = rows(data, 'loo'), burned = rows(data, 'addis_burned')
  const probs = useMemo(() => [
    ...loo.map((r): Row => ({ group: r.true_burned ? 'burned references (LOO)' : 'unburned references (LOO)', p: r.p_burned })),
    ...burned.map((r): Row => ({ group: 'Addis filters (final model)', p: r.p_burned })),
  ], [loo, burned])
  const overall = rows(data, 'loo_overall'), pc = (m: string) => num(overall.find((r) => String(r.measure).startsWith(m))?.['%'] ?? null)
  const archive = rows(data, 'archive_scores'), addisP = archive.find((r) => String(r.collection).startsWith('Addis'))
  const mixture = rows(data, 'mixture'), fit = rows(data, 'mixture_fit')
  const mvs = rows(data, 'mixture_vs_season')
  const lowDry = num(mvs.find((r) => r.season === 'Dry Season')?.['low-similarity'] ?? null), nDry = lowDry != null ? lowDry + (num(mvs.find((r) => r.season === 'Dry Season')?.normal ?? null) ?? 0) : null
  const byT = rows(data, 'addis_by_transform'), project = byT.find((r) => String(r.transform).startsWith('PROJECT')), airspec = byT.find((r) => r.transform === 'AIRSpec + SNV')
  return <>
    {analogue.length > 0
      ? <Figure kind="histogram" id="mlc-analogue" data={analogue} value="best_r" series="season" bins={24} xLabel="best-match r against the charcoal library" colors={colors}
        title={`Charcoal-adjacent, not charcoal: Addis matches the library at median r ${f2(addisRef?.['50%'])}, where held-out charcoal matches it at ${f2(heldOut?.['50%'])}`}
        subtitle={`char_02: each Addis filter's best Pearson r against the 3,924 temperature-labelled charcoal spectra (crop → detrend → SNV on both sides, 1430–3500 cm⁻¹), by the notebook's season. The method check: ${heldOut?.count ?? 300} held-out charcoal spectra matched against the rest score median ${f2(heldOut?.['50%'])} (IQR ${f2(heldOut?.['25%'])}–${f2(heldOut?.['75%'])}); Cross-collection temperature recovery on that check: median |error| ${rows(data, 'recovery')[0]?.value ?? '?'} °C. Best collection: ${rows(data, 'analogue_collection').map((r) => `${r.best_collection} ${r.filters}`).join(', ')}.`}
        provenance={<>{NB.c02} [15] left and middle. {provenance(data, ['analogue', 'analogue_reference', 'analogue_collection'])}</>} />
      : <Figure kind="table" id="mlc-analogue" data={ref} title={`Charcoal-adjacent, not charcoal: Addis matches the library at median r ${f2(addisRef?.['50%'])}, where held-out charcoal matches it at ${f2(heldOut?.['50%'])}`}
        subtitle="char_02's printed summary; the per-filter matches need its CSV (see the intro)." provenance={<>{NB.c02} [15]. {provenance(data, ['analogue_reference'])}</>} />}

    {implied.length > 0 && <Figure kind="heatmap" id="mlc-implied" data={implied} row="season" column="T" value="share" ramp="sequential" domain={[0, 1]} format="pct" valueLabel="share of the season's filters"
      title={`The implied charring temperature is the 400 °C tier in every season (${Math.round(100 * implied.filter((r) => r.T === '400 °C').reduce((a, r) => a + (num(r.share) ?? 0), 0) / 3)} % of filters on average), at best ±100 °C`}
      subtitle="char_02: each filter's implied temperature, the median over its closest library matches, as a share of the season's filters. It sits inside the library's range, not pinned to its lowest tier, which is what the AIRSpec baseline changed (char_02 §0)."
      provenance={<>{NB.c02} [15] right. {provenance(data, ['analogue'])}</>} />}

    {probs.length > 0 && <Figure kind="histogram" id="mlc-burned" data={probs} value="p" series="group" bins={20} xLabel="P(burned)" colors={colors}
      title={`The burned/unburned classifier calls ${addisP?.['% classed burned'] ?? '?'} % of Addis burned (median P ${f2(addisP?.['median P(burned)'])}), but recovers unburned material only ${pc('unburned') ?? '?'} % of the time`}
      subtitle={`char_03: PCA (8 components) + balanced logistic regression on McCall's 24 raw and 216 charred spectra, cross-validated by leaving out one feedstock at a time (overall accuracy ${pc('overall')} %, burned recall ${pc('burned')} %), then refitted on all of them and applied to Addis. Addis sits off every burned→unburned band axis and beyond the training cloud's 99th percentile for most filters (Mahalanobis above), so its confident "unburned" is extrapolation.`}
      provenance={<>{NB.c03} [16]. {provenance(data, ['loo', 'addis_burned', 'loo_overall'])}</>} />}

    <Figure kind="table" id="mlc-burned-tables" data={[...rows(data, 'loo_accuracy').map((r): Row => ({ what: `LOO accuracy · ${r.feedstock}`, value: r.accuracy })),
      ...archive.map((r): Row => ({ what: `median P(burned) · ${r.collection} (${r['% classed burned']} % classed burned)`, value: r['median P(burned)'] }))]}
      columns={[{ key: 'what', label: 'char_03' }, { key: 'value', label: 'Value', format: 'f3' }]}
      title="The classifier reads the rest of the archive (all ≥ 200 °C) as burned, as it should, and Addis as unburned"
      subtitle="Leave-one-feedstock-out accuracy per held-out feedstock, then the final model on every other collection and on Addis (char_03's sanity check)."
      provenance={<>{NB.c03} [14][15]. {provenance(data, ['loo_accuracy', 'archive_scores'])}</>} />

    {mixture.length > 0 && <Figure kind="histogram" id="mlc-mixture" data={mixture} value="r_char" series="component" bins={30} xLabel="best r against McCall char (r_char)" colors={colors}
      title={`Two populations: the Gaussian mixture splits r_char at means ${String(fit[1]?.means ?? '?').split(/\s+/).join(' and ')} and puts ${lowDry ?? '?'} of ${nDry ?? '?'} dry-season filters in the low-similarity one`}
      subtitle={`char_06: a two-component Gaussian mixture on r_char alone, fitted without the season, preferred over one component (BIC ${f2(fit[1]?.BIC)} against ${f2(fit[0]?.BIC)}). Its split, compared with the calendar it never saw: ${mvs.map((r) => `${String(r.season).split(' ')[0]} ${r['low-similarity']} low / ${r.normal} normal`).join(', ')}. The per-filter charts that use this label (by season, lot and loading, and char_07's exclusions) are live on Research → Charcoal similarity.`}
      provenance={<>{NB.c06} [11]. {provenance(data, ['mixture', 'mixture_fit', 'mixture_vs_season'])}</>} />}

    <Figure kind="intervals" id="mlc-transform" data={byT.map((r): Row => ({ ...r, label: `${String(r.transform).replace(/^PROJECT DEFAULT \((.*)\)$/, 'project default: $1')} · ${num(r['modal implied T (°C)'])} °C` }))} row="label" aggregate={{ stat: 'median', interval: 'none' }} labelWidth={330}
      panels={[{ value: 'median best r', label: 'Addis median best r against the library' }, { value: 'share at modal T', label: 'Share of filters at the modal implied T' }, { value: 'best match is the 0 °C tier', label: 'Share whose best match is unburned (0 °C)', reference: 0 }]}
      title={`The transform that matches Addis to charcoal best (truncated AIRSpec, r ${f2(airspec?.['median best r'])}) is the one that destroys the most chemistry; the project default keeps r ${f2(project?.['median best r'])} and the 400 °C answer`}
      subtitle="char_11: what each transform concludes for Addis; each row names the modal implied temperature it gives. Plain SNV sends almost every filter to the 0 °C tier, which holds 24 of 2,340 library spectra (char_03's failure mode); EMSC collapses the correlation. Read with the benchmark above: a high Addis–charcoal r is not evidence a transform is right."
      provenance={<>{NB.c11} [24]. {provenance(data, ['addis_by_transform'])}</>} />
  </>
}
