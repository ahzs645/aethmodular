/**
 * Charcoal archive B: its unsupervised structure (char_04 silhouette and KMeans passes A and B, char_02's
 * Mahalanobis distance and baseline sensitivity, char_11's transform benchmark). Static: the notebooks' tables.
 */
import { Figure } from '@/components/Figure'
import { f2 } from '@/lib/format'
import { num, type Row } from '@/lib/table'
import type { SpectralExport } from '../../../research/pages/spectralCommon'
import { provenance, rows } from '../../common'

const NB = {
  c02: 'research/ftir_etad_char/char_02_addis_vs_charcoal.ipynb',
  c03: 'research/ftir_etad_char/char_03_burned_vs_unburned.ipynb',
  c04: 'research/ftir_etad_char/char_04_spectral_typology.ipynb',
  c11: 'research/ftir_etad_char/char_11_normalization_alternatives.ipynb',
}
const pct0 = (v: unknown) => (typeof v === 'number' ? `${v.toFixed(0)} %` : '—')
const PASS_COLORS = { 'Pass A · charcoal only': '#4e79a7', 'Pass B · charcoal + Addis': '#b07aa1' }

export function Structure({ data }: { data: SpectralExport }) {
  const chosen = rows(data, 'chosen_k')
  const kA = chosen.find((r) => String(r.pass).startsWith('Pass A')), kB = chosen.find((r) => String(r.pass).startsWith('Pass B'))
  const assoc = rows(data, 'association'), V = (a: string) => num(assoc.find((r) => r.against === a)?.["Cramér's V"] ?? null)
  const addis = rows(data, 'passB_addis'), top = addis[0]
  const purity = rows(data, 'passB_purity').find((r) => r.cluster === top?.cluster)
  const passB = rows(data, 'passB_source')
  const dist = rows(data, 'mahalanobis')
  const sens = rows(data, 'baseline_sensitivity'), sensOf = (m: string, col: string) => num(sens.find((r) => String(r.measure).startsWith(m))?.[col] ?? null)
  const [airCol, rawCol] = Object.keys(sens[0] ?? {}).filter((k) => k !== 'measure')
  const bench = rows(data, 'benchmark'), current = bench.find((r) => String(r.transform).includes('current')), airspec = bench.find((r) => r.transform === 'AIRSpec + SNV')
  return <>
    <Figure kind="sweep" id="mlc-silhouette" data={rows(data, 'silhouette')} x="k" y="silhouette" series="pass" xLabel="clusters (k)" yLabel="silhouette (KMeans on PCA scores)" height={240}
      colors={PASS_COLORS} chosen={chosen.map((r) => ({ series: String(r.pass), x: r.k }))}
      title={`The silhouette picks k = ${kA?.k ?? '?'} types in the charcoal archive (${f2(kA?.silhouette)}) and k = ${kB?.k ?? '?'} once Addis joins (${f2(kB?.silhouette)}), and neither is a sharp optimum`}
      subtitle="char_04: KMeans on the leading PCA scores of the SNV spectra, k = 2–8. Pass A is the charcoal archive alone (5,889 spectra × 1,274 points, 952–3498 cm⁻¹); pass B adds the 239 Addis filters on the window they share (6,128 × 1,036, 1430–3500 cm⁻¹). The ringed k is the one each pass uses."
      provenance={<>{NB.c04} [5]. {provenance(data, ['silhouette', 'chosen_k'])}</>} />

    <Figure kind="heatmap" id="mlc-passA-temperature" data={rows(data, 'passA_temperature')} row="temperature" column="type" value="share" ramp="sequential" domain={[0, 1]} format="pct" valueLabel="share of the tier's spectra"
      title={`Pass A's types follow the charring temperature (Cramér's V ${f2(V('temperature'))}) more than the collection (${f2(V('source collection'))}) or the species (${f2(V('species/feedstock'))})`}
      subtitle="Share of each temperature tier's spectra in each pass-A type (rows sum to 1). Unburned and low-temperature material is type 1; types 0 and 2 split the charred tiers, and 400 °C straddles all three, where char_01 found the aromatic C=C band turns over. Types are KMeans labels: their numbers carry no order."
      provenance={<>{NB.c04} [7]. {provenance(data, ['passA_temperature', 'association'])}</>} />

    <Figure kind="heatmap" id="mlc-passA-collection" data={rows(data, 'passA_collection')} row="source" column="type" value="share" ramp="sequential" domain={[0, 1]} format="pct" valueLabel="share of the collection's spectra"
      title="Every collection spreads over all three types: the archive's structure is charring stage, not which lab measured it"
      subtitle="Share of each collection's spectra in each pass-A type (rows sum to 1). The McCall sets lean to type 1 because they hold the unburned and low-temperature material."
      provenance={<>{NB.c04} [7]. {provenance(data, ['passA_collection'])}</>} />

    {passB.length > 0
      ? <Figure kind="heatmap" id="mlc-passB" data={passB} row="source" column="type" value="share" ramp="sequential" domain={[0, 1]} format="pct" valueLabel="share of the source's spectra"
        title={`Add Addis and it forms its own type: ${Math.round(100 * (num(top?.share ?? null) ?? NaN))} % of the filters land in ${top?.cluster ?? 'one type'}, which is ${Math.round(100 * (num(purity?.['Addis share'] ?? null) ?? NaN))} % Addis`}
        subtitle="Pass B: share of each source's spectra in each of the six types (rows sum to 1). Addis does not join any charcoal type; the charcoal collections share the other five."
        provenance={<>{NB.c04} [11]. {provenance(data, ['passB_source', 'passB_addis', 'passB_purity'])}</>} />
      : <Figure kind="table" id="mlc-passB" data={addis} columns={[{ key: 'cluster', label: 'Pass-B type' }, { key: 'share', label: 'Share of Addis filters', format: 'pct' }]}
        title={`Add Addis and it forms its own type: ${Math.round(100 * (num(top?.share ?? null) ?? NaN))} % of the filters land in ${top?.cluster ?? 'one type'}`}
        subtitle="Pass B's composition by source needs char_04's CSV (see the intro); the executed notebook shows Addis's row."
        provenance={<>{NB.c04} [11]. {provenance(data, ['passB_addis'])}</>} />}

    <Figure kind="table" id="mlc-mahalanobis" data={dist}
      columns={[{ key: 'comparison', label: 'Cloud' }, { key: 'reference median', label: 'Reference median', format: 'f2' }, { key: 'reference 99th pct', label: 'Reference 99th pct', format: 'f2' },
        { key: 'Addis median', label: 'Addis median', format: 'f2' }, { key: 'Addis beyond the 99th pct (%)', label: 'Addis beyond the 99th pct', format: 'percent' }]}
      title={`Addis sits on the far edge of the charcoal cloud (median distance ${f2(dist[0]?.['Addis median'])} against ${f2(dist[0]?.['reference median'])}), and outside the burned/unburned training cloud for ${dist[1]?.['Addis beyond the 99th pct (%)'] ?? '?'} % of filters`}
      subtitle="Mahalanobis distance in PCA score space: char_02 projects Addis into a 6-PC model of the charcoal archive; char_03 into the 8-PC model its burned/unburned classifier was trained on. Distance-based, so unlike the correlation results it depends on the normalization (char_11). The per-filter distances and the PCA scores were not saved (see the intro)."
      provenance={<>{NB.c02} [20]; {NB.c03} §5. {provenance(data, ['mahalanobis'])}</>} />

    <Figure kind="table" id="mlc-baseline" data={rows(data, 'baseline_sensitivity')}
      title={`Baselining Addis with AIRSpec lowers its match to charcoal (${f2(sensOf('median best-match r', rawCol))} → ${f2(sensOf('median best-match r', airCol))}) and moves it further out (${f2(sensOf('median Mahalanobis', rawCol))} → ${f2(sensOf('median Mahalanobis', airCol))}), but not the implied temperature`}
      subtitle="char_02 §5: the whole chain repeated on raw offset-corrected Addis spectra (detrended) instead of the AIRSpec-baselined ones."
      provenance={<>{NB.c02} [23]. {provenance(data, ['baseline_sensitivity'])}</>} />

    <Figure kind="intervals" id="mlc-benchmark" data={bench} row="transform" aggregate={{ stat: 'median', interval: 'none' }} labelWidth={200}
      panels={[{ value: '% within 100 °C', label: 'Cross-collection recovery: % within 100 °C' }, { value: 'MAE (°C)', label: 'Recovery MAE (°C)', reference: 0 },
        { value: 'sil(temperature) ↑', label: 'Silhouette by temperature (higher is better)', reference: 0 }, { value: 'sil(collection) ↓', label: 'Silhouette by collection (lower is better)', reference: 0 }]}
      title={`AIRSpec on the charcoal side halves cross-collection recovery (${pct0(current?.['% within 100 °C'])} → ${pct0(airspec?.['% within 100 °C'])} within 100 °C) and sends the temperature silhouette negative`}
      subtitle="char_11's benchmark on the 2,340 absorbance spectra with a known temperature: each transform's nearest-analogue temperature recovery across collections, and how well its PCA space separates temperature tiers rather than labs. Under Pearson r the plain normalizations (SNV, vector, area, none) are identical to ~1e-15, so only transforms that change the shape are compared."
      provenance={<>{NB.c11} [21]. {provenance(data, ['benchmark'])}</>} />

    <Figure kind="table" id="mlc-airspec" data={[...rows(data, 'airspec_coverage'), ...rows(data, 'airspec_check').map((r): Row => ({ collection: r.spectra, 'ramp score': r['ramp score'] }))]}
      columns={[{ key: 'collection', label: 'Collection or spectra' }, { key: 'max wavenumber', label: 'Max wavenumber (cm⁻¹)', format: 'f2' }, { key: 'reaches the 3710 anchor', label: 'Reaches the 3710 anchor' }, { key: 'ramp score', label: 'Ramp score', format: 'f3' }]}
      title="AIRSpec cannot be applied to the charcoal side: only McCall barley reaches its 3710 cm⁻¹ anchor, and a truncated fit overcorrects"
      subtitle="char_11: each collection's upper wavenumber against AIRSpec's segment-1 anchor, then the ramp score (correlation of the mean spectrum with wavenumber; 0 = no residual slope) as published, after a truncated AIRSpec, and for Addis. The fitted baselines themselves were not saved (see the intro)."
      provenance={<>{NB.c11} [13][15]. {provenance(data, ['airspec_coverage', 'airspec_check'])}</>} />
  </>
}
