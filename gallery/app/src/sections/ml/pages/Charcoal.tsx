import type { SectionPageProps } from '../../types'
import { MlWaiting, PageIntro, useMl } from '../common'
import { Matching } from './charcoal/Matching'
import { Structure } from './charcoal/Structure'

/**
 * The charcoal archive's model-fitting side (research/ftir_etad_char char_02, char_03, char_04, char_06, char_11):
 * its unsupervised structure (PCA + KMeans passes, silhouette, Mahalanobis distance, the transform benchmark) and
 * the analogue matcher, the burned/unburned classifier and the Gaussian mixture applied to Addis. Static: the
 * notebooks' tables (gallery/data/ml_pages/charcoal.py). Nothing here depends on the MAC. The per-filter similarity
 * scores these feed are live on Research → Charcoal similarity.
 */
export function Charcoal({ meta }: SectionPageProps) {
  const { data, error } = useMl('charcoal')
  if (!data) return <MlWaiting error={error} what="the charcoal archive results" />
  return <>
    <PageIntro id="mlc-intro" title="Charcoal archive: structure, analogue matching and classifiers" data={data}
      notebooks="research/ftir_etad_char char_02_addis_vs_charcoal, char_03_burned_vs_unburned, char_04_spectral_typology, char_06_dry_season_anomaly, char_11_normalization_alternatives">
      <p>Six published charcoal and biochar collections (5,889 spectra; research/ftir_hips_chem/charcoal_ftir_sources) against the 239 Addis filter spectra, AIRSpec-baselined, cropped to 1430–3500 cm⁻¹, detrended and SNV-scaled on both sides. Everything here is a fitted model (PCA, KMeans, Gaussian mixture, logistic regression) or a nearest-neighbour search, so it is the notebooks' numbers with their seasons; nothing is refitted and nothing depends on the MAC. The per-filter scores and their season, lot and loading checks are live on <a href="#tab=research&aep=charcoal">Research → Charcoal similarity</a>.</p>
    </PageIntro>
    <Structure data={data} />
    <Matching data={data} meta={meta} />
  </>
}
