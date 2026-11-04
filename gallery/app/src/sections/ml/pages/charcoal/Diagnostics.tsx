import { SavedFigure } from '@/components/SavedFigure'
import type { SpectralExport } from '../../../research/pages/spectralCommon'
import { useMemo } from 'react'
import { rowsOf } from '@/lib/table'

/** Model score spaces stay separate: different PCA fits must never share axes. */
export function CharcoalDiagnostics({ data }: { data: SpectralExport }) {
  const withIds = useMemo(() => ({ ...data, tables: {
    ...data.tables,
    ...(data.tables.mahalanobis_per_filter ? { mahalanobis_per_filter: rowsOf(data.tables.mahalanobis_per_filter).map((r) => ({ ...r, id: r.ExternalFilterId ?? null })) } : {}),
    ...(data.tables.per_filter_escape ? { per_filter_escape: rowsOf(data.tables.per_filter_escape).map((r) => ({ ...r, id: r.ExternalFilterId ?? null })) } : {}),
  } }), [data])
  return <>
    <SavedFigure bundle={data} table="pca_scores" kind="scatter" id="mlc-pca" x="PC1" y="PC2" color="population" filterIdColumn="id"
      tipColumns={['population', 'collection', 'temp_c', 'sample', 'season_label']}
      xLabel="PC1 score" yLabel="PC2 score" title="Charcoal PCA with Addis projected into the archive model"
      subtitle="PCA fitted to the temperature-labelled charcoal archive. Common window 1430–3500 cm⁻¹; detrended and SNV-normalized. Addis projections show the raw and spline-baselined treatments." />
    <SavedFigure bundle={data} table="typology_scores" kind="scatter" id="mlc-typology-pca" x="PC1" y="PC2" color="source" filterIdColumn="id"
      tipColumns={['source', 'cluster', 'temp_c', 'sample']}
      choose={[{ column: 'analysis', label: 'PCA fit' }]} xLabel="PC1 score" yLabel="PC2 score" title="Spectral typology in its fitted PCA space" />
    <SavedFigure bundle={withIds} table="mahalanobis_per_filter" kind="ecdf" id="mlc-distance-cdf" value="distance" series="population" filterIdColumn="id"
      xLabel="Mahalanobis distance (not squared)" title="Distance from the charcoal reference cloud" />
    <SavedFigure bundle={data} table="envelope_spectra" kind="spectrum" id="mlc-envelope" wn="wn" y="mean" lo="p05" hi="p95" series="population"
      choose={[{ column: 'representation', label: 'Spectra' }]} yLabel="Absorbance (AU), or derivative (AU / grid-step²)" bandLabel="line: mean · band: 5th–95th percentiles"
      variants={[{ key: 'region', label: 'Window', default: 'full', choices: [
        { value: 'full', label: 'Full range' },
        { value: 'CH', label: 'C–H (2800–3000)', patch: { window: [3000, 2800] } },
        { value: 'CO', label: 'Carbonyl / aromatic (1450–1850)', patch: { window: [1850, 1450] } },
      ] }]}
      title="IMPROVE and Addis spectral envelopes"
      subtitle="Raw: absorbance (AU). Second derivative: AU per grid-step² on the 2 cm⁻¹ grid (Savitzky–Golay, 13 points, polynomial order 2)." />
    <SavedFigure bundle={withIds} table="per_filter_escape" kind="ecdf" id="mlc-escape" value="escape" series="season_label" filterIdColumn="id"
      xLabel="Fraction outside the IMPROVE 5–95 % envelope" title="Envelope escape per Addis filter" />
  </>
}
