import { useMemo } from 'react'
import { ExportState } from '@/components/ExportState'
import type { SectionPageProps } from '../../types'
import { Agreement } from './filterOnly/Agreement'
import { COMMAND, FoMarkedProvider, SettingsControls, useFilterOnly, useRuns, useSettings } from './filterOnly/data'
import { CoefficientInfluence, CommonFilterInfluence } from './filterOnly/Influence'
import { Id11Errors, Id11Support } from './filterOnly/Id11'
import { DenominatorSensitivity, PopulationHeat, PopulationModels, Weighting } from './filterOnly/Populations'
import { ConcentrationRanges, HeldOutScatter, MetadataErrors } from './filterOnly/Predictions'
import { PairedProportional, QuarterBias, QuarterMae, TestWeights } from './filterOnly/Quarters'

/**
 * Filter-only transfer limits: how well HIPS Fabs carries from FTIR-predicted EC across reported-date quarters at
 * Addis, Beijing, Delhi and JPL, from the filter-only release's frozen 545 diagnostic filters
 * (gallery/data/export_research_filter_only.py). Every model is refitted here (common/filterOnly.ts), so the
 * charts follow Samples (the sites, dates and seasons in view, joined by filter id; a quarter left below the minimum
 * training n says so as any other), the population, the minimum training n, Filters → excluded (the export's
 * exclusions and the reader's hand exclusions, filterOnly/data useFilterOnly) and, for HIPS in BC units, the MAC. The
 * settings sit in the dock (population and minimum n on the Page tab, HIPS units on the Plot tab under the MAC); the
 * last panel checks the refit against the release's own tables, on all 545 as the release ran, whatever Samples says.
 */
export function FilterOnly({ meta, rows }: SectionPageProps) {
  const { data, error } = useFilterOnly(rows)
  const s = useSettings()
  const all = useRuns(data?.points ?? [], s)
  // the sites Samples leaves filters at; a chart with none says what to pick (filterOnly/data NO_FILTERS)
  const runs = useMemo(() => all.filter((r) => r.diagnostic.length), [all])
  if (!data) return <ExportState title="Filter-only transfer limits" what="the filter-only release export" error={error} command={COMMAND} />
  const props = { runs, s, meta }
  return <FoMarkedProvider value={data.marked}>
    <SettingsControls />
    <QuarterMae {...props} />
    <QuarterBias {...props} />
    <PairedProportional {...props} />
    <HeldOutScatter {...props} />
    <ConcentrationRanges {...props} />
    <MetadataErrors {...props} />
    <PopulationHeat points={data.points} s={s} />
    <PopulationModels points={data.points} s={s} meta={meta} />
    <Weighting {...props} />
    <TestWeights {...props} />
    <CoefficientInfluence points={data.points} s={s} meta={meta} omitDefault={data.omitDefault} />
    <CommonFilterInfluence points={data.points} s={s} meta={meta} omitDefault={data.omitDefault} />
    <Id11Errors {...props} />
    <Id11Support {...props} />
    <DenominatorSensitivity points={data.points} s={s} meta={meta} />
    <Agreement all={data.all} />
  </FoMarkedProvider>
}
