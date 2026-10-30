import { ChartFrame, Note } from '@/components/ChartFrame'
import { ExportState } from '@/components/ExportState'
import type { SectionPageProps } from '../../types'
import { Agreement } from './filterOnly/Agreement'
import { COMMAND, NB_DIAG, NB_PROP, NB_STAB, SettingsControls, useFilterOnly, useRuns, useSettings } from './filterOnly/data'
import { CoefficientInfluence, CommonFilterInfluence } from './filterOnly/Influence'
import { Id11Errors, Id11Support } from './filterOnly/Id11'
import { DenominatorSensitivity, PopulationHeat, PopulationModels, Weighting } from './filterOnly/Populations'
import { ConcentrationRanges, HeldOutScatter, MetadataErrors } from './filterOnly/Predictions'
import { PairedProportional, QuarterBias, QuarterMae, TestWeights } from './filterOnly/Quarters'

/**
 * Filter-only transfer limits: how well HIPS Fabs carries from FTIR-predicted EC across reported-date quarters at
 * Addis, Beijing, Delhi and JPL, from the filter-only release's frozen 545 diagnostic filters
 * (gallery/data/export_research_filter_only.py). Every model is refitted here (common/filterOnly.ts), so the
 * charts follow the population, the minimum training n, the reader's hand exclusions and, for HIPS in BC units,
 * the MAC; the last panel checks the refit against the release's own tables.
 */
export function FilterOnly({ meta }: SectionPageProps) {
  const { data, error } = useFilterOnly()
  const s = useSettings()
  const runs = useRuns(data?.points ?? [], s)
  if (!data) return <ExportState title="Filter-only transfer limits" what="the filter-only release export" error={error} command={COMMAND} />
  const props = { runs, s, meta }
  return <>
    <ChartFrame id="fo-settings" title="Filter-only transfer limits — how far the HIPS–FTIR EC relationship carries across reported-date quarters" exportable={false} pageOnly
      subtitle="545 diagnostic filters (480 with a ratio-eligible EC) at Addis Ababa, Beijing, Delhi and JPL · HIPS Fabs predicted from FTIR-predicted EC by OLS, the training median, or a through-origin fit, on withheld calendar quarters"
      provenance={<>{NB_STAB} (9 figures), {NB_PROP} (8 figures), {NB_DIAG} (figure 05)</>}
      controls={<SettingsControls />}>
      <Note>
        Reported dates are not verified sampling intervals, and the FTIR predictor is not an independent EC reference. The population, minimum training n and units apply to every chart below;
        filters excluded by hand on any page are left out of every fit.{data.note ? ` ${data.note}` : ''} Fabs is in Mm⁻¹ and needs no MAC; “BC at MAC” divides every HIPS value, prediction and error by the reader's MAC, which leaves percentages unchanged.
      </Note>
    </ChartFrame>
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
  </>
}
