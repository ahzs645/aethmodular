import type { FilterRow, MetaFile } from '@/lib/types'
import { CompositionTreemap } from '@/charts/partOfWhole/CompositionTreemap'
import { FTIR_GROUP_ORDER } from '@/lib/palettes'

/**
 * The FTIR organic functional groups stacked per site (or season), in the order and colours of the
 * IMPROVE functional-group figure the group reads them against (lib/palettes FTIR_GROUP_COLOURS, the
 * "FTIR functional groups" palette): alkane CH at the bottom, then carboxylic COOH, carbonyl CO and
 * alcohol COH. That figure's amine NH2 goes on top; the SPARTAN FTIR product reports no amine group,
 * so its slot is left out (the species list keeps only fields the data has).
 */
export function FunctionalGroups({ rows, meta, id = 'fgroups' }: { rows: FilterRow[]; meta: MetaFile; id?: string }) {
  return <CompositionTreemap rows={rows} meta={meta} id={id} stackedOnly species={FTIR_GROUP_ORDER} palette="FTIR functional groups" legendTopDown
    title="Functional groups — the FTIR organic functional groups, stacked per site and season"
    subtitle="FTIR functional groups (UC Davis calibration) stacked bottom to top as alkane CH, carboxylic COOH, non-acid carbonyl CO and alcohol COH, in the colours of the IMPROVE functional-group figure. That figure has amine NH2 on top; the SPARTAN FTIR product has no amine group. Concentration stacks the selected mean or median in µg/m³; Composition (%) divides each group by the sum of the groups shown, so hiding a group changes the denominator. Missing values are omitted, never filled with zero."
    provenance="SPARTAN FTIR functional groups (lab batches 2–3 and the public release) · order and colours after Atmos. Environ. 2017, Fig. 7 (IMPROVE OM/OC by functional group)" />
}
