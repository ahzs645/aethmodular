/**
 * Fields that are arithmetic on another field, so a correlation between the
 * two is 1.000 by construction and says nothing. `HIPS BC` is `HIPS Fabs`
 * divided by MAC (config.MAC_VALUE) — the exporter ships both under different
 * names precisely so nobody mistakes a unit change for a measurement.
 *
 * The correlogram drops these pairs and the ranking marks them, rather than
 * letting "HIPS BC predicts HIPS Fabs with R² = 1" top the list.
 */
export const DERIVED_FROM: Record<string, string> = {
  'HIPS BC': 'HIPS Fabs',
}

export function byConstruction(a: string, b: string): boolean {
  return DERIVED_FROM[a] === b || DERIVED_FROM[b] === a
}
