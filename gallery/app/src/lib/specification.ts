export type HighlightRule = 'guardrails' | 'intercept' | 'cv-slope' | 'custom'
export interface HighlightLimits { intercept: number; slopeMin: number; slopeMax: number; cvMin: number }
export const DEFAULT_HIGHLIGHT_LIMITS: HighlightLimits = { intercept: 0.5, slopeMin: 0.7, slopeMax: 1.3, cvMin: 0.85 }
export interface HighlightMetrics { intercept: number | null; slope: number | null; cv: number | null; guardrails: boolean }

/** Missing measurements cannot satisfy a criterion that requires them. */
export function highlightMatches(metrics: HighlightMetrics, rule: HighlightRule, limits: HighlightLimits): boolean {
  if (rule === 'guardrails') return metrics.guardrails
  const nearZero = metrics.intercept !== null && Number.isFinite(metrics.intercept) && Math.abs(metrics.intercept) <= limits.intercept
  if (rule === 'intercept') return nearZero
  const cvSlope = metrics.slope !== null && Number.isFinite(metrics.slope) && metrics.slope >= limits.slopeMin && metrics.slope <= limits.slopeMax
    && metrics.cv !== null && Number.isFinite(metrics.cv) && metrics.cv >= limits.cvMin
  return cvSlope && (rule !== 'custom' || nearZero)
}

/** Preserve intercept ordering within each selection population; never pair unlike configurations. */
export function selectionPanels(shown: number[], codes: number[], categories: string[]) {
  return categories.map((key, code) => ({ key, indices: shown.filter(i => codes[i] === code) })).filter(p => p.indices.length > 0)
}
