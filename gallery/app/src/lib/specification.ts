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

type ChoiceDimension = 'co' | 'sp' | 'sel' | 'mo' | 'lot' | 'cut' | 'k'
interface ChoiceFile extends Record<ChoiceDimension, (number | null)[]> { n: number; masks: string[] }

/** Identify the setting each saved mask represents without parsing its display label.
 * Cohort takes precedence over cutoff when a complete cohort has a unique size.
 * Unknown compound choices retain their own independent requirement.
 */
export function choiceFilterDimensions(file: ChoiceFile): string[] {
  const keys: ChoiceDimension[] = ['co', 'sp', 'sel', 'mo', 'lot', 'k', 'cut']
  return file.masks.map((mask, c) => keys.find(key => {
    const bits = new Map<number | null, string>()
    for (let i = 0; i < file.n; i++) {
      const value = file[key][i], bit = mask[i]
      if (bits.has(value) && bits.get(value) !== bit) return false
      bits.set(value, bit)
    }
    return new Set(bits.values()).size > 1
  }) ?? `choice-${c}`)
}

/** Alternatives within one setting are combined; different settings narrow the search. */
export function choiceFilterSets(required: ReadonlySet<number>, dimensions: readonly string[]): number[][] {
  const groups = new Map<string, number[]>()
  for (const c of required) {
    const dimension = dimensions[c] ?? `choice-${c}`
    const choices = groups.get(dimension) ?? []
    choices.push(c)
    groups.set(dimension, choices)
  }
  return [...groups.values()]
}

export function matchesChoiceFilters(masks: readonly string[], i: number, groups: readonly number[][]): boolean {
  return groups.every(choices => choices.some(c => masks[c][i] === '1'))
}
