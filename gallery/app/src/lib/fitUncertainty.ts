import type { MetaFile } from './types'
import { monthBlocks } from './regressionCI.ts'

/** Relative per-filter HIPS errors; the common y scale retains the chart's Deming λ. */
export function fitUncertainty(rows: { site: string; date: string; [field: string]: unknown }[], x: string, y: string, meta: MetaFile, lambda = 1) {
  const errors = (field: string) => {
    if (field !== 'HIPS BC' && field !== 'HIPS Fabs') return undefined
    const values = rows.map(r => Number(r['HIPS Fabs uncertainty']) / (field === 'HIPS BC' ? meta.mac_value : 1))
    // Partial coverage cannot be described as a measured uncertainty vector.
    if (!values.length || values.some(v => !Number.isFinite(v) || v <= 0)) return undefined
    const sorted = values.slice().sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    return values.map(v => v / median)
  }
  const sx = errors(x), sy = errors(y)
  return { errorsInVariables: true, lambda, sigmaX: sx, sigmaY: sy?.map(v => v * Math.sqrt(lambda)), bootstrapGroups: monthBlocks(rows) }
}
