import { num, type Row } from './table.ts'

/** Re-express a saved MAC-10 residual using each filter's actual Fabs, never
 * rescale a residual or a fitted model's prediction by the reader's MAC. */
export function residualRows(rows: Row[], mac: number, sourceMac = 10): Row[] {
  if (!(mac > 0) || !Number.isFinite(mac) || !(sourceMac > 0) || !Number.isFinite(sourceMac)) throw new Error('MAC must be finite and positive')
  return rows.map((r) => {
    const fabs = num(r.Fabs), saved = num(r.residual_ugm3)
    const predicted = num(r.prediction_ugm3) ?? (fabs != null && saved != null ? saved + fabs / sourceMac : null)
    const observed = fabs != null ? fabs / mac : null
    return { ...r, observed, predicted, residual: observed != null && predicted != null ? predicted - observed : null }
  })
}
