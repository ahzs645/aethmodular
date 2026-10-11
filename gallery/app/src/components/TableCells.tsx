import { INK } from '@/lib/theme'
import { coef, coefSigned } from '@/lib/format'

/**
 * Table cells the research pages repeat, at the agreed precision (3 dp for
 * coefficients, lib/format `coef`): a fitted line with its quality, and a
 * median with its spread.
 */
export function FitCell({ slope, intercept, r2, n, bias, slopeCI, stacked = false, separator = 'x' }: {
  slope: unknown; intercept: unknown; r2?: unknown; n?: unknown
  /** mean residual, when the table reports it */
  bias?: unknown
  /** 95% interval for the slope, when shown */
  slopeCI?: [unknown, unknown] | null
  /** quality on a second, smaller line */
  stacked?: boolean
  /** 'x' writes "0.820x −1.060"; '/' writes "0.820 / −1.060" */
  separator?: 'x' | '/'
}) {
  if (slope == null && intercept == null) return <>—</>
  const quality = [
    slopeCI ? `95% slope ${coef(slopeCI[0])}–${coef(slopeCI[1])}` : null,
    r2 !== undefined ? `R² ${coef(r2)}` : null,
    bias !== undefined ? `bias ${coefSigned(bias)}` : null,
    n !== undefined ? `n ${n}` : null,
  ].filter(Boolean).join(stacked ? ' · ' : ', ')
  const line = separator === 'x' ? `${coef(slope)}x ${coefSigned(intercept)}` : `${coef(slope)} / ${coefSigned(intercept)}`
  return <>{line}{quality && (stacked
    ? <><br /><span style={{ color: INK.muted, fontSize: 11 }}>{quality}</span></>
    : <> <span style={{ color: INK.muted }}>{quality}</span></>)}</>
}

/** "median [p10–p90]" with the interval muted; `signed` for differences and intercepts. */
export function BandCell({ median, lo, hi, signed = false }: { median: unknown; lo: unknown; hi: unknown; signed?: boolean }) {
  const f = signed ? coefSigned : coef
  if (median == null) return <>—</>
  return <>{f(median)} <span style={{ color: INK.muted }}>[{f(lo)}–{f(hi)}]</span></>
}
