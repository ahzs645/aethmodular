import { createContext, useContext, useMemo, type ReactNode } from 'react'
import { fmt, type RegressionStats } from './stats'

import { ESTIMATORS, FIT_STYLE, type Estimator } from './regressionMethods'
export { ESTIMATORS, FIT_STYLE, parseEstimators, estimatorParam, type Estimator } from './regressionMethods'

const Ctx = createContext<{ methods: Estimator[]; setMethods: (v: Estimator[]) => void; showCI: boolean; setShowCI: (v: boolean) => void }>({ methods: ['deming'], setMethods: () => {}, showCI: false, setShowCI: () => {} })
export function RegressionProvider({ methods, setMethods, showCI, setShowCI, children }: { methods: Estimator[]; setMethods: (v: Estimator[]) => void; showCI: boolean; setShowCI: (v: boolean) => void; children: ReactNode }) {
  const value = useMemo(() => ({ methods, setMethods, showCI, setShowCI }), [methods, setMethods, showCI, setShowCI])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
export const useRegressions = () => useContext(Ctx)

export interface Fit { method: Estimator; slope: number; intercept: number; label: string; color: string; dash?: string; assumption?: string }
export function fitsFor(stats: RegressionStats | null, methods: Estimator[]): Fit[] {
  if (!stats) return []
  return methods.flatMap(method => {
    const slope = method === 'ols' ? stats.slope : method === 'deming' ? stats.demingSlope : stats.yorkSlope
    const intercept = method === 'ols' ? stats.intercept : method === 'deming' ? stats.demingIntercept : stats.yorkIntercept
    return slope !== null && intercept !== null && Number.isFinite(slope) && Number.isFinite(intercept)
      ? [{ method, slope, intercept, ...FIT_STYLE[method], assumption: method === 'york' ? stats.yorkAssumption : undefined }] : []
  })
}
export const equation = (fit: Fit, digits = 3) => `${fit.label} y = ${fmt(fit.slope, digits)}x ${fit.intercept < 0 ? '−' : '+'} ${fmt(Math.abs(fit.intercept), digits)}`

export function RegressionControls() {
  const { methods, setMethods, showCI, setShowCI } = useRegressions()
  return <fieldset className="regression-controls">
    <legend>Regression fits</legend>
    <div className="regression-options">{ESTIMATORS.map(method => <label key={method}>
      <input type="checkbox" checked={methods.includes(method)} onChange={() => setMethods(ESTIMATORS.filter(e => e === method ? !methods.includes(e) : methods.includes(e)))} />
      <span style={{ color: FIT_STYLE[method].color }}>{FIT_STYLE[method].label}</span>
    </label>)}</div>
    <p>Compare fits or deselect all to hide them. Deming is the default.</p>
    <label><input type="checkbox" checked={showCI} onChange={e => setShowCI(e.target.checked)} /> 95% fit uncertainty</label>
    <p>Show line confidence bands and coefficient intervals where available. These describe the fit, not individual measurement errors or prediction ranges.</p>
    {methods.includes('york') && <p>York uses point uncertainties when supplied; otherwise it uses constant errors with the Deming error ratio. Constant-error York and Deming coincide.</p>}
    <button className="btn quiet" type="button" onClick={() => { setMethods(['deming']); setShowCI(false) }}>Reset fits</button>
  </fieldset>
}

/** Saved-summary charts have no per-filter uncertainty vectors. Keep the assumption visible. */
const Active = createContext<Estimator>('deming')
export const useEstimator = () => useContext(Active)
export function EstimatorViews({ children, available = ESTIMATORS }: { children: (method: Estimator) => ReactNode; available?: readonly Estimator[] }) {
  const { methods, setMethods, showCI, setShowCI } = useRegressions()
  if (!methods.length) return <p className="chart-note">Regression fits hidden. Select an estimator in the sidebar to show this comparison.</p>
  return <>{methods.map(method => <Active.Provider key={method} value={method}>
    <Ctx.Provider value={{ methods: [method], setMethods, showCI, setShowCI }}>
    <div className="estimator-view" data-estimator={method}>
      {/* the caption tells stacked copies apart; Deming alone is the default view and needs none */}
      {(methods.length > 1 || method !== 'deming') && <p className="estimator-caption" style={{ color: FIT_STYLE[method].color }}>{FIT_STYLE[method].label}
        {method === 'york' && available.includes(method) && ' · constant-error fit (same error ratio as saved Deming)'}
        {method !== 'deming' && available.includes(method) && ' · saved Deming guardrails unchanged'}
      </p>}
      {showCI && <p className="chart-note">95% intervals are shown where the export contains them or paired data support resampling. Coefficient-only summaries have no saved confidence intervals; no line band is inferred from marginal coefficient bounds.</p>}
      {available.includes(method) ? children(method) : <p className="chart-note">{FIT_STYLE[method].label} is not available in this saved analysis. No substitute fit is shown.</p>}
    </div>
    </Ctx.Provider>
  </Active.Provider>)}</>
}

/** Project saved summaries without changing the original Deming guardrails or scores. */
export function projectGrid<T extends { dm: number | null; db: number | null; om: number | null; ob: number | null }>(row: T, method: Estimator): T {
  if (method !== 'ols') return row
  const r = row as T & { adm?: number | null; adb?: number | null; aom?: number | null; aob?: number | null; db6?: number | null; ob6?: number | null }
  return { ...r, guardDm: r.dm, guardDb: r.db, dm: r.om, db: r.ob, adm: r.aom ?? null, adb: r.aob ?? null, db6: r.ob6 ?? null }
}

export function projectRun<T extends import('./types').CalibRun>(run: T, method: Estimator): T {
  const project = (rows: import('./types').MetricRow[]) => rows.map(r => method === 'ols' ? { ...r, deming_slope: r.ols_slope, deming_intercept: r.ols_intercept } : r)
  return { ...run, metrics: project(run.metrics), split_check: run.split_check.map(s => ({ ...s, metrics: project(s.metrics) })) }
}

export const estimatorText = (text: string, method: Estimator) => {
  const at = text.toLowerCase().indexOf('guardrail')
  return at < 0 ? text.replace(/Deming/g, FIT_STYLE[method].label) : text.slice(0, at).replace(/Deming/g, FIT_STYLE[method].label) + text.slice(at)
}
export function EstimatorName() { return <>{FIT_STYLE[useEstimator()].label}</> }
