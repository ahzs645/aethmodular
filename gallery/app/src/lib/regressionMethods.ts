export const ESTIMATORS = ['ols', 'deming', 'york'] as const
export type Estimator = typeof ESTIMATORS[number]
export const FIT_STYLE = {
  ols: { label: 'OLS', color: '#2171b5', dash: undefined },
  deming: { label: 'Deming', color: '#c026d3', dash: '7 3' },
  york: { label: 'York', color: '#0f766e', dash: '2 3' },
} as const
export const parseEstimators = (value: string | null): Estimator[] => {
  if (value === 'none') return []
  const parsed = value === null ? [] : ESTIMATORS.filter(e => value.split('|').includes(e))
  return parsed.length ? parsed : ['deming']
}
export const estimatorParam = (methods: Estimator[]) => methods.length === 1 && methods[0] === 'deming' ? null : methods.length ? methods.join('|') : 'none'

