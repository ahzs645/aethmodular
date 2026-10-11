import { INK } from './theme.ts'
export const ESTIMATORS = ['ols', 'deming', 'york'] as const
export type Estimator = typeof ESTIMATORS[number]
export const FIT_STYLE = {
  ols: { label: 'OLS', color: INK.accent, dash: undefined },
  deming: { label: 'Deming', color: INK.deming, dash: '7 3' },
  york: { label: 'York', color: INK.good, dash: '2 3' },
} as const
export const parseEstimators = (value: string | null): Estimator[] => {
  if (value === 'none') return []
  const parsed = value === null ? [] : ESTIMATORS.filter(e => value.split('|').includes(e))
  return parsed.length ? parsed : ['deming']
}
export const estimatorParam = (methods: Estimator[]) => methods.length === 1 && methods[0] === 'deming' ? null : methods.length ? methods.join('|') : 'none'

