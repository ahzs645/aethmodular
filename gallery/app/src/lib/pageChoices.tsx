import { createContext, useContext, type ReactNode } from 'react'
import { ChartFrame } from '@/components/ChartFrame'
import { useSearchParam } from '@/lib/url'

/**
 * Choices a whole page shares (the baseline, the season), set once at the top instead of a
 * picker on every chart. Each chart reads the page's value through `usePageChoice`; when its
 * own data does not have that value (a baseline one analysis did not run), it shows the first
 * one it has and says so, rather than drawing nothing.
 *
 *   <PageChoices defs={[{ key: 'baseline', urlKey: 'mbase', label: 'Baseline', default: 'AIRSpec',
 *     options: methods, legacy: ['mrmethod', …], render: (v, set) => <BaselinePicker … /> }]}>
 *     …charts…
 *   </PageChoices>
 *   const { value, substituted } = usePageChoice('baseline', methods)
 */
export interface PageChoiceDef {
  key: string
  /** one URL key for the page (old per-chart keys in `legacy` still open the same view) */
  urlKey: string
  label: string
  default: string
  options: readonly string[]
  legacy?: readonly string[]
  /** a bespoke control (a baseline picker); a plain select otherwise */
  render: (value: string, set: (v: string) => void) => ReactNode
}

interface Entry { value: string; set: (v: string) => void; label: string; render: PageChoiceDef['render'] }
const Ctx = createContext<Record<string, Entry>>({})

/** `panel={false}` when the page has its own place for page-wide controls (draw them with PageChoiceControls) */
export function PageChoices({ defs, title = 'For every chart on this page', panel = true, children }: { defs: readonly PageChoiceDef[]; title?: string; panel?: boolean; children: ReactNode }) {
  // a fixed list of choices per page, so the hooks run in the same order every render
  const states = defs.map((d) => useSearchParam<string>(d.urlKey, legacyValue(d) ?? d.default, d.options)) // eslint-disable-line react-hooks/rules-of-hooks
  const value = Object.fromEntries(defs.map((d, i) => [d.key, { value: states[i][0], set: states[i][1], label: d.label, render: d.render }]))
  return (
    <Ctx.Provider value={value}>
      {panel && <ChartFrame title={title} exportable={false} controls={<PageChoiceControls />}>
        <p className="chart-note" style={{ marginTop: 0 }}>Set once here: every chart below follows. A chart whose data lacks the choice shows the nearest one it has, and says so.</p>
      </ChartFrame>}
      {children}
    </Ctx.Provider>
  )
}

/** the page's shared controls, wherever the page puts them (its panel, or a sidebar) */
export function PageChoiceControls() {
  const all = useContext(Ctx)
  return <>{Object.entries(all).map(([k, e]) => <span key={k}>{e.render(e.value, e.set)}</span>)}</>
}

/** the page's value for this chart, among the values the chart's data has */
export function usePageChoice(key: string, available?: readonly string[]) {
  const c = useContext(Ctx)[key]
  const chosen = c?.value ?? ''
  const value = !available?.length || available.includes(chosen) ? chosen : available[0]
  return {
    value, chosen, set: c?.set ?? (() => {}), substituted: value !== chosen,
    /** a sentence for the chart when it had to substitute */
    note: value !== chosen ? `${c?.label ?? key}: ${chosen} is not in this analysis, so it shows ${value}.` : null,
  }
}

function legacyValue(d: PageChoiceDef): string | null {
  const q = new URL(window.location.href).searchParams
  if (q.get(d.urlKey)) return null
  for (const k of d.legacy ?? []) { const v = q.get(k); if (v && d.options.includes(v)) return v }
  return null
}
