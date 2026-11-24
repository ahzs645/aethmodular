import { useMemo } from 'react'
import { Select } from '@/components/Controls'
import { ScrubField } from '@/components/ScrubField'
import { useExport } from '@/lib/data'
import { fmt } from '@/lib/format'
import { useFirstSampleSite } from '@/lib/gallerySamples'
import { EVEN, FULL_ALPHA, IMPROVE_HOLES, WHITE_ALPHA, curveModel, screenModel, type HipsModel, type ModelFit } from '@/lib/hipsPattern'
import type { LinFilter } from '@/lib/hipsLinearise'
import { useExclusionSplit } from '@/lib/markedRows'
import { SidebarControls } from '@/lib/pageSidebar'
import { RESEARCH_SERIES } from '@/lib/palettes'
import { usePageSite } from '@/lib/siteFocus'
import { useSearchParam } from '@/lib/url'
import { SIDE, useHipsSiteFilters, type Ceil, type Linearise } from '@/pages/HipsLinearisePage'

/** What the "Warren White's model and ours" page's charts share: its data, the corrections offered, and the page's choice. */

export const WARREN_COMMAND = 'uv run python gallery/data/export_hips_warren.py'
export const WARREN_FILE = 'hips_linearise/warren.json'
/** the private Phoenix pairs: never in a public build (the name is dropped from it), and gitignored */
export const PRIVATE_BUILD = import.meta.env.VITE_PUBLIC_BUILD !== '1'
const DRAIN_FILE = PRIVATE_BUILD ? 'research/hips_drain_disc/pairs.json' : null

export interface WarrenSet { key: string; label: string; why: string; filters: number; sites: number; groups: number; tau_inf: number; lo: number; hi: number; past_cut: number; past_limit: number }
export interface WarrenFit { key: string; label: string; free: string; params: number[]; rms_log: number; aic: number; residual_pct: number[] }
export interface Warren {
  n_boot: number; hole_fraction: number; limit_tau: number; cut: number
  areas: { area_cm2: number; filters: number }[]
  sets: WarrenSet[]; medians: { n: number; loading_ug: number; tau: number }[]; fits: WarrenFit[]
  splits: { alpha: number; c: number; rms_log: number; aic: number }[]
  shares: { population: string; filters: number; past_cut: number; past_limit: number }[]
}
export interface DrainPair { d: string; t1: number; t5: number; /** Fabs, collocated over routine */ r: number; mass: number | null; ec: number | null; fe: number | null; drain: boolean }
export interface Drain {
  note: string; window: [string, string]; pairs: DrainPair[]
  dark: { n: number; ratio: number; ci: [number, number]; other_n: number; other_ratio: number }
  alpha: { value: number; ci: [number, number] }
  ceilings: { dates: string; sampler: string; pairs: number; tau_inf: number; ci: [number, number] }[]
  hips_relative_error: { all_pairs: number; without_drain: number }
}
export const useWarren = () => useExport<Warren>(WARREN_FILE)
/** null in a public build, or in a clone without the file */
export const useDrain = () => useExport<Drain>(DRAIN_FILE).data ?? null

export const C = { none: RESEARCH_SERIES[7], alpha10: RESEARCH_SERIES[0], full: RESEARCH_SERIES[4], curve: RESEARCH_SERIES[1], drain: RESEARCH_SERIES[3], custom: RESEARCH_SERIES[2] } as const
export type ModelKey = keyof typeof C
const KEYS: readonly ModelKey[] = ['none', 'alpha10', 'full', 'curve', 'drain', 'custom']
export const ADDIS = 'ETAD'

/** the corrections the page offers, and the one it draws (URL `wp.model`; `wp.h` and `wp.a` for a screen of the reader's own) */
export function useModels(ceil: Ceil, drain: Drain | null) {
  const [key, setKey] = useSearchParam<ModelKey>('wp.model', 'alpha10', KEYS)
  const [hText, setH] = useSearchParam<string>('wp.h', '0.5')
  const [aText, setA] = useSearchParam<string>('wp.a', '10')
  const h = Math.min(0.95, Math.max(0.1, Number(hText) || 0.5)), alpha = Math.min(100, Math.max(1, Number(aText) || 10))
  const drainAlpha = drain?.alpha.value ?? null
  const models = useMemo<(HipsModel & { key: ModelKey })[]>(() => [
    { ...EVEN, key: 'none' as const },
    ...(drainAlpha != null ? [screenModel('drain', `White's screen, α = ${fmt(drainAlpha, 1)} (Phoenix drain discs)`, `α ${fmt(drainAlpha, 1)}`, drainAlpha)] : []),
    screenModel('alpha10', `White's screen, α = ${WHITE_ALPHA} (published)`, `α ${WHITE_ALPHA}`, WHITE_ALPHA),
    screenModel('full', "White's screen, fully pixelated (his upper bound)", 'fully pixelated', FULL_ALPHA),
    curveModel('curve', `Our curve, τ∞ ${fmt(ceil.ceiling, 2)}`, 'our curve', ceil.ceiling),
    screenModel('custom', `A screen of your own: holes ${fmt(h, 2)}, α ${fmt(alpha, 1)}`, 'your screen', alpha, h),
  ] as (HipsModel & { key: ModelKey })[], [ceil.ceiling, drainAlpha, h, alpha])
  const model = models.find((m) => m.key === key) ?? models.find((m) => m.key === 'alpha10')!
  /** pick a screen on the mesh chart */
  const setScreen = (nh: number, na: number) => { setH(String(+nh.toFixed(3))); setA(String(+na.toFixed(2))); setKey('custom') }
  return { models, model, key: model.key, setKey, h, alpha, setH, setA, setScreen }
}
export type Models = ReturnType<typeof useModels>

export function ModelOptions({ m }: { m: Models }) {
  return <SidebarControls title="Correction" tabNote={m.model.short}>
    <div role="group" aria-label="Correction" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
      <Select label="" value={m.key} options={m.models.map((x) => x.key)} onChange={(v) => m.setKey(v as ModelKey)} optionLabel={(k) => m.models.find((x) => x.key === k)?.label ?? k}
        title="The correction the Addis crossplot, the site strip and the factor curve draw. A screen is White et al. (2025): a share of the area over the holes holds a deposit α times as thick as the rest." />
      {m.key === 'custom' && <>
        <span className="control" title="Share of the support screen that is holes (IMPROVE 0.638, CSN 0.283)">Holes <ScrubField label="Share of the screen that is holes" value={m.h} min={0.1} max={0.95} step={0.01} width={76} format={(v) => fmt(v, 2)} onChange={(v) => m.setH(String(+v.toFixed(2)))} /></span>
        <span className="control" title="How many times thicker the deposit is over a hole than between holes">α <ScrubField label="Amplitude α" value={m.alpha} min={1} max={100} step={0.5} width={76} format={(v) => fmt(v, 1)} onChange={(v) => m.setA(String(+v.toFixed(1)))} /></span>
      </>}
    </div>
  </SidebarControls>
}

/** one site's filters: Addis is the closure workflow's set and its λ, any other the public rows and λ = 1 */
export function useSiteSet(data: Linearise) {
  const { bySite, sites, loaded } = useHipsSiteFilters(data)
  const pick = usePageSite(sites, { prefer: useFirstSampleSite() ?? ADDIS })
  const code = pick.code || ADDIS, addis = code === ADDIS
  const raw = useMemo<LinFilter[]>(() => (addis ? data.filters : bySite.get(code)?.filters ?? []), [addis, data.filters, bySite, code])
  const { counted } = useExclusionSplit(addis ? [] : raw, (f) => f.id)
  const filters = useMemo(() => (addis ? raw : counted).filter((f) => f.ftir != null), [addis, raw, counted])
  const full = sites.find((s) => s.code === code)?.name ?? 'Addis Ababa'
  return { filters, lambda: addis ? data.lambda : 1, code, addis, name: full.split(',')[0], pick, sites, bySite, loaded }
}
export type SiteSet = ReturnType<typeof useSiteSet>
export const siteOption = (sites: SiteSet['sites']) => (v: string) => { const s = sites.find((x) => x.code === v); return s ? `${s.name} (${s.n})` : v }

export const line = (f: { slope: number; intercept: number } | null | undefined) => (f ? `${fmt(f.slope, 2)}x ${f.intercept < 0 ? '−' : '+'} ${fmt(Math.abs(f.intercept), 2)}` : '—')
export const macText = (f: ModelFit | null | undefined) => (!f || !Number.isFinite(f.mac) ? '—' : f.mac > 200 ? '> 200' : fmt(f.mac, 1))
export const pct = (v: number, d = 0) => `${fmt(100 * v, d)} %`
export const HALF = SIDE * 0.8

/** a button that scrolls to a chart on this page */
export function ToChart({ id, children }: { id: string; children: string }) {
  return <button type="button" className="link-btn" onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>{children}</button>
}
export { IMPROVE_HOLES }
