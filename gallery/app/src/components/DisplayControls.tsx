import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { AutoThemeIcon, ExplainIcon, MoonIcon, SunIcon, TextSizeIcon, XIcon } from '@/components/Icons'
import { useFloatingMenu } from '@/hooks/useFloatingMenu'
import { EXPLAIN_AVAILABLE, usePrefs } from '@/lib/prefs'
import { setThemePref, useThemePref } from '@/lib/colorTheme'
import { DISPLAY_SIDES, mostVisibleChart, type ChartDisplay, type StatsPlace, type VisibleChart } from '@/lib/chartDisplay'
import { resetChartDisplay, setSharedChartDisplay, setSingleChartDisplay, useChartDisplayPrefs } from '@/lib/chartDisplayPrefs'
import { useSingleView } from '@/lib/singleView'
import { SpartanDataSettings, useSpartanSourceLabel } from '@/components/SpartanDataSettings'
import { CHART_BUILDER_AVAILABLE, setChartBuilderEnabled, useChartBuilderEnabled } from '@/lib/chartBuilderPrefs'
import { useHeaderNav } from '@/lib/headerNav'

const SIDE_LABEL = { top: 'Above', right: 'Right', left: 'Left' }
const THEME_NEXT = { light: 'dark', dark: 'auto', auto: 'light' } as const

function visibleChart(): VisibleChart | null {
  const headerBottom = document.querySelector('.app-header')?.getBoundingClientRect().bottom ?? 0
  const charts = [...document.querySelectorAll<HTMLElement>('[data-chart-key]')].filter(el => el.getClientRects().length).map(el => {
    const box = (el.querySelector('.frame-chart') ?? el).getBoundingClientRect()
    const rect = { left: box.left, right: box.right, top: box.top, bottom: box.bottom }
    // Dashboard tiles can scroll or clip their contents independently of the page.
    for (let parent = el.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent), bounds = parent.getBoundingClientRect()
      if (/(auto|scroll|hidden|clip)/.test(style.overflowY)) { rect.top = Math.max(rect.top, bounds.top); rect.bottom = Math.min(rect.bottom, bounds.bottom) }
      if (/(auto|scroll|hidden|clip)/.test(style.overflowX)) { rect.left = Math.max(rect.left, bounds.left); rect.right = Math.min(rect.right, bounds.right) }
    }
    return { key: el.dataset.chartKey!, title: el.dataset.chartTitle!, rect, focused: el.classList.contains('focused') }
  })
  return mostVisibleChart(charts, { left: 0, right: window.innerWidth, top: headerBottom, bottom: window.innerHeight })
}

/** A click changes all charts. Right-click (or the visible menu button) offers explicit scope. */
export function DisplayControls() {
  const { density, setDensity, explain, setExplain } = usePrefs()
  const theme = useThemePref()
  const prefs = useChartDisplayPrefs()
  const single = useSingleView()
  const sourceLabel = useSpartanSourceLabel()
  const builderEnabled = useChartBuilderEnabled()
  const nav = useHeaderNav()
  const placement = single.on ? single.side : prefs.controlsSide
  const [open, setOpen] = useState(false)
  const [panel, setPanel] = useState<'display' | 'data' | 'interface'>('display')
  const [target, setTarget] = useState<VisibleChart | null>(null)
  const [scope, setScope] = useState<'all' | 'chart'>('all')
  const root = useRef<HTMLSpanElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const returnFocus = useRef<HTMLElement | null>(null)
  const menuId = useId()
  const place = useFloatingMenu(open, root, menu, 560)
  const shared: ChartDisplay = { controlsSide: prefs.controlsSide, statsPlace: prefs.statsPlace ?? 'options', density, explain, theme }
  const override = target ? prefs.charts[target.key] : undefined
  // a chart can keep its statistics above the plot by default (ChartFrame statsPlace): the frame says what it uses
  const chartStats = target ? document.querySelector<HTMLElement>(`[data-chart-key="${CSS.escape(target.key)}"]`)?.dataset.statsPlace as StatsPlace | undefined : undefined
  const settings = scope === 'chart' ? { ...shared, ...(chartStats ? { statsPlace: chartStats } : {}), ...override } : shared
  const showMenu = (from: HTMLElement) => {
    returnFocus.current = from
    setTarget(visibleChart())
    setScope('all')
    setOpen(true)
  }
  const close = () => { setOpen(false); returnFocus.current?.focus({ preventScroll: true }) }
  useEffect(() => {
    if (open && place) menu.current?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true })
  }, [open, !!place])
  useEffect(() => {
    if (!open) return
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node) && !menu.current?.contains(event.target as Node)) setOpen(false)
    }
    const escape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); setOpen(false); returnFocus.current?.focus({ preventScroll: true }) }
    }
    window.addEventListener('pointerdown', outside)
    window.addEventListener('keydown', escape)
    return () => { window.removeEventListener('pointerdown', outside); window.removeEventListener('keydown', escape) }
  }, [open])
  const apply = (patch: Partial<ChartDisplay>, onlyChart = false) => {
    if (onlyChart && target) { setSingleChartDisplay(target.key, patch); return }
    // Persist exceptions before a global theme change remounts the app.
    setSharedChartDisplay(patch)
    if (patch.density) setDensity(patch.density)
    if (patch.explain !== undefined) setExplain(patch.explain)
    if (patch.theme) setThemePref(patch.theme)
  }
  const trapFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab') return
    const items = [...menu.current!.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], summary')].filter(el => el.getClientRects().length)
    const at = items.indexOf(document.activeElement as HTMLElement)
    if (event.shiftKey && at === 0 || !event.shiftKey && at === items.length - 1) {
      event.preventDefault(); items[event.shiftKey ? items.length - 1 : 0]?.focus()
    }
  }
  return <span className="display-controls" ref={root}
    onContextMenu={event => { event.preventDefault(); showMenu(event.target instanceof HTMLElement ? event.target.closest('button') ?? trigger.current! : trigger.current!) }}
    onKeyDown={event => { if (event.key === 'ContextMenu' || event.shiftKey && event.key === 'F10') { event.preventDefault(); showMenu(document.activeElement as HTMLElement) } }}>
    {/* in the single-chart view the placement is the view's own (side= in the link), so a shared link keeps it */}
    <button type="button" className={`btn quiet icon-btn${placement !== 'top' ? ' on' : ''}`} aria-label="Options placement"
      onClick={() => { const next = DISPLAY_SIDES[(DISPLAY_SIDES.indexOf(placement) + 1) % DISPLAY_SIDES.length]; if (single.on) single.setSide(next); else apply({ controlsSide: next }) }}
      title={single.on ? `Options: ${SIDE_LABEL[placement].toLowerCase()} in the one-chart view (kept in the link). Click to cycle.` : `Options: ${SIDE_LABEL[placement].toLowerCase()} for all charts. Click to cycle; right-click for settings and this chart only.`}>
      <OptionsPlacementIcon side={placement} />
    </button>
    <button type="button" ref={trigger} className={`btn quiet icon-btn display-menu-trigger${open ? ' on' : ''}`} aria-label="Display and data settings"
      aria-haspopup="dialog" aria-expanded={open} aria-controls={open ? menuId : undefined}
      title="Display and data settings: text size, theme, interface and SPARTAN live data" onClick={event => open ? close() : showMenu(event.currentTarget)}>
      <TextSizeIcon />{sourceLabel && <span className="spartan-source-label">{sourceLabel}</span>}
    </button>
    {EXPLAIN_AVAILABLE && <button type="button" className={`btn quiet icon-btn${explain ? ' on' : ''}`} aria-pressed={explain} aria-label="Explain"
      onClick={() => apply({ explain: !explain })} title="Explanations for all charts. Click to toggle; right-click for settings and this chart only."><ExplainIcon /></button>}
    <button type="button" className={`btn quiet icon-btn${theme !== 'light' ? ' on' : ''}`} aria-label={`Theme: ${theme}`}
      onClick={() => apply({ theme: THEME_NEXT[theme] })} title="Theme for all charts: light, dark, or system. Right-click for settings and this chart only. Exports stay light.">
      {theme === 'dark' ? <MoonIcon /> : theme === 'auto' ? <AutoThemeIcon /> : <SunIcon />}
    </button>
    {open && createPortal(<div ref={menu} id={menuId} className="display-menu" role="dialog" aria-modal="true" aria-label="Display and data settings" onKeyDown={trapFocus}
      style={place ?? { position: 'fixed', visibility: 'hidden' }}>
      <header><strong>Display and data</strong><button type="button" className="btn quiet x-btn" aria-label="Close display settings" onClick={close}><XIcon /></button></header>
      <div className="seg" role="group" aria-label="Settings section">
        <button type="button" className={panel === 'display' ? 'active' : ''} aria-pressed={panel === 'display'} onClick={() => setPanel('display')}>Display</button>
        <button type="button" className={panel === 'data' ? 'active' : ''} aria-pressed={panel === 'data'} onClick={() => setPanel('data')}>SPARTAN data</button>
        {CHART_BUILDER_AVAILABLE && <button type="button" className={panel === 'interface' ? 'active' : ''} aria-pressed={panel === 'interface'} onClick={() => setPanel('interface')}>Interface</button>}
      </div>
      {panel === 'interface' ? <>
        <DisplayChoice label="Free-form chart builder" value={builderEnabled ? 'on' : 'off'} choices={[{ value: 'off', label: 'Off' }, { value: 'on', label: 'On' }]}
          onChange={value => setChartBuilderEnabled(value === 'on')} />
        <p>Explore combinations of measures and site or season facets with Graphic Walker. Adds Chart builder to the Charts menu.</p>
        {builderEnabled && <button type="button" className="btn" onClick={() => { close(); nav?.goTo({ tab: 'builder' }) }}>Open chart builder</button>}
        <p className="display-footnote">Optional and remembered in this browser. The builder loads when opened. Download a draft to keep or share its chart definitions.</p>
      </> : panel === 'data' ? <SpartanDataSettings /> : <>
      <fieldset className="display-scope"><legend>Apply changes to</legend>
        <div className="seg"><button type="button" className={scope === 'all' ? 'active' : ''} aria-pressed={scope === 'all'} onClick={() => setScope('all')}>All charts</button>
          <button type="button" className={scope === 'chart' ? 'active' : ''} aria-pressed={scope === 'chart'} disabled={!target} onClick={() => setScope('chart')}>This chart only</button></div>
        <p>{scope === 'chart' ? target?.title : 'Every chart, including charts on other tabs.'}</p>
        {target ? <p className="display-target">Most visible: <strong>{target.title}</strong></p> : <p>No chart is visible.</p>}
      </fieldset>
      <DisplayChoice label="Options placement" value={settings.controlsSide} choices={DISPLAY_SIDES.map(value => ({ value, label: SIDE_LABEL[value] }))}
        onChange={controlsSide => apply({ controlsSide }, scope === 'chart')} />
      <DisplayChoice label="Fit statistics, one-chart view" value={settings.statsPlace} choices={[{ value: 'options', label: 'With the options' }, { value: 'plot', label: 'Above the plot' }]}
        onChange={statsPlace => apply({ statsPlace }, scope === 'chart')} />
      <DisplayChoice label="Text size" value={settings.density} choices={[{ value: 'compact', label: 'Compact' }, { value: 'readable', label: 'Readable' }]}
        onChange={density => apply({ density }, scope === 'chart')} />
      {EXPLAIN_AVAILABLE && <DisplayChoice label="Explanations" value={settings.explain ? 'show' : 'hide'} choices={[{ value: 'hide', label: 'Hide' }, { value: 'show', label: 'Show' }]}
        onChange={value => apply({ explain: value === 'show' }, scope === 'chart')} />}
      <DisplayChoice label="Theme" value={settings.theme} choices={[{ value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }, { value: 'auto', label: 'System' }]}
        onChange={theme => apply({ theme }, scope === 'chart')} />
      {scope === 'chart' && target && <button type="button" className="btn" disabled={!override} onClick={() => resetChartDisplay(target.key)}>Use shared settings for this chart</button>}
      <p className="display-footnote">Remembered in this browser. Exported figures stay light.</p>
      </>}
    </div>, document.body)}
  </span>
}

function DisplayChoice<T extends string>({ label, value, choices, onChange }: { label: string; value: T; choices: { value: T; label: string }[]; onChange: (value: T) => void }) {
  return <fieldset className="display-choice"><legend>{label}</legend><div className="seg">{choices.map(choice =>
    <button key={choice.value} type="button" className={choice.value === value ? 'active' : ''} aria-pressed={choice.value === value} onClick={() => onChange(choice.value)}>{choice.label}</button>,
  )}</div></fieldset>
}

function OptionsPlacementIcon({ side }: { side: ChartDisplay['controlsSide'] }) {
  const bar = side === 'top' ? { x: 2, y: 2, width: 12, height: 3 } : { x: side === 'right' ? 10 : 2, y: 2, width: 4, height: 12 }
  return <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><rect x="1" y="1" width="14" height="14" rx="2" fill="none" stroke="currentColor" strokeWidth="1.3" /><rect {...bar} rx="0.8" fill="currentColor" /></svg>
}
