import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select, Toggle } from '@/components/ChartFrame'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { ColorLegend } from '@/components/ColorLegend'
import type { Config } from './common'
import { useEstimator, FIT_STYLE } from '@/lib/regressionPrefs'
import { COHORT, CV_PROTOCOL, PREPROCESSING, SELECTION_SPACE, label } from '@/lib/labels'
import { choiceFilterDimensions, choiceFilterSets, matchesChoiceFilters, DEFAULT_HIGHLIGHT_LIMITS, highlightMatches, type HighlightLimits, type HighlightRule } from '@/lib/specification'
import { SpecCurvePlots, type SpecLayout } from './SpecCurvePlots'

/** spec_curve.json, written by gallery/data/export_spec_curve.py */
export interface SpecCurveFile {
  n: number
  n_pass: number
  guardrails: string
  choices: string[]
  /** optional headings: rows start..end-1 sit under title */
  groups?: { title: string; start: number; end: number }[]
  masks: string[]
  ok: string
  intercept: number[]
  olsIntercept?: (number | null)[]
  olsSlope?: (number | null)[]
  slope: (number | null)[]
  heldout: (number | null)[]
  cut: (number | null)[]
  k: (number | null)[]
  categories: { co: string[]; sp: string[]; sel: string[]; mo: string[]; lot: string[] }
  co: number[]
  sp: number[]
  sel: number[]
  mo: number[]
  lot: number[]
}

const BASE = `${import.meta.env.BASE_URL}data`
type FilterKey = 'co' | 'sp' | 'sel' | 'mo' | 'lot' | 'cut' | 'k'
const ALL = '*'
const NO_CUTOFF = 'complete'
const initialFilters = (): Record<FilterKey, string> => ({ co: ALL, sp: ALL, sel: ALL, mo: ALL, lot: ALL, cut: ALL, k: ALL })
const BAND = 0.5
const RULE_LABELS: Record<HighlightRule, string> = { guardrails: 'Saved guardrails', intercept: 'Near-zero intercept', 'cv-slope': 'Slope + cross-validation', custom: 'Intercept + slope + cross-validation' }
/** strip heat map: share of specifications using a choice, white (none) to dark red (all) */
const HEAT = (t: number) => d3.interpolateReds(0.12 + 0.88 * t)

/**
 * The specification curve (Simonsohn et al. 2020) over the whole Addis
 * search: every scored configuration, every k, sorted by the Deming intercept
 * it produces, with the analytic choices that produced it underneath. The
 * interactive twin of research/ftir_ec_phase3/output/plots/pathways/
 * 01_specification_curve.png; row selection, guardrails and the choice list
 * come from the same spec_curve.py module, so the counts agree.
 *
 * 20k columns do not fit in 1,000 px, so each pixel of the choice strip is the
 * share of the specifications in it that use the choice (ink = all of them),
 * rather than whichever single column a raster happens to sample.
 */
export function SpecCurve({
  selected, onSelect, src = 'spec_curve.json', id = 'spec-curve', title, subtitle, provenance, controls, defaultLayout = 'combined', selectHint = 'click to load into the k-sweep below', excludePreprocessing = [], defaultComparison = 'full',
}: {
  selected?: Config
  onSelect?: (c: Config) => void
  /** another export in the same schema (e.g. meeting/spec_curve_20260917.json) */
  src?: string
  id?: string
  title?: string
  subtitle?: (f: SpecCurveFile | null) => string
  provenance?: string
  controls?: React.ReactNode
  defaultLayout?: SpecLayout
  selectHint?: string
  /** Hide archived correction/selection variants while retaining the frozen export. */
  excludePreprocessing?: readonly string[]
  defaultComparison?: 'full' | 'standard' | 'all'
}) {
  const method = useEstimator()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const [data, setData] = useState<SpecCurveFile | null | 'missing'>(null)
  const [required, setRequired] = useState<Set<number>>(new Set())
  const [passOnly, setPassOnly] = useState(false)
  const [filters, setFilters] = useState(initialFilters)
  const [layout, setLayout] = useState<SpecLayout>(defaultLayout)
  const [activeSelection, setActiveSelection] = useState('airspec')
  const [comparison, setComparison] = useState<string>(defaultComparison)
  const [showDots, setShowDots] = useState(true)
  const [rule, setRule] = useState<HighlightRule>('guardrails')
  const [limits, setLimits] = useState<HighlightLimits>({ ...DEFAULT_HIGHLIGHT_LIMITS })
  const excludedKey = excludePreprocessing.join('|')

  useEffect(() => {
    const excluded = excludedKey.split('|')
    setFilters(current => excluded.includes(current.sp) || excluded.includes(current.sel)
      ? { ...current, sp: excluded.includes(current.sp) ? ALL : current.sp, sel: excluded.includes(current.sel) ? ALL : current.sel }
      : current)
    if (excluded.includes('vibes')) setComparison(current => current === 'standard' ? 'full' : current)
  }, [excludedKey])

  useEffect(() => {
    let cancelled = false
    setData(null)
    setRequired(new Set())
    setFilters(initialFilters())
    fetch(`${BASE}/${src}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => !cancelled && setData(d))
      .catch(() => !cancelled && setData('missing'))
    return () => { cancelled = true }
  }, [src])

  const file = useMemo(() => {
    if (!data || data === 'missing') return null
    if (method !== 'ols') return data
    if (!data.olsIntercept || !data.olsSlope) return null
    const order = data.olsIntercept.map((_, i) => i).filter(i => data.olsIntercept![i] !== null && data.olsSlope![i] !== null)
      .sort((a, b) => data.olsIntercept![a]! - data.olsIntercept![b]!)
    const pick = <T,>(a: T[]) => order.map(i => a[i])
    const bits = (s: string) => order.map(i => s[i]).join('')
    return { ...data, n: order.length, n_pass: order.filter(i => data.ok[i] === '1').length,
      intercept: pick(data.olsIntercept) as number[], slope: pick(data.olsSlope), heldout: pick(data.heldout), cut: pick(data.cut), k: pick(data.k),
      co: pick(data.co), sp: pick(data.sp), sel: pick(data.sel), mo: pick(data.mo), lot: pick(data.lot), masks: data.masks.map(bits), ok: bits(data.ok) }
  }, [data, method])

  const highlighted = useMemo(() => file ? file.intercept.map((intercept, i) => highlightMatches({
    intercept, slope: file.slope[i], cv: file.heldout[i], guardrails: file.ok[i] === '1',
  }, rule, limits)) : [], [file, rule, limits])

  const choiceDimensions = useMemo(() => file ? choiceFilterDimensions(file) : [], [file])
  const choiceGroups = useMemo(() => choiceFilterSets(required, choiceDimensions), [required, choiceDimensions])

  // the specifications left after the choice and guardrail filters, still in intercept order
  const shown = useMemo(() => {
    if (!file) return [] as number[]
    const out: number[] = []
    for (let i = 0; i < file.n; i++) {
      if (excludePreprocessing.includes(file.categories.sp[file.sp[i]]) || excludePreprocessing.includes(file.categories.sel[file.sel[i]])) continue
      if (comparison !== 'all' && file.categories.sel.some(s => s.startsWith('vibes'))) {
        const vibes = comparison === 'full' ? (file.categories.sel.includes('vibes-full-r70') ? 'vibes-full-r70' : 'vibes-full') : 'vibes'
        if (!['airspec', vibes].includes(file.categories.sel[file.sel[i]])) continue
      }
      if (passOnly && !highlighted[i]) continue
      if ((['co', 'sp', 'sel', 'mo', 'lot'] as const).some(key => filters[key] !== ALL && file.categories[key][file[key][i]] !== filters[key])) continue
      if (filters.cut !== ALL && (file.cut[i] === null ? NO_CUTOFF : String(file.cut[i])) !== filters.cut) continue
      if (filters.k !== ALL && String(file.k[i]) !== filters.k) continue
      if (matchesChoiceFilters(file.masks, i, choiceGroups)) out.push(i)
    }
    return out
  }, [file, choiceGroups, passOnly, filters, highlighted, comparison, excludePreprocessing])

  const setFilter = (key: FilterKey, value: string) => {
    if ((key === 'sel' || key === 'sp') && (value === 'vibes' || value.startsWith('vibes-'))) setComparison('all')
    setFilters(current => ({ ...current, [key]: value }))
    tip.hide()
  }
  const categorySelect = (key: 'co' | 'sp' | 'sel' | 'mo' | 'lot', name: string, labels: Record<string, string>) => file && (
    <Select label={name} value={filters[key]} options={[ALL, ...file.categories[key].filter(value => !['sp', 'sel'].includes(key) || !excludePreprocessing.includes(value))]} onChange={value => setFilter(key, value)}
      optionLabel={value => value === ALL ? 'All' : label(labels, value)} />
  )
  const numericOptions = (key: 'cut' | 'k') => file ? [...new Set(file[key].filter((v): v is number => v !== null))].sort((a, b) => a - b).map(String) : []

  const selectedIndices = useMemo(() => new Set(shown.filter(i => selected && file
    && file.categories.co[file.co[i]] === selected.co && file.cut[i] === selected.cut
    && file.categories.sel[file.sel[i]] === (selected.sel ?? 'raw') && file.categories.sp[file.sp[i]] === selected.sp)), [file, shown, selected])
  const availableSelections = file ? file.categories.sel.filter((_, k) => shown.some(i => file.sel[i] === k)) : []
  const selection = availableSelections.includes(activeSelection) ? activeSelection : availableSelections[0]
  const displayed = layout === 'single' && file ? shown.filter(i => file.categories.sel[file.sel[i]] === selection) : shown
  const m = displayed.length
  const passCount = file ? displayed.filter(i => file.ok[i] === '1').length : 0
  const matchCount = displayed.filter(i => highlighted[i]).length
  const availableCount = file ? file.sp.filter((sp, i) => !excludePreprocessing.includes(file.categories.sp[sp]) && !excludePreprocessing.includes(file.categories.sel[file.sel[i]])).length : 0
  const highlightDescription = rule === 'guardrails' ? 'saved guardrails'
    : [rule !== 'cv-slope' ? `|intercept| ≤ ${limits.intercept} µg/m³` : '', rule !== 'intercept' ? `slope ${limits.slopeMin}–${limits.slopeMax}; cross-validation R² ≥ ${limits.cvMin}` : ''].filter(Boolean).join('; ')
  const numberControl = (key: keyof HighlightLimits, name: string, min: number, max?: number) => <label className="control">
    {name}<input aria-label={name} className="select" type="number" min={min} max={max} step="0.05" value={limits[key]} style={{ width: 78 }}
      onChange={e => { const value = e.target.valueAsNumber; if (Number.isFinite(value) && value >= min && (max === undefined || value <= max)) setLimits(current => ({ ...current, [key]: value })) }} />
  </label>

  // helpers below run only once the file has loaded (the body is empty until then)
  const f = file as SpecCurveFile
  const cfgOf = (i: number): Config => ({
    co: f.categories.co[f.co[i]],
    cut: f.cut[i],
    sel: f.categories.sel[f.sel[i]],
    sp: f.categories.sp[f.sp[i]],
  })
  const describe = (i: number) => {
    const c = cfgOf(i)
    const mode = f.categories.mo[f.mo[i]]
    const lot = f.categories.lot[f.lot[i]]
    return `${label(COHORT, c.co)}${c.cut !== null ? ' · cutoff ' + c.cut : ''} · ${label(SELECTION_SPACE, c.sel)} × ${label(PREPROCESSING, c.sp)} · ${label(CV_PROTOCOL, mode)} · ${f.k[i] ?? '?'} PLS factors${lot !== 'all' ? ' · calibration lot ' + lot : ''}`
  }

  const toggleChoice = (c: number) => {
    setRequired((s) => { const n = new Set(s); if (n.has(c)) n.delete(c); else n.add(c); return n })
    tip.hide()
  }

  return (
    <ChartFrame
      id={id}
      title={title ?? 'Specification curve — every Addis configuration, sorted by the intercept it produces'}
      subtitle={subtitle ? subtitle(file) : `All ${file ? f.n.toLocaleString() + ' ' : ''}scored Addis specifications (every cohort, cutoff, baseline, protocol and k), sorted by Deming intercept at MAC 10, with the analytic choices underneath: a darker strip means more of the specifications at that position used the choice. Click choices to filter: alternatives within the same setting are combined, while different settings must all match. Red points use the selected highlight rule; the default is every saved guardrail${file ? ': ' + f.guardrails : ''}.`}
      provenance={provenance ?? 'calibration_explorer batch results via research/ftir_ec_phase3/scripts/spec_curve.py · figure twin: pathways/01_specification_curve.png'}
      controls={
        <>
          {controls}
          <Select label="Graph view" value={layout} options={['single', 'separate', 'stacked', 'overlay', 'combined']} onChange={v => setLayout(v as SpecLayout)}
            optionLabel={v => ({ single: 'One selection at a time', separate: 'Side by side', stacked: 'Stacked selections', overlay: 'Combined overlay', combined: 'Combined ordered curve' }[v] ?? v)} />
          {layout === 'single' && availableSelections.length > 0 && <Select label="Display selection" value={selection} options={availableSelections} onChange={setActiveSelection} optionLabel={v => label(PREPROCESSING, v)} />}
          {file?.categories.sel.some(s => s.startsWith('vibes')) && <Select label="Comparison baselines" value={comparison}
            options={[...(file.categories.sel.some(s => s === 'vibes-full' || s === 'vibes-full-r70') ? ['full'] : []), ...(file.categories.sel.includes('vibes') && !excludePreprocessing.includes('vibes') ? ['standard'] : []), 'all']} onChange={setComparison}
            optionLabel={v => v === 'full' ? 'Spline + full-range VIBES' : v === 'standard' ? 'Spline + archived VIBES (4000–1425)' : excludePreprocessing.includes('vibes') ? 'All active baseline variants' : 'All active and archived variants'} />}
          {file && <>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', width: '100%' }}>
              <strong className="control">Calibration cohort</strong>
              {categorySelect('co', 'Selection method', COHORT)}
              {categorySelect('sel', 'Selection spectra', SELECTION_SPACE)}
              <Select label="Cohort cutoff" value={filters.cut} options={[ALL, ...(file.cut.includes(null) ? [NO_CUTOFF] : []), ...numericOptions('cut')]}
                onChange={value => setFilter('cut', value)} optionLabel={value => value === ALL ? 'All' : value === NO_CUTOFF ? 'Complete cohort (no cutoff)' : `${value} filters`}
                title="Requested number of filters in the selected calibration cohort; complete cohorts have no cutoff." />
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', width: '100%' }}>
              <strong className="control">Calibration settings</strong>
              {categorySelect('sp', 'Model spectra', PREPROCESSING)}
              {categorySelect('mo', 'Cross-validation', CV_PROTOCOL)}
              <Select label="PLS factors" value={filters.k} options={[ALL, ...numericOptions('k')]} onChange={value => setFilter('k', value)} optionLabel={value => value === ALL ? 'All' : value} />
              {file.categories.lot.length > 1 && categorySelect('lot', 'calibration lot', { all: 'All lots' })}
            </div>
          </>}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', width: '100%' }}>
            <Toggle label="Show red dots" checked={showDots} onChange={setShowDots} />
            <Select label="Red-dot criteria" value={rule} options={Object.keys(RULE_LABELS)} onChange={v => setRule(v as HighlightRule)} optionLabel={v => RULE_LABELS[v as HighlightRule]} />
            {rule !== 'guardrails' && <>
              {rule !== 'cv-slope' && numberControl('intercept', '|intercept| ≤ (µg/m³)', 0)}
              {rule !== 'intercept' && <>{numberControl('slopeMin', 'slope ≥', 0)}{numberControl('slopeMax', 'slope ≤', 0)}{numberControl('cvMin', 'cross-validation R² ≥', 0, 1)}</>}
              <button type="button" className="btn quiet" onClick={() => setLimits({ ...DEFAULT_HIGHLIGHT_LIMITS })}>Reset criteria</button>
            </>}
          </div>
          <Toggle label={rule === 'guardrails' ? 'Guardrail passers only' : 'Highlight matches only'} checked={passOnly} onChange={setPassOnly} />
          {(required.size > 0 || Object.values(filters).some(value => value !== ALL) || passOnly || (file?.categories.sel.some(s => s.startsWith('vibes')) && comparison !== 'all')) && <button type="button" className="btn quiet" onClick={() => { setRequired(new Set()); setFilters(initialFilters()); setPassOnly(false); setComparison('all'); tip.hide() }}>Reset filters</button>}
          {file && <span className="control">{m.toLocaleString()} of {availableCount.toLocaleString()} shown · {passCount.toLocaleString()} saved guardrail passers{rule !== 'guardrails' && ` · ${matchCount.toLocaleString()} highlight matches`}{availableCount < f.n && ` · ${(f.n - availableCount).toLocaleString()} archived configurations hidden`}</span>}
          {file && required.size > 0 && <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', width: '100%' }} aria-label="Selected choice filters">
            {[...required].map(c => <button key={c} type="button" className="btn quiet" aria-label={`Remove choice filter: ${file.choices[c]}`} onClick={() => toggleChoice(c)}>{file.choices[c]} ×</button>)}
          </div>}
        </>
      }
    >
      <div ref={wrapRef} className="chart-wrap">
        {data === null ? (
          <Empty>Loading the specification curve…</Empty>
        ) : !file ? (
          <Empty>{method === 'ols' && data && data !== 'missing' ? 'OLS coefficients are not available in this saved specification export.' : 'No specification export available.'}</Empty>
        ) : m === 0 ? (
          <Empty>No saved configuration matches these filters{passOnly ? ' and the highlight criteria' : ''}.</Empty>
        ) : (
          <SpecCurvePlots file={f} shown={shown} width={width} id={id + '-' + method} layout={layout} activeSelection={selection}
            highlighted={highlighted} showDots={showDots} highlightLabel={highlightDescription} band={BAND}
            required={required} toggleChoice={toggleChoice} selectedIndices={selectedIndices}
            showTip={tip.show} hideTip={tip.hide} select={onSelect ? i => onSelect(cfgOf(i)) : undefined}
            describe={i => [describe(i),
              `${FIT_STYLE[method].label} intercept ${fmt(f.intercept[i], 2)} µg/m³ · slope ${fmt(f.slope[i], 2)}`,
              `IMPROVE cross-validation R² ${fmt(f.heldout[i], 2)} · ${f.ok[i] === '1' ? 'passes every saved guardrail' : 'fails a saved guardrail'}`,
              `${highlighted[i] ? 'Matches' : 'Does not match'} ${RULE_LABELS[rule]}`, ...(onSelect ? [selectHint] : [])]} />
        )}
        {file && m > 0 && layout !== 'overlay' && (
          <div className="legend">
            <ColorLegend scale={d3.scaleSequential((t: number) => HEAT(t)).domain([0, 100])} width={180}
              label="strip colour: % of the specifications at that position that use the choice" format={(v) => `${v.toFixed(0)}%`} />
          </div>
        )}
        {layout === 'overlay' && <p className="chart-note">Each selection is sorted independently on a percentile axis. Curves compare distributions, not matched configurations. Choice strips are shown in the separate and combined ordered views.</p>}
        {rule !== 'guardrails' && <p className="chart-note">Custom highlights use the displayed fit's intercept, slope and IMPROVE cross-validation R². They do not apply the saved guardrails' out-of-domain or negative-prediction limits; saved guardrail results stay unchanged.</p>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}
