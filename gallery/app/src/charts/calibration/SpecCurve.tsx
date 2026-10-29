import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react'
import { loadJson } from '@/lib/data'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented, Select } from '@/components/ChartFrame'
import { AxisChoices } from '@/components/AxisChoices'
import { useColourScheme } from '@/lib/colourSchemes'
import { PencilIcon } from '@/components/Icons'
import { useChartOptions, type OptionChoice } from '@/lib/chartOptions'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { ColorLegend } from '@/components/ColorLegend'
import type { Config } from './common'
import { useEstimator, FIT_STYLE } from '@/lib/regressionPrefs'
import { CALIBRATION_LOT as LOT_LABELS, COHORT, CV_PROTOCOL, PREPROCESSING, PREPROCESSING_SHORT, SELECTION_SPACE, label } from '@/lib/labels'
import { choiceFilterDimensions, choiceFilterSets, matchesChoiceFilters, mergeSpecFiles } from '@/lib/specification'
import { TEMPLATES, decodeGuardrail, describeGuardrail, encodeGuardrail, evaluateGuardrail, guardFields, guardRows, templatesFor, type Guardrail } from '@/lib/guardrails'
import { useMac } from '@/lib/mac'
// the editor (react-querybuilder) loads only when it is opened
const GuardrailEditor = lazy(() => import('@/components/GuardrailEditor').then((m) => ({ default: m.GuardrailEditor })))
import { useSearchParam } from '@/lib/url'
import { SpecCurvePlots, type SpecLayout } from './SpecCurvePlots'
import { ChartAside } from '@/lib/singleView'

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
  /** extra readouts (export_spec_curve.py EXTRA_FIELDS), in the export's row order */
  extra?: Record<string, (number | null)[]>
  co: number[]
  sp: number[]
  sel: number[]
  mo: number[]
  lot: number[]
}

type FilterKey = 'co' | 'sp' | 'sel' | 'mo' | 'lot' | 'cut' | 'k'
const ALL = '*'
const NO_CUTOFF = 'complete'
const initialFilters = (): Record<FilterKey, string> => ({ co: ALL, sp: ALL, sel: ALL, mo: ALL, lot: ALL, cut: ALL, k: ALL })
/** what the curve sorts and plots: the test set's Deming intercept (default), its slope, or IMPROVE cross-validation R² */
export type SpecMetric = 'intercept' | 'slope' | 'heldout'
export interface SpecMetricInfo { axis: string; rank: string; ref: number; band: [number, number] }
const METRICS: Record<SpecMetric, { name: string } & SpecMetricInfo> = {
  intercept: { name: 'Deming intercept', axis: 'Deming intercept, MAC 10 (µg/m³)', rank: 'Configuration rank by intercept', ref: 0, band: [-0.5, 0.5] },
  slope: { name: 'Deming slope', axis: 'Deming slope, MAC 10 (1 = agreement)', rank: 'Configuration rank by slope', ref: 1, band: [0.7, 1.3] },
  heldout: { name: 'IMPROVE cross-validation R²', axis: 'IMPROVE cross-validation R² (site-grouped only)', rank: 'Configuration rank by IMPROVE cross-validation R²', ref: 0.85, band: [0.85, 1] },
}
/** the red dots: the saved guardrails (the export's own pass bit), a template rule, the reader's rule, or none */
const RULE_LABELS: Record<string, string> = { ...Object.fromEntries(TEMPLATES.map((t) => [t.key, t.label])), rule: 'Custom guardrails' }
type DotRule = 'none' | 'rule' | (typeof TEMPLATES)[number]['key']
/** the baselines a search compares: a preset pair, every baseline, or one baseline ('only:<key>') */
type Comparison = 'full' | 'standard' | 'all'

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
  selected, onSelect, src = 'spec_curve.json', id = 'spec-curve', title, subtitle, provenance, controls, defaultLayout = 'combined', selectHint = 'click to load into the k-sweep below', excludePreprocessing = [], defaultComparison = 'full', baselines: baselinesProp,
}: {
  selected?: Config
  onSelect?: (c: Config) => void
  /** another export in the same schema (e.g. meeting/spec_curve_20260917.json), or several, merged into one curve (lib/specification mergeSpecFiles) */
  src?: string | readonly { src: string; name: string }[]
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
  /**
   * The baselines shown, set by the page (its own menu) instead of the chart's: 'full' (spline +
   * full-range VIBES), 'all', or 'only:<key>' (one baseline; the model spectra where selection and
   * model are not paired). The chart then hides its own Baselines control and Model spectra filter.
   */
  baselines?: string
}) {
  const method = useEstimator()
  // strip colour: the share of the specifications at a position that use a choice, white (none) to
  // the palette's darkest (all); the palette is picked from the strip ramp (default Reds)
  const stripColours = useColourScheme(`${id}-strips`, 'sequential', 'Reds')
  const heat = useMemo(() => d3.interpolateRgbBasis(stripColours.stops), [stripColours.stops.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef, { large: true })
  const [data, setData] = useState<SpecCurveFile | null | 'missing'>(null)
  const [required, setRequired] = useState<Set<number>>(new Set())
  const [filters, setFilters] = useState(initialFilters)
  // the reader's own guardrail, written in the editor; kept in the URL so a link reopens it
  const [ruleToken, setRuleToken] = useSearchParam<string>(`${id}.rule`, '')
  const userRule = useMemo(() => decodeGuardrail(ruleToken), [ruleToken])
  const [editing, setEditing] = useState(false)
  // the rule being written in the editor, drawn on the chart until it is applied or cancelled
  const [draft, setDraft] = useState<Guardrail | null>(null)
  const excludedKey = excludePreprocessing.join('|')
  const loaded = data && data !== 'missing' ? data : null
  const selections = (loaded?.categories.sel ?? []).filter(s => !excludePreprocessing.includes(s))
  const hasVibes = selections.some(s => s.startsWith('vibes'))
  // the 17 Sep searches select and model on the same baseline, so one control picks both
  const paired = !!loaded && loaded.sel.every((s, i) => loaded.categories.sel[s] === loaded.categories.sp[loaded.sp[i]])
  const fullVibes = selections.includes('vibes-full-r70') ? 'vibes-full-r70' : 'vibes-full'

  const o = useChartOptions(id, [
    // the y axis's readout, picked from its title; interleaved cross-validation holds out no IMPROVE
    // site, so with the protocol filtered to it the held-out R² is not offered (it would draw nothing)
    { key: 'metric', axis: 'y', label: 'Sort and plot by', default: 'intercept', persist: true,
      choices: (Object.keys(METRICS) as SpecMetric[]).map(k => ({ value: k, label: METRICS[k].name, when: k === 'heldout' ? (c) => !c.interleavedOnly : undefined })) },
    { key: 'view', label: 'View', default: defaultLayout, persist: true,
      title: 'One curve over every configuration shown; one panel per baseline (each sorted on its own, shared vertical scale); or every baseline\'s curve overlaid on a percentile axis.',
      choices: [{ value: 'combined', label: 'One curve' }, { value: 'stacked', label: 'Per baseline' }, { value: 'overlay', label: 'Overlay' }] },
    { key: 'baselines', label: 'Baselines', style: 'select', default: defaultComparison, persist: true, when: (c) => !!c.hasVibes && !baselinesProp,
      choices: [
        { value: 'full', label: 'Spline + full-range VIBES', when: () => selections.includes(fullVibes) },
        { value: 'standard', label: 'Spline + archived VIBES (4000–1425)', when: () => selections.includes('vibes') },
        // one baseline alone, where selection and model share it (otherwise the spectra filters below do this)
        ...selections.map((s): OptionChoice => ({ value: `only:${s}`, label: label(PREPROCESSING, s), when: (c) => !!c.paired })),
        { value: 'all', label: excludePreprocessing.includes('vibes') ? 'All active baselines' : 'All active and archived baselines' },
      ] },
    // "Custom…" opens the guardrail editor; once chosen, a pencil beside the menu reopens it
    { key: 'dots', label: 'Red dots', style: 'select', default: 'guardrails', persist: true,
      title: 'Which configurations are drawn as red dots on the curve; Custom writes your own rule',
      choices: [{ value: 'none', label: 'None' }, ...TEMPLATES.map(t => ({ value: t.key, label: t.label })), { value: 'rule', label: (c) => (c.hasRule ? 'Custom' : 'Custom…') }],
      onChange: (v: string) => { if (v === 'rule' && !editing) setEditing(true) } },
    { key: 'edit', when: (c) => c.dots === 'rule', node: () => <button type="button" className="btn quiet icon-btn" onClick={() => setEditing(true)}
      aria-label="Edit the custom guardrails" title="Edit the custom guardrails: conditions on any readout, AND / OR groups, calculated fields"><PencilIcon /></button> },
    { key: 'only', label: 'Red dots only', toggle: true, default: false, hidden: false, when: (c) => c.dots !== 'none',
      title: 'Keep only the configurations drawn as red dots' },
  ], { interleavedOnly: filters.mo.startsWith('app'), hasVibes, paired, hasRule: !!userRule })
  const metric = o.values.metric as SpecMetric
  const layout = o.values.view as SpecLayout
  const comparison = (baselinesProp ?? o.values.baselines) as Comparison | `only:${string}`
  const dots = o.values.dots as DotRule
  const passOnly = o.values.only as boolean

  useEffect(() => {
    const excluded = excludedKey.split('|')
    setFilters(current => excluded.includes(current.sp) || excluded.includes(current.sel)
      ? { ...current, sp: excluded.includes(current.sp) ? ALL : current.sp, sel: excluded.includes(current.sel) ? ALL : current.sel }
      : current)
  }, [excludedKey])

  const srcKey = typeof src === 'string' ? src : src.map(p => p.src).join('|')
  useEffect(() => {
    let cancelled = false
    setData(null)
    setRequired(new Set())
    setFilters(initialFilters())
    const parts = typeof src === 'string' ? [{ src, name: src }] : src
    Promise.all(parts.map(p => loadJson<SpecCurveFile>(p.src).then(file => ({ file, name: p.name }))))
      .then((files) => !cancelled && setData(mergeSpecFiles(files)))
      .catch(() => !cancelled && setData('missing'))
    return () => { cancelled = true }
  }, [srcKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // the search at the MAC set under Filters (lib/mac): the export's slopes are against Fabs/10, so each is
  // re-expressed (× MAC/10; intercepts, R² and held-out R² do not move), and the saved pass bit, which
  // judged the slope at MAC 10, is rebuilt from the saved guardrails' own rule where the export carries
  // its inputs (lib/guardrails: that rule reproduces the bit exactly). RMSE against HIPS stays at MAC 10.
  const gallery = useMac()
  const atMac = useMemo(() => {
    if (!data || data === 'missing' || gallery.mac === gallery.exportMac) return { spec: data, okRebuilt: null as boolean | null }
    const k = gallery.mac / gallery.exportMac
    const sc = (a: (number | null)[] | undefined) => a?.map((v) => (v == null ? v : v * k))
    const extra = data.extra && Object.fromEntries(Object.entries(data.extra).map(([key, col]) => [key, /Slope$/.test(key) ? sc(col)! : col]))
    const spec: SpecCurveFile = { ...data, slope: sc(data.slope)!, olsSlope: sc(data.olsSlope), extra }
    if (!templatesFor(spec).some((t) => t.key === 'guardrails')) return { spec, okRebuilt: false }
    const names = guardFields(spec).filter((x) => x.kind === 'number').map((x) => x.name)
    const pass = evaluateGuardrail(TEMPLATES[0].rule, guardRows(spec), names).pass
    return { spec: { ...spec, ok: pass.map((p) => (p ? '1' : '0')).join(''), n_pass: pass.filter(Boolean).length }, okRebuilt: true }
  }, [data, gallery.mac, gallery.exportMac])

  const macNote = atMac.okRebuilt === null ? '' : ` Slopes are re-expressed at ${gallery.label} (Filters → units; slope × MAC/10, intercepts unchanged)${atMac.okRebuilt
    ? ', and the saved guardrails are re-judged on them; RMSE against HIPS stays at MAC 10.'
    : '; this export lacks the inputs to re-judge the saved guardrails, so their pass bit still judges the slope at MAC 10.'}`

  const file = useMemo(() => {
    const data = atMac.spec
    if (!data || data === 'missing') return null
    if (method === 'ols' && (!data.olsIntercept || !data.olsSlope)) return null
    const intercept = method === 'ols' ? data.olsIntercept! : data.intercept
    const slope = method === 'ols' ? data.olsSlope! : data.slope
    const value = { intercept, slope, heldout: data.heldout }[metric]
    if (method !== 'ols' && metric === 'intercept') return { ...data, value: data.intercept }
    // re-sort by the chosen measure; configurations without it (no CV split, no slope) drop out
    const order = value.map((_, i) => i).filter(i => value[i] !== null && intercept[i] !== null && slope[i] !== null)
      .sort((a, b) => value[a]! - value[b]!)
    const pick = <T,>(a: T[]) => order.map(i => a[i])
    const bits = (s: string) => order.map(i => s[i]).join('')
    return { ...data, n: order.length, n_pass: order.filter(i => data.ok[i] === '1').length, value: pick(value) as number[],
      intercept: pick(intercept) as number[], slope: pick(slope), heldout: pick(data.heldout), cut: pick(data.cut), k: pick(data.k),
      co: pick(data.co), sp: pick(data.sp), sel: pick(data.sel), mo: pick(data.mo), lot: pick(data.lot), masks: data.masks.map(bits), ok: bits(data.ok),
      extra: data.extra && Object.fromEntries(Object.entries(data.extra).map(([key, col]) => [key, pick(col)])) }
  }, [atMac, method, metric])

  // the rule behind the red dots: the saved guardrails are the export's own pass bit; the
  // templates and the reader's rule run on the displayed fit (lib/guardrails)
  const rows = useMemo(() => file ? guardRows(file) : [], [file])
  const previewing = editing && !!draft
  const showDots = dots !== 'none' || previewing
  // the red dots before "Custom…" was picked: the editor starts from them, and Cancel returns to them
  // when no custom rule has been applied yet
  const lastPreset = useRef<DotRule>('guardrails')
  const applied = useRef(false)
  if (dots !== 'rule') lastPreset.current = dots
  const activeRule: Guardrail | null = previewing ? draft : dots === 'rule' ? userRule : TEMPLATES.find(t => t.key === dots)?.rule ?? null
  const evaluated = useMemo(() => {
    if (!file || (dots === 'none' && !previewing)) return { pass: [] as boolean[], error: null as string | null }
    if ((dots === 'guardrails' && !previewing) || !activeRule) return { pass: Array.from(file.ok, b => b === '1'), error: null }
    return evaluateGuardrail(activeRule, rows, guardFields(file).filter(x => x.kind === 'number').map(x => x.name))
  }, [file, rows, dots, activeRule, previewing])
  const highlighted = evaluated.pass

  const choiceDimensions = useMemo(() => file ? choiceFilterDimensions(file) : [], [file])
  const choiceGroups = useMemo(() => choiceFilterSets(required, choiceDimensions), [required, choiceDimensions])

  // the specifications left after the choice and guardrail filters, still in intercept order
  // the configurations the chart's choices and filters keep; `shown` then drops the non-red ones
  // when "Red dots only" is on (the guardrail editor counts over `inView`)
  const inView = useMemo(() => {
    if (!file) return [] as number[]
    const out: number[] = []
    for (let i = 0; i < file.n; i++) {
      if (excludePreprocessing.includes(file.categories.sp[file.sp[i]]) || excludePreprocessing.includes(file.categories.sel[file.sel[i]])) continue
      if (comparison.startsWith('only:')) {
        // one baseline: the paired searches' shared baseline, otherwise the model spectra
        const want = comparison.slice(5)
        if ((paired ? file.categories.sel[file.sel[i]] : file.categories.sp[file.sp[i]]) !== want) continue
      } else if (hasVibes && comparison !== 'all') {
        const sel = file.categories.sel[file.sel[i]]
        if (!['airspec', comparison === 'full' ? fullVibes : 'vibes'].includes(sel)) continue
      }
      if ((['co', 'sp', 'sel', 'mo', 'lot'] as const).some(key => filters[key] !== ALL && file.categories[key][file[key][i]] !== filters[key])) continue
      if (filters.cut !== ALL && (file.cut[i] === null ? NO_CUTOFF : String(file.cut[i])) !== filters.cut) continue
      if (filters.k !== ALL && String(file.k[i]) !== filters.k) continue
      if (matchesChoiceFilters(file.masks, i, choiceGroups)) out.push(i)
    }
    return out
  }, [file, choiceGroups, filters, comparison, excludePreprocessing, hasVibes, fullVibes, paired])
  const shown = useMemo(() => passOnly ? inView.filter(i => highlighted[i]) : inView, [inView, passOnly, highlighted])

  const setFilter = (key: FilterKey, value: string) => {
    // a spectra filter on one baseline would be emptied by a baseline pair that leaves it out
    if ((key === 'sel' || key === 'sp') && value !== ALL && hasVibes && !baselinesProp) o.set('baselines', 'all')
    setFilters(current => ({ ...current, [key]: value }))
    tip.hide()
  }
  // `anyLabel` names the no-filter choice where 'All' would read as one of the values (lots: 'Any', beside 'Both lots')
  const categorySelect = (key: 'co' | 'sp' | 'sel' | 'mo' | 'lot', name: string, labels: Record<string, string>, anyLabel = 'All') => file && (
    <Select label={name} value={filters[key]} options={[ALL, ...file.categories[key].filter(value => !['sp', 'sel'].includes(key) || !excludePreprocessing.includes(value))]} onChange={value => setFilter(key, value)}
      optionLabel={value => value === ALL ? anyLabel : label(labels, value)} />
  )
  const numericOptions = (key: 'cut' | 'k') => file ? [...new Set(file[key].filter((v): v is number => v !== null))].sort((a, b) => a - b).map(String) : []

  const selectedIndices = useMemo(() => new Set(shown.filter(i => selected && file
    && file.categories.co[file.co[i]] === selected.co && file.cut[i] === selected.cut
    && file.categories.sel[file.sel[i]] === (selected.sel ?? 'raw') && file.categories.sp[file.sp[i]] === selected.sp)), [file, shown, selected])
  const m = shown.length
  const passCount = file ? shown.filter(i => file.ok[i] === '1').length : 0
  const matchCount = shown.filter(i => highlighted[i]).length
  const availableCount = file ? file.sp.filter((sp, i) => !excludePreprocessing.includes(file.categories.sp[sp]) && !excludePreprocessing.includes(file.categories.sel[file.sel[i]])).length : 0
  const highlightDescription = (dots === 'guardrails' && !previewing) || !activeRule || !file ? 'saved guardrails'
    : `${previewing ? 'preview, ' : ''}${describeGuardrail(activeRule, guardFields(file, activeRule.formulas))}`

  // helpers below run only once the file has loaded (the body is empty until then)
  const f = file as SpecCurveFile & { value: number[] }
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
    return `${label(COHORT, c.co)}${c.cut !== null ? ' · cutoff ' + c.cut : ''} · ${label(SELECTION_SPACE, c.sel)} × ${label(PREPROCESSING, c.sp)} · ${label(CV_PROTOCOL, mode)} · ${f.k[i] ?? '?'} PLS factors${lot !== 'all' ? ' · ' + label(LOT_LABELS, lot).toLowerCase() : ''}`
  }

  const toggleChoice = (c: number) => {
    setRequired((s) => { const n = new Set(s); if (n.has(c)) n.delete(c); else n.add(c); return n })
    tip.hide()
  }

  // what the chart draws, shared by the chart and the guardrail editor's live preview of it
  const curve = () => ({
    heat,
    file: f, metric: { ...METRICS[metric], axis: METRICS[metric].axis.replace('MAC 10', gallery.label) }, shown, layout, highlighted, showDots, highlightLabel: highlightDescription,
    // panels per baseline: the shared one where selection and model are paired, else the model spectra
    panelBy: (paired ? 'sel' : 'sp') as 'sel' | 'sp',
    required, toggleChoice, selectedIndices,
    describe: (i: number) => [describe(i),
      `${FIT_STYLE[method].label} intercept ${fmt(f.intercept[i], 2)} µg/m³ · slope ${fmt(f.slope[i], 2)}`,
      `IMPROVE cross-validation R² ${fmt(f.heldout[i], 2)} · ${f.ok[i] === '1' ? 'passes every saved guardrail' : 'fails a saved guardrail'}`,
      ...(showDots ? [`${highlighted[i] ? 'Matches' : 'Does not match'} ${previewing ? 'the rule being edited' : RULE_LABELS[dots] ?? dots}`] : []), ...(onSelect ? [selectHint] : [])],
  })

  // anything narrowing the curve beyond the chart's defaults
  const filtered = required.size > 0 || Object.values(filters).some(value => value !== ALL) || passOnly || (hasVibes && !baselinesProp && comparison !== defaultComparison)
  const resetFilters = () => { setRequired(new Set()); setFilters(initialFilters()); o.set('only', false); if (hasVibes && !baselinesProp) o.set('baselines', defaultComparison); tip.hide() }

  return (
    <ChartFrame
      id={id}
      title={title ?? 'Specification curve — every Addis configuration, sorted by the intercept it produces'}
      subtitle={(subtitle ? subtitle(file) : `All ${file ? f.n.toLocaleString() + ' ' : ''}scored Addis specifications (every cohort, cutoff, baseline, protocol and k), sorted by Deming intercept at ${gallery.label}, with the analytic choices underneath: a darker strip means more of the specifications at that position used the choice. Click choices to filter: alternatives within the same setting are combined, while different settings must all match. Red points use the selected highlight rule; the default is every saved guardrail${file ? ': ' + f.guardrails : ''}.`) + macNote}
      provenance={provenance ?? 'calibration_explorer batch results via research/ftir_ec_phase3/scripts/spec_curve.py · figure twin: pathways/01_specification_curve.png'}
      controls={
        <>
          {controls}
          {o.controls()}
          {file && <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', width: '100%' }}>
            <strong className="control">Filters</strong>
            {categorySelect('co', 'Cohort', COHORT)}
            <Select label="Cohort size" value={filters.cut} options={[ALL, ...(file.cut.includes(null) ? [NO_CUTOFF] : []), ...numericOptions('cut')]}
              onChange={value => setFilter('cut', value)} optionLabel={value => value === ALL ? 'All' : value === NO_CUTOFF ? 'Complete cohort (no cutoff)' : `${value} filters`}
              title="Requested number of filters in the selected calibration cohort; complete cohorts have no cutoff." />
            {!paired && categorySelect('sel', 'Selection spectra', SELECTION_SPACE)}
            {!paired && !baselinesProp && categorySelect('sp', 'Model spectra', PREPROCESSING)}
            {categorySelect('mo', 'Cross-validation', CV_PROTOCOL)}
            <Select label="PLS factors" value={filters.k} options={[ALL, ...numericOptions('k')]} onChange={value => setFilter('k', value)} optionLabel={value => value === ALL ? 'All' : value} />
            {file.categories.lot.length > 1 && categorySelect('lot', 'Calibration lot', LOT_LABELS, 'Any')}
            {filtered && <button type="button" className="btn quiet" onClick={resetFilters}>Reset filters</button>}
          </div>}
          {evaluated.error && <span className="control" style={{ color: 'var(--danger)' }}>Red-dot rule: {evaluated.error}</span>}
          {file && <span className="control">{m.toLocaleString()} of {availableCount.toLocaleString()} shown · {passCount.toLocaleString()} saved guardrail passers{showDots && (dots !== 'guardrails' || previewing) && ` · ${matchCount.toLocaleString()} red dots${previewing ? ' (preview)' : ''}`}{availableCount < f.n && ` · ${(f.n - availableCount).toLocaleString()} archived configurations hidden`}</span>}
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
          <Empty>No saved configuration matches these filters{passOnly ? ' and the red-dot rule' : ''}.</Empty>
        ) : (
          <AxisChoices {...o.axisChoices()} noun="measures"><SpecCurvePlots {...curve()} width={width} id={id + '-' + method}
            showTip={tip.show} hideTip={tip.hide} select={onSelect ? i => onSelect(cfgOf(i)) : undefined} /></AxisChoices>
        )}
        {file && m > 0 && layout !== 'overlay' && (
          <div className="legend">
            <ColorLegend scale={d3.scaleSequential(heat).domain([0, 100])} width={180} scheme={stripColours.scheme}
              label="strip colour: % of the specifications at that position that use the choice" format={(v) => `${v.toFixed(0)}%`} />
          </div>
        )}
        {layout === 'overlay' && <ChartAside><p className="chart-note">Each baseline is sorted independently on a percentile axis. Curves compare distributions, not matched configurations. Choice strips are shown in the One curve and Per baseline views.</p></ChartAside>}
        {file && data && data !== 'missing' && data.n > file.n && <ChartAside><p className="chart-note">{(data.n - file.n).toLocaleString()} configurations have no {METRICS[metric].name}{metric === 'heldout' ? ' (interleaved cross-validation holds no whole IMPROVE site out)' : ''} and are left out of this view.</p></ChartAside>}
        {showDots && <ChartAside><p className="chart-note">Red dots: {highlightDescription}.{layout === 'stacked' ? ' Shared vertical scale; each panel is sorted independently.' : ''} Rules run on the displayed fit's intercept and slope; the saved guardrail results (field "Passes the saved guardrails") stay unchanged.</p></ChartAside>}
        {editing && file && <Suspense fallback={null}><GuardrailEditor title={title ?? 'Specification curve'} src={file} scope={inView}
          preview={inView.length > 0 ? <PreviewCurve {...curve()} shown={inView} id={`${id}-${method}-preview`} /> : <Empty>No configuration matches the chart's filters.</Empty>}
          initial={userRule && dots === 'rule' ? userRule : TEMPLATES.find(t => t.key === lastPreset.current)?.rule ?? TEMPLATES[0].rule}
          onPreview={setDraft} onApply={(r) => { applied.current = true; setRuleToken(encodeGuardrail(r)); o.set('dots', 'rule') }}
          onClose={() => {
            setEditing(false); setDraft(null)
            // cancelled with no custom rule to fall back on: back to the red dots chosen before
            if (!applied.current && !userRule) o.set('dots', lastPreset.current)
            applied.current = false
          }} /></Suspense>}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/**
 * The curve alone (no choice strips), sized to the guardrail editor, with its own tooltip.
 * "Show" picks both baselines or one of them, for the preview only: the rule still counts
 * every configuration the chart's filters keep.
 */
function PreviewCurve(props: Omit<React.ComponentProps<typeof SpecCurvePlots>, 'width' | 'showTip' | 'hideTip' | 'curveOnly'>) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref, { large: true })
  const { file, shown } = props
  const by = props.panelBy ?? 'sel'
  const present = useMemo(() => file.categories[by].filter((_, k) => shown.some(i => file[by][i] === k)), [file, shown, by])
  const [only, setOnly] = useState('both')
  const pick = present.includes(only) ? only : 'both'
  const curve = pick === 'both' ? shown : shown.filter(i => file.categories[by][file[by][i]] === pick)
  return <div ref={ref} className="chart-wrap" style={{ position: 'relative' }}>
    {present.length > 1 && <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 2 }}>
      {/* buttons up to five choices, a dropdown past that (the gallery's rule) */}
      {(() => {
        const showLabel = (v: string) => v === 'both' ? (present.length === 2 ? 'Both' : 'All') : PREPROCESSING_SHORT[v] ?? label(PREPROCESSING, v)
        return present.length + 1 > 5
          ? <Select label="Show" value={pick} options={['both', ...present]} onChange={setOnly} optionLabel={showLabel} />
          : <Segmented label="Show" value={pick} options={['both', ...present]} onChange={setOnly} optionLabel={showLabel} />
      })()}
    </div>}
    {width > 0 && <SpecCurvePlots {...props} shown={curve} width={width} showTip={tip.show} hideTip={tip.hide} curveOnly={{ height: pick === 'both' && present.length > 3 ? 115 : 170 }} />}
    {tip.node}
  </div>
}
