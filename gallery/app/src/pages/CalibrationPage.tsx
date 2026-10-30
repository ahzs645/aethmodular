import { EstimatorViews, ESTIMATORS, projectGrid, projectRun, useRegressions } from '@/lib/regressionPrefs'
import { useMemo, useState } from 'react'
import { ChartFrame, Select } from '@/components/ChartFrame'
import { Combobox } from '@/components/Combobox'
import { PageToc } from '@/components/PageToc'
import { SpecCurve } from '@/charts/calibration/SpecCurve'
import { CutoffSweep } from '@/charts/calibration/CutoffSweep'
import { KSweep } from '@/charts/calibration/KSweep'
import { SlopeInterceptTrap } from '@/charts/calibration/SlopeInterceptTrap'
import { CrossSiteHeatmap } from '@/charts/calibration/CrossSiteHeatmap'
import { SpectraDumbbell } from '@/charts/calibration/SpectraDumbbell'
import { RunCrossplot } from '@/charts/calibration/RunCrossplot'
import { RunSeries } from '@/charts/calibration/RunSeries'
import { CVCurve } from '@/charts/calibration/CVCurve'
import { SplitCheck } from '@/charts/calibration/SplitCheck'
import { CohortComposition } from '@/charts/calibration/CohortComposition'
import { SelectionRanking } from '@/charts/calibration/SelectionRanking'
import { OverlapMatrix } from '@/charts/calibration/OverlapMatrix'
import { SpectraOverlay, type SpectrumSeries } from '@/charts/calibration/SpectraOverlay'
import { AnalogLab } from '@/charts/calibration/AnalogLab'
import { HipsYork } from '@/charts/calibration/HipsYork'
import { CrossSiteTransfer } from '@/charts/calibration/CrossSiteTransfer'
import { calibAtMac, cohortColor, configLabel, presetName, type Config } from '@/charts/calibration/common'
import { runCalendar, seasonalRun } from '@/charts/calibration/seasonView'
import { useMac } from '@/lib/mac'
import { useCountedFilter } from '@/lib/markedRows'
import { INK } from '@/lib/theme'
import { usePrefs } from '@/lib/prefs'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, CalibFile, CalibRunsFile, MetaFile } from '@/lib/types'

/**
 * The phase-3 FTIR-EC calibration work, in two halves.
 *
 * "The grid" reads the explorer's batch results: what every configuration
 * did, at the rule k, at every site. "One configuration, every filter" reads
 * the explorer's own per-run readouts for a short list of named
 * configurations: the crossplot, residuals, dated series, blind-half check
 * and CV curve the Calibrate and Series tabs show. "Cross-site transfer"
 * applies one calibration to every SPARTAN site side by side. Below that, the cohort
 * diagnostics (composition, ranking, overlap, spectra, analog lab, HIPS).
 *
 * This is the reading view; for a new configuration, a new batch, or a k
 * you cannot find here, run `python calibration_explorer/app.py` (port 5058).
 */
/** both searches, merged into one curve for 'All baselines' (named for the Search strip rows) */
const SPEC_BOTH = [
  { src: 'spec_curve.json', name: 'spline, raw and 2nd derivative' },
  { src: 'meeting-r70/spec_curve_20260917.json', name: 'spline and VIBES' },
] as const

/** The specification curve's baselines menu: grouped by the search they come from, named by spectra. */
const SPEC_VIEWS = [
  { key: 'both:all', label: 'Spline, VIBES, raw and 2nd derivative', group: 'All baselines' },
  { key: 'vibes:full', label: 'Spline + VIBES', group: 'Spline and VIBES' },
  { key: 'vibes:only:airspec', label: 'Spline only', group: 'Spline and VIBES' },
  { key: 'vibes:only:vibes-full-r70', label: 'VIBES only', group: 'Spline and VIBES' },
  { key: 'vibes:only:vibes-full-cut-r70', label: 'VIBES, predicting on 4000–1425 only', group: 'Spline and VIBES' },
  { key: 'vibes:all', label: 'All three', group: 'Spline and VIBES' },
  { key: 'orig:all', label: 'Spline + raw + 2nd derivative', group: 'Spline, raw and 2nd derivative' },
  { key: 'orig:only:raw', label: 'Raw only', group: 'Spline, raw and 2nd derivative' },
  { key: 'orig:only:airspec', label: 'Spline only', group: 'Spline, raw and 2nd derivative' },
  { key: 'orig:only:deriv2', label: '2nd derivative only', group: 'Spline, raw and 2nd derivative' },
]

export function CalibrationPage({ calib, runs, meta, knownIds, seasonSel = null, pmfSel = null, pmfLabelOfDate, galleryRows, sourceRows, galleryMeta }: {
  calib: CalibFile | null
  runs: CalibRunsFile | null
  meta: MetaFile
  /** the gallery's rows and meta under the reader's view, for the crossplot's explorer panel */
  galleryRows?: FilterRow[]
  /** every filter's row before the view's options (a filter's own values, e.g. its deployed SPARTAN EC) */
  sourceRows?: FilterRow[]
  galleryMeta?: MetaFile
  knownIds: Set<string>
  seasonSel?: Set<string> | null
  /** PMF-source groups kept in the Filters panel (Addis dates only), and a filter's group from its date */
  pmfSel?: Set<string> | null
  pmfLabelOfDate?: (date: string | null | undefined) => string
}) {
  const { methods } = useRegressions()
  // the grid at the MAC set under Filters (lib/mac): slopes re-expressed before the estimator projection,
  // so the slope box and the guardrails judge the slope the reader sees
  const gallery = useMac()
  const atMac = useMemo(() => (calib ? calibAtMac(calib, gallery.mac) : null), [calib, gallery.mac])
  const calibrationViews = useMemo(() => Object.fromEntries(ESTIMATORS.map(method => [method, atMac ? { ...atMac, grid: atMac.grid.map(r => projectGrid(r, method)), sweeps: atMac.sweeps.map(r => projectGrid(r, method)) } : null])) as Record<typeof ESTIMATORS[number], CalibFile | null>, [atMac])
  const [selected, setSelected] = useState<Config>({ co: 'ocec', cut: 800, sel: 'raw', sp: 'airspec' })
  const [presetKey, setPresetKey] = useState<string>('ocec800_airspec')
  const [targetId, setTargetId] = useState<string>('addis')
  // the specification curve's baselines: one menu, named by the spectra compared. Spline and VIBES
  // come from the spline-vs-VIBES search (completed rank-70 VIBES), raw and 2nd derivative from the
  // original search; the value is '<search>:<baselines>' for SpecCurve's `baselines`
  const [specView, setSpecView] = useSearchParam<string>('cbase', 'vibes:full', SPEC_VIEWS.map(v => v.key))
  const specOriginal = specView.startsWith('orig:')
  const specBoth = specView.startsWith('both:')
  const { explain } = usePrefs()

  const presets = runs?.presets ?? []
  const preset = presets.find((p) => p.key === presetKey) ?? presets[0]
  const targetIds = useMemo(() => [...new Set((runs?.runs ?? []).filter((r) => r.preset === preset?.key).map((r) => r.target))], [runs, preset])
  const target = targetIds.includes(targetId) ? targetId : targetIds[0]
  const run = runs?.runs.find((r) => r.preset === preset?.key && r.target === target) ?? null

  // Season filter for the per-filter section comes from the Filters panel. The
  // panel carries the Addis calendar (Addis-only tab); an Ethiopian target uses it
  // as is, and another site under the per-site default uses its own calendar, where
  // the Addis season names do not apply and the filter is left off.
  const runSite = target ? runs?.targets[target]?.site : undefined
  const ethiopian = runSite === 'Addis Ababa' || runSite === 'Bishoftu'
  const isLocal = meta.season_convention === 'local'
  const runSeasons = useMemo(
    () => (ethiopian || !isLocal ? meta.seasons : runCalendar(runSite, meta, 'Site local calendar')),
    [ethiopian, isLocal, meta, runSite]
  )
  const runKeep = useMemo(() => {
    if (!seasonSel) return null
    const kept = new Set(runSeasons.map((x) => x.name).filter((n) => seasonSel.has(n)))
    return kept.size === 0 || kept.size === runSeasons.length ? null : kept
  }, [seasonSel, runSeasons])
  // the PMF solution is Addis's, joined by date, so it filters only an Addis target
  const runPmf = runSite === 'Addis Ababa' && pmfSel && pmfLabelOfDate ? { labelOf: pmfLabelOfDate, keep: pmfSel } : undefined
  const runView = useMemo(
    () => (run ? seasonalRun(run, runSeasons, runKeep, calib?.deming_lambda_mac10 ?? gallery.lambda10, runPmf) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [run, runSeasons, runKeep, calib, pmfSel, pmfLabelOfDate, runSite, gallery.lambda10]
  )

  // the filters the crossplot and dated series count: the season and PMF subset, then Filters → excluded's one rule
  const isCounted = useCountedFilter()
  const nRunCounted = useMemo(() => (runView ? runView.eval.id.filter((id) => isCounted(id)).length : 0), [runView, isCounted])

  // Clicking a point in the grid charts also moves the per-filter section to
  // the nearest exported preset, when one exists for that configuration.
  const selectConfig = (c: Config) => {
    setSelected(c)
    const hit = presets.find((p) => p.cohort === c.co && p.cutoff === c.cut && p.spectra === c.sp)
    if (hit) setPresetKey(hit.key)
  }

  const cohortSpectra = useMemo<Record<string, SpectrumSeries[]>>(() => {
    if (!runs) return {}
    const out: Record<string, SpectrumSeries[]> = {}
    const colours = ['eth_shaped', 'analogs', 'ocec']
    for (const [space, cs] of Object.entries(runs.cohort_spectra)) {
      out[space] = [
        ...cs.series.map((s, i) => ({ label: s.label, color: cohortColor(colours[i] ?? 'pool'), wn: cs.wn, median: s.median, q25: s.q25, q75: s.q75 })),
        { label: cs.reference_label, color: INK.text, wn: cs.wn, median: cs.reference.median, q25: cs.reference.q25, q75: cs.reference.q75, dash: '6 3' },
      ]
    }
    return out
  }, [runs])
  const siteSpectra = useMemo<Record<string, SpectrumSeries[]>>(() => {
    if (!runs) return {}
    const out: Record<string, SpectrumSeries[]> = {}
    for (const [space, ss] of Object.entries(runs.site_spectra)) {
      out[space] = ss.series.filter((s) => s.name in runs.targets).map((s) => ({ label: `${runs.targets[s.name]?.site ?? s.label} (n=${s.n})`, color: runs.targets[s.name]?.color ?? INK.muted, wn: s.wn, median: s.median, q25: s.q25, q75: s.q75 }))
    }
    return out
  }, [runs])

  if (!calib && !runs) {
    return (
      <ChartFrame title="Calibration" exportable={false}>
        <p className="frame-empty">
          No calibration export found. Run <code>python gallery/data/export_calibration.py</code> (batch grid) and <code>python gallery/data/export_calibration_runs.py</code> (per-filter readouts).
        </p>
      </ChartFrame>
    )
  }

  return (
    <PageToc>
      {calib && (
        <>
          <SectionBar>
            <span className="pill">The grid · {calib.grid.length.toLocaleString()} configuration × site readouts</span>
            <span style={{ color: 'var(--ink-muted)' }}>selected: <strong style={{ color: 'var(--ink-text)' }}>{configLabel(selected, calib)}</strong></span>
            {explain && <span style={{ color: 'var(--ink-muted)' }}>MAC {gallery.mac}{gallery.mac !== calib.mac_value ? ` (slopes re-expressed from the export's ${calib.mac_value}; RMSE recomputed from each evaluation set's HIPS BC mean and variance, solved from its saved fits${atMac?.rmse_unmoved ? `, unavailable for ${atMac.rmse_unmoved.toLocaleString()} rows whose set could not be solved` : ''})` : ''} · Deming λ* {calib.deming_lambda_mac10} at MAC 10{gallery.mac !== 10 ? `, ${+gallery.lambda.toFixed(3)} at MAC ${gallery.mac}` : ''} · slope box {calib.slope_box.join('–')} · IMPROVE cross-validation R² floor {calib.heldout_floor}</span>}
          </SectionBar>
          <EstimatorViews>{() => <SpecCurve
            defaultLayout="stacked"
            excludePreprocessing={['vibes']}
            src={specBoth ? SPEC_BOTH : specOriginal ? 'spec_curve.json' : 'meeting-r70/spec_curve_20260917.json'}
            baselines={specView.slice(specView.indexOf(':') + 1)}
            selected={specOriginal ? selected : undefined}
            onSelect={specOriginal ? selectConfig : undefined}
            controls={<Combobox label="Baselines" value={specView} options={SPEC_VIEWS} onChange={setSpecView}
              title="Spline and VIBES come from one search (completed rank-70 VIBES, cohort selection and model on the same baseline); raw and 2nd derivative from another, so spline appears in both" />}
            subtitle={specBoth ? f => `${f ? f.n.toLocaleString() + ' ' : ''}Addis configurations from both searches on one curve: the spline, raw and 2nd-derivative search and the spline-vs-VIBES search (completed rank-70 VIBES). The two searches used different cohort designs, so the Search rows in the strips mark where each configuration came from; spline configurations from both share the Spline panel. The k-sweep and configuration readouts below use the spline, raw and 2nd-derivative search.`
              : specOriginal ? undefined : f => `${f ? f.n.toLocaleString() + ' ' : ''}Addis configurations from the spline-vs-VIBES search (completed rank-70 VIBES): every cohort, cutoff, cross-validation protocol and number of PLS factors, with the same baseline for cohort selection and model spectra. The k-sweep and configuration readouts below use the spline, raw and 2nd-derivative search.`}
            provenance={specBoth ? 'spec_curve.json + completed rank-70 migration category_grid.jsonl, merged in the browser · shared specification guardrails'
              : specOriginal ? undefined : 'Completed rank-70 migration · category_grid.jsonl · shared specification guardrails'}
          />}</EstimatorViews>
          <EstimatorViews>{method => <CutoffSweep calib={calibrationViews[method]!} selected={selected} onSelect={selectConfig} />}</EstimatorViews>
          <EstimatorViews>{method => <KSweep calib={calibrationViews[method]!} selected={selected} onSelect={selectConfig} />}</EstimatorViews>
          <EstimatorViews>{method => <SlopeInterceptTrap calib={calibrationViews[method]!} selected={selected} onSelect={selectConfig} />}</EstimatorViews>
          <EstimatorViews>{method => <SpectraDumbbell calib={calibrationViews[method]!} selected={selected} onSelect={selectConfig} />}</EstimatorViews>
          <EstimatorViews>{method => <CrossSiteHeatmap calib={calibrationViews[method]!} selected={selected} onSelect={selectConfig} />}</EstimatorViews>
        </>
      )}

      {runs && (
        <>
          <section className="configuration-section" aria-label="Selected configuration and its diagnostics"><SectionBar>
            <span className="pill">Configuration for the charts below</span>
            <Select label="Configuration" value={preset?.key ?? ''} options={presets.map((p) => p.key)} onChange={setPresetKey} optionLabel={(k) => { const p = presets.find((q) => q.key === k); return p ? presetName(p) : k }} title={presets.map((p) => `${p.key}: ${presetName(p)}`).join('\n')} />
            <Select label="Test set" value={target ?? ''} options={targetIds} onChange={setTargetId} optionLabel={(t) => runs.targets[t]?.label ?? t} />
            <span style={{ color: 'var(--ink-muted)' }}>{preset ? presetName(preset) : null}</span>
            <span className="season-row" style={{ color: 'var(--ink-muted)' }}>
              {runKeep || runPmf
                ? <>{runKeep && <>seasons: <strong style={{ color: 'var(--ink-text)' }}>{[...runKeep].join(', ')}</strong> · </>}{runPmf && <>PMF: <strong style={{ color: 'var(--ink-text)' }}>{runPmf.keep.size === 5 || (runPmf.keep.size === 2 && runPmf.keep.has('Marine') && runPmf.keep.has('Combustion')) ? 'PMF days only (every source)' : [...runPmf.keep].join(', ')}</strong> · </>}crossplot and dated series refitted on these {nRunCounted} filters (change in Filters)</>
                : (seasonSel && !ethiopian && isLocal) || (pmfSel && runSite !== 'Addis Ababa')
                  ? <>the Filters panel's Addis seasons and PMF sources do not apply to {runSite}; showing every filter</>
                  : <>every season · pick seasons in the Filters panel to refit the crossplot and dated series</>}
            </span>
            {explain && (
              <span style={{ marginLeft: 'auto', color: 'var(--ink-muted)' }}>
                exported {runs.generated.slice(0, 10)} · other configurations: <code>python calibration_explorer/app.py</code>
              </span>
            )}
          </SectionBar>
          {run ? (
            <>
              <CVCurve run={run} runs={runs} />
              <RunCrossplot lambda10={calib?.deming_lambda_mac10} run={runView ?? run} runs={runs} meta={meta} knownIds={knownIds} galleryRows={galleryRows} sourceRows={sourceRows} galleryMeta={galleryMeta} />
              <RunSeries run={runView ?? run} runs={runs} meta={meta} knownIds={knownIds} />
              {/* the blind-half readouts are the explorer's own, on every season */}
              <EstimatorViews>{method => <SplitCheck run={projectRun(run, method)} runs={runs} />}</EstimatorViews>
            </>
          ) : (
            <ChartFrame title="No run" exportable={false}><p className="frame-empty">No exported run for this configuration at this site.</p></ChartFrame>
          )}

          </section><SectionBar>
            <span className="pill">Cross-site transfer</span>
            <span style={{ color: 'var(--ink-muted)' }}>one IMPROVE-trained calibration, applied unchanged to every SPARTAN site</span>
          </SectionBar>
          <EstimatorViews>{method => <CrossSiteTransfer runs={{ ...runs, runs: runs.runs.map(r => projectRun(r, method)) }} galleryRows={galleryRows} sourceRows={sourceRows} galleryMeta={galleryMeta} />}</EstimatorViews>

          <SectionBar>
            <span className="pill">Cohort diagnostics</span>
            {explain && <span style={{ color: 'var(--ink-muted)' }}>what the cohorts are, before anything is fitted</span>}
          </SectionBar>
          <CohortComposition info={preset ? runs.cohorts[preset.key] : undefined} label={preset ? presetName(preset) : ''} />
          <SelectionRanking runs={runs} calib={calib} />
          <OverlapMatrix rows={runs.overlap} />
          <SpectraOverlay
            id="cohort-spectra"
            title="Cohort spectra — the three selections against the Addis median"
            subtitle="Median spectrum with the interquartile band for each selection cohort at its locked cutoff, and the Addis test set (dashed). Ann's multi-cohort ask: which selection actually looks like Addis, and does that survive baseline correction?"
            provenance="calibration_explorer /api/spectra compare · Selection tab"
            bySpace={cohortSpectra}
            fixedSet="the explorer's saved median spectra of each cohort and of the Addis test set"
          />
          <SpectraOverlay
            id="site-spectra"
            title="Site spectra — every SPARTAN evaluation site"
            subtitle="Median spectrum with the interquartile band per site. The ~1600 cm⁻¹ band that distinguishes Addis is visible here against Beijing's and JPL's; it is what the offset argument rests on."
            provenance="calibration_explorer /api/site_spectra · Sites tab"
            bySpace={siteSpectra}
            fixedSet="the explorer's saved median spectra of every filter at each site"
          />
          {Object.keys(runs.analog).length > 0 && <AnalogLab bySpace={runs.analog} cutoff={calib?.default_cutoff.analogs ?? 500} />}
          {runs.hips.york && methods.includes('york') && <HipsYork runs={runs} />}
        </>
      )}
    </PageToc>
  )
}

function SectionBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="panel" style={{ padding: '12px 16px', display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', fontSize: 12.5 }}>
      {children}
    </div>
  )
}
