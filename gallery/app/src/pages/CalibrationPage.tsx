import { useMemo, useState } from 'react'
import { ChartFrame, Select } from '@/components/ChartFrame'
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
import { cohortColor, configLabel, type Config } from '@/charts/calibration/common'
import { INK } from '@/lib/theme'
import { usePrefs } from '@/lib/prefs'
import type { CalibFile, CalibRunsFile, MetaFile } from '@/lib/types'

/**
 * The phase-3 FTIR-EC calibration work, in two halves.
 *
 * "The grid" reads the explorer's batch results: what every configuration
 * did, at the rule k, at every site. "One configuration, every filter" reads
 * the explorer's own per-run readouts for a short list of named
 * configurations: the crossplot, residuals, dated series, blind-half check
 * and CV curve the Calibrate and Series tabs show. Below that, the cohort
 * diagnostics (composition, ranking, overlap, spectra, analog lab, HIPS).
 *
 * This is the reading view; for a new configuration, a new batch, or a k
 * you cannot find here, run `python calibration_explorer/app.py` (port 5058).
 */
export function CalibrationPage({ calib, runs, meta, knownIds }: { calib: CalibFile | null; runs: CalibRunsFile | null; meta: MetaFile; knownIds: Set<string> }) {
  const [selected, setSelected] = useState<Config>({ co: 'ocec', cut: 800, sel: 'raw', sp: 'airspec' })
  const [presetKey, setPresetKey] = useState<string>('ocec800_airspec')
  const [targetId, setTargetId] = useState<string>('addis')
  const { explain } = usePrefs()

  const presets = runs?.presets ?? []
  const preset = presets.find((p) => p.key === presetKey) ?? presets[0]
  const targetIds = useMemo(() => [...new Set((runs?.runs ?? []).filter((r) => r.preset === preset?.key).map((r) => r.target))], [runs, preset])
  const target = targetIds.includes(targetId) ? targetId : targetIds[0]
  const run = runs?.runs.find((r) => r.preset === preset?.key && r.target === target) ?? null

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
    <>
      {calib && (
        <>
          <SectionBar>
            <span className="pill">The grid · {calib.grid.length.toLocaleString()} configuration × site readouts</span>
            <span style={{ color: 'var(--ink-muted)' }}>selected: <strong style={{ color: 'var(--ink-text)' }}>{configLabel(selected, calib)}</strong></span>
            {explain && <span style={{ color: 'var(--ink-muted)' }}>MAC {calib.mac_value} · Deming λ* {calib.deming_lambda_mac10} · slope box {calib.slope_box.join('–')} · held-out floor {calib.heldout_floor}</span>}
          </SectionBar>
          <CutoffSweep calib={calib} selected={selected} onSelect={selectConfig} />
          <KSweep calib={calib} selected={selected} onSelect={selectConfig} />
          <SlopeInterceptTrap calib={calib} selected={selected} onSelect={selectConfig} />
          <SpectraDumbbell calib={calib} selected={selected} onSelect={selectConfig} />
          <CrossSiteHeatmap calib={calib} selected={selected} onSelect={selectConfig} />
        </>
      )}

      {runs && (
        <>
          <SectionBar>
            <span className="pill">One configuration, every filter</span>
            <Select label="configuration" value={preset?.key ?? ''} options={presets.map((p) => p.key)} onChange={setPresetKey} title={presets.map((p) => `${p.key}: ${p.label}`).join('\n')} />
            <Select label="evaluate on" value={target ?? ''} options={targetIds} onChange={setTargetId} />
            <span style={{ color: 'var(--ink-muted)' }}>{preset?.label}</span>
            {explain && (
              <span style={{ marginLeft: 'auto', color: 'var(--ink-muted)' }}>
                exported {runs.generated.slice(0, 10)} · other configurations: <code>python calibration_explorer/app.py</code>
              </span>
            )}
          </SectionBar>
          {run ? (
            <>
              <CVCurve run={run} runs={runs} />
              <RunCrossplot run={run} runs={runs} meta={meta} knownIds={knownIds} />
              <RunSeries run={run} runs={runs} meta={meta} knownIds={knownIds} />
              <SplitCheck run={run} runs={runs} />
            </>
          ) : (
            <ChartFrame title="No run" exportable={false}><p className="frame-empty">No exported run for this configuration at this site.</p></ChartFrame>
          )}

          <SectionBar>
            <span className="pill">Cohort diagnostics</span>
            {explain && <span style={{ color: 'var(--ink-muted)' }}>what the cohorts are, before anything is fitted</span>}
          </SectionBar>
          <CohortComposition info={preset ? runs.cohorts[preset.key] : undefined} label={preset?.label ?? ''} />
          <SelectionRanking runs={runs} calib={calib} />
          <OverlapMatrix rows={runs.overlap} />
          <SpectraOverlay
            id="cohort-spectra"
            title="Cohort spectra — the three selections against the Addis median"
            subtitle="Median spectrum with the interquartile band for each selection cohort at its locked cutoff, and the Addis evaluation set (dashed). Ann's multi-cohort ask: which selection actually looks like Addis, and does that survive baseline correction?"
            provenance="calibration_explorer /api/spectra compare · Selection tab"
            bySpace={cohortSpectra}
          />
          <SpectraOverlay
            id="site-spectra"
            title="Site spectra — every SPARTAN evaluation site"
            subtitle="Median spectrum with the interquartile band per site. The ~1600 cm⁻¹ band that distinguishes Addis is visible here against Beijing's and Pasadena's; it is what the offset argument rests on."
            provenance="calibration_explorer /api/site_spectra · Sites tab"
            bySpace={siteSpectra}
          />
          {Object.keys(runs.analog).length > 0 && <AnalogLab bySpace={runs.analog} cutoff={calib?.default_cutoff.analogs ?? 500} />}
          {runs.hips.york && <HipsYork runs={runs} />}
        </>
      )}

      {calib && (
        <ChartFrame
          title="The write-ups"
          subtitle="Dated result documents in calibration_explorer/, newest first. The heatmap's numbers are superseded for citation by OFFSET_ADJUDICATION_2026-08-23.md in research/ftir_ec_phase3/."
          exportable={false}
        >
          <table className="placement">
            <tbody>
              {calib.docs.map((d) => (
                <tr key={d.file}>
                  <td className="u" style={{ whiteSpace: 'nowrap', verticalAlign: 'top' }}>{d.date}</td>
                  <td style={{ verticalAlign: 'top' }}>
                    <div style={{ fontWeight: 600 }}>{d.title}</div>
                    <div className="u" style={{ fontSize: 11.5 }}>{d.summary}</div>
                    <code style={{ fontSize: 10.5, color: 'var(--ink-muted)' }}>{d.file}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </ChartFrame>
      )}
    </>
  )
}

function SectionBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="panel" style={{ padding: '12px 16px', display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', fontSize: 12.5 }}>
      {children}
    </div>
  )
}
