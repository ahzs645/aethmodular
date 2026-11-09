import type { CalibFile, CalibRunsFile } from './types'
import type { Config } from '../charts/calibration/common'

/** Both saved searches use the same VIP ranking under different identifiers. */
export const calibrationCohort = (cohort: string) => cohort === 'vip_analogs' ? 'analogs' : cohort

/** A view of saved results; protocol changes apply to the grid, not the fixed-protocol PLS-factor exports. */
export function calibrationSubset(data: CalibFile, cohort: string, protocol = ''): CalibFile {
  if (!cohort && !protocol) return data
  return { ...data,
    grid: data.grid.filter(r => (!cohort || r.co === cohort) && (!protocol || r.mo === protocol)),
    sweeps: cohort ? data.sweeps.filter(r => r.co === cohort) : data.sweeps,
    default_cutoff: cohort ? Object.fromEntries(Object.entries(data.default_cutoff).filter(([co]) => co === cohort)) : data.default_cutoff,
  }
}

/** Preserve the chosen preprocessing and size where the requested cohort has that saved preset. */
export function cohortPreset(presets: CalibRunsFile['presets'], key: string, cohort: string) {
  const current = presets.find(p => p.key === key)
  const available = cohort ? presets.filter(p => p.cohort === cohort) : presets
  return available.find(p => p.key === key)
    ?? available.find(p => p.spectra === current?.spectra && p.cutoff === current?.cutoff)
    ?? available.find(p => p.spectra === current?.spectra)
    ?? available[0]
}

/** Pick an existing configuration in a newly selected cohort; full-cohort exports retain their null cutoff. */
export function cohortConfig(current: Config, data: CalibFile | null, cohort: string): Config {
  if (!cohort || current.co === cohort || !data) return current
  const choices = data.sweeps.filter(r => r.co === cohort)
  const cut = data.default_cutoff[cohort] ?? null
  const row = choices.find(r => r.sp === current.sp && r.cut === cut && r.sel === current.sel)
    ?? choices.find(r => r.sp === current.sp && r.cut === cut)
    ?? choices.find(r => r.cut === cut)
    ?? choices[0]
    ?? data.grid.find(r => r.co === cohort)
  return row ? { co: row.co, cut: row.cut, sel: row.sel, sp: row.sp } : current
}
