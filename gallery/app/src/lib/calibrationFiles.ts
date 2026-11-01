/**
 * The calibration explorer's two exports, loaded once per session under one cache key each and with the
 * gallery's site naming (lib/metricDefinition normalizeSiteIdentity: Pasadena reads as JPL). The Calibration
 * tab (hooks/useGalleryData), the dashboard and the Research/ML pages all go through here, so a download and
 * its relabelled copy are shared.
 */
import { loadDerived, loadJson, useLoaded } from './data.ts'
import { normalizeSiteIdentity } from './metricDefinition.ts'
import type { CalibFile, CalibRunsFile } from './types.ts'

const relabelled = <T,>(name: string) => loadDerived<T>(`gallery:${name}`, async () => normalizeSiteIdentity(await loadJson<T>(`${name}.json`)))

/** calibration.json: the explorer's summary grid */
export const loadCalibration = () => relabelled<CalibFile>('calibration')
/** calibration_runs.json: per-preset CV curves and per-filter readouts */
export const loadCalibrationRuns = () => relabelled<CalibRunsFile>('calibration_runs')

/** calibration.json for a component (`useLoaded`'s data, error, loading) */
export const useCalibrationFile = () => useLoaded('gallery:calibration', loadCalibration)
/** calibration_runs.json for a component */
export const useCalibrationRuns = () => useLoaded('gallery:calibration_runs', loadCalibrationRuns)
