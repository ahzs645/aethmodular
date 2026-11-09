/**
 * Contract for gallery/data/export_addis_study.py. Statistics are saved hourly
 * summaries; the gallery does not recompute QC, source fractions or study means.
 */
export type CalendarId = 'belg_feb' | 'dry_feb'

export interface StudyStats {
  expected_hours: number
  snapshot_hours: number
  valid_hours: number
  coverage_percent: number | null
  snapshot_coverage_percent: number | null
  mean_bc_ug_m3: number | null
  median_bc_ug_m3: number | null
  sd_bc_ug_m3: number | null
  max_bc_ug_m3: number | null
  q25_bc_ug_m3: number | null
  q75_bc_ug_m3: number | null
}
export interface TimedStats extends StudyStats { start_local: string; end_local_exclusive: string }
export interface StudySeason extends StudyStats { id: string; label: string; months: number[]; color: string }
export interface StudyHour extends StudyStats { season_id: string; hour: number }
export interface StudyCalendar {
  id: CalendarId
  label: string
  festival_exclusions_applied: false
  seasons: StudySeason[]
  diurnal: StudyHour[]
}
export interface StudyEvent extends TimedStats {
  id: string
  label: string
  date_local: string
  window_rule: string
  available_hours: number
  incomplete: boolean
  unavailable: boolean
  trace: { elapsed_hour: number; local: string; bc_ug_m3: number | null; source_present: boolean }[]
}
export interface ReportedSite {
  label: string
  bc_mean_ug_m3: number
  bc_sd_ug_m3: number
  bc_max_ug_m3: number
  completeness_percent: number
}
export interface StudyPaper {
  id: string
  citation: { title: string; doi: string; url: string; supporting_information_url: string }
  start_local: string
  end_local_exclusive: string
  instrument_model: string
  inlet: string
  season_calendar: string
  source_model: { alpha_fossil: number; alpha_biomass: number; paper_wavelength_pair_nm: number[] | null; paper_fraction_aggregation: string | null }
  reported: { central: ReportedSite; jacros: ReportedSite; biomass_bc_percent_discussion: number }
  unresolved_for_exact_reproduction: string[]
}
export interface AddisStudyFile {
  schema_version: 1
  generated: string
  paper: StudyPaper
  paper_reproduction_complete: false
  independent_validation: false
  deployment: { id: 'jacros'; label: string; site_code: string; serial: string; instrument_model: string; timezone: string; channel: string; unit: 'ug/m3' }
  period: { start_local: string; end_local_exclusive: string; timezone: string }
  source: {
    kind: string; input_file: string; metadata_file: string; input_sha256: string; metadata_sha256: string
    profile_file: string; profile_sha256: string; generated: string; upstream_record: unknown
    timestamp_basis: string; source_unit: string; output_unit: string; min_valid_ir_minutes: number
    finite_handling: string; additional_value_screen: false
    first_local: string; last_local: string; end_local_exclusive: string
  }
  coverage: StudyStats & { outside_snapshot_hours: number; missing_within_snapshot_hours: number; first_valid_local: string | null; last_valid_local: string | null }
  summary: TimedStats
  daily: TimedStats[]
  monthly: TimedStats[]
  calendars: Record<CalendarId, StudyCalendar>
  events: StudyEvent[]
  capabilities: { central: { available: false; reason: string }; source_fractions: { available: false; reason: string } }
  limitations: string[]
}

const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
const text = (v: unknown): v is string => typeof v === 'string' && !!v.trim()
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const count = (v: unknown): v is number => finite(v) && Number.isInteger(v) && v >= 0
const nullable = (v: unknown) => v === null || finite(v)
const localTime = (v: unknown): v is string => text(v) && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+03:00$/.test(v) && Number.isFinite(Date.parse(v))
const VALUES = ['mean_bc_ug_m3', 'median_bc_ug_m3', 'sd_bc_ug_m3', 'max_bc_ug_m3', 'q25_bc_ug_m3', 'q75_bc_ug_m3']

function stats(v: unknown): boolean {
  if (!object(v) || !count(v.expected_hours) || !count(v.snapshot_hours) || !count(v.valid_hours)) return false
  if (v.valid_hours > v.snapshot_hours || v.snapshot_hours > v.expected_hours || !VALUES.every((k) => nullable(v[k]))) return false
  if (v.valid_hours === 0 && VALUES.some((k) => v[k] !== null)) return false
  if (v.valid_hours > 0 && VALUES.filter((k) => k !== 'sd_bc_ug_m3').some((k) => !finite(v[k]))) return false
  return ['coverage_percent', 'snapshot_coverage_percent'].every((k) => v[k] === null || finite(v[k]) && v[k] >= 0 && v[k] <= 100)
}
function timed(v: unknown): boolean {
  return object(v) && stats(v) && localTime(v.start_local) && localTime(v.end_local_exclusive) && Date.parse(v.start_local) < Date.parse(v.end_local_exclusive)
}
function timeSeries(v: unknown): boolean {
  if (!Array.isArray(v) || !v.every(timed)) return false
  return v.every((r, i) => i === 0 || Date.parse(v[i - 1].end_local_exclusive) <= Date.parse(r.start_local))
}

/** Fail closed on an unsupported or incomplete export, rather than display missing data as zero. */
export function checkAddisStudy(v: unknown): string | null {
  if (!object(v) || v.schema_version !== 1) return 'unsupported Addis study export version'
  if (v.paper_reproduction_complete !== false || v.independent_validation !== false) return 'unsupported study evidence status'
  const deployment = v.deployment
  if (!object(deployment) || deployment.id !== 'jacros' || deployment.unit !== 'ug/m3' || deployment.timezone !== 'Africa/Addis_Ababa'
    || !['label', 'site_code', 'serial', 'instrument_model', 'channel'].every((k) => text(deployment[k]))) return 'invalid deployment or BC units'
  if (!object(v.period) || !localTime(v.period.start_local) || !localTime(v.period.end_local_exclusive)
    || Date.parse(v.period.start_local) >= Date.parse(v.period.end_local_exclusive) || v.period.timezone !== 'Africa/Addis_Ababa') return 'invalid study window or timezone'
  if (!object(v.paper) || v.paper.id !== 'shlipak_2026_addis' || !localTime(v.paper.start_local) || !localTime(v.paper.end_local_exclusive)
    || !object(v.paper.citation) || !text(v.paper.citation.title) || !text(v.paper.citation.url)
    || !object(v.paper.reported) || !object(v.paper.source_model) || !finite(v.paper.source_model.alpha_fossil) || !finite(v.paper.source_model.alpha_biomass)
    || !finite(v.paper.reported.biomass_bc_percent_discussion) || !Array.isArray(v.paper.unresolved_for_exact_reproduction)
    || !v.paper.unresolved_for_exact_reproduction.every(text)) return 'missing published study context'
  for (const id of ['central', 'jacros']) {
    const s = v.paper.reported[id]
    if (!object(s) || !text(s.label) || !['bc_mean_ug_m3', 'bc_sd_ug_m3', 'bc_max_ug_m3', 'completeness_percent'].every((k) => finite(s[k]))) return 'invalid published benchmark'
  }
  if (!object(v.source) || !localTime(v.source.first_local) || !localTime(v.source.last_local) || !localTime(v.source.end_local_exclusive)
    || v.source.output_unit !== 'ug/m3' || !count(v.source.min_valid_ir_minutes) || !text(v.source.input_file) || !text(v.source.input_sha256)
    || !text(v.source.finite_handling) || !text(v.source.timestamp_basis)) return 'missing saved-record provenance'
  if (!object(v.coverage) || !stats(v.coverage) || !count(v.coverage.outside_snapshot_hours) || !count(v.coverage.missing_within_snapshot_hours)
    || !timed(v.summary) || !timeSeries(v.daily) || !timeSeries(v.monthly)) return 'invalid hourly support or timeline summaries'
  if (!object(v.calendars)) return 'missing named season calendars'
  for (const id of ['belg_feb', 'dry_feb']) {
    const c = v.calendars[id]
    if (!object(c) || c.id !== id || !text(c.label) || c.festival_exclusions_applied !== false || !Array.isArray(c.seasons) || !Array.isArray(c.diurnal)) return 'invalid named season calendar'
    if (!c.seasons.every((s) => object(s) && stats(s) && text(s.id) && text(s.label) && text(s.color) && Array.isArray(s.months)
      && s.months.every((m) => count(m) && m >= 1 && m <= 12))) return 'invalid seasonal statistics'
    const months = c.seasons.flatMap((s) => s.months).sort((a, b) => a - b)
    if (months.length !== 12 || months.some((m, i) => m !== i + 1)) return 'season calendar must assign each month once'
    const ids = new Set(c.seasons.map((s) => s.id))
    if (ids.size !== c.seasons.length || !c.diurnal.every((h) => object(h) && stats(h) && ids.has(h.season_id) && count(h.hour) && h.hour < 24)
      || new Set(c.diurnal.map((h) => `${h.season_id}:${h.hour}`)).size !== c.diurnal.length) return 'invalid local-hour summaries'
  }
  if (!Array.isArray(v.events) || !v.events.every((e) => object(e) && timed(e) && text(e.id) && text(e.label) && text(e.date_local) && text(e.window_rule)
    && count(e.valid_hours) && e.expected_hours === 24 && e.available_hours === e.valid_hours
    && localTime(e.start_local) && localTime(e.end_local_exclusive) && Date.parse(e.end_local_exclusive) - Date.parse(e.start_local) === 24 * 3600000
    && e.incomplete === (e.valid_hours < e.expected_hours) && e.unavailable === (e.valid_hours === 0) && Array.isArray(e.trace)
    && e.trace.every((h) => object(h) && count(h.elapsed_hour) && h.elapsed_hour < 24 && localTime(h.local) && nullable(h.bc_ug_m3) && typeof h.source_present === 'boolean')
    && new Set(e.trace.map((h) => h.elapsed_hour)).size === e.trace.length)) return 'invalid festival windows'
  const capabilities = v.capabilities
  if (!object(capabilities) || !['central', 'source_fractions'].every((k) => object(capabilities[k]) && capabilities[k].available === false && text(capabilities[k].reason))) return 'missing pending-input status'
  if (!Array.isArray(v.limitations) || !v.limitations.every(text)) return 'missing study limitations'
  return null
}

/**
 * Display only: put the declared Addis wall clock on a UTC axis. This avoids
 * browser timezone/DST shifts; no observations are joined or aggregated here.
 */
export const localAxisDate = (value: string): Date => new Date(`${value.slice(0, 19)}Z`)
const DATE = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
export const localDateLabel = (value: string): string => DATE.format(localAxisDate(value))
export const localRangeLabel = (start: string, endExclusive: string): string => {
  const a = localDateLabel(start), b = DATE.format(new Date(+localAxisDate(endExclusive) - 1))
  return a === b ? a : `${a}–${b}`
}
export const localClockLabel = (value: string): string => `${localDateLabel(value)} ${value.slice(11, 16)}`

/** Preserve all missing hours as gaps; never join across an absent hourly bin. */
export function eventHours(event: StudyEvent) {
  const byHour = new Map(event.trace.map((h) => [h.elapsed_hour, h]))
  return Array.from({ length: 24 }, (_, hour) => ({ hour, observation: byHour.get(hour) ?? null }))
}
