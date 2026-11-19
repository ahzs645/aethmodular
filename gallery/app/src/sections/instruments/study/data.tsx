import { ChartFrame, Segmented } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { useExport } from '@/lib/data'
import { fmt, percent } from '@/lib/format'
import { namedColor } from '@/lib/siteColours'
import { useSearchParam } from '@/lib/url'
import { checkAddisStudy, type AddisStudyFile, type CalendarId, type StudyCalendar, type StudyStats } from './model'

export * from './model'
export const STUDY_PATH = 'aethalometer/addis-study.json'
export const STUDY_PROVENANCE = 'Shlipak et al. (2026), DOI 10.1021/acsestair.6c00140 · gallery/data/export_addis_study.py · saved Jacros hourly export'
export const useAddisStudy = () => useExport<AddisStudyFile>(STUDY_PATH, checkAddisStudy)
export const CALENDARS: CalendarId[] = ['belg_feb', 'dry_feb']
const LABELS: Record<CalendarId, string> = { belg_feb: 'Paper calendar · February in Belg', dry_feb: 'Sensitivity · February in Dry' }

/** The page's two calendar views move together; a Dashboard tile gets its own URL scope. */
export function useStudyCalendar() {
  return useSearchParam<CalendarId>('aeth-addis-study.calendar', 'belg_feb', CALENDARS)
}
export function CalendarControl({ value, onChange }: { value: CalendarId; onChange: (v: CalendarId) => void }) {
  return <Segmented label="Season calendar" value={value} options={CALENDARS} onChange={onChange} optionLabel={(v) => LABELS[v]}
    title="The study puts February in Belg. The alternative moves February into the longer Dry season to show sensitivity to that choice." />
}
export function StudyLoading({ id, title, error }: { id: string; title: string; error: string }) {
  return <ChartFrame id={id} title={title} exportable={false}>
    <LoadState what="the Addis study comparison" error={error} command="uv run python gallery/data/export_addis_study.py" />
  </ChartFrame>
}
export const seasonColour = (calendar: StudyCalendar, id: string) => namedColor(calendar.seasons.map((s) => ({ name: s.id, color: s.color })), id)
export const hoursLabel = (n: number) => n.toLocaleString('en-GB')
export const summaryTip = (s: StudyStats) => [
  `${hoursLabel(s.valid_hours)} valid hourly values / ${hoursLabel(s.expected_hours)} calendar hours (${percent(s.coverage_percent, 1)})`,
  `${hoursLabel(s.snapshot_hours)} slots in the saved record; ${percent(s.snapshot_coverage_percent, 1)} have a finite IR BC value`,
  `Available-hour mean ${fmt(s.mean_bc_ug_m3, 2)} µg/m³; median ${fmt(s.median_bc_ug_m3, 2)} µg/m³`,
  `SD ${fmt(s.sd_bc_ug_m3, 2)}; maximum ${fmt(s.max_bc_ug_m3, 2)} µg/m³`,
]
