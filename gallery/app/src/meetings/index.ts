import type { MeetingSpec } from './types'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
/** a meeting's name in the picker: its date as month, day, year, e.g. "Oct 28, 2026" (from the ISO date, no time zone) */
export const meetingLabel = (iso: string) => {
  const [y, m, d] = iso.split('-').map(Number)
  return `${MONTHS[m - 1]} ${d}, ${y}`
}

/** Every `*.meeting.ts` in this folder, newest first, each named by its date (meetingLabel). */
const modules = import.meta.glob<{ default: MeetingSpec }>('./*.meeting.ts', { eager: true })
export const SPEC_MEETINGS: MeetingSpec[] = Object.values(modules).map((m) => ({ ...m.default, label: meetingLabel(m.default.date) }))
  .sort((a, b) => b.date.localeCompare(a.date))
