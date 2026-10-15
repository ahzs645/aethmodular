import type { MeetingSpec } from './types'

/** Every `*.meeting.ts` in this folder, newest first; the Meeting tab's picker lists them after the hand-built pages. */
const modules = import.meta.glob<{ default: MeetingSpec }>('./*.meeting.ts', { eager: true })
export const SPEC_MEETINGS: MeetingSpec[] = Object.values(modules).map((m) => m.default).sort((a, b) => b.date.localeCompare(a.date))
