import type { FigureSpec } from '@/components/Figure'

/**
 * A meeting page, declared instead of coded: sections of text and figures,
 * drawn by pages/MeetingSpecPage. The data is one bundle written by a Python
 * exporter (gallery/data/meeting_bundle.py); figures name its tables and
 * columns. Text is Markdown, or "@key" for a paragraph the exporter wrote
 * from the numbers (so prose cannot drift from the charts).
 *
 * To add a meeting: write gallery/data/export_meeting_<date>.py (tables + text
 * through write_bundle), then a `<id>.meeting.ts` file here. It appears in the
 * Meeting tab's picker by itself (meetings/index.ts).
 */
export interface MeetingSpec {
  /** also the bundle's file name: data/meetings/<id>.json */
  id: string
  /** picker label, the date as "Oct 9, 2026" (meetings/index meetingLabel sets it from `date`) */
  label: string
  /** ISO date of the meeting; the picker lists newest first */
  date: string
  title: string
  /** command that rebuilds the bundle, shown if it cannot be loaded */
  command: string
  intro?: string
  sections: MeetingSection[]
}
export interface MeetingSection {
  id: string
  title: string
  text?: string
  figures?: FigureSpec[]
  /** figures side by side on wide screens */
  columns?: 1 | 2
}
