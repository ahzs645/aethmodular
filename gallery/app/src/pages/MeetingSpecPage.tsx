import { useEffect, useState, type ReactNode } from 'react'
import { ExportState } from '@/components/ExportState'
import { BundleContext, Figure, type MeetingBundle } from '@/components/Figure'
import { PageToc } from '@/components/PageToc'
import { useExport } from '@/lib/data'
import { usePrefs } from '@/lib/prefs'
import { Markdown } from '@/lib/markdown'
import type { MeetingSpec } from '@/meetings/types'

/**
 * Draws a declared meeting (src/meetings/*.meeting.ts): each section's text as
 * a panel, then its figures. The bundle is the exporter's tables and text; a
 * paragraph that is just "@key" is replaced by the bundle's text for that key.
 * The page intro and the sections' text (answers, open items) are explanation:
 * each section's title always shows, its text behind the section's "?" (open on
 * every section at once with the header's Explain toggle); the figures always show.
 */
export function MeetingSpecPage({ spec }: { spec: MeetingSpec }) {
  const { data, error } = useExport<MeetingBundle>(`meetings/${spec.id}.json`, (b) =>
    b.schema_version !== 1 ? `schema_version ${b.schema_version}, expected 1` : !b.tables ? 'no tables' : null)
  if (!data) return <ExportState title={spec.title} what={`the ${spec.label} meeting data`} error={error} command={spec.command} />
  const text = (t: string) => t.split(/\n\s*\n/).map((p) => {
    const m = /^@([\w-]+)$/.exec(p.trim())
    return m ? data.text[m[1]] ?? `*(no text “${m[1]}” in the bundle)*` : p
  }).join('\n\n')
  return (
    <BundleContext.Provider value={data}>
      <PageToc>
        <SectionText title={spec.title} note={`Data: ${data.generated_from}`}>
          {spec.intro && <Markdown text={text(spec.intro)} />}
        </SectionText>
        {spec.sections.map((s) => <div key={s.id} id={`sec-${spec.id}-${s.id}`}>
          <SectionText title={s.title}>{s.text ? <Markdown text={text(s.text)} /> : null}</SectionText>
          <div className={s.columns === 2 ? 'figure-pair' : undefined}>
            {(s.figures ?? []).map((f, i) => <Figure key={f.id ?? i} {...f} />)}
          </div>
        </div>)}
      </PageToc>
    </BundleContext.Provider>
  )
}

/** A section's title, with its text behind a "?" like a chart's explanation; the header's Explain toggle opens them all. */
function SectionText({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  const { explain } = usePrefs()
  const [open, setOpen] = useState(explain)
  useEffect(() => setOpen(explain), [explain])
  const has = !!children || !!note
  return (
    <section className="panel">
      <header className="frame-head">
        <div><h2>{title}</h2>{open && note && <p className="subtitle">{note}</p>}</div>
        {has && <div className="frame-actions">
          <button type="button" className={`btn quiet ${open ? 'on' : ''}`} aria-pressed={open}
            title={open ? 'Hide the text' : 'What we found in this section'} onClick={() => setOpen((v) => !v)}>?</button>
        </div>}
      </header>
      {open && children}
    </section>
  )
}
