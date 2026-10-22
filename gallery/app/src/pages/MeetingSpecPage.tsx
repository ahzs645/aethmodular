import type { ReactNode } from 'react'
import { ExportState } from '@/components/ExportState'
import { BundleContext, Figure, type MeetingBundle } from '@/components/Figure'
import { PageToc } from '@/components/PageToc'
import { useExport } from '@/lib/data'
import { Markdown } from '@/lib/markdown'
import type { MeetingSpec } from '@/meetings/types'

/**
 * Draws a declared meeting (src/meetings/*.meeting.ts): each section's text as
 * a panel, then its figures. The bundle is the exporter's tables and text; a
 * paragraph that is just "@key" is replaced by the bundle's text for that key.
 * The page intro and text-only sections (open items, next steps) show as panels.
 * A section with figures shows its title as a heading and puts its text under
 * the first figure's "?" (open on every chart at once with the header's Explain
 * toggle), with that figure's own description.
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
        <TextPanel title={spec.title} note={`Data: ${data.generated_from}`}>
          {spec.intro && <Markdown text={text(spec.intro)} />}
        </TextPanel>
        {spec.sections.map((s) => {
          const figures = s.figures ?? []
          const body = s.text ? <Markdown text={text(s.text)} /> : null
          // a text-only section is a panel of its own; otherwise its text is the first figure's explanation
          return <div key={s.id} id={`sec-${spec.id}-${s.id}`}>
            {figures.length ? <h2 className="meeting-section-title">{s.title}</h2> : <TextPanel title={s.title}>{body}</TextPanel>}
            <div className={s.columns === 2 ? 'figure-pair' : undefined}>
              {figures.map((f, i) => <Figure key={f.id ?? i} {...f} about={i === 0 ? body : undefined} />)}
            </div>
          </div>
        })}
      </PageToc>
    </BundleContext.Provider>
  )
}

/** Text with nothing to explain it further: the page intro and sections without figures. */
function TextPanel({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <section className="panel">
      <header className="frame-head"><div><h2>{title}</h2>{note && <p className="subtitle">{note}</p>}</div></header>
      {children}
    </section>
  )
}
