import { ChartFrame, ExplainOnly } from '@/components/ChartFrame'
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
 * The page intro and the sections' text (answers, open items) are explanation:
 * shown with Explain on; the figures always show.
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
        <ExplainOnly><ChartFrame title={spec.title} exportable={false} subtitle={`Data: ${data.generated_from}`}>
          {spec.intro && <Markdown text={text(spec.intro)} />}
        </ChartFrame></ExplainOnly>
        {spec.sections.map((s) => <div key={s.id} id={`sec-${spec.id}-${s.id}`}>
          <ExplainOnly><ChartFrame title={s.title} exportable={false}>{s.text ? <Markdown text={text(s.text)} /> : null}</ChartFrame></ExplainOnly>
          <div className={s.columns === 2 ? 'figure-pair' : undefined}>
            {(s.figures ?? []).map((f, i) => <Figure key={f.id ?? i} {...f} />)}
          </div>
        </div>)}
      </PageToc>
    </BundleContext.Provider>
  )
}
