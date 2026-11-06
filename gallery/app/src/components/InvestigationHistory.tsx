import { useId, useMemo } from 'react'
import { Empty } from '@/components/ChartFrame'
import { Select } from '@/components/Controls'
import { downloadBlob } from '@/lib/export'
import { useSearchParam } from '@/lib/url'
import { EVIDENCE_LABELS, filterHistory, historyMermaid, historyNeighborhood, historySourceUrl, relationLabel,
  type HistoryEdge, type HistoryRecord, type InvestigationHistory as History } from '@/lib/investigationHistory'

const dateText = (date: string | null) => date ? new Date(date + 'T12:00:00Z').toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }) : 'Date unknown'
const Badge = ({ record }: { record: HistoryRecord }) => <span className={`history-badge history-evidence-${record.evidence}`}>{EVIDENCE_LABELS[record.evidence]}</span>
const SourceLink = ({ source, children }: { source: string; children: React.ReactNode }) => <a href={historySourceUrl(source)} target="_blank" rel="noreferrer">{children}</a>

/** A navigable graph of all direct neighbours, including neighbours outside the catalog filters. */
function ExperimentMap({ history, selected, onSelect }: { history: History; selected: HistoryRecord; onSelect: (id: string) => void }) {
  const marker = useId().replace(/:/g, '')
  const { records, edges } = useMemo(() => historyNeighborhood(history, selected.id), [history, selected.id])
  const before = records.filter(r => r.id !== selected.id && edges.some(e => e.from === r.id && e.to === selected.id))
  const after = records.filter(r => r.id !== selected.id && edges.some(e => e.to === r.id && e.from === selected.id))
  const height = Math.max(before.length, after.length, 2) * 120 + 56, middle = height / 2 - 50
  const neighbours = [...before.map((r, i) => ({ r, x: 16, y: 40 + i * 120, incoming: true })),
    ...after.map((r, i) => ({ r, x: 704, y: 40 + i * 120, incoming: false }))]
  return <section className="history-map" aria-label="Experiment connections">
    <div className="history-list-heading"><h3>Connections</h3><span>{records.length} records · {edges.length} connections</span></div>
    <p className="subtitle">Select a connected experiment to follow the research. Direct neighbours stay visible across filters; arrow direction follows the curated record.</p>
    {edges.length === 0 ? <Empty>No documented connections for this record yet. Its position in the timeline does not imply a dependency.</Empty> : <>
      <div className="history-map-scroll" tabIndex={0} aria-label="Scrollable experiment diagram">
        <svg width={960} height={height} viewBox={`0 0 960 ${height}`} role="group" aria-label={`Research connections for ${selected.id}`}>
          <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" /></marker></defs>
          <text x={16} y={22} className="history-map-label">CITED EARLIER WORK</text><text x={360} y={22} className="history-map-label">SELECTED EXPERIMENT</text><text x={704} y={22} className="history-map-label">CONNECTED FOLLOW-UPS</text>
          {neighbours.map(({ r, x, y, incoming }) => {
            const own = edges.filter(e => incoming ? e.from === r.id && e.to === selected.id : e.from === selected.id && e.to === r.id)
            const revised = own.some(e => ['qualifies_claim', 'revises_interpretation'].includes(e.type))
            return <g key={`${incoming}-${r.id}`}>
              <path d={incoming ? `M 256 ${y + 50} C 308 ${y + 50}, 308 ${middle + 50}, 360 ${middle + 50}` : `M 600 ${middle + 50} C 652 ${middle + 50}, 652 ${y + 50}, 704 ${y + 50}`}
                className={`history-map-edge${revised ? ' revised' : ''}${own.every(e => e.type === 'related_work') ? ' related' : ''}`} markerEnd={`url(#${marker})`} />
              <foreignObject x={x} y={y} width={240} height={106}>
                <button type="button" className="history-map-node" onClick={() => onSelect(r.id)} title={r.title}>
                  <small>{r.id}</small><strong>{r.title}</strong><span>{[...new Set(own.map(e => relationLabel(e.type)))].join(' · ')}</span>
                </button>
              </foreignObject>
            </g>
          })}
          <foreignObject x={360} y={middle} width={240} height={112}><div className="history-map-node selected"><small>{selected.id}</small><strong>{selected.title}</strong><Badge record={selected} /></div></foreignObject>
        </svg>
      </div>
      <p className="history-map-key"><span>→ Data dependency</span><span>⇢ Related work</span><span className="history-correction-ink">→ Qualification or revision</span></p>
    </>}
    <details className="history-mermaid"><summary>Mermaid diagram</summary><p className="subtitle">This exports the connections currently shown, with their relationship types.</p>
      <button type="button" className="btn" onClick={() => downloadBlob(new Blob([historyMermaid(records, edges)], { type: 'text/plain' }), `investigation-${selected.id}.mmd`)}>Download Mermaid</button>
      <pre>{historyMermaid(records, edges)}</pre>
    </details>
  </section>
}

function RecordDetails({ record, history, onSelect, onOpen, canOpen }: {
  record: HistoryRecord; history: History; onSelect: (id: string) => void
  onOpen: (tab: string, page?: string) => void; canOpen: (tab: string) => boolean
}) {
  const edges = history.edges.filter(e => e.from === record.id || e.to === record.id)
  const charts = record.charts.filter(c => canOpen(c.tab))
  const corrections = record.corrections ?? []
  const hasWarnings = corrections.some(c => c.from === record.id) || record.caveats.length > 0
  const notebook = record.notebook
  return <aside className="panel history-detail" aria-label="Selected experiment">
    <div className="history-detail-kicker">{record.id} · {history.families.find(f => f.id === record.family)?.title}</div>
    <h3>{record.title}</h3><Badge record={record} />
    <p className="subtitle">{record.sourceDate ? `Source updated ${dateText(record.sourceDate)}. Run date not established by this index.` : 'No source-history date is available.'}</p>
    {record.question && <><h4>Question</h4><p>{record.question}</p></>}
    <h4>Evidence graphs</h4>
    <p className="history-context">Open the existing analysis to inspect its graphs, cohort and settings. A live diagnostic can differ from a saved historical result.</p>
    {charts.length ? <div className="history-chart-links">{charts.map(c => <div key={`${c.tab}/${c.page}`}><button type="button" className="btn" onClick={() => onOpen(c.tab, c.page ?? undefined)}>{c.label} ↗</button>{c.missingData?.length ? <small className="history-context">Results unavailable in this checkout</small> : null}</div>)}</div>
      : <p className="subtitle">No graph page is mapped for this record in this build. Its source remains listed below.</p>}
    {hasWarnings && <div className="history-caution"><h4>Interpretation and limits</h4>
      {record.caveats.map((note, i) => <p key={i}>{note}</p>)}
      {corrections.map((c, i) => <div key={i}><strong>{c.from === record.id ? 'Later qualification' : 'Revises earlier work'} · {c.claim}</strong><p>{c.reading}</p><button className="area-card-link" type="button" onClick={() => onSelect(c.from === record.id ? c.to : c.from)}>Inspect {c.from === record.id ? c.to : c.from} ›</button> · <SourceLink source={c.source}>Correction source</SourceLink></div>)}
    </div>}
    {!hasWarnings && corrections.length > 0 && <div className="history-caution"><h4>What this changed</h4>{corrections.map((c, i) => <p key={i}>{c.reading} <button className="area-card-link" type="button" onClick={() => onSelect(c.from)}>Earlier record: {c.from}</button></p>)}</div>}
    {notebook ? <details open={!hasWarnings && corrections.length === 0}><summary>Historical notebook notes</summary>
      <p className="history-context">Extracted from the notebook; the qualifications above take precedence over older conclusions.</p>
      <p>{record.summary || 'No result summary was recovered. Open the source to inspect its methods and saved evidence.'}</p>
    </details> : <><h4>Recorded finding or readiness</h4><p>{record.summary || 'No result summary was recovered. Open the source to inspect its methods and saved evidence.'}</p></>}
    <h4>Evidence record</h4>
    {notebook && <p>{notebook.executed}/{notebook.cells} code cells have execution counts; {notebook.outputs} have saved outputs. {notebook.errors > 0 ? `${notebook.errors} saved error${notebook.errors === 1 ? '' : 's'}.` : 'No saved errors found.'} Saved cells do not verify a complete successful run.</p>}
    {record.evidenceLabel && <p>{record.evidenceLabel} (retrospective label).</p>}
    {record.source && <p className="history-source">{record.tracked ? <SourceLink source={record.source}>{record.source}</SourceLink> : <code>{record.source}</code>}{!record.sourceAvailable && <span> · Source missing in this checkout</span>}{!record.tracked && <span> · Not in the source revision</span>}</p>}
    <details><summary>Connected checks ({edges.length})</summary>{edges.length ? <ul className="history-edges">{edges.map((e, i) => <Connection key={i} edge={e} selected={record.id} onSelect={onSelect} />)}</ul> : <p>No documented connection yet.</p>}</details>
  </aside>
}
function Connection({ edge, selected, onSelect }: { edge: HistoryEdge; selected: string; onSelect: (id: string) => void }) {
  const other = edge.from === selected ? edge.to : edge.from
  return <li><button type="button" className="area-card-link" onClick={() => onSelect(other)}>{edge.from} → {edge.to}</button><small>{relationLabel(edge.type)}</small><p>{edge.reason}</p>{edge.sourceTracked === false ? <p className="history-source"><code>{edge.source}</code> · Evidence file is outside the source revision</p> : <SourceLink source={edge.source}>Connection source</SourceLink>}</li>
}

export function InvestigationHistory({ history, view, onOpen, canOpen }: {
  history: History; view: 'timeline' | 'connections'; onOpen: (tab: string, page?: string) => void; canOpen: (tab: string) => boolean
}) {
  const [search, setSearch] = useSearchParam<string>('investigations.search', '')
  const [family, setFamily] = useSearchParam<string>('investigations.family', 'all', ['all', ...history.families.map(f => f.id)])
  const [evidence, setEvidence] = useSearchParam<string>('investigations.evidence', 'all', ['all', ...Object.keys(EVIDENCE_LABELS)])
  const defaultId = history.records.find(r => r.evidence === 'verified')?.id ?? history.records[0]?.id ?? ''
  const [id, setId] = useSearchParam<string>('investigations.record', defaultId, history.records.map(r => r.id))
  const records = useMemo(() => filterHistory(history.records, search, family, evidence), [history, search, family, evidence])
  const selected = records.find(r => r.id === id) ?? records[0]
  const select = (next: string) => {
    if (!records.some(r => r.id === next)) { setSearch(''); setFamily('all'); setEvidence('all') }
    setId(next)
  }
  const clear = () => { setSearch(''); setFamily('all'); setEvidence('all') }
  return <>
    <div className="panel history-controls">
      <label className="history-search">Search experiments<input type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Question, method, notebook or finding…" /></label>
      <Select label="Research family" value={family} onChange={setFamily} options={['all', ...history.families.map(f => f.id)]} optionLabel={v => history.families.find(f => f.id === v)?.title ?? 'All families'} />
      <Select label="Evidence" value={evidence} onChange={setEvidence} options={['all', ...Object.keys(EVIDENCE_LABELS)]} optionLabel={v => EVIDENCE_LABELS[v as keyof typeof EVIDENCE_LABELS] ?? 'All evidence'} />
      <span className="subtitle" role="status">{records.length} of {history.records.length} records</span><button type="button" className="btn" onClick={clear}>Reset filters</button>
      <button type="button" className="btn" title="Download a Mermaid diagram of every record and connection in this catalog" onClick={() => downloadBlob(new Blob([historyMermaid(history.records, history.edges)], { type: 'text/plain' }), 'investigation-history.mmd')}>Export full Mermaid</button>
    </div>
    <details className="history-scope"><summary>Coverage and how to read this history</summary><p>{history.coverage}</p><p>{history.chronology}</p><p>Relationships come from the reviewed retrospective. Later notebooks without curated relationships stay in the catalog; no connections are inferred from dates or similar titles.</p></details>
    {selected ? <div className={`history-workspace ${view}`}>
      <div className="panel history-browser">{view === 'connections' ? <ExperimentMap history={history} selected={selected} onSelect={select} /> : <>
        <div className="history-list-heading"><h3>Research timeline</h3><span>Latest source change first</span></div>
        <p className="subtitle">Source dates are shown below. Historical run dates are not established.</p>
        <ol className="history-timeline" aria-label="Research timeline">{records.map(r => <li key={r.id}>
          <button type="button" className={`history-event${r.id === selected.id ? ' selected' : ''}`} aria-pressed={r.id === selected.id} onClick={() => select(r.id)}>
            <span className="history-event-meta"><time dateTime={r.sourceDate ?? undefined}>{dateText(r.sourceDate)}</time><span>{r.id}</span></span>
            <strong>{r.title}</strong><span className="history-event-foot"><Badge record={r} /><span>{r.charts.filter(c => canOpen(c.tab)).length} graph pages</span></span>
          </button>
        </li>)}</ol>
      </>}</div>
      <RecordDetails record={selected} history={history} onSelect={select} onOpen={onOpen} canOpen={canOpen} />
    </div> : <div className="panel"><Empty>No research records match these filters.</Empty><button type="button" className="btn" onClick={clear}>Show all records</button></div>}
  </>
}
