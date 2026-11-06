/** Evidence catalog semantics shared by the private index and the Investigations UI. */
export type EvidenceState = 'saved' | 'source' | 'documented' | 'missing' | 'verified' | 'diagnostic' | 'blocked'
export interface HistoryChart { tab: string; page: string | null; label: string; missingData?: string[] }
export interface HistoryCorrection { from: string; to: string; claim: string; reading: string; source: string }
export interface HistoryRecord {
  id: string; title: string; family: string; source: string | null; sourceAvailable: boolean
  sourceDate: string | null; tracked: boolean; evidence: EvidenceState; evidenceLabel?: string
  summary: string; question?: string; caveats: string[]; charts: HistoryChart[]; corrections?: HistoryCorrection[]
  notebook?: { cells: number; executed: number; outputs: number; errors: number }
}
export interface HistoryEdge { from: string; to: string; type: string; reason: string; source: string; sourceTracked?: boolean }
export interface InvestigationHistory {
  schema: number; coverage: string; chronology: string
  families: { id: string; title: string; purpose: string }[]
  records: HistoryRecord[]; edges: HistoryEdge[]
}
export const EVIDENCE_LABELS: Record<EvidenceState, string> = {
  saved: 'Saved notebook outputs', source: 'Source only', documented: 'Documented record', missing: 'Missing source',
  verified: 'Reproduction verified', diagnostic: 'Diagnostic completed', blocked: 'Blocked',
}
export const RELATION_LABELS: Record<string, string> = {
  data_dependency: 'Data dependency', related_work: 'Related work', qualifies_claim: 'Qualifies claim', revises_interpretation: 'Revises interpretation',
}
export const relationLabel = (type: string) => RELATION_LABELS[type] ?? type.replace(/_/g, ' ')
export function filterHistory(records: HistoryRecord[], search: string, family = 'all', evidence = 'all') {
  const words = search.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean)
  return records.filter(r => (family === 'all' || r.family === family) && (evidence === 'all' || r.evidence === evidence)
    && words.every(w => [r.id, r.title, r.source ?? '', r.summary, ...r.caveats, ...(r.corrections ?? []).map(c => c.reading)].join(' ').toLocaleLowerCase().includes(w)))
}
/** Only documented edges are used. Chronology and family membership never create edges. */
export function historyNeighborhood(history: InvestigationHistory, id: string) {
  const edges = history.edges.filter(e => e.from === id || e.to === id)
  const ids = new Set([id, ...edges.flatMap(e => [e.from, e.to])])
  return { records: history.records.filter(r => ids.has(r.id)), edges }
}
export function historyMermaid(records: HistoryRecord[], edges: HistoryEdge[]) {
  const ids = new Map(records.map((r, i) => [r.id, `n${i}`]))
  const quote = (s: string) => s.replace(/[&"<>\[\]{}|\r\n]/g, c => `#${c.charCodeAt(0)};`)
  return ['flowchart LR', ...records.map(r => `  ${ids.get(r.id)}["${quote(r.id + ': ' + r.title)}"]`),
    ...edges.filter(e => ids.has(e.from) && ids.has(e.to)).map(e => `  ${ids.get(e.from)} ${e.type === 'related_work' ? '-.->' : '-->'}|"${quote(relationLabel(e.type))}"| ${ids.get(e.to)}`)].join('\n') + '\n'
}
export const historySourceUrl = (source: string) => `https://github.com/ahzs645/aethmodular/blob/${import.meta.env?.VITE_SOURCE_REVISION ?? 'main'}/${source.split('/').map(encodeURIComponent).join('/')}`
