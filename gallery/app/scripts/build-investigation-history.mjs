#!/usr/bin/env node
/** Index research sources and curated connections without running any analyses.
 * The checked-in index lives inside the private Research section, so public builds omit it.
 * node scripts/build-investigation-history.mjs [--check]
 */
import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const ROOT = path.resolve(APP, '../..')
const OUT = path.join(APP, 'src/sections/research/history.generated.json')
const read = p => fs.readFileSync(path.join(ROOT, p), 'utf8')
const json = p => JSON.parse(read(p))
const base = 'docs/openresearch-retrospective/'
const families = json(base + 'families.json')
const curation = json(base + 'curation.json')
const updates = json(base + 'claim_updates.json').updates
const contracts = json(base + 'experiments/contracts.json').experiments
const git = (...args) => execFileSync('git', ['-C', ROOT, ...args], { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 }).trim()
const tracked = new Set(git('ls-files', '-z').split('\0'))
// One history pass, rather than launching Git once per source. These are source dates, never run dates.
const dates = new Map()
let date = null
for (const line of git('log', '--format=@%cs', '--name-only', '--', 'research', 'docs', 'deliverables').split('\n')) {
  if (/^@\d{4}-\d{2}-\d{2}$/.test(line)) date = line.slice(1)
  else if (line && !dates.has(line)) dates.set(line, date)
}
const plain = s => s.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/<[^>]+>/g, '').replace(/[*`#]/g, '').replace(/\s+/g, ' ').trim()
const member = new Map(families.families.flatMap(f => f.records.map(id => [id, f.id])))
const records = new Map()
const notes = id => [curation.overrides[id]?.note, families.record_notes[id]].filter(Boolean).map(plain)
function add(id, source, title, more = {}) {
  const available = fs.existsSync(path.join(ROOT, source))
  records.set(id, { id, title, family: member.get(id) ?? 'recent', source, sourceAvailable: available,
    sourceDate: dates.get(source) ?? null, tracked: tracked.has(source), evidence: available ? 'documented' : 'missing',
    summary: '', caveats: notes(id), charts: [], ...more })
}
for (const area of ['research/ftir_hips_chem', 'research/ftir_ec_phase3']) {
  for (const filename of fs.readdirSync(path.join(ROOT, area)).filter(f => f.endsWith('.ipynb')).sort()) {
    const source = `${area}/${filename}`, nb = json(source)
    const markdown = nb.cells.filter(c => c.cell_type === 'markdown').map(c => Array.isArray(c.source) ? c.source.join('') : c.source ?? '')
    const code = nb.cells.filter(c => c.cell_type === 'code')
    const id = filename.match(/^ftir_(\d+)_/)?.[1]
    const key = id ? `ftir_${id.padStart(2, '0')}` : filename === 'ftir_vibes_vs_airspec.ipynb' ? 'vibes_comparison' : filename.slice(0, -6)
    const title = markdown.flatMap(s => s.split('\n')).find(s => s.startsWith('# '))?.slice(2) ?? key.replaceAll('_', ' ')
    const summary = markdown.find(s => /^#{1,4}\s+.*(?:tl;dr|summary|conclusion|takeaway|verdict)/im.test(s)) ?? markdown[0] ?? ''
    const outputs = code.filter(c => c.outputs?.length).length
    const errors = code.flatMap(c => c.outputs ?? []).filter(o => o.output_type === 'error').length
    add(key, source, plain(title), { evidence: outputs ? 'saved' : 'source', summary: plain(summary.replace(/^#{1,6}.*$/gm, '')).slice(0, 1400),
      notebook: { cells: code.length, executed: code.filter(c => c.execution_count != null).length, outputs, errors } })
  }
}
for (const r of curation.additional_records) {
  add(r.id, r.source, r.title, { summary: plain(curation.overrides[r.id]?.note ?? ''), evidenceLabel: r.evidence_status.replaceAll('_', ' ') })
}
for (const r of contracts) {
  const evidence = r.status === 'blocked' ? 'blocked' : r.status === 'local_reproduction_verified' ? 'verified' : 'diagnostic'
  add(r.id, base + 'experiments/' + r.file, r.title, { family: r.family, evidence, summary: r.readiness, question: r.question })
}
// Family entries missing from this checkout stay visible as evidence gaps.
for (const [id, family] of member) if (!records.has(id)) records.set(id, {
  id, title: id.replaceAll('_', ' '), family, source: null, sourceAvailable: false, tracked: false,
  sourceDate: null, evidence: 'missing', summary: 'The retrospective names this record, but its source was not located in this checkout.', caveats: notes(id), charts: [],
})

const edges = new Map()
function edge(e) {
  if (!records.has(e.from) || !records.has(e.to)) return
  const type = e.type ?? 'related_work'
  const source = e.evidence ?? base + 'curation.json'
  edges.set(`${e.from}/${e.to}/${type}`, { from: e.from, to: e.to, type, reason: e.reason, source, sourceTracked: tracked.has(source) })
}
curation.links.forEach(edge)
families.dependency_evidence.forEach(edge)
families.revisions.forEach(edge)
// Later dependencies verified in the cited source introductions/scripts; never inferred from titles.
const laterDependencies = [
  ['decision_workbook_20261022', 'experiments_20261022', 'research/ftir_hips_chem/experiments_20261022.ipynb', 'The introduction states that the follow-up experiments copy the decision workbook’s data, toolbox and out-of-site predictions, using the same models and folds.'],
  ['decision_workbook_20261022', 'transfer_diagnosis_20261102', 'research/ftir_hips_chem/workflows/run_transfer_diagnosis_20261102.py', 'The transfer diagnosis reads the 22 October decision bundle for the IMPROVE, Addis and network spectra.'],
  ['transfer_diagnosis_20261102', 'season_composition_20261103', 'research/ftir_hips_chem/workflows/run_season_composition_20261103.py', 'The seasonal analysis explicitly reuses the transfer-diagnosis fits and reads its output tables; it adds no new calibration.'],
  ['transfer_diagnosis_20261102', 'vibes_newton_refit_20261103', 'research/ftir_hips_chem/workflows/vibes_newton_refit_20261103.py', 'The Newton follow-up reuses transfer-diagnosis scoring and its saved continuum-injection comparison. This is a later, different population from the September 12-case check.'],
  ['vibes_full_colab', 'vibes_followup_experiments', 'research/ftir_hips_chem/vibes_followup_experiments.ipynb', 'The context identifies the completed September Colab run as the source of saved training, test and Addis spectra for common-test selection and routing.'],
]
laterDependencies.forEach(([from, to, evidence, reason]) => edge({ from, to, evidence, reason, type: 'data_dependency' }))
for (const r of contracts) for (const from of r.sources) edge({ from, to: r.id,
  type: r.data_sources?.includes(from) ? 'data_dependency' : 'related_work',
  reason: r.dependency_reason ?? 'The experiment contract cites this earlier work.', evidence: base + 'experiments/' + r.file })
for (const u of updates) {
  const correction = { from: u.from, to: u.to, claim: u.claim_scope, reading: u.current_reading, source: u.evidence }
  for (const id of [u.from, u.to]) {
    const r = records.get(id)
    if (r) (r.corrections ??= []).push(correction)
  }
}
// Graph destinations come from the page registry, rather than duplicating its labels and page keys.
for (const [file, tab] of [['research', 'research'], ['ml', 'ml']]) {
  const pages = read(`gallery/app/src/sections/${file}/pages.ts`)
  for (const m of pages.matchAll(/\{ key: '([^']+)', label: '([^']+)', title: '([^']+)'/g)) {
    for (const r of records.values()) if (m[3].includes(r.id) && (r.id.startsWith('ftir_') ? new RegExp(`${r.id}(?!\\d)`).test(m[3]) : true)) {
      r.charts.push({ tab, page: m[1], label: m[2] })
    }
  }
}
const chartGroups = [
  [['filter_only_diagnostics', 'filter_relationship_stability', 'filter_proportionality', 'september_release'], 'research', 'filter-only', 'Filter-only transfer limits'],
  [['vibes_comparison', 'vibes_full_colab', 'vibes_subgroup_audit', 'next_vibes_loading_trace'], 'baseline', null, 'Spline baseline vs VIBES'],
  [['next_airspec_reproduction', 'airspec_port'], 'ml', 'protocols', 'Protocols and transfer'],
  [['next_addis_independent_validation'], 'experimental', 'tor', 'Adama TOR study'],
  [['decision_workbook_20261022'], 'ml', 'decision', 'Decision workbook'],
  [['experiments_20261022'], 'experimental', 'attribution', 'Attribution grid'],
  [['experiments_20261022'], 'experimental', 'analogs', 'Loading-matched analogs'],
  [['experiments_20261022'], 'experimental', 'selector', 'Error-predicting selector'],
  [['experiments_20261022'], 'experimental', 'forward', 'Earlier → later years'],
  [['addis_feb_season_sensitivity_20261028'], 'experimental', 'fixes', 'February calendar sensitivity'],
  [['addis_reference_closure_20261110'], 'research', 'hips-linearise', 'Linearising HIPS at Addis'],
  [['season_composition_20261103'], 'experimental', 'season', 'Season or composition?'],
  [['vibes_newton_refit_20261103'], 'experimental', 'vibes-newton', 'VIBES solved to convergence'],
  [['transfer_diagnosis_20261102'], 'experimental', 'continuum', 'Continuum and the baseline'],
  [['ann_followup_20261009'], 'meeting', 'ann-20261009', '9 October meeting'],
  [['meeting_followup_20260917'], 'meeting', 'followup', '17 September follow-up'],
]
for (const [ids, tab, page, label] of chartGroups) for (const id of ids) records.get(id)?.charts.push({ tab, page, label })
const experimentalFiles = { attribution: 'attribution', analogs: 'analogs', selector: 'selector', forward: 'forward', fixes: 'fixes', season: 'season_composition', 'vibes-newton': 'vibes_newton', continuum: 'continuum_baseline', tor: 'tor' }
// Describe primary exports, not the indexer's local data snapshot. The reader's
// shared data loader checks these on the active host when a record is selected.
for (const r of records.values()) for (const chart of r.charts) {
  const files = chart.tab === 'baseline' ? ['baseline_paired_r70.json']
    : chart.tab === 'experimental' && experimentalFiles[chart.page] ? [`experimental/${experimentalFiles[chart.page]}.json`]
      : chart.tab === 'ml' && chart.page ? [`ml/${chart.page}.json`] : []
  if (files.length) chart.primaryExports = files
}

const result = {
  schema: 2,
  coverage: 'Top-level FTIR/HIPS and phase-3 notebooks, the retrospective’s release and audit records, and its experiment contracts. Archived notebooks and other research folders are outside this index.',
  chronology: 'Dates show the last committed source change, not when an experiment ran. Undated records remain visible. Saved outputs are historical evidence, not a new reproduction or independent validation.',
  families: [...families.families.map(({ id, title, purpose }) => ({ id, title, purpose })), { id: 'recent', title: 'Later investigations', purpose: 'Later source records not yet assigned to a retrospective family.' }],
  records: [...records.values()].sort((a, b) => (b.sourceDate ?? '').localeCompare(a.sourceDate ?? '') || a.id.localeCompare(b.id)),
  edges: [...edges.values()],
}
const output = JSON.stringify(result, null, 2) + '\n'
if (process.argv.includes('--check')) {
  if (!fs.existsSync(OUT) || fs.readFileSync(OUT, 'utf8') !== output) { console.error('Investigation history is out of date; run npm run index:investigations'); process.exitCode = 1 }
} else {
  fs.writeFileSync(OUT, output)
  console.log(`Indexed ${result.records.length} research records and ${result.edges.length} connections.`)
}
