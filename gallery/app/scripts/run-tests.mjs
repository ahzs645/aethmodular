/** Run data-independent checks separately from scientific snapshot reconciliation.
 * Missing inputs and skipped assertions never count as a successful release check.
 */
import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const APP = fileURLToPath(new URL('../', import.meta.url))
const ROOT = path.resolve(APP, '../..')
const MANIFEST = JSON.parse(fs.readFileSync(new URL('test-inputs.json', import.meta.url), 'utf8'))
const SUITES = ['source', 'data-public', 'data-research', 'all']

function walk(dir) {
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => {
    const file = path.join(dir, e.name)
    return e.isDirectory() ? walk(file) : e.isFile() && e.name.endsWith('.test.mjs') ? [file] : []
  })
}

export function testFiles(app = APP) {
  return ['src/lib', 'src/charts', 'src/sections', 'scripts']
    .flatMap(dir => walk(path.join(app, dir)))
    .map(file => path.relative(app, file).split(path.sep).join('/')).sort()
}

function inputFiles(root, rel) {
  if (typeof rel !== 'string' || path.isAbsolute(rel) || rel.split('/').includes('..')) {
    throw new Error(`test input must be repository-relative: ${rel}`)
  }
  if (!rel.includes('*')) return fs.existsSync(path.join(root, rel)) && fs.statSync(path.join(root, rel)).isFile() ? [rel] : []
  const folder = path.posix.dirname(rel), base = path.posix.basename(rel)
  if (folder.includes('*')) throw new Error(`only a final-component wildcard is supported: ${rel}`)
  if (!fs.existsSync(path.join(root, folder))) return []
  const pattern = new RegExp(`^${base.split('*').map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`)
  return fs.readdirSync(path.join(root, folder), { withFileTypes: true })
    .filter(e => e.isFile() && pattern.test(e.name)).map(e => `${folder}/${e.name}`).sort()
}

export function planTests({ app = APP, root = ROOT, manifest = MANIFEST, suite = 'all' } = {}) {
  if (!SUITES.includes(suite)) throw new Error(`unknown test suite ${suite}; choose ${SUITES.join(', ')}`)
  const files = testFiles(app), known = new Set(files), entries = manifest.tests
  if (manifest.schema_version !== 1 || !entries) throw new Error('unsupported test-inputs manifest')
  for (const [file, entry] of Object.entries(entries)) {
    if (!known.has(file)) throw new Error(`test-inputs.json names a missing test file: ${file}`)
    if (!['public', 'research'].includes(entry.scope) || !entry.inputs?.length || !entry.reason) {
      throw new Error(`incomplete data-test declaration: ${file}`)
    }
  }
  // A new data reader must be classified deliberately, even when its current
  // checkout happens to have the export and would otherwise pass as source-only.
  for (const file of files.filter(f => f.startsWith('src/') && !entries[f])) {
    const code = fs.readFileSync(path.join(app, file), 'utf8')
    if (/(?:readFile|readdir|statSync)/.test(code) && /(?:public\/data\/|data\/spartan\/|\.\.\/research\/.*(?:tables|output))/.test(code)) {
      throw new Error(`declare data inputs for ${file} in scripts/test-inputs.json`)
    }
  }
  const sectionsFile = path.join(app, 'src/sections/sections.json')
  const privateDirs = fs.existsSync(sectionsFile) ? Object.values(JSON.parse(fs.readFileSync(sectionsFile, 'utf8')).sections)
    .filter(section => !section.public).flatMap(section => section.data ?? []) : []
  const prefix = 'gallery/app/public/data/'
  for (const [file, entry] of Object.entries(entries).filter(([, e]) => e.scope === 'public')) {
    for (const input of entry.inputs) {
      if (!input.startsWith(prefix) || privateDirs.some(dir => input.slice(prefix.length).startsWith(`${dir.replace(/\/$/, '')}/`))) {
        throw new Error(`public test ${file} depends on a research-only input: ${input}`)
      }
    }
  }
  const selected = files.filter(file => suite === 'all' || (suite === 'source' ? !entries[file] : entries[file]?.scope === suite.slice(5)))
  const ready = [], blocked = [], inputs = new Set()
  for (const file of selected) {
    const missing = []
    for (const input of entries[file]?.inputs ?? []) {
      const found = inputFiles(root, input)
      if (!found.length) missing.push(input)
      found.forEach(p => inputs.add(p))
    }
    if (missing.length) blocked.push({ file, missing })
    else ready.push(file)
  }
  return { suite, selected, ready, blocked, inputs: [...inputs].sort(), omitted: files.filter(file => !selected.includes(file)) }
}

const fingerprint = (root, file) => {
  const body = fs.readFileSync(path.join(root, file))
  return { path: file, bytes: body.length, sha256: createHash('sha256').update(body).digest('hex') }
}

export function runSuite({ app = APP, root = ROOT, manifest = MANIFEST, suite = 'all', receipt, list = false, quiet = false } = {}) {
  const plan = planTests({ app, root, manifest, suite })
  const say = message => { if (!quiet) process.stdout.write(`${message}\n`) }
  say(`${suite}: ${plan.selected.length} test files selected, ${plan.ready.length} ready, ${plan.blocked.length} unavailable; ${plan.omitted.length} outside this scope.`)
  for (const { file, missing } of plan.blocked) say(`UNAVAILABLE ${file}\n${missing.map(input => `  missing: ${input}`).join('\n')}`)
  if (list) {
    for (const file of plan.ready) say(`READY ${file}`)
    return { code: plan.blocked.length ? 1 : 0, ...plan }
  }
  const started = new Date().toISOString()
  const testIdentity = plan.selected.map(file => fingerprint(app, file))
  const inputIdentity = plan.inputs.map(file => fingerprint(root, file))
  const command = ['--test', '--test-reporter=tap', '--test-concurrency=4', ...plan.ready]
  // The gate's own regression tests launch a real child runner. Do not inherit
  // Node's internal child-test protocol, which suppresses the requested TAP output.
  const env = { ...process.env }
  delete env.NODE_TEST_CONTEXT
  const child = plan.ready.length ? spawnSync(process.execPath, command, { cwd: app, env, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }) : null
  if (!quiet && child) {
    process.stdout.write(child.stdout ?? '')
    process.stderr.write(child.stderr ?? '')
    if (child.error) process.stderr.write(`${child.error.message}\n`)
  }
  const counts = {}
  for (const match of (child?.stdout ?? '').matchAll(/^# (tests|pass|fail|cancelled|skipped|todo) (\d+)\s*$/gm)) counts[match[1]] = Number(match[2])
  const failed = child && (child.status !== 0 || child.error || !('tests' in counts))
  const unavailable = plan.blocked.length > 0 || (counts.skipped ?? 0) > 0 || (counts.todo ?? 0) > 0 || !plan.selected.length
  const changed = inputIdentity.filter(input => !fs.existsSync(path.join(root, input.path)) || fingerprint(root, input.path).sha256 !== input.sha256).map(input => input.path)
  const result = {
    schema_version: 1, started, finished: new Date().toISOString(), suite,
    status: failed || changed.length ? 'failed' : unavailable ? 'unavailable' : 'passed',
    node: process.version, command: ['node', ...command], counts,
    test_files: testIdentity, input_files: inputIdentity, changed_inputs_during_check: changed,
    unavailable_test_files: plan.blocked, omitted_test_files: plan.omitted,
    test_process_exit: child?.status ?? null, test_process_error: child?.error?.message ?? null,
    note: 'A scoped check is not scientific qualification of omitted analyses. Missing inputs, skips and TODOs are not passes.',
  }
  if (receipt) {
    fs.mkdirSync(path.dirname(receipt), { recursive: true })
    fs.writeFileSync(receipt, `${JSON.stringify(result, null, 2)}\n`)
  }
  if (counts.skipped || counts.todo) say(`UNAVAILABLE: ${counts.skipped ?? 0} skipped and ${counts.todo ?? 0} TODO assertions. Restore their inputs; this is not a passing data check.`)
  if (changed.length) say(`FAILED: inputs changed during verification: ${changed.join(', ')}`)
  say(`${suite}: ${result.status.toUpperCase()}`)
  return { code: result.status === 'passed' ? 0 : 1, ...result }
}

function main(argv) {
  let suite = 'all', receipt, list = false
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--suite') suite = argv[++i]
    else if (argv[i] === '--receipt') receipt = path.resolve(argv[++i])
    else if (argv[i] === '--list') list = true
    else throw new Error(`unknown argument ${argv[i]}`)
  }
  receipt ??= path.join(ROOT, `research/ftir_hips_chem/output/tables/gallery_rebuild/node-${suite}.json`)
  return runSuite({ suite, receipt, list }).code
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { process.exitCode = main(process.argv.slice(2)) }
  catch (error) { console.error(error.message); process.exitCode = 2 }
}
