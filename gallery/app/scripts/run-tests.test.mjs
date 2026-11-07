import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { planTests, runSuite } from './run-tests.mjs'

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gallery-test-gate-'))
  const app = path.join(root, 'gallery/app')
  fs.mkdirSync(path.join(app, 'src/lib'), { recursive: true })
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const write = (rel, text) => {
    const file = path.join(root, rel)
    fs.mkdirSync(path.dirname(file), { recursive: true })
    fs.writeFileSync(file, text)
  }
  return { root, app, write, manifest: { schema_version: 1, tests: {} }, quiet: true }
}

test('a newly added source regression produces a failing gate, with no filename exemptions', t => {
  const f = fixture(t)
  f.write('gallery/app/src/lib/newRegression.test.mjs', "import assert from 'node:assert/strict'; assert.equal(1, 2, 'unexpected regression')")
  const result = runSuite({ ...f, suite: 'source' })
  assert.equal(result.code, 1)
  assert.equal(result.status, 'failed')
  assert.ok(result.counts.fail > 0)
})

test('missing scientific inputs are unavailable; restored inputs are verified and malformed values fail', t => {
  const f = fixture(t)
  f.write('gallery/app/src/lib/behavior.test.mjs', "import assert from 'node:assert/strict'; assert.equal(1 + 1, 2)")
  f.write('gallery/app/src/lib/reconcile.test.mjs', `
    import fs from 'node:fs'; import assert from 'node:assert/strict'
    const input = JSON.parse(fs.readFileSync(new URL('../../public/data/fixture.json', import.meta.url)))
    assert.equal(input.expected, 5)
  `)
  const input = 'gallery/app/public/data/fixture.json'
  f.manifest.tests['src/lib/reconcile.test.mjs'] = { scope: 'public', inputs: [input], reason: 'Reconcile the saved value.' }
  assert.equal(runSuite({ ...f, suite: 'source' }).status, 'passed')
  const missing = runSuite({ ...f, suite: 'data-public' })
  assert.equal(missing.code, 1)
  assert.equal(missing.status, 'unavailable')
  assert.deepEqual(missing.unavailable_test_files, [{ file: 'src/lib/reconcile.test.mjs', missing: [input] }])
  assert.deepEqual(missing.counts, {}, 'unavailable assertions are not counted as passes')
  f.write(input, '{"expected":5}')
  const restored = runSuite({ ...f, suite: 'data-public' })
  assert.equal(restored.code, 0)
  assert.equal(restored.input_files.length, 1)
  assert.match(restored.input_files[0].sha256, /^[0-9a-f]{64}$/)
  f.write(input, '{"expected":6}')
  assert.equal(runSuite({ ...f, suite: 'data-public' }).status, 'failed')
})

test('a conditional data skip is not a passing scientific check even when the file exists', t => {
  const f = fixture(t)
  f.write('gallery/app/src/lib/conditional.test.mjs', "import test from 'node:test'; test('required table absent', {skip:true}, () => {})")
  f.write('gallery/app/public/data/fixture.json', '{}')
  f.manifest.tests['src/lib/conditional.test.mjs'] = {
    scope: 'research', inputs: ['gallery/app/public/data/fixture.json'], reason: 'Needs a table in this export.',
  }
  const result = runSuite({ ...f, suite: 'data-research' })
  assert.equal(result.counts.skipped, 1)
  assert.equal(result.code, 1)
  assert.equal(result.status, 'unavailable')
})

test('new direct data readers need declared inputs, and data scopes partition the test population', t => {
  const f = fixture(t)
  const reader = "import fs from 'node:fs'; fs.readFileSync(new URL('../../public/data/fixture.json', import.meta.url))"
  f.write('gallery/app/src/lib/read.test.mjs', reader)
  assert.throws(() => planTests({ ...f, suite: 'source' }), /declare data inputs/)
  f.manifest.tests['src/lib/read.test.mjs'] = { scope: 'public', inputs: ['gallery/app/public/data/fixture.json'], reason: 'Saved export.' }
  f.write('gallery/app/src/lib/private.test.mjs', "import assert from 'node:assert/strict'; assert.equal(2, 2)")
  f.manifest.tests['src/lib/private.test.mjs'] = { scope: 'research', inputs: ['research/evidence.csv'], reason: 'Saved analysis table.' }
  f.write('gallery/app/src/lib/source.test.mjs', "import assert from 'node:assert/strict'; assert.equal(3, 3)")
  const suites = ['source', 'data-public', 'data-research'].flatMap(suite => planTests({ ...f, suite }).selected)
  assert.equal(new Set(suites).size, suites.length)
  assert.deepEqual(suites.sort(), planTests({ ...f, suite: 'all' }).selected)
})

test('an input modified by a test invalidates its receipt', t => {
  const f = fixture(t)
  f.write('gallery/app/src/lib/mutate.test.mjs', "import fs from 'node:fs'; fs.writeFileSync(new URL('../../public/data/fixture.json', import.meta.url), '{\"new\":true}')")
  const input = 'gallery/app/public/data/fixture.json'
  f.write(input, '{}')
  f.manifest.tests['src/lib/mutate.test.mjs'] = { scope: 'public', inputs: [input], reason: 'Must remain frozen during verification.' }
  const result = runSuite({ ...f, suite: 'data-public' })
  assert.equal(result.code, 1)
  assert.deepEqual(result.changed_inputs_during_check, [input])
})
