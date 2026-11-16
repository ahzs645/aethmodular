// The "Which VIBES setting" meeting page (meetings/vibes-settings-20261116.meeting.ts) draws rings on the
// cross-validation curve at the k the selection rule picks. The rings are written in the spec, the picks are data
// (table "rule" of the page's bundle), so check they agree, and that every table the spec names is in the bundle.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const spec = fs.readFileSync(new URL('../meetings/vibes-settings-20261116.meeting.ts', import.meta.url), 'utf8')
const bundle = JSON.parse(fs.readFileSync(new URL('../../public/data/meetings/vibes-settings-20261116.json', import.meta.url), 'utf8'))

test('the rings on the cross-validation curve are the k the saved rule picked', () => {
  const picks = Object.fromEntries([...spec.matchAll(/^ {2}(whole|qc): \{(.*?)\},$/gm)].map(([, key, body]) =>
    [key, Object.fromEntries([...body.matchAll(/'(tau [\d.]+)': (\d+)/g)].map(([, tau, k]) => [tau, Number(k)]))]))
  const rule = bundle.tables.rule
  assert.deepEqual(Object.keys(picks), ['whole', 'qc'])
  for (const [key, row] of [['whole', rule.training.findIndex((t) => t.startsWith('Whole'))], ['qc', rule.training.findIndex((t) => t.startsWith('Without'))]]) {
    assert.ok(row >= 0, `the rule table has the ${key} row`)
    for (const tau of ['tau 0.05', 'tau 0.1', 'tau 0.2']) assert.equal(picks[key][tau], rule[tau][row], `${key}, ${tau}`)
  }
})

test('every table and text key the page names is in its bundle', () => {
  const tables = [...spec.matchAll(/table: '([\w]+)'/g)].map((m) => m[1])
  assert.ok(tables.length >= 15)
  for (const name of tables) assert.ok(bundle.tables[name], `table "${name}"`)
  for (const [, key] of spec.matchAll(/'@([\w-]+)'/g)) assert.ok(bundle.text[key], `text "${key}"`)
})
