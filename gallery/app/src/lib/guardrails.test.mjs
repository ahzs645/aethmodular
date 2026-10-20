import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { FormulaError, evalFormula, parseFormula } from './formula.ts'
import { TEMPLATES, decodeGuardrail, describeGuardrail, encodeGuardrail, evaluateGuardrail, formulaName, guardFields, guardRows, templatesFor } from './guardrails.ts'
import { DEFAULT_HIGHLIGHT_LIMITS, highlightMatches } from './specification.ts'

const EXPORTS = ['public/data/spec_curve.json', 'public/data/meeting/spec_curve_20260917.json', 'public/data/meeting-r70/spec_curve_20260917.json']
const numeric = (src, formulas = []) => guardFields(src, formulas).filter((f) => f.kind === 'number').map((f) => f.name)

test('formulas parse to arithmetic over known fields only', () => {
  const f = parseFormula('abs(intercept) * 2 - -slope ^ 2', ['intercept', 'slope'])
  assert.equal(evalFormula(f, { intercept: -0.5, slope: 3 }), 1 + 9)
  assert.equal(evalFormula(parseFormula('max(a, b, 3) / 2', ['a', 'b']), { a: 1, b: 8 }), 4)
  assert.equal(evalFormula(parseFormula('a + 1', ['a']), { a: null }), null, 'a missing input makes the result missing')
  assert.equal(evalFormula(parseFormula('1 / a', ['a']), { a: 0 }), null, 'a non-finite result is missing')
  assert.throws(() => parseFormula('constructor', ['a']), FormulaError)
  assert.throws(() => parseFormula('abs(a, a)', ['a']), /abs takes 1 argument/)
  assert.throws(() => parseFormula('a +', ['a']), /ends too soon/)
  assert.throws(() => parseFormula('a; alert(1)', ['a']), /unexpected ';'/)
})

test('calculated field names are identifiers that do not collide', () => {
  assert.equal(formulaName('Intercept at MAC 6!', ['x']), 'interceptAtMAC6')
  assert.equal(formulaName('slope', ['slope', 'slope2']), 'slope3')
  assert.equal(formulaName('9 lives', []), 'lives')
})

const tiny = {
  n: 3, ok: '010', intercept: [-0.2, 0.4, null], slope: [0.9, 1.5, 1], heldout: [null, 0.9, 0.95], cut: [300, null, 800], k: [5, 9, 12],
  categories: { co: ['ocec', 'pool'], sp: ['airspec'], sel: ['airspec'], mo: ['app', 'site_heldout'], lot: ['all'] },
  co: [0, 1, 0], sp: [0, 0, 0], sel: [0, 0, 0], mo: [0, 1, 1], lot: [0, 0, 0], extra: { qres: [1, 2, null] },
}

test('a missing value fails a comparison instead of reading as zero', () => {
  const rows = guardRows(tiny)
  const run = (query) => evaluateGuardrail({ query, formulas: [] }, rows, numeric(tiny)).pass
  assert.deepEqual(run({ combinator: 'and', rules: [{ field: 'heldout', operator: '<=', value: '0.92' }] }), [false, true, false])
  assert.deepEqual(run({ combinator: 'and', rules: [{ field: 'absIntercept', operator: '<=', value: '0.5' }] }), [true, true, false])
  assert.deepEqual(run({ combinator: 'and', rules: [{ field: 'slope', operator: 'between', value: '0.7,1.3' }] }), [true, false, true])
  assert.deepEqual(run({ combinator: 'or', rules: [{ field: 'co', operator: 'in', value: 'pool' }, { field: 'ok', operator: '=', value: true }] }), [false, true, false])
  assert.deepEqual(run({ combinator: 'and', not: true, rules: [{ field: 'mo', operator: '=', value: 'app' }] }), [false, true, true])
  assert.deepEqual(run({ combinator: 'and', rules: [{ field: 'slope', operator: '>', value: 'heldout', valueSource: 'field' }] }), [false, true, true])
  assert.deepEqual(run({ combinator: 'and', rules: [] }), [true, true, true], 'no conditions: everything passes')
})

test('calculated fields are worked out per row before the rule runs', () => {
  const rows = guardRows(tiny)
  const rule = { formulas: [{ name: 'gap', label: 'gap', formula: 'slope - heldout' }], query: { combinator: 'and', rules: [{ field: 'gap', operator: '>', value: '0.5' }] } }
  assert.deepEqual(evaluateGuardrail(rule, rows, numeric(tiny)).pass, [false, true, false])
  const bad = evaluateGuardrail({ ...rule, formulas: [{ name: 'gap', label: 'Gap', formula: 'slope -' }] }, rows, numeric(tiny))
  assert.match(bad.error, /^Gap: /)
})

test('a rule survives the URL and reads back in words', () => {
  const rule = { formulas: [{ name: 'g', label: 'g', formula: 'abs(slope - 1)' }], query: TEMPLATES[0].rule.query }
  const back = decodeGuardrail(encodeGuardrail(rule))
  assert.deepEqual(back.formulas, rule.formulas)
  assert.equal(back.query.rules.length, 5)
  assert.match(encodeGuardrail(rule), /^[A-Za-z0-9_-]+$/)
  assert.equal(decodeGuardrail('not base64!'), null)
  assert.match(describeGuardrail(TEMPLATES[1].rule, guardFields(tiny)), /\|Deming intercept\|/)
})

for (const path of EXPORTS) {
  test(`templates reproduce the saved checks on ${path}`, { skip: !existsSync(path) && 'export not generated' }, () => {
    const src = JSON.parse(readFileSync(path, 'utf8'))
    const rows = guardRows(src)
    const names = numeric(src)
    assert.ok(templatesFor(src).some((t) => t.key === 'guardrails'), 'the export carries the saved guardrails\' inputs')
    // the saved guardrails, written out as an editable rule, pass exactly the configurations the export marks
    const saved = evaluateGuardrail(TEMPLATES[0].rule, rows, names).pass
    const mismatch = saved.map((p, i) => p !== (src.ok[i] === '1')).filter(Boolean).length
    assert.equal(mismatch, 0)
    // the old fixed highlights agree row for row with their templates
    for (const t of TEMPLATES.slice(1)) {
      const pass = evaluateGuardrail(t.rule, rows, names).pass
      const want = rows.map((r, i) => highlightMatches({ intercept: src.intercept[i], slope: src.slope[i], cv: src.heldout[i], guardrails: false }, t.key, DEFAULT_HIGHLIGHT_LIMITS))
      assert.deepEqual(pass, want, t.key)
    }
  })
}

test('conditions apply in order, with what each removes, alone and shared', async () => {
  const { stepStats, ruleSteps, ruleUpTo } = await import('./guardrails.ts')
  // 5 configurations; A fails rows 0,1; B fails rows 1,2; C fails row 4 (row 3 passes all)
  const A = [false, false, true, true, true], B = [true, false, false, true, true], C = [true, true, true, true, false]
  const s = stepStats([A, B, C], [0, 1, 2, 3, 4])
  assert.deepEqual(s.fails, [2, 2, 1])
  assert.deepEqual(s.atStep, [2, 1, 1], 'B only removes row 2 once A has removed row 1')
  assert.deepEqual(s.left, [3, 2, 1])
  assert.deepEqual(s.only, [1, 1, 1], 'row 1 fails both A and B, so neither removes it alone')
  assert.deepEqual(s.shared, [[2, 1, 0], [1, 2, 0], [0, 0, 1]])
  assert.deepEqual(stepStats([A, B], [1, 3]).left, [1, 1], 'only the rows in scope count')
  const or = stepStats([A, B], [0, 1, 2, 3, 4], 'or')
  assert.deepEqual(or.atStep, [3, 1], 'with OR, steps add configurations')
  assert.deepEqual(or.left, [3, 4])
  const rule = { formulas: [], query: { combinator: 'and', rules: [{ field: 'a', operator: '>', value: '1' }, { combinator: 'or', rules: [] }, { field: 'b', operator: '<', value: '2' }] } }
  assert.equal(ruleSteps(rule).length, 3)
  assert.equal(ruleUpTo(rule, 2).query.rules.length, 2)
})
