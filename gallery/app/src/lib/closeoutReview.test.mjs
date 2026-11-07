import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { closeoutReview } from './closeoutReview.ts'
import { decodeSection, encodeSection, itemHref, sameSection } from './sections.ts'

test('closeout review shares six resolvable figures and preserves comparator settings and evidence limits', async () => {
  const section = closeoutReview('2026-11-04T00:00:00Z', 'EPO rerun pending')
  const reopened = await decodeSection(await encodeSection(section))
  assert.ok(sameSection(section, reopened))
  assert.equal(reopened.items.length, 6)
  assert.equal(new Set(reopened.items.map(item => item.chart)).size, 6)
  // Experimental is deliberately unlisted, so these frames are resolved on
  // their page rather than registered in the public search catalog.
  const page = readFileSync(new URL('../pages/experimental/ContinuumStudy.tsx', import.meta.url), 'utf8')
  const chartIds = new Set([...page.matchAll(/<ChartFrame\b[^>]*\bid="([^"]+)"/g)].map(match => match[1]))
  for (const item of reopened.items) {
    assert.ok(chartIds.has(item.chart), `${item.chart} must resolve on its pinned experimental page`)
    assert.equal(item.view.h.tab, 'experimental')
    assert.equal(item.view.h.xp, 'continuum')
  }
  const last = reopened.items.at(-1)
  assert.equal(last.view.q['xc-addis.x'], 'aeth_ir')
  assert.equal(last.view.q['xc-addis.before'], 'vibes k18')
  assert.equal(last.view.q['xc-addis.after'], 'vibes_anchorhp k18')
  assert.match(last.note, /absolute Addis EC accuracy remains unresolved/)
  assert.match(reopened.items.find(item => item.chart === 'xc-source').note, /EPO source scores remain withheld/)
  assert.match(reopened.about, /not a frozen data package/)
  assert.match(reopened.about, /2026-11-04T00:00:00Z/)
  globalThis.window ??= { location: { pathname: '/gallery/' } }
  const href = new URL(itemHref(last, true), 'http://localhost')
  assert.equal(href.searchParams.get('chart'), 'xc-addis')
  assert.equal(href.searchParams.get('xc-addis.x'), 'aeth_ir')
})

test('a verified source review includes corrected EPO scores without promoting target accuracy', async () => {
  const section = closeoutReview('2026-11-07T12:00:00Z', 'corrected retrospective evaluation', true)
  const reopened = await decodeSection(await encodeSection(section))
  const note = reopened.items.find(item => item.chart === 'xc-source').note
  assert.match(note, /Verified fold-safe EPO source scores are included/)
  assert.match(note, /common physical-filter comparison/)
  assert.match(note, /does not establish Addis EC accuracy/)
  assert.doesNotMatch(note, /remain withheld/)
  assert.match(reopened.items.at(-1).note, /absolute Addis EC accuracy remains unresolved/)
  // A status label alone must not turn a legacy review into a verified one.
  const unverified = closeoutReview('2026-11-07T12:00:00Z', 'corrected retrospective evaluation')
  assert.match(unverified.items.find(item => item.chart === 'xc-source').note, /remain withheld/)
})
