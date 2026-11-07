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
  assert.match(reopened.about, /not a frozen data package/)
  assert.match(reopened.about, /2026-11-04T00:00:00Z/)
  globalThis.window ??= { location: { pathname: '/gallery/' } }
  const href = new URL(itemHref(last, true), 'http://localhost')
  assert.equal(href.searchParams.get('chart'), 'xc-addis')
  assert.equal(href.searchParams.get('xc-addis.x'), 'aeth_ir')
})
