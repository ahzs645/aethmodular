// The part-by-part HIPS page (pages/HipsPartsPage.tsx) borrows figures from other meeting pages by id. An id that no
// longer exists there would only show as an error panel in the browser, so check every borrowed id against its spec.
// Source-only: reads the page and the meeting specs, no data.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8')
const page = read('../pages/HipsPartsPage.tsx')

// const CLOSURE = 'closure-20261110' → { CLOSURE: 'closure-20261110' }
const bundles = Object.fromEntries([...page.matchAll(/^const ([A-Z]+) = '([a-z]+(?:-[a-z]+)*-\d{8})'$/gm)].map((m) => [m[1], m[2]]))

test('every figure the part-by-part page borrows is declared on the meeting page it is borrowed from', () => {
  const borrows = [...page.matchAll(/<Borrow from=\{([A-Z]+)\} figures=\{\[(.*?)\]\}/gs)]
  assert.ok(borrows.length >= 8, 'the page borrows from the 10 and 11 Nov meeting pages')
  let checked = 0
  for (const [, name, list] of borrows) {
    const bundle = bundles[name]
    assert.ok(bundle, `${name} names a meeting page`)
    const spec = read(`../meetings/${bundle}.meeting.ts`)
    // the borrowed ids: bare strings, and the id of each { id, about } entry (an `about` key is bundle text, not a figure)
    const ids = [...list.matchAll(/(?:^|[\[,{]\s*)(?:id:\s*)?'([a-z0-9]+-[a-z0-9-]+)'/g)].map((m) => m[1])
    assert.ok(ids.length, `figures listed for ${name}`)
    for (const id of ids) {
      assert.ok(spec.includes(`id: '${id}'`), `${id} is declared in meetings/${bundle}.meeting.ts`)
      checked++
    }
  }
  assert.ok(checked >= 20)
})

test('the bundle text shown under a borrowed figure is a section text of that meeting page', () => {
  for (const [, name, list] of page.matchAll(/<Borrow from=\{([A-Z]+)\} figures=\{\[(.*?)\]\}/gs)) {
    const spec = read(`../meetings/${bundles[name]}.meeting.ts`)
    for (const [, key] of list.matchAll(/about:\s*'([\w-]+)'/g)) assert.ok(new RegExp(`@${key}(?![\\w-])`).test(spec), `@${key} is a text of ${bundles[name]}`)
  }
})
