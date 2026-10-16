#!/usr/bin/env node
/**
 * The ⌘K search's index (src/chartIndex.json): every chart title in the gallery, the tab (and
 * meeting page) it sits on, and the component that draws it, read from the source so a new
 * ChartFrame is searchable without being registered. A file's tab comes from where it lives
 * (FILE_TABS), else from the nearest file that imports it.
 *
 *   node scripts/build-chart-index.mjs          write src/chartIndex.json
 *   node scripts/build-chart-index.mjs --check  exit 1 if it is out of date (npm test runs the same check)
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src')
const OUT = path.join(SRC, 'chartIndex.json')

/** where a file is drawn: a tab, or 'meeting:<key>' for a meeting page; first match wins */
const FILE_TABS = [
  [/^charts\/(correlation|distribution|evolution|ranking|partOfWhole|map|flow|pmf|calibration|weather)\//, (m) => m[1]],
  [/^pages\/WeatherPage/, () => 'weather'],
  [/^pages\/CalibrationPage/, () => 'calibration'],
  [/^pages\/Baseline|^components\/FullRangeVibes/, () => 'baseline'],
  [/^pages\/SimilarityPage|^components\/SimilaritySpectra/, () => 'similarity'],
  [/^pages\/MeetingFollowupPage/, () => 'meeting:followup'],
  [/^pages\/SatoshiFollowupPage|^components\/SatoshiSpectra/, () => 'meeting:satoshi'],
  [/^pages\/BlanksPage/, () => 'blanks'],
  [/^pages\/CensusPage/, () => 'census'],
  [/^meetings\/(.+)\.meeting\.ts$/, (m) => `meeting:${meetingId(m.input)}`],
]

const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
  const p = path.join(dir, e.name)
  return e.isDirectory() ? walk(p) : /\.(tsx|ts)$/.test(e.name) && !e.name.endsWith('.d.ts') ? [p] : []
})
const rel = (p) => path.relative(SRC, p).split(path.sep).join('/')
const read = (r) => fs.readFileSync(path.join(SRC, r), 'utf8')
function meetingId(r) { return read(r).match(/\bid:\s*'([^']+)'/)?.[1] ?? path.basename(r, '.meeting.ts') }

/** a title as written: a string, or a template literal with its ${…} parts shown as "…" */
function literal(s) {
  const m = s.match(/^\s*(?:"([^"]*)"|'((?:[^'\\]|\\.)*)'|\{\s*(?:'((?:[^'\\]|\\.)*)'|"([^"]*)"|`((?:[^`\\]|\\.)*)`)\s*\}|`((?:[^`\\]|\\.)*)`)/)
  if (!m) return null
  const t = m.slice(1).find((x) => x !== undefined)
  return t.replace(/\$\{[^}]*\}/g, '…').replace(/\\(.)/g, '$1').replace(/\s+/g, ' ').trim()
}

/** the first string inside a {…} expression (a ternary, a `?? 'fallback'`), or the literal a variable was set to */
const FIRST_STRING = /"([^"]+)"|'((?:[^'\\]|\\.)+)'|`((?:[^`\\]|\\.)+)`/
/** a template literal's text from its opening backtick, every ${…} (nested templates included) as "…" */
function templateText(s, i) {
  let out = ''
  for (i += 1; i < s.length && s[i] !== '`'; i++) {
    if (s[i] === '\\') { out += s[++i] ?? ''; continue }
    if (s[i] === '$' && s[i + 1] === '{') {
      let depth = 0
      for (i += 1; i < s.length; i++) {
        if (s[i] === '`') { i = skipTemplate(s, i); continue }
        if (s[i] === '{') depth++
        else if (s[i] === '}' && --depth === 0) break
      }
      out += '…'
      continue
    }
    out += s[i]
  }
  return out.replace(/\s+/g, ' ').trim()
}
function skipTemplate(s, i) {
  for (i += 1; i < s.length && s[i] !== '`'; i++) {
    if (s[i] === '\\') i++
    else if (s[i] === '$' && s[i + 1] === '{') { let d = 0; for (i += 1; i < s.length; i++) { if (s[i] === '`') { i = skipTemplate(s, i); continue } if (s[i] === '{') d++; else if (s[i] === '}' && --d === 0) break } }
  }
  return i
}

function titleOf(s, before) {
  const t = s.match(/^\s*\{?\s*`/)
  if (t) return templateText(s, t[0].length - 1)
  const direct = literal(s)
  if (direct) return direct
  if (!s.startsWith('{')) return null
  let depth = 0, i = 0
  for (; i < s.length; i++) { if (s[i] === '{') depth++; else if (s[i] === '}' && --depth === 0) break }
  const expr = s.slice(1, i).trim()
  const inner = expr.match(FIRST_STRING)
  if (inner) return literal(inner[0])
  if (/^\w+$/.test(expr)) {
    // a default parameter (title = '…'), else the last const / let before the chart
    const param = [...before.matchAll(new RegExp(`[({,]\\s*${expr}\\s*=\\s*(['"][\\s\\S]{0,300})`, 'g'))].pop()
    const pl = param?.[1].match(FIRST_STRING)
    if (pl) return literal(pl[0])
    const set = [...before.matchAll(new RegExp(`\\b(?:const|let)\\s+${expr}\\s*=\\s*([\\s\\S]{0,400})`, 'g'))].pop()
    const lit = set?.[1].match(FIRST_STRING)
    if (lit) return literal(lit[0])
  }
  return null
}

function chartsIn(r) {
  const src = read(r)
  const out = []
  // <ChartFrame … title=… id=…>
  for (const m of src.matchAll(/<ChartFrame\b/g)) {
    const tail = src.slice(m.index, m.index + 3000)
    const end = tail.search(/\n\s*>\s*\n|\/?>\s*\n\s*</)
    const head = end > 0 ? tail.slice(0, end + 2) : tail
    const tm = head.match(/\btitle=/)
    const title = tm ? titleOf(head.slice(tm.index + 6), src.slice(0, m.index)) : null
    if (!title) continue
    const id = head.match(/\bid="([^"]+)"/)?.[1]
    const before = src.slice(0, m.index)
    const comp = [...before.matchAll(/function\s+([A-Z]\w*)\s*[(<]/g)].pop()?.[1]
    out.push({ title, ...(id ? { id } : {}), ...(comp ? { component: comp } : {}) })
  }
  // declared meeting pages: figures' `title: '…'` (src/meetings/*.meeting.ts)
  if (r.endsWith('.meeting.ts')) {
    for (const m of src.matchAll(/\bkind:\s*'(\w+)'[^\n]*?\bid:\s*'([^']+)'[\s\S]{0,600}?\btitle:\s*('(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g)) {
      out.push({ title: literal(m[3]), id: m[2], component: m[1] })
    }
  }
  return out
}

export function buildIndex() {
  const files = walk(SRC).map(rel)
  const importers = new Map()
  for (const f of files) {
    for (const m of read(f).matchAll(/from\s+'(@\/|\.\.?\/)([^']+)'/g)) {
      const target = m[1] === '@/' ? m[2] : path.posix.normalize(path.posix.join(path.posix.dirname(f), m[1] + m[2]))
      const hit = files.find((x) => x === target || x === `${target}.tsx` || x === `${target}.ts`)
      if (hit) importers.set(hit, [...(importers.get(hit) ?? []), f])
    }
  }
  const direct = (f) => { for (const [re, fn] of FILE_TABS) { const m = f.match(re); if (m) return fn(m) } return null }
  // a shared component takes the place(s) of the nearest files that import it
  const placesOf = (f, seen = new Set()) => {
    const d = direct(f)
    if (d) return [d]
    if (seen.has(f)) return []
    seen.add(f)
    return [...new Set((importers.get(f) ?? []).flatMap((g) => placesOf(g, seen)))]
  }
  const charts = []
  for (const f of files.sort()) {
    const found = chartsIn(f)
    if (!found.length) continue
    const places = placesOf(f)
    for (const c of found) for (const p of places.length ? places : ['']) {
      const [tab, meeting] = p.startsWith('meeting:') ? ['meeting', p.slice(8)] : [p, undefined]
      if (!tab) continue
      charts.push({ ...c, tab, ...(meeting ? { meeting } : {}), file: f })
    }
  }
  return { note: 'Generated by scripts/build-chart-index.mjs; do not edit by hand.', charts }
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isMain) {
  const next = JSON.stringify(buildIndex(), null, 1) + '\n'
  if (process.argv.includes('--check')) {
    const cur = fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8') : ''
    if (cur !== next) { console.error('src/chartIndex.json is out of date: run npm run index:charts'); process.exit(1) }
  } else {
    fs.writeFileSync(OUT, next)
    console.log(`wrote src/chartIndex.json (${JSON.parse(next).charts.length} charts)`)
  }
}
