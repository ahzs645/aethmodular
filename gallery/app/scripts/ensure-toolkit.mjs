#!/usr/bin/env node
/**
 * The maps and scrub fields come from @pgmaps/geo-toolkit, pinned as the git submodule
 * gallery/packages/geo-toolkit and linked by package.json (file:../packages/geo-toolkit). The link
 * resolves to the toolkit's compiled dist/, and its dependencies to its own node_modules, so both
 * must exist before Vite runs. npm install / npm ci run this as postinstall; run it by hand after
 * moving the submodule pointer:
 *
 *   node scripts/ensure-toolkit.mjs           install + build when dist/ is missing or older than src/
 *   node scripts/ensure-toolkit.mjs --force   always rebuild
 */
import { execSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const KIT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../packages/geo-toolkit')
const force = process.argv.includes('--force')

if (!fs.existsSync(path.join(KIT, 'package.json'))) {
  console.error(`geo-toolkit submodule is missing at ${path.relative(process.cwd(), KIT)}.\n`
    + 'Run: git submodule update --init gallery/packages/geo-toolkit')
  process.exit(1)
}

/** newest mtime under a directory (files only) */
function newest(dir) {
  let t = 0
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name)
    t = Math.max(t, e.isDirectory() ? newest(p) : fs.statSync(p).mtimeMs)
  }
  return t
}

const run = (cmd) => execSync(cmd, { cwd: KIT, stdio: 'inherit' })
const dist = path.join(KIT, 'dist')
if (force || !fs.existsSync(path.join(KIT, 'node_modules'))) run('npm ci --no-audit --no-fund')
if (force || !fs.existsSync(path.join(dist, 'styles.css')) || newest(path.join(KIT, 'src')) > newest(dist)) run('npm run build')
else console.log('geo-toolkit: dist is current')
