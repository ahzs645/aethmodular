#!/usr/bin/env node
/** Offline build guard: changing the dependency also requires reviewing the gallery adapter. */
import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'

const app = fileURLToPath(new URL('..', import.meta.url))
const read = p => JSON.parse(fs.readFileSync(path.join(app, p), 'utf8'))
const contract = read('src/lib/graphicWalkerContract.json')
const manifest = read('package.json')
const lock = read('package-lock.json')
const require = createRequire(new URL('../package.json', import.meta.url))
const entry = require.resolve(contract.package)
let dir = path.dirname(entry), installed
while (!installed) {
  const candidate = path.join(dir, 'package.json')
  if (fs.existsSync(candidate)) {
    const pkg = JSON.parse(fs.readFileSync(candidate, 'utf8'))
    if (pkg.name === contract.package) installed = pkg
  }
  if (installed) break
  const parent = path.dirname(dir)
  if (parent === dir) throw new Error(`Cannot locate the installed manifest for ${contract.package}.`)
  dir = parent
}
const versions = {
  reviewed: contract.testedVersion,
  declared: manifest.dependencies[contract.package],
  locked: lock.packages[`node_modules/${contract.package}`]?.version,
  installed: installed.version,
}
const errors = Object.entries(versions).filter(([, version]) => version !== versions.reviewed)
  .map(([source, version]) => `${source}: ${version ?? 'missing'}; reviewed adapter: ${versions.reviewed}`)
if (!contract.readableDraftVersions?.includes(versions.reviewed)) errors.push('The reviewed version must be in readableDraftVersions so newly exported drafts can reopen.')
for (const version of contract.readableDraftVersions ?? []) {
  if (!fs.existsSync(path.join(app, `scripts/fixtures/graphic-walker-${version}.json`))) errors.push(`Missing saved-chart compatibility fixture for ${version}.`)
}
try {
  execFileSync('npm', ['ls', contract.package, 'react', 'react-dom', 'styled-components', '--all', '--json'], { cwd: app, stdio: 'pipe' })
} catch {
  errors.push('Peer dependency check failed. Run npm ls @kanaries/graphic-walker react react-dom styled-components --all.')
}
if (errors.length) {
  console.error(`Graphic Walker compatibility check failed:\n${errors.map(e => `  ${e}`).join('\n')}\n`
    + 'Review the adapter and draft compatibility, update graphicWalkerContract.json with the exact version, then run the type, unit and browser checks in docs/graphic-walker-integration.md.')
  process.exitCode = 1
} else {
  console.log(`Graphic Walker ${installed.version}: manifest, lockfile, reviewed adapter and installed peers agree.`)
}
