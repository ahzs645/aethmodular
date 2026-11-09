import test from 'node:test'
import { execFileSync } from 'node:child_process'

// Regenerating the catalog reads the research notebooks and their git history, so this check is declared as a
// research-scope test in scripts/test-inputs.json: the Pages deploy (a gallery-only, depth-1 checkout) leaves it
// out instead of failing on the missing research/ folders, and the full-history investigation-history workflow
// runs it by name.
test('the checked-in catalog can be regenerated without executing research', () => {
  execFileSync(process.execPath, ['scripts/build-investigation-history.mjs', '--check'], { cwd: new URL('../../', import.meta.url), stdio: 'pipe' })
})
