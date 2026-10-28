import { readFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const src = (p: string) => fileURLToPath(new URL(`./src/${p}`, import.meta.url))

/**
 * VITE_PUBLIC_BUILD=1 (the GitHub Pages deploy, build_share.py) leaves out every section that
 * src/sections/sections.json marks private ("public": false): its import resolves to src/sections/omitted.ts,
 * so none of its code is compiled. An "unlisted" section is compiled (App hides it from the menu). Any other
 * build includes every section.
 */
function omittedSections(mode: string): string[] {
  if (loadEnv(mode, process.cwd(), 'VITE_').VITE_PUBLIC_BUILD !== '1' && process.env.VITE_PUBLIC_BUILD !== '1') return []
  const manifest = JSON.parse(readFileSync(src('sections/sections.json'), 'utf8')) as { sections: Record<string, { public: boolean | 'unlisted' }> }
  return Object.entries(manifest.sections).filter(([, s]) => !s.public).map(([key]) => key)
}

export default defineConfig(({ mode }) => {
  const omitted = omittedSections(mode)
  if (omitted.length) console.log(`public build: leaving out section(s) ${omitted.join(', ')}`)
  return {
    plugins: [react()],
    resolve: {
      alias: [
        ...omitted.map((key) => ({ find: new RegExp(`^@/sections/${key}$`), replacement: src('sections/omitted.ts') })),
        { find: '@', replacement: src('') },
      ],
      // the toolkit is linked from gallery/packages/geo-toolkit, whose own node_modules hold dev copies
      // of these: one React and one MapLibre for the whole page
      dedupe: ['react', 'react-dom', 'maplibre-gl'],
    },
    server: { port: 5178, open: true },
    build: { outDir: 'dist' },
  }
})
