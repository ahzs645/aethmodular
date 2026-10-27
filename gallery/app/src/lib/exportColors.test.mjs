import test from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_EXPORT_COLORS, effectiveExportColors, contrastRatio, hexColor, paintColor, readExportColors, recommendExportColors, replacePaint, siteExportColor } from './exportColors.ts'

const palette = {
  seasons: [],
  sites: Array.from({ length: 40 }, (_, i) => ({ key: `S${i}`, label: `Site ${i}`, color: '#2171b5' })),
  fits: ['ols', 'deming', 'york'].map(key => ({ key, label: key, color: '#2171b5', mixed: false })),
}
test('saved colors reject malformed values and accept valid hex without letting CSS through', () => {
  assert.equal(hexColor(' #AbC '), '#aabbcc')
  assert.equal(hexColor('var(--color)'), null)
  assert.deepEqual(readExportColors({ sameSites: 'yes', shared: '#fff', sites: { A: '#123456', B: 'red' }, fits: { deming: '#789', other: '#fff' } }),
    { mode: 'custom', consolidateSeasons: false, seasons: {}, sameSites: false, shared: '#ffffff', sites: { A: '#123456' }, fits: { deming: '#778899' } })
  assert.deepEqual(readExportColors(null), DEFAULT_EXPORT_COLORS)
})
test('site overrides, shared mode and reset resolve independently of fit colors', () => {
  const prefs = { ...DEFAULT_EXPORT_COLORS, sites: { S0: '#ff0000' }, fits: { deming: '#00ff00' } }
  assert.equal(siteExportColor(palette.sites[0], prefs), '#ff0000')
  assert.equal(siteExportColor(palette.sites[1], prefs), '#2171b5')
  assert.equal(siteExportColor(palette.sites[0], { ...prefs, sameSites: true, shared: '#222222' }), '#222222')
  assert.equal(siteExportColor(palette.sites[0], DEFAULT_EXPORT_COLORS), '#2171b5')
})
test('paint substitution preserves opacity, transparent marks and unrelated colors', () => {
  assert.equal(paintColor('rgb(33, 113, 181)'), '#2171b5')
  assert.equal(replacePaint('rgba(33, 113, 181, 0.25)', '#2171b5', '#cc0000'), 'rgba(204, 0, 0, 0.25)')
  assert.equal(replacePaint('#fff', '#2171b5', '#cc0000'), '#fff')
  assert.equal(replacePaint('transparent', '#000000', '#cc0000'), 'transparent')
  assert.equal(replacePaint('none', 'none', '#cc0000'), 'none')
})
test('recommendations give 40 unique site colors and separate readable colors for all enabled fits', () => {
  const prefs = recommendExportColors(palette, false)
  const siteColors = Object.values(prefs.sites), fits = Object.values(prefs.fits)
  assert.equal(new Set(siteColors).size, 40)
  assert.equal(new Set([...siteColors, ...fits]).size, 43)
  assert.ok(siteColors.every(c => contrastRatio(c) >= 3))
  assert.ok(fits.every(c => contrastRatio(c) >= 4.5))
  assert.deepEqual(recommendExportColors(palette, false), prefs)
  const shared = recommendExportColors(palette, true)
  assert.deepEqual(shared.sites, {})
  assert.ok(contrastRatio(shared.shared) >= 3)
  assert.ok(!Object.values(shared.fits).includes(shared.shared))
})

test('standard colors bypass saved custom colors without losing them, including after storage reload', () => {
  const custom = readExportColors({sameSites:true, shared:'#123456', sites:{S0:'#abcdef'}, seasons:{winter:'#234567'}, fits:{deming:'#345678'}, consolidateSeasons:true})
  assert.equal(custom.mode, 'custom') // migrate preferences saved before scheme choices existed
  const standard = readExportColors({...custom, mode:'standard'})
  const effective = effectiveExportColors(standard)
  assert.deepEqual(effective.sites,{})
  assert.deepEqual(effective.seasons,{})
  assert.deepEqual(effective.fits,{})
  assert.equal(effective.sameSites,false)
  assert.equal(effective.consolidateSeasons,true)
  assert.equal(standard.sites.S0,'#abcdef')
  assert.deepEqual(effectiveExportColors({...standard,mode:'custom'}),custom)
  assert.equal(readExportColors({mode:'unknown'}).mode,'standard')
})

test('a dark-page export keeps site and season colours, even when a dark ramp step shares the hex', async () => {
  const { registerPrintColours } = await import('./theme.ts')
  const { fixedColours, printPaint } = await import('./exportColors.ts')
  // a dark colour-scheme stop that is exactly Beijing's red, as lib/colourSchemes registers one
  registerPrintColours(['#e74c3c'], ['#df4532'])
  const fixed = fixedColours([{ code: 'CHTS', name: 'Beijing', color: '#E74C3C' }], [{ color: 'rgb(52, 152, 219)' }])
  assert.deepEqual([...fixed], ['#e74c3c', '#3498db'])
  assert.equal(printPaint('#E74C3C', fixed), null)
  assert.equal(printPaint('rgb(231, 76, 60)', fixed), null, 'a computed HTML swatch colour too')
  assert.equal(printPaint('#e74c3c'), '#df4532', 'the same hex on a heat-map cell still prints light')
  assert.equal(printPaint('#e3e7ec', fixed), '#1f2933', 'INK dark text still prints light')
  assert.equal(printPaint('none', fixed), null)
})
