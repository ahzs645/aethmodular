import test from 'node:test'
import assert from 'node:assert/strict'
import { paletteSeason, seasonColorRows, seasonExportColor } from './exportSeasons.ts'
import { DEFAULT_EXPORT_COLORS, readExportColors, recommendExportColors } from './exportColors.ts'

const season = (site, label = 'Cold (Apr–Sep)', months = [4,5,6,7,8,9], color = '#6f86ba') => paletteSeason({name: `${site} · ${label}`, site, months, color})
const santiago = season('Santiago, Chile'), talca = season('Talca, Chile')
test('only equal season names and month memberships consolidate across sites', () => {
  assert.equal(santiago.group, talca.group)
  assert.equal(season('X', 'Cold (Apr-Sep)', [9,8,7,6,5,4]).group, santiago.group)
  assert.notEqual(season('X', 'Cold (Apr–Sep)', [4,5,6]).group, santiago.group)
  assert.notEqual(season('X', 'Wet (Apr–Sep)').group, santiago.group)
  assert.equal(seasonColorRows([santiago,talca], false).length, 2)
  assert.deepEqual(seasonColorRows([santiago,talca], true).map(s => s.label), ['Cold (Apr–Sep)'])
})
test('consolidated colors share an override while independent choices survive toggling', () => {
  const seasons = [santiago,talca], prefs = {...DEFAULT_EXPORT_COLORS, seasons:{[santiago.key]:'#123456',[talca.key]:'#abcdef'}}
  assert.equal(seasonExportColor(talca,prefs,seasons), '#abcdef')
  assert.equal(seasonExportColor(talca,{...prefs,consolidateSeasons:true},seasons), '#123456')
  assert.equal(seasonExportColor(talca,{...prefs,consolidateSeasons:true,seasons:{...prefs.seasons,[talca.group]:'#ff0000'}},seasons), '#ff0000')
  assert.deepEqual(readExportColors({consolidateSeasons:true,seasons:{[santiago.group]:'#abc',bad:'url(unsafe)'}}).seasons, {[santiago.group]:'#aabbcc'})
})
test('recommended season colors follow consolidation and remain separate from enabled fits', () => {
  const palette = {sites:[],seasons:[santiago,talca,season('Santiago, Chile','Warm (Oct–Mar)',[10,11,12,1,2,3])],fits:[{key:'deming',label:'Deming',color:'#000000',mixed:false}]}
  const separate = recommendExportColors(palette,false,false), combined = recommendExportColors(palette,false,true)
  assert.equal(Object.keys(separate.seasons).length,3)
  assert.equal(Object.keys(combined.seasons).length,2)
  assert.equal(new Set([...Object.values(combined.seasons),...Object.values(combined.fits)]).size,3)
})
