// Run the original package's deck builder after the checked import.
// A fresh output name is required; the reviewed deck is never overwritten.
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const repo = path.resolve(__dirname, '../../..');
const source = path.join(repo, 'deliverables/ann_weekly_2026-09-17/bundle/presentation_source/build_deck.js');
const output = process.argv[2];
if (!fs.existsSync(source)) {
  console.error('Import the approved meeting ZIP with import_ann_weekly_20260917.py first.');
  process.exit(1);
}
if (!output || path.extname(output).toLowerCase() !== '.pptx') {
  console.error('Usage: node research/ftir_hips_chem/workflows/build_ann_weekly_20260917.cjs <new-output.pptx>');
  process.exit(1);
}
const absoluteOutput = path.resolve(output);
if (fs.existsSync(absoluteOutput)) {
  console.error('Refusing to overwrite an existing presentation: ' + absoluteOutput);
  process.exit(1);
}
fs.mkdirSync(path.dirname(absoluteOutput), {recursive:true});
const child = spawnSync(process.execPath, [source], {
  cwd:repo, env:{...process.env, DECK_OUT:absoluteOutput}, stdio:'inherit'
});
if (child.error) { console.error(child.error.message); process.exit(1); }
process.exit(child.status === null ? 1 : child.status);
