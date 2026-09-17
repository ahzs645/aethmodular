// Replace the audit's tables/text with figures exported by its Jupyter notebook.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const root = process.env.AETHMODULAR_REPO_ROOT ?? process.cwd();
const skill = process.env.AETH_PRESENTATIONS_SKILL_DIR;
const python = process.env.AETH_RUNTIME_PYTHON;
if (!skill || !python) throw new Error('Set AETH_PRESENTATIONS_SKILL_DIR and AETH_RUNTIME_PYTHON');
const {importRuntimeModule} = await import(pathToFileURL(path.join(skill,'container_tools/runtime_helpers.mjs')));
const {FileBlob,PresentationFile} = await importRuntimeModule('@oai/artifact-tool');
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const workspaceDir = path.join(root,'deliverables/ann_weekly_audit_2026-09-10');
const build = path.join(workspaceDir,'.build/notebook_revision');
await fs.mkdir(build,{recursive:true});
const source = path.join(workspaceDir,'output/ann_weekly_with_audit.pptx');
const expectedSource = 'b1ca33f7955afe8346e976c0114aaf0315f6fe60c0117f9a8b381404ad02fb31';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
if (hash(await fs.readFile(source)) !== expectedSource) throw new Error('Reinspect the changed source deck');
const active = path.join(root,'research/ftir_hips_chem');
const notebook = path.join(active,'notebooks/archive/executed/matched_sample_audit_figures.ipynb');
const nb = JSON.parse(await fs.readFile(notebook,'utf8'));
const sections = nb.cells.filter(c=>c.cell_type==='markdown' && c.source.join('').startsWith('## ')
  && !c.source.join('').startsWith('## Source definitions')).map(c=>c.source.join(''));
const provenance = JSON.parse(await fs.readFile(path.join(active,'output/tables/matched_sample_audit_figures/figure_manifest.json'),'utf8'));
if (sections.length!==8 || provenance.figures.length!==8) throw new Error('Execute the eight-figure notebook first');
for (const f of provenance.figures) {
  if (hash(await fs.readFile(f.png)) !== f.png_sha256) throw new Error(`Figure changed after notebook execution: ${f.png}`);
}
const specs = [
  ['546 filter pairs, 347 IR date candidates','Date candidates still need verified sampling intervals. Counts precede registry exclusions.'],
  ['317 HIPS filters have collection windows','Window availability reflects the inspected sources. Active sampling periods remain unverified.'],
  ['Saved timestamps cluster at 15:00','The source code is corrected. Existing saved averages still need rebuilding.'],
  ['Saved completeness needs an observation audit','This saved field describes input availability. Observed IR coverage remains unverified.'],
  ['23 timing issues in 453 portal records','Agreement within 0.05 h checks metadata consistency. It does not establish continuous operation.'],
  ['20 filters span several days of collection','Points group identical durations. Dashed line: equal hours. All 453 records contribute.'],
  ['24 sampled hours within a 192-hour window','These bars compare totals. The actual on/off periods remain unknown.'],
  ['500 filters contain conflicting EC values','Original values remain separate. Independent EC validation needs a separate reference.'],
];
const p = await PresentationFile.importPptx(await FileBlob.load(source));
if (p.slides.items.length !== 16) throw new Error('Expected the inspected 16-slide source');
const original = [...p.slides.items];
const inspect = await p.inspect({kind:'slide,textbox,table,chart,layout',maxChars:55000});
await fs.writeFile(path.join(build,'source-inspection.ndjson'),inspect.ndjson);
await fs.writeFile(path.join(build,'source-structure.json'),JSON.stringify({
  masters:p.masters.items.map(m=>({id:m.id,name:m.name})),
  originalSlideIds:original.map(s=>s.id),
},null,2));
const records=inspect.ndjson.split('\n').filter(Boolean).map(line=>JSON.parse(line));
const chartOwners=records.filter(r=>r.kind==='chart').map(r=>r.slide>=14?r.slide+4:r.slide);
const tableOwners=records.filter(r=>r.kind==='table' && r.slide<10).map(r=>r.slide);
const family='Arial',ink='#17384A',muted='#536574';
function text(s,value,x,y,w,h,size=25,color=ink,bold=false) {
  const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},
    fill:'none',line:{fill:'none',width:0}});
  t.text=value;t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};
}
for (const [i,[title,footnote]] of specs.entries()) {
  const s=i<4?original[9+i]:p.slides.add();
  if (i>=4) s.moveTo(9+i);
  s.shapes.deleteAll();
  for (const t of [...s.tables.items]) s.tables.deleteById(t.id);
  for (const c of [...s.charts.items]) s.charts.deleteById(c.id);
  s.background.fill='#FFFFFF';
  text(s,title,62,42,1150,86,43,ink,true);
  const f=provenance.figures[i];
  s.images.add({blob:new Uint8Array(await fs.readFile(f.png)),contentType:'image/png',
    alt:title,fit:'contain',position:{left:40,top:132,width:1200,height:503}});
  text(s,footnote,65,644,1100,48,20,muted);
  text(s,String(10+i),1190,660,40,30,18,muted);
  let note=sections[i].split('\n\n').slice(1).join('\n\n');
  if(i===0) note+=' One registered Delhi exclusion leaves 545 eligible HIPS/FTIR diagnostic pairs.';
  note+='\n\nSources:\n'+path.join(root,'docs/matched-sample-audit-2026-09-10.md');
  if(i===7) note+='\n'+path.join(root,'docs/open-items.md');
  note+='\n'+notebook+'\n'+f.png;
  note+='\n'+path.join(active,'output/tables/matched_sample_audit_figures');
  s.speakerNotes.textFrame.setText(note);
  if(p.slides.items[9+i].id!==s.id)throw new Error('Audit insertion order changed');
}
for (const r of records.filter(x=>x.kind==='textbox' && x.slide>=14 && x.bbox?.[0]===1190 &&
    x.bbox?.[1]===660 && x.text===String(x.slide))) {
  p.resolve(r.id).text.replace(String(r.slide),String(r.slide+4));
}
for(let i=0;i<original.length;i++) {
  if (i>=9 && i<=12) continue;
  if(p.slides.items[i<9?i:i+4].id!==original[i].id) throw new Error('Original slide order changed');
}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const copied=execFileSync(python,[path.join(active,'workflows/preserve_ann_chart_parts_20260910.py'),
  source,candidate,JSON.stringify([[3,3],[14,18],[15,19],[16,20]])],{encoding:'utf8'});
await fs.writeFile(path.join(build,'chart-preservation.json'),copied);
const name=process.env.AETH_AUDIT_DECK_NAME??'ann_weekly_with_audit_graphs.pptx';
if(path.basename(name)!==name || !name.endsWith('.pptx'))throw new Error('Expected a PPTX filename');
const finalPath=path.join(workspaceDir,'output',name);
// The user asked for notebook-generated graphs to be added to the slides.
// Those eight figures are notebook images; existing native charts/workbooks stay intact.
const result=await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath,
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:20,requiredNativeTableOwnerSlides:tableOwners,
  requiredNativeChartOwnerSlides:chartOwners,requiredEmbeddedWorkbookChartOwnerSlides:chartOwners,
  materializeLiteralChartWorkbooks:false,
  fontPolicy:{basis:'reference',families:[family],referencePath:source,referenceSha256:expectedSource},
  verifyArtifactToolImport:true,receiptPath:path.join(build,`${name}.validation.json`),
});
console.log(JSON.stringify({finalPath,sha256:result.finalSha256}));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(path.join(build,'final-rendered'),{recursive:true});
for(let i=0;i<final.slides.items.length;i++) {
  const png=await final.export({slide:final.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,'final-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),
    new Uint8Array(await png.arrayBuffer()));
}
const montage=await final.export({format:'webp',montage:true,scale:0.6});
await fs.writeFile(path.join(build,'final-montage.webp'),new Uint8Array(await montage.arrayBuffer()));
