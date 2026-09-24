// The user explicitly requested notebook PNGs instead of editable charts.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {skill,pythonExecutable,runtimeHelpers,artifactToolUtils} from './codex_presentations_runtime.mjs';
const {importRuntimeModule}=await runtimeHelpers();
const {resolvePresentationFont,finalizePresentation}=await artifactToolUtils();
const {PresentationFile,FileBlob}=await importRuntimeModule('@oai/artifact-tool');
const root=process.env.AETHMODULAR_REPO_ROOT??process.cwd();
const workspaceDir=path.join(root,'deliverables/ann_weekly_2026-09-10');
const build=path.join(workspaceDir,'.build/notebook_revision');
await fs.mkdir(build,{recursive:true});
const source=path.join(workspaceDir,'output/ann_weekly_2026-09-10.pptx');
const content=JSON.parse(await fs.readFile(path.join(workspaceDir,'.build/visual_revision/content.json'),'utf8'));
const figures=path.join(root,'research/ftir_hips_chem/output/plots/ann_weekly_20260910_notebook');
const notebook=path.join(root,'research/ftir_hips_chem/notebooks/archive/executed/ann_weekly_20260910_figures.ipynb');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const family=resolvePresentationFont({fontFamily:'Arial'});
const ink='#17384A',muted='#536574';
function text(s,value,x,y,w,h,size=26,color=ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};
}
for(const [i,spec] of content.slides.entries()){
 const s=p.resolve(spec.anchor);
 let note=spec.notes;
 if(i>0){
  s.shapes.deleteAll();
  for(const t of [...s.tables.items])s.tables.deleteById(t.id);
  for(const c of [...s.charts.items])s.charts.deleteById(c.id);
  text(s,spec.title,62,42,1150,86,43,ink,true);
  const subtitle=i<9?spec.subtitle:null;
  if(subtitle)text(s,subtitle,65,132,1140,45,25,muted);
  const imagePath=path.join(figures,`slide_${String(i+1).padStart(2,'0')}.png`);
  s.images.add({blob:new Uint8Array(await fs.readFile(imagePath)),contentType:'image/png',
   alt:spec.title,fit:'contain',position:{left:60,top:subtitle?186:154,width:1160,height:subtitle?448:480}});
  let footnote=spec.footnote;
  if(i===7)footnote='Dots: estimates. Bars: 95% CIs. Fixed calibration, MAC and λ = 3.34.';
  if(i>=9){
   footnote='Thin lines: individual spectra. Thick lines: medians. Shading: excluded from selection.';
   note=note.replace('Thin lines show the minimum and maximum at each wavenumber, and thick lines show the median.',
                     'Thin lines show every individual spectrum, and the thick line in each panel shows the median.');
   note=note.replace('This is a range summary, not a confidence interval. The report supplies every individual spectral trace.',
                     'Every individual training and Addis trace is shown on shared axes. These are spectra, not confidence intervals.');
  }
  if(i===1)note=note.replace('The displayed gold trace starts at 3500','The gold shading starts at 3500');
  if(i===7)note=note.replace('These horizontal intervals show the uncertainty around the seasonal Deming estimates.',
                            'The dots show the seasonal Deming estimates and the horizontal error bars show their 95% confidence intervals.');
  if(footnote)text(s,footnote,65,644,1100,48,20,muted);
  text(s,String(i+1),1190,660,40,30,18,muted);
  note+=`\nFigure format: PNG exported from the Jupyter notebook, as requested; regenerate in the notebook to edit the graph.\n${notebook}\n${imagePath}`;
 }
 s.speakerNotes.textFrame.setText(note);
}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const name=process.env.AETHMODULAR_WEEKLY_DECK_NAME??'ann_weekly_2026-09-10_notebook_plots.pptx';
if(path.basename(name)!==name||!name.endsWith('.pptx'))throw new Error('Expected a PPTX filename');
const finalPath=path.join(workspaceDir,'output',name);
// The user's explicit notebook-image request supersedes native-chart ownership.
// Retain package/layout/font/import checks; no native-chart claim is made.
const receipt=await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath,
 pythonExecutable,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:[family],referencePath:source,
  referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify({finalPath,sha256:receipt.finalSha256}));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(path.join(build,'final-rendered'),{recursive:true});
for(let i=0;i<final.slides.items.length;i++){
 const png=await final.export({slide:final.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,'final-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
