import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {importRuntimeModule} from '/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations/container_tools/runtime_helpers.mjs';
import {resolvePresentationFont,applyPresentationChartFont,finalizePresentation} from '/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations/container_tools/artifact_tool_utils.mjs';
const {PresentationFile,FileBlob}=await importRuntimeModule('@oai/artifact-tool');
const workspaceDir=path.join(process.env.AETHMODULAR_REPO_ROOT??process.cwd(),'deliverables/ann_weekly_2026-09-10');
const build=path.join(workspaceDir,'.build/visual_revision');
const source=path.join(workspaceDir,'output/ann_weekly_2026-09-10.pptx');
const skill='/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const content=JSON.parse(await fs.readFile(path.join(build,'content.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const sourceHash=createHash('sha256').update(await fs.readFile(source)).digest('hex');
if(p.slides.items.length!==12)throw new Error('Expected the original twelve-slide source');
const family=resolvePresentationFont({fontFamily:'Arial'});
const ink='#17384A',muted='#536574';
const chartOwners=[];

function text(s,value,x,y,w,h,size=26,color=ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return t;
}
function chart(s,b){
 if(b.grouping==='stacked')b.series.forEach((series,i)=>{
  series.dataLabelOverrides=series.values.map((_,idx)=>({idx,showValue:true,position:'center',
   textStyle:{typeface:family,fontSize:26,fill:i===0?'#FFFFFF':ink,bold:true}}));
 });
 const c=s.charts.add(b.type,{position:{left:b.x,top:b.y,width:b.w,height:b.h},
  categories:b.categories,series:b.series,hasLegend:b.legend??false,
  ...(b.type==='bar'?{barOptions:{direction:'column',grouping:b.grouping??'clustered',gapWidth:b.gapWidth??75,overlap:b.grouping==='stacked'?100:0}}:{}),
  ...(b.type==='scatter'?{scatterOptions:{style:b.scatterStyle??'lineWithMarkers'}}:{}),
  xAxis:b.xAxis??{textStyle:{typeface:family,fontSize:22,fill:ink},majorGridlines:null},
  yAxis:b.yAxis??{textStyle:{typeface:family,fontSize:22,fill:ink},min:0,majorGridlines:{fill:'#E3E9EE',width:1}},
  legend:{position:'bottom',overlay:false,textStyle:{typeface:family,fontSize:20,fill:ink}},
  dataLabels:b.dataLabels??{showValue:b.type==='bar',position:'outEnd',textStyle:{typeface:family,fontSize:24,fill:ink}},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
 applyPresentationChartFont(c,{fontFamily:family});return c;
}
for(const [i,spec] of content.slides.entries()){
 const s=p.resolve(spec.anchor);
 if(s!==p.slides.items[i])throw new Error(`Source slide ${i+1} differs from inspected anchor`);
 if(spec.preserve!=='all'){
  s.shapes.deleteAll();
  for(const t of [...s.tables.items])s.tables.deleteById(t.id);
  if(spec.preserve!=='charts')for(const c of [...s.charts.items])s.charts.deleteById(c.id);
  text(s,spec.title,62,42,1150,86,43,ink,true);
  if(spec.subtitle)text(s,spec.subtitle,65,134,1140,62,25,muted);
  for(const b of spec.blocks??[]){
   if(b.kind==='text')text(s,b.text,b.x,b.y,b.w,b.h,b.size,b.color,b.bold);
   if(b.kind==='chart')chart(s,b);
  }
  if(spec.footnote)text(s,spec.footnote,65,638,1100,55,20,muted);
  text(s,String(i+1),1190,660,40,30,18,muted);
 }
 if(s.charts.items.length)chartOwners.push(i+1);
 s.speakerNotes.textFrame.setText(spec.notes);
}

const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.mkdir(path.join(build,'rendered'),{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,'rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
// Allow inspection before the final package is written.
if(process.env.AETHMODULAR_VISUAL_DRAFT_ONLY!=='1'){
 const name=process.env.AETHMODULAR_WEEKLY_DECK_NAME??'ann_weekly_2026-09-10_visual.pptx';
 if(path.basename(name)!==name||!name.endsWith('.pptx'))throw new Error('Expected a PPTX filename');
 const finalPath=path.join(workspaceDir,'output',name);
 const retained=path.join(build,'candidate-with-original-workbooks.pptx');
 execFileSync('/Users/ahmadjalil/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  [path.join(import.meta.dirname,'preserve_ann_visual_appendix.py'),source,candidate,retained],{stdio:'inherit'});
 const receipt=await finalizePresentation({workspaceDir,candidatePath:retained,finalPath,
  pythonExecutable:'/Users/ahmadjalil/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:chartOwners,
  requiredEmbeddedWorkbookChartOwnerSlides:[10,11,12],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[family],referencePath:source,referenceSha256:sourceHash},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
 console.log(JSON.stringify({finalPath,sha256:receipt.finalSha256,charts:chartOwners}));
 const verified=await PresentationFile.importPptx(await FileBlob.load(finalPath));
 await fs.mkdir(path.join(build,'final-rendered'),{recursive:true});
 for(let i=0;i<verified.slides.items.length;i++){
  const png=await verified.export({slide:verified.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,'final-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 }
}
