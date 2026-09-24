import fs from 'node:fs/promises';
import path from 'node:path';
import {skill,pythonExecutable,runtimeHelpers,artifactToolUtils} from './codex_presentations_runtime.mjs';
const {importRuntimeModule}=await runtimeHelpers();
const {Presentation, PresentationFile, FileBlob}=await importRuntimeModule('@oai/artifact-tool');
const {resolvePresentationFont, applyPresentationChartFont, finalizePresentation}=await artifactToolUtils();
const workspaceDir=path.join(process.env.AETHMODULAR_REPO_ROOT??process.cwd(),'deliverables/ann_weekly_2026-09-10');
const content=JSON.parse(await fs.readFile(path.join(workspaceDir,'.build/content.json'),'utf8'));
const family=resolvePresentationFont({fontFamily:'Arial'});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#17384A', blue='#246A9B', muted='#536574';
const chartOwners=[],tableOwners=[];
function text(s,value,x,y,w,h,size=26,color=ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return t;
}
function table(s,b){
 const data=b.values;const t=s.tables.add({rows:data.length,columns:data[0].length,left:b.x??66,top:b.y??210,width:b.w??1148,height:b.h??300,values:data,columnWidths:b.widths});
 for(let r=0;r<data.length;r++) for(let c=0;c<data[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#E8F0F5':'#FFFFFF';
  cell.text.style={typeface:family,fontSize:b.fontSize??25,color:ink,bold:r===0};
 }
 t.borders.assign({style:'solid',fill:'#D4DFE7',width:0.5});
 tableOwners.push(p.slides.items.length);
}
function chart(s,b){
 const c=s.charts.add(b.type,{position:{left:b.x??72,top:b.y??200,width:b.w??1100,height:b.h??365},
  categories:b.categories,series:b.series,hasLegend:b.legend??false,
  ...(b.type==='bar'?{barOptions:{direction:'column',grouping:'clustered',gapWidth:65}}:{}),
  ...(b.type==='scatter'?{scatterOptions:{style:b.scatterStyle??'line'}}:{}),
  xAxis:b.xAxis??{textStyle:{typeface:family,fontSize:22,fill:ink},majorGridlines:null},
  yAxis:b.yAxis??{textStyle:{typeface:family,fontSize:22,fill:ink},min:0,majorGridlines:{fill:'#E3E9EE',width:1}},
  legend:{position:'bottom',overlay:false,textStyle:{typeface:family,fontSize:20,fill:ink}},
  dataLabels:b.dataLabels??{showValue:b.type==='bar',position:'outEnd',textStyle:{typeface:family,fontSize:24,fill:ink}},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
 applyPresentationChartFont(c,{fontFamily:family});chartOwners.push(p.slides.items.length);return c;
}
for(const [i,spec] of content.slides.entries()){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 if(spec.cover){
  text(s,spec.title,75,175,1080,145,58,ink,true);
  text(s,spec.subtitle,78,350,1070,130,31,muted);
  text(s,'Ahmad Jalil\n10 September 2026',78,557,800,90,26,blue);
 }else{
  text(s,spec.title,62,42,1150,86,43,ink,true);
  if(spec.subtitle)text(s,spec.subtitle,65,134,1140,68,25,muted);
  for(const b of spec.blocks??[]){
   if(b.kind==='text')text(s,b.text,b.x??70,b.y??230,b.w??1110,b.h??250,b.size??29,b.color??ink,b.bold??false);
   if(b.kind==='table')table(s,b);
   if(b.kind==='chart')chart(s,b);
  }
  if(spec.footnote)text(s,spec.footnote,65,632,1100,60,20,muted);
  text(s,String(i+1),1190,660,40,30,18,muted);
 }
 s.speakerNotes.textFrame.setText(spec.notes??'');
}
const candidate=path.join(workspaceDir,'.build/candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.mkdir(path.join(workspaceDir,'.build/rendered'),{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.items[i];
 const image=await p.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(workspaceDir,'.build/rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));
}
const deckName=process.env.AETHMODULAR_WEEKLY_DECK_NAME??'ann_weekly_2026-09-10.pptx';
if(path.basename(deckName)!==deckName || !deckName.endsWith('.pptx')) throw new Error('Deck name must be a .pptx filename');
const out=path.join(workspaceDir,'output',deckName);
const result=await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath:out,
 pythonExecutable,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
  ...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:content.slides.length,
 requiredNativeTableOwnerSlides:[...new Set(tableOwners)],requiredNativeChartOwnerSlides:[...new Set(chartOwners)],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,
 receiptPath:path.join(workspaceDir,'.build/validation.json')});
console.log(JSON.stringify(result));

// Render the exact validated final package for visual review.
const verified=await PresentationFile.importPptx(await FileBlob.load(out));
await fs.mkdir(path.join(workspaceDir,'.build/final-rendered'),{recursive:true});
for(let i=0;i<verified.slides.items.length;i++){
 const png=await verified.export({slide:verified.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(workspaceDir,'.build/final-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
