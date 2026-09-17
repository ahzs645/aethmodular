// Append notebook figures; user explicitly requested images, not native charts.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {importRuntimeModule} from '/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations/container_tools/runtime_helpers.mjs';
import {resolvePresentationFont,finalizePresentation} from '/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations/container_tools/artifact_tool_utils.mjs';
const {PresentationFile,FileBlob}=await importRuntimeModule('@oai/artifact-tool');
const root=process.env.AETHMODULAR_REPO_ROOT??process.cwd();
const workspaceDir=path.join(root,'deliverables/ann_weekly_2026-09-10');
const build=path.join(workspaceDir,'.build/diagnostics');
const source=path.join(workspaceDir,'output/ann_weekly_2026-09-10_notebook_plots.pptx');
const skill='/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const specs=JSON.parse(await fs.readFile(path.join(build,'content.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const family=resolvePresentationFont({fontFamily:'Arial'});
const ink='#17384A',muted='#536574';
function text(s,value,x,y,w,h,size=26,color=ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};
}
for(const spec of specs){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,spec.title,62,42,1150,94,42,ink,true);
 s.images.add({blob:new Uint8Array(await fs.readFile(spec.image)),contentType:'image/png',
  alt:spec.title,fit:'contain',position:{left:60,top:160,width:1160,height:470}});
 text(s,spec.footer,65,644,1100,48,20,muted);
 text(s,String(spec.number),1190,660,40,30,18,muted);
 s.speakerNotes.textFrame.setText(spec.notes);
}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const name=process.env.AETHMODULAR_WEEKLY_DECK_NAME??'ann_weekly_2026-09-10_expanded.pptx';
if(path.basename(name)!==name||!name.endsWith('.pptx'))throw new Error('Expected PPTX filename');
const finalPath=path.join(workspaceDir,'output',name);
const receipt=await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath,
 pythonExecutable:'/Users/ahmadjalil/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:[family],referencePath:source,
  referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:path.join(build,`${name}.validation.json`)});
console.log(JSON.stringify({finalPath,sha256:receipt.finalSha256}));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(path.join(build,'final-rendered'),{recursive:true});
for(let i=0;i<final.slides.items.length;i++){
 const png=await final.export({slide:final.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,'final-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
