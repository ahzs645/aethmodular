// Append the measured audit to a copy of the Ann weekly deck, before its appendix.
// The caller provides the bundled presentation runtime through the environment.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const repo = process.env.AETHMODULAR_REPO_ROOT ?? process.cwd();
const skill = process.env.AETH_PRESENTATIONS_SKILL_DIR;
const python = process.env.AETH_RUNTIME_PYTHON;
if (!skill || !python) throw new Error('Set AETH_PRESENTATIONS_SKILL_DIR and AETH_RUNTIME_PYTHON');
const {importRuntimeModule} = await import(pathToFileURL(path.join(skill, 'container_tools/runtime_helpers.mjs')));
const {FileBlob, PresentationFile} = await importRuntimeModule('@oai/artifact-tool');
const {finalizePresentation} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const workspaceDir = path.join(repo, 'deliverables/ann_weekly_audit_2026-09-10');
const buildDir = path.join(workspaceDir, '.build');
const data = JSON.parse(await fs.readFile(path.join(buildDir, 'audit-slide-data.json'), 'utf8'));
const sourceBytes = await fs.readFile(data.source_pptx);
if (crypto.createHash('sha256').update(sourceBytes).digest('hex') !== data.source_sha256) {
  throw new Error('Source deck changed. Inspect the new version before adding slides.');
}
const p = await PresentationFile.importPptx(await FileBlob.load(data.source_pptx));
if (p.slides.items.length !== 12) throw new Error('Expected the inspected 12-slide source');
const originalSlides = [...p.slides.items];
const snapshot = await p.inspect({kind:'slide,textbox,table,chart', maxChars:45000});
const records = snapshot.ndjson.split('\n').filter(Boolean).map(line => JSON.parse(line));
const chartOwners = records.filter(x => x.kind === 'chart').map(x => x.slide > 9 ? x.slide + 4 : x.slide);
const tableOwners = records.filter(x => x.kind === 'table').map(x => x.slide > 9 ? x.slide + 4 : x.slide);
const ink = '#17384A', muted = '#536574', blue = '#246A9B', family = 'Arial';

function text(slide, value, x, y, width, height, size=29, color=ink, bold=false) {
  const shape = slide.shapes.add({geometry:'textbox', position:{left:x, top:y, width, height},
    fill:'none', line:{fill:'none', width:0}});
  shape.text = value;
  shape.text.style = {typeface:family, fontSize:size, color, bold, autoFit:'none'};
  return shape;
}

function table(slide, values, widths, y=227, height=316, size=25, total=false) {
  const t = slide.tables.add({rows:values.length, columns:values[0].length,
    left:66, top:y, width:1148, height, values, columnWidths:widths});
  for (let r=0; r<values.length; r++) {
    for (let c=0; c<values[0].length; c++) {
      const cell = t.getCell(r,c);
      cell.fill = r === 0 ? '#E8F0F5' : '#FFFFFF';
      cell.text.style = {typeface:family, fontSize:size, color:ink,
        bold:r === 0 || (total && r === values.length-1)};
    }
  }
  t.borders.assign({style:'solid', fill:'#D4DFE7', width:0.5});
  return t;
}

const sourceNotes = (keys, explanation) => [explanation, '', 'Sources:', ...keys.map(k=>data.sources[k])].join('\n');
const specs = [
  {
    title:'546 filters have both HIPS and FTIR EC',
    subtitle:'Audit of the legacy four-site dataset. Separate from the 239-filter calibration cohort.',
    table:{values:[['Site','HIPS/FTIR\npairs','HIPS + IR\ndate candidate','HIPS + reported\ncollection window'], ...data.sample_rows],
      widths:[240,270,305,333], y:218, height:325, total:true},
    body:'347 date-window candidates still need matching to actual sampling intervals.',
    bodyY:560, bodySize:27,
    footnote:'Counts precede exclusions. One registered exclusion leaves 545 HIPS/FTIR diagnostic pairs.',
    notes:sourceNotes(['selection','audit_report'],
      'The catalog retains 1,055 site/base FilterId pairs and all 44,493 original measurement rows. '
      +'The three numeric columns describe separate overlapping subsets, not sequential attrition. '
      +'The 317 HIPS filters with a positive-duration reported collection window comprise 187 Addis and 130 JPL filters. '
      +'Only current Addis/JPL portal files were available in the inspected directory. '
      +'A zero for Beijing/Delhi means no window recovered from these inputs, not proof that metadata do not exist. '
      +'Same-filter HIPS/FTIR agreement is a measurement diagnostic, not independent validation of FTIR-predicted EC.'),
  },
  {
    title:'The saved daily windows need rebuilding',
    subtitle:'The historical “9 AM” builder created 15:00 boundaries for three sites.',
    table:{values:[['Site','Saved local timestamp','Coverage limitation'],
      ['Beijing','15:00','Input availability only'],
      ['Delhi','15:00','Input availability only'],
      ['JPL','Mostly 15:00','One day reaches 104.0%'],
      ['Addis Ababa','09:00','No coverage field']],
      widths:[240,390,518], y:220, height:315},
    body:'The source now uses local 09:00 bounds and the actual elapsed minutes.',
    bodyY:558, bodySize:27,
    footnote:'Saved aggregates await regeneration. Filter metadata must determine the final matching intervals.',
    notes:sourceNotes(['inventory','audit_report'],
      'Beijing has 590 timestamps at 15:00 and Delhi has 289. JPL has 762 at 15:00, two at 14:00 and one at 16:00. '
      +'All three saved inputs also contain duplicate datetime_local columns. Addis has 515 timestamps at 09:00. '
      +'The old builder subtracted 15 hours, aggregated midnight bins and added 15 hours. '
      +'The revised source uses local [start,end) calendar intervals labelled at their end. '
      +'It counts distinct valid minutes per channel and allows 23/25-hour DST days. '
      +'Without upstream observation flags, non-null input availability still cannot establish observed coverage. '
      +'The saved pickles have not been regenerated or relabelled.'),
  },
  {
    title:'Collection can span several days',
    subtitle:`Portal exports restored ${data.windows} positive-duration windows and flagged ${data.timing_issues} timing issues.`,
    table:{values:[['Example filter','Collection dates\n(local time)','Elapsed\nhours','Sampled\nhours'],
      [data.examples[0].id,'7–15 Aug 2024',data.examples[0].elapsed_h,data.examples[0].sampled_h],
      [data.examples[1].id,'6–16 Mar 2024',data.examples[1].elapsed_h,data.examples[1].sampled_h],
      [data.examples[2].id,'24 Feb 2024, same start/end',data.examples[2].elapsed_h,data.examples[2].sampled_h]],
      widths:[280,448,210,210], y:226, height:290},
    body:'The average must represent the hours when the sampler actually operated.',
    bodyY:552, bodySize:28,
    footnote:'Sampled hours do not identify active periods. ETAD-0243 also has a one-day start-date disagreement.',
    notes:sourceNotes(['metadata','audit_report'],
      'Seven Addis and sixteen JPL portal records have sampled hours inconsistent with the elapsed collection window or invalid timing. '
      +'Five Addis windows span 192–216 hours with 24 sampled hours. Fifteen JPL windows span 237–238 hours with 48 sampled hours. '
      +'ETAD-0163 separately reports 22.8 sampled hours over 24 hours. USPA-0243 reports 24 sampled hours over 23 hours. '
      +'The zero-length ETAD-0178 record remains flagged and does not count among the 452 positive-duration windows. '
      +'ETAD-0243 starts on 2024-08-24 in the portal but has SampleDate 2024-08-25 in the unified table. '
      +'Hours agreeing within 0.05 h establish metadata consistency only, not an observed on/off log.\n'
      + JSON.stringify(data.examples, null, 2)),
  },
  {
    title:'EC provenance and the next comparison',
    subtitle:`${data.ec_conflicts} ChemSpec EC groups contain conflicting values under the same parameter name.`,
    blocks:[
      {y:232,h:94,text:'ChemSpec EC repeats the FTIR EC product.\nIndependent EC validation requires a separate reference.',size:29},
      {y:359,h:90,text:'The audit preserves each original value and source row.\nThe conflicts still need tracing to their original parameter definitions.',size:29},
      {y:495,h:106,text:'Next milestone: verified sampling intervals and optical conversions,\nfollowed by withheld-period checks of any proposed adjustment.',size:29,color:blue},
    ],
    footnote:'This section audits legacy four-site inputs. The seasonal calibration results use a separate cohort.',
    notes:sourceNotes(['provenance','audit_report','selection'],
      'The 500 conflicting ChemSpec EC groups are Beijing 156, Addis 175, Delhi 27 and JPL 142. '
      +'For example, CHTS-0658 has 0.93 and 0.06 micrograms per cubic metre under the same parameter name. '
      +'The audit neither averages conflicting values nor chooses the first record. '
      +'Existing repo provenance work identifies ChemSpec EC concentration as FTIR EC and ChemSpec BC as HIPS Fabs/10, subject to rounding. '
      +'Before an optical comparison, verify the instrument absorption/eBC conversion history, wavelength, particle size and reference conditions. '
      +'Keep each physical filter in a single validation group and assess adjustments in withheld calendar blocks. '
      +'Missing inputs still include Beijing/Delhi collection metadata in the inspected source set and active sampling logs for intermittent windows. '
      +'This audit does not re-estimate the separate seasonal calibration shown earlier in the presentation.'),
  },
];

for (const [i,spec] of specs.entries()) {
  const slide = p.slides.add();
  slide.background.fill = '#FFFFFF';
  text(slide,spec.title,62,42,1150,86,43,ink,true);
  text(slide,spec.subtitle,65,134,1140,68,25,muted);
  if (spec.table) {
    const b=spec.table;
    table(slide,b.values,b.widths,b.y,b.height,25,b.total);
    tableOwners.push(10+i);
  }
  if (spec.body) text(slide,spec.body,75,spec.bodyY,1110,62,spec.bodySize);
  for (const b of spec.blocks??[]) text(slide,b.text,75,b.y,1110,b.h,b.size,b.color??ink);
  text(slide,spec.footnote,65,632,1100,60,20,muted);
  text(slide,String(10+i),1190,660,40,30,18,muted);
  slide.speakerNotes.textFrame.setText(spec.notes);
  slide.moveTo(9+i);
  if (p.slides.items[9+i].id !== slide.id) throw new Error('Unexpected slide insertion semantics');
}

// Preserve original slides and change only the appendix page numbers.
for (const record of records.filter(x => x.kind==='textbox' && x.slide>=10 &&
    x.bbox?.[0]===1190 && x.bbox?.[1]===660 && x.text===String(x.slide))) {
  p.resolve(record.id).text.replace(String(record.slide),String(record.slide+4));
}
for (let i=0;i<originalSlides.length;i++) {
  if (p.slides.items[i<9?i:i+4].id !== originalSlides[i].id) throw new Error('Original slide order changed');
}
await fs.mkdir(path.join(buildDir,'draft-rendered'),{recursive:true});
for (let i=9;i<13;i++) {
  const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(buildDir,'draft-rendered',`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidatePath=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
// Imported chart bindings do not survive the current Artifact Tool exporter.
// Restore the existing source chart/workbook parts, preserving exact values,
// formulas and lineage. Do not synthesize replacement workbook snapshots.
const preservation = execFileSync(python,[
  path.join(repo,'research/ftir_hips_chem/workflows/preserve_ann_chart_parts_20260910.py'),
  data.source_pptx,candidatePath],{encoding:'utf8'});
await fs.writeFile(path.join(buildDir,'chart-preservation.json'),preservation);
const finalPath=path.join(workspaceDir,'output',process.env.AETH_AUDIT_DECK_NAME??'ann_weekly_with_audit.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:16,
  requiredNativeTableOwnerSlides:[...new Set(tableOwners)],
  requiredNativeChartOwnerSlides:chartOwners,
  requiredEmbeddedWorkbookChartOwnerSlides:chartOwners,
  materializeLiteralChartWorkbooks:false,
  fontPolicy:{basis:'reference',families:[family],referencePath:data.source_pptx,referenceSha256:data.source_sha256},
  verifyArtifactToolImport:true,receiptPath:path.join(buildDir,`${path.basename(finalPath)}.validation.json`),
});
console.log(JSON.stringify(result));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(path.join(buildDir,'final-rendered'),{recursive:true});
for (let i=0;i<final.slides.items.length;i++) {
  const png=await final.export({slide:final.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(buildDir,'final-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const montage=await final.export({format:'webp',montage:true,scale:0.5});
await fs.writeFile(path.join(buildDir,'final-montage.webp'),new Uint8Array(await montage.arrayBuffer()));
