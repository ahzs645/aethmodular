/** Revise the Adama deck while retaining its source theme and slide structure. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {importRuntimeModule} from '/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations/container_tools/runtime_helpers.mjs';
import {finalizePresentation} from '/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations/container_tools/artifact_tool_utils.mjs';
const {PresentationFile,FileBlob}=await importRuntimeModule('@oai/artifact-tool');
const repo=process.env.AETHMODULAR_REPO_ROOT??process.cwd();
const workspaceDir=path.join(repo,'deliverables/adama_summary_2026-08-25');
const build=path.join(workspaceDir,'.build_20260910');
const output=path.join(workspaceDir,'output');
const tables=path.join(repo,'research/ftir_hips_chem/output/tables/adama_summary_20260910');
const figures=path.join(repo,'research/ftir_hips_chem/output/plots/adama_summary_20260910');
const source=path.join(workspaceDir,'adama_summary_2026-09-01.pptx');
const skill='/Users/ahmadjalil/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const summary=JSON.parse(await fs.readFile(path.join(tables,'summary.json'),'utf8'));
const spectra=JSON.parse(await fs.readFile(path.join(tables,'spectral_overlay_manifest.json'),'utf8'));
const pairs=JSON.parse(await fs.readFile(path.join(tables,'adama_pairs.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const anchors=['sl/y90nupkv','sl/hwbqtkby','sl/ofy9wn61','sl/jyx0ra1s','sl/i107q5of',
 'sl/x8f69ofe','sl/gnmp4jqx','sl/fu1gfa1s','sl/udsvah03'];
const originals=anchors.map(a=>p.resolve(a));
if(p.slides.items.length!==9)throw new Error('Expected the inspected nine-slide source');
if(originals.some((s,i)=>s!==p.slides.items[i]))throw new Error('Source anchors changed');
const family='Calibri'; // Source theme's major and minor Latin typeface.
const ink='#22252A',muted='#666B72';
const clone=()=>originals[0].duplicate();
const ordered=[...originals.slice(0,5),clone(),clone(),clone(),...originals.slice(5,8),clone(),clone(),clone(),originals[8]];
ordered.forEach((s,i)=>s.moveTo(i));
if(ordered.some((s,i)=>p.slides.items[i]!==s))throw new Error('Slide reordering failed');
const localNote=(...names)=>names.map(n=>path.join(tables,n)).join('\n');
const flags='Open rings mark the 9 July timing flag and 30 July volume flag. All five pairs remain visible.';
const method='Paired PTFE and quartz samplers, July 2024. HIPS and FTIR share the PTFE filter. Quartz is a separate co-located filter. All comparisons use each sampler\'s own reported air volume. FTIR is the AMOD Batch 54 deployed EC export, CalibrationSetId 26. HIPS Fabs/MAC is an assumed EC-equivalent, not a direct EC measurement.';
const specs=[
 {title:'Adama absorption overlaps Addis, while filter optical depth is lower',
  subtitle:'Five July 2024 Adama filters added to the existing Addis and Bishoftu comparison',figure:'context',
  foot:'Different collection periods and samplers. Median sampled volumes: Addis 7.25 m³, Bishoftu 7.17 m³, Adama 2.88 m³.',
  notes:'Meeting 05:55–08:09 and 14:11–14:50: add Adama to the concentration/loading comparison. Density histograms make the unequal sample sizes explicit. The Adama vertical lines are the five observations, not a density estimate. Median Fabs is 40.51 Mm⁻¹ at Adama, 47.11 at Addis and 26.94 at Bishoftu. Optical depth medians are 0.294, 0.954 and 0.547, respectively. HIPS optical depth is a filter loading diagnostic. Air volume, deposit area and aerosol properties all matter. The site periods are not matched, so these distributions do not establish a city-wide or seasonal difference.\n'+localNote('adama_pairs.csv','addis_hips_population.csv','sources.json')},
 {title:'Adama TOR OC/EC falls within the IMPROVE distribution',
  subtitle:'Adama median OC/EC = 6.08 across five dates, compared with 199 IMPROVE site medians',figure:'ratio_oc_ec',
  foot:'IMPROVE and Adama use quartz TOR. The four SPARTAN city markers use deployed FTIR products, so the methods differ.',
  notes:'Meeting 04:11–05:55 and 17:20–17:58. Each IMPROVE marker is a site median over at least 100 valid days. The Adama median is at approximately the 65th percentile of those site medians, rather than exactly at their median. Addis has FTIR OC/EC 1.34 and is at the lower edge of this reference. This is not an independent validation of FTIR at Addis because its plotted carbon products come from that calibration. It does not prove a universal composition difference from five July dates. Central sample exclusions apply to the SPARTAN markers, with positive finite ratio eligibility recorded.\n'+localNote('site_ratio_medians.csv','improve_oc_ec_site_medians.csv','adama_pairs.csv')},
 {title:'Adama can join the OC-to-absorption comparison',
  subtitle:'The Adama marker combines quartz TOR OC with HIPS on the paired PTFE filter',figure:'ratio_oc_fabs',
  foot:'Adama median OC/Fabs is 0.39 using TOR OC and 0.17 using FTIR OC. The marker depends on the OC measurement method.',
  notes:'Meeting 21:00: add OC/Fabs. The older deck says Adama HIPS is unavailable. Batch 54 contains these measurements. The updated cloud has 198 eligible IMPROVE sites, not the older hardcoded count of 199. Adama all-five median TOR OC/Fabs = 0.38791. The FTIR OC/Fabs median is 0.1689. Mixed media and different OC methods limit interpretation. Pairing flags remain in the source table.\n'+localNote('site_ratio_medians.csv','improve_oc_fabs_site_medians.csv','adama_pairs.csv')},
 {title:'Absorption per PM₂.₅ mass still needs paired Adama gravimetry',
  subtitle:'Existing HIPS and filter-mass data provide context for Addis and Bishoftu',figure:'ratio_fabs_pm',
  foot:'Adama HIPS is available. A matched PM₂.₅ mass value for each Adama PTFE filter is still missing from these source exports.',
  notes:'The original absorption-per-mass comparison is retained with refreshed source medians and the correct count of eligible IMPROVE sites. No FTIR product enters this ratio. The Adama absence is a missing PM2.5 denominator, not missing optics. Regional BAM or PurpleAir PM may give ambient context, but cannot be substituted directly for paired gravimetric mass on these five filters.\n'+localNote('site_ratio_medians.csv','improve_fabs_pm_site_medians.csv','bishoftu_mass_pairs.csv','sources.json')},
 {title:'Addis and Adama against the same HIPS reference',
  subtitle:'Both panels use concentrations, identical axes and the configured MAC of 10 m²/g',figure:'same_reference',
  foot:'Adama n = 5, with no fitted line. Open rings identify the two pairing flags. HIPS / MAC depends on the assumed MAC.',
  notes:'Meeting 14:50–16:34: preserve Addis and add Adama FTIR versus HIPS. '+method+' '+flags+' Addis includes 190 pairs from 7 December 2022 to 21 September 2024. The larger HIPS population continues to January 2026. OLS slope 1.898, intercept −4.170, R² 0.764. The Deming error model uses median measured HIPS uncertainty / MAC = '+summary.sigma_x.toFixed(4)+' µg/m³ and approximate FTIR sigma_y = 0.531 µg/m³ from the prior AIRSpec held-out TOR RMSE. This gives lambda '+summary.addis_fit.deming_lambda.toFixed(3)+', slope '+summary.addis_fit.deming_slope.toFixed(3)+' and intercept '+summary.addis_fit.deming_intercept.toFixed(3)+'. Lambda is a sensitivity assumption, not an exact error model for this deployed calibration.\n'+localNote('addis_ec_pairs.csv','adama_pairs.csv','summary.json')},
 {title:'The five Adama pairs connect FTIR, HIPS and quartz TOR',
  subtitle:'All three pairwise comparisons use the same five dates and a 1:1 reference',figure:'three_crossplots',
  foot:'Concentrations in µg/m³. HIPS uses MAC = 10 m²/g. Open rings flag pairing issues, with no exclusion or correction.',
  notes:'Meeting 14:50–16:34 requested all three measurement relationships. Separate panels avoid a dual-axis scale ambiguity. '+method+' '+flags+' Quartz TOR EC spans 1.86–3.38 µg/m³. The approximately 4.8–8.9 values in the old crossplot were micrograms per filter, not ambient concentrations. HIPS / 10 exceeds quartz TOR EC on all five dates. These five observations cannot establish a population regression or rule out an Addis-like discrepancy in other Adama seasons.\n'+localNote('adama_pairs.csv','legacy_calibrations_concentration_ratios.csv')},
 {title:'The Addis comparison mostly contains one sampled-volume range',
  subtitle:'189 of 190 pairs span 6.78–7.36 m³. One filter has a recorded volume of 2.07 m³.',figure:'volume_crossplot',
  foot:'Color shows recorded air volume. The solid line is pooled OLS. Residuals include every paired filter.',
  notes:'Meeting 07:02–08:09: color FTIR EC versus Fabs by sampled air volume. HIPS volume and the FTIR metadata volume agree for all 239 HIPS evaluation rows. The 190 deployed-EC pairs contain one low-volume record: ETAD-0248-8, 7 August 2024, 2.0736 m³. Display strata (<6, 6–8 m³) are not exclusions or documented sampling regimes. This plot does not have two well-populated volume regimes and cannot validate a before/after loading effect. The OLS / Deming assumptions are the same as slide 5.\n'+localNote('addis_ec_pairs.csv','addis_volume_counts.csv','summary.json')},
 {title:'Later HIPS samples have no deployed FTIR EC in this subset',
  subtitle:'The HIPS population extends to January 2026. The 190 deployed-EC pairs end in September 2024.',figure:'volume_coverage',
  foot:'The weak residual–optical-depth association is descriptive. It does not establish that loading or volume has no effect.',
  notes:'Meeting 07:19 and 12:30–12:49: distinguish the data available from the paired subset used by this deck. There are 239 HIPS/spectra filters but only 190 with a deployed EC product in the canonical unified dataset. No aethalometer flow-fix dates have been used as filter-sampler intervention dates. The optical-depth residual screen has Spearman r = '+summary.residual_tau_spearman.toFixed(3)+'. Residuals are computed from the pooled y-on-x OLS in slide 5, so this is an exploratory screen, not a causal test. Obtain the filter sampling protocol/logs and missing deployed EC products before claiming a regime comparison.\n'+localNote('addis_hips_population.csv','addis_ec_pairs.csv','summary.json')},
 {title:'Adama FTIR/TOR ratios depend on calibration and sampled volume',
  subtitle:'Deployed FTIR/TOR median = 0.69 on a concentration basis, with a range of 0.58–1.56',figure:'calibration_char',
  foot:'The three unflagged deployed ratios span 0.64–0.81. Open rings identify the two pairing flags.',
  notes:'Meeting 09:39–14:11: retain the production/biomass comparison and thermal-fraction diagnostic. The old deck reported a median of 0.77 from FTIR mass / quartz mass. Because PTFE and quartz volumes differ, that is not a concentration ratio. New values divide each model mass by PTFE air volume, then by quartz TOR concentration. The production mass reproduces the AMOD set-26 EC concentration to numerical precision. Char/soot here is only the existing empirical fraction ratio (EC1−OPTR)/(EC2+EC3). All five ratios are below one (0.018–0.580). That does not independently identify a combustion source or resolve the calibration mechanism. '+flags+'\n'+localNote('legacy_calibrations_concentration_ratios.csv','adama_pairs.csv')},
 {title:'At MAC 10, HIPS EC-equivalent exceeds Adama quartz EC',
  subtitle:'For the three unflagged pairs, implied MAC medians are 16.5 versus TOR and 20.8 versus TOT',figure:'three_method_mac',
  foot:'Open rings show flagged dates. The MAC 6–10 band illustrates how the assumed MAC changes the HIPS EC-equivalent.',
  notes:'The previous three-method content is retained with the measured definitions and explicit sample flags. '+method+' The inferred MAC is Fabs divided by the selected EC definition, with units m²/g. Bars use only the three pairs with no comparability flag and all five points remain displayed. The earlier comparison to an Addis value near 47 is omitted from the claim because that number is a composition-bridge inference, not a same-site quartz measurement. Pure-EC universal upper-bound claims are not needed for this evidence.\n'+localNote('adama_pairs.csv','summary.json')},
 {title:'The Addis-selected models predict more EC at Adama',
  subtitle:'Median model/TOR ratios for three unflagged pairs: locked 800 = 1.40, winner 440 = 2.00',figure:'locked_models',
  foot:'Exploratory transfer check. The spectrum-to-FilterId mapping used by these earlier model predictions remains provisional.',
  notes:'Preserves the September 1 deck\'s locked-calibration comparison, with separate TOR and TOT panels and readable labels. The source predictions come from ftir_44 and were not refitted for this deck. Model/TOR medians are 0.69 deployed, 1.40 locked 800, 2.00 winner 440, and 1.01 TOT-target. Paired-sampler flags apply. The older model workflow inferred spectrum-to-FilterId order from spectral rank agreement. That does not replace a definitive lab crosswalk, so these model values remain provisional. This is not enough evidence to declare the models generally invalid or to tune a new calibration.\n'+localNote('locked_model_ratios.csv')+'\n'+path.join(repo,'research/ftir_ec_phase3/output/tables/ftir44/adama_three_method_with_phase3_models.csv')+'\n'+path.join(repo,'research/ftir_ec_phase3/output/tables/ftir44/id_mapping_audit.csv')},
 {title:'Five Adama spectra over the Addis evaluation set',
  subtitle:`Adama in color (n = 5). Addis in light gray (n = ${spectra.n_addis_all}), December 2022–January 2026.`,figure:'spectra_all',
  foot:'Raw absorbance, with no normalization or baseline correction. Legend labels 4744–4748 are Adama source spectrum IDs.',
  notes:'Requested in Ann Dillner\'s pasted email. Five Adama spectra overlay every filter in the existing 239-filter Addis spectra/HIPS evaluation set. Addis replicate FTIR scans are averaged by MediaId through the canonical phase3_common loader. No additional exclusions. Both slide 12 and slide 13 use identical x and y limits in each corresponding panel. The detailed panel enlarges 3500–1500 cm⁻¹ without changing the exported absorbance. Native wavenumber grids are plotted without interpolation. Addis spans 3998–500 cm⁻¹, Adama extends to 420 cm⁻¹. No confirmed row-ID-to-FilterId crosswalk is present, so the curves carry source IDs only. Raw differences mix baseline, substrate and loading effects with aerosol features, and are not a source diagnosis.\n'+localNote('spectral_overlay_manifest.json','spectral_overlay_addis_population.csv','spectral_overlay_adama_ids.csv')+'\n'+spectra.sources.join('\n')},
 {title:'Five Adama spectra over summer Addis',
  subtitle:`Summer = Kiremt, June–September. Addis in light gray (n = ${spectra.n_addis_summer}), pooled across 2023–2025.`,figure:'spectra_summer',
  foot:'The same five July 2024 Adama spectra and axis limits as the previous slide. Summer selection uses the repo’s canonical calendar.',
  notes:'Ann requested a summer-only Addis overlay because Adama was sampled in July. Summer is explicitly operationalized as Kiremt (June–September), via config.ETHIOPIA_SEASONS under season convention dry_feb. The alternate registered calendar changes February only, so it does not change this summer selection. '+spectra.n_addis_summer+' Addis filters qualify, spanning '+spectra.summer_dates.join(' to ')+'. This pools available summers and is not restricted to July 2024 or the 190 deployed-EC pairs. The plots show raw spectra with identical corresponding panel limits. Adama source row IDs do not assert a physical-filter match.\n'+localNote('spectral_overlay_manifest.json','spectral_overlay_addis_population.csv','spectral_overlay_adama_ids.csv')},
 {title:'Two Adama pairs need sampler-log review',
  subtitle:'Paired samplers use different filters and recorded air volumes. Both flagged pairs remain in every main comparison.',table:true,
  foot:'Volume ratio = PTFE / quartz. Start offset = quartz start minus PTFE start. Flags follow the existing >5 min and >20% rules.',
  notes:'Flags reproduced from ftir_41 using the source timestamps and sampler volumes. 9 July has a quartz start 39.73 minutes after PTFE. 30 July has PTFE volume only 0.456 times quartz volume. Neither is automatically excluded or corrected. No sampler-log explanation is available in the input. Review whether the volume discrepancy reflects true collection or a metadata error before correcting any concentration.\n'+localNote('adama_pairs.csv')},
 {title:'Sampling priorities from the meeting',
  subtitle:'The five Adama dates inform the campaign design. They do not resolve the Addis discrepancy.',actions:true,
  foot:'PM context and sampler-log review remain open. Collection scope, duration and costs need agreement with the sampling team.',
  notes:'Meeting 18:27–21:00: inspect BAM availability through the MAIA portal and consider PurpleAir for relative PM context. Meeting 21:46–25:50: prioritize quartz in Addis. Same-site co-location is preferred, but the meeting described the existing southern Addis site as a more feasible alternative. Meeting 26:00–29:11: Ann will investigate the available paired Delhi filters. The earlier 36-filter sizing remains a separate proposal, not a decision of this meeting. No messages have been sent.\nPublic data route checked 10 September 2026: NASA/JPL MAIA GIVT supports download of quality-controlled daily surface PM observations: https://maia.jpl.nasa.gov/mmgis/ . This confirms a source route, not a date-matched Adama BAM/PurpleAir data series. No ambient PM city ratio is presented without verified station, method, date and completeness coverage. NASA product announcement: https://www.earthdata.nasa.gov/data/alerts-outages/multi-angle-imager-aerosols-maia-surface-monitor-data-product-released .\nMeeting source: user-pasted transcript, Adama discussion 03:04–30:05. Subsequent pasted email from Ann requests slides 12–13.'}
];

function text(s,value,x,y,w,h,size=22,color=ink,bold=false){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return t;
}
function title(s,value){
 const t=s.shapes.items[0];t.text=value;
 t.position={left:43.2,top:21.12,width:1200,height:91.2};
 t.text.style={typeface:family,fontSize:28,bold:true,color:ink,autoFit:'none'};
}
for(let i=0;i<ordered.length;i++){
 const s=ordered[i],spec=specs[i];
 if(spec.actions){
  s.shapes.deleteAll();
  text(s,spec.title,43.2,21.12,1200,91.2,28,ink,true);
 }else title(s,spec.title);
 text(s,spec.subtitle,53,107,1170,49,22,muted);
 // Duplicated source slides share image references. Give each slide its own
 // image rather than replacing a shared reference and changing every clone.
 for(const im of [...s.images.items])s.images.deleteById(im.id);
 if(spec.figure){
  s.images.add({blob:new Uint8Array(await fs.readFile(path.join(figures,spec.figure+'.png'))),
    contentType:'image/png',fit:'contain',alt:spec.title,
    position:{left:54.38,top:174,width:1171.2,height:446}});
 }else if(spec.table){
  const rows=[['July 2024','PTFE / quartz','PTFE\nm³','Quartz\nm³','Volume\nratio','Offset\nmin','Review note'],
   ...pairs.map(r=>[String(new Date(r.date).getUTCDate()),r.FilterId_ptfe+' / '+r.FilterId_quartz,
    r.Volume_m3.toFixed(3),(r.Volume_liters/1000).toFixed(3),r.volume_ratio.toFixed(3),
    (r.start_offset_min>=0?'+':'')+r.start_offset_min.toFixed(2),
    r.pairing_flag.includes('start')?'Start time':r.pairing_flag?'Air volume':'No flag'])];
  const tab=s.tables.add({rows:6,columns:7,left:60,top:198,width:1150,height:360,
                         columnWidths:[125,245,130,130,140,140,240],values:rows});
  tab.borders.assign({fill:'#D9D9D9',width:.5});
  for(let r=0;r<6;r++)for(let c=0;c<7;c++){
   const cell=tab.getCell(r,c);cell.fill=r===0?'#EFEFEF':'#FFFFFF';
   cell.text.style={typeface:family,fontSize:r===0?20:22,color:ink,bold:r===0,autoFit:'none'};
  }
 }else if(spec.actions){
  const blocks=[
   ['Quartz in Addis','Prioritize paired quartz and PTFE at Addis. If the SPARTAN location is infeasible, assess the existing southern Addis site and document the site differences.'],
   ['Paired Delhi filters','Follow up on the quartz/PTFE material discussed with Ann. Use a known thermal reference to test whether the Addis spectral pattern appears elsewhere.'],
   ['Regional PM context','Check MAIA daily BAM data for Adama, Bishoftu and Addis. Use comparable PurpleAir records if needed, with matched dates and clear method labels.'],
   ['Sampler metadata','Resolve the two Adama pairing flags and obtain the Addis filter sampling protocol. Confirm the Adama spectrum-to-FilterId crosswalk with the lab.']
  ];
  blocks.forEach(([heading,body],j)=>{
   text(s,heading,64,186+j*107,258,34,24,ink,true);
   text(s,body,337,182+j*107,862,81,24,ink);
  });
 }
 text(s,spec.foot,54,638,1171,66,18,muted);
 s.speakerNotes.textFrame.setText(spec.notes);
}
await fs.mkdir(output,{recursive:true});
await fs.mkdir(path.join(build,'draft-rendered'),{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(build,'content.json'),JSON.stringify(specs,null,2));
await fs.writeFile(path.join(build,'inspect.ndjson'),(await p.inspect({kind:'slide,textbox,image,table,layout',maxChars:50000})).ndjson);
for(let i=0;i<ordered.length;i++){
 const blob=await ordered[i].export({format:'png',scale:1});
 await fs.writeFile(path.join(build,'draft-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log('Draft rendered: '+candidate);
if(process.env.AETHMODULAR_ADAMA_DRAFT_ONLY!=='1'){
 const name=process.env.AETHMODULAR_ADAMA_OUTPUT??'adama_summary_2026-09-10.pptx';
 if(path.basename(name)!==name||!name.endsWith('.pptx'))throw new Error('Expected a PPTX filename');
 const finalPath=path.join(output,name);
 const result=await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath,
  pythonExecutable:'/Users/ahmadjalil/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12191695,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','14'],
  explicitTotalSlideCount:15,requiredNativeTableOwnerSlides:[14],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:[family],referencePath:source,
              referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},
  verifyArtifactToolImport:true,receiptPath:path.join(build,name+'.validation.json')});
 console.log(JSON.stringify({finalPath,sha256:result.finalSha256}));
 const verified=await PresentationFile.importPptx(await FileBlob.load(finalPath));
 await fs.mkdir(path.join(build,'final-rendered'),{recursive:true});
 for(let i=0;i<verified.slides.items.length;i++){
  const blob=await verified.slides.items[i].export({format:'png',scale:1});
  await fs.writeFile(path.join(build,'final-rendered',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
 }
}
