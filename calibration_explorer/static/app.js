/* Calibration Iteration Explorer — frontend logic.
   Talks to the Flask API in ../app.py; all plots via Plotly (responsive).
   Layout: global toolbar (presets + configuration + toggles) over four tabs. */

const $ = id => document.getElementById(id);
const SEASON_COLOUR = {Dry:'#B23327', Belg:'#7A4FA3', Kiremt:'#2C6E9E', unknown:'#9a9a9a'};
const seasonColour = s => SEASON_COLOUR[Object.keys(SEASON_COLOUR).find(k => String(s).startsWith(k)) || 'unknown'];
const PCFG = {displayModeBar:false, responsive:true};
const RANKED = ['eth_shaped', 'analogs', 'ocec'];
const LEGEND_TOP = {orientation:'h', y:1, yanchor:'bottom', x:0, xanchor:'left', font:{size:11}};
const SPECTRA_LABEL = {raw:'raw', airspec:'AIRSpec-corrected', deriv2:'SG 2nd derivative'};
const SPECTRA_SHORT = {raw:'raw', airspec:'AIRSpec', deriv2:'D2'};
const MODE_LABEL = {site_heldout:'Option A — site-grouped 5-fold, first major minimum',
                    app:'Option B — interleaved 10-fold, within 5% of minimum',
                    app_fmm:'Option B2 — interleaved 10-fold, first major minimum'};
const MODE_SHORT = {site_heldout:'A (site-grouped)', app:'B (interleaved 5%)', app_fmm:'B2 (interleaved FMM)'};
let spectraMode = 'single';

let last = null;
let lastSweep = null;   // rows of the most recent k sweep (belongs to that run's config)
let lastSweepCfg = null;
let pins = JSON.parse(localStorage.getItem('calib_explorer_pins') || '[]');
let customPresets = JSON.parse(localStorage.getItem('calib_explorer_presets') || '{}');
let toggles = {mac:'10', est:'deming', evalset:'fixed'};
let defaults = {};
let ready = false;

/* The six setup-matrix rows as built-in presets (site-held-out protocol, rule k). */
const BUILTIN_PRESETS = [
  {name:'Entire IMPROVE network (no selection)', cfg:{cohort:'pool', spectra:'raw', selection_space:'raw', mode:'site_heldout', kmode:'auto'}},
  {name:'Biomass-smoke (906)',                   cfg:{cohort:'smoke', spectra:'raw', selection_space:'raw', mode:'site_heldout', kmode:'auto'}},
  {name:'Ethiopia-shaped smoke (300)',           cfg:{cohort:'eth_shaped', cutoff:300, spectra:'raw', selection_space:'raw', mode:'site_heldout', kmode:'auto'}},
  {name:'Spectral analogs (500)',                cfg:{cohort:'analogs', cutoff:500, spectra:'raw', selection_space:'raw', mode:'site_heldout', kmode:'auto'}},
  {name:'Lowest-OC/EC (800)',                    cfg:{cohort:'ocec', cutoff:800, spectra:'raw', selection_space:'raw', mode:'site_heldout', kmode:'auto'}},
  {name:'Lowest-OC/EC + AIRSpec (800)',          cfg:{cohort:'ocec', cutoff:800, spectra:'airspec', selection_space:'raw', mode:'site_heldout', kmode:'auto'}},
];

/* ---- helpers --------------------------------------------------------------- */
function busy(on){ $('busy').style.display = on ? 'block' : 'none'; }
function showError(msg){
  const e = $('err');
  if(!msg){ e.style.display='none'; return; }
  e.textContent = msg; e.style.display='block';
}
async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  return await r.json();
}
function cfg(){
  return {
    cohort: $('cohort').value,
    cutoff: $('cutoff').disabled ? null : parseInt($('cutoff').value) || null,
    selection_space: $('selection_space').disabled ? 'raw' : $('selection_space').value,
    spectra: $('spectra').value,
    mode: $('mode').value,
    lot: $('lot').value,
    target: $('target').value || 'addis',
    eval_lot: $('eval_lot').disabled ? 'all' : ($('eval_lot').value || 'all'),
  };
}
function syncEvalLot(){
  // filter-lot info exists only for the built-in Addis target
  const addis = $('target').value === 'addis';
  $('eval_lot').disabled = !addis;
  if(!addis) $('eval_lot').value = 'all';
}
const GROUP_PALETTE = ['#2C6E9E', '#B23327', '#7A4FA3', '#8F8C84', '#C49442', '#548C66'];
function groupColour(g, order){
  const k = Object.keys(SEASON_COLOUR).find(k => String(g).startsWith(k));
  return k ? SEASON_COLOUR[k] : GROUP_PALETTE[order % GROUP_PALETTE.length];
}
function metricRow(m, refKind){
  if(!m || !m.length) return null;
  const kind = refKind ?? (last && last.target ? last.target.ref_kind : 'fabs');
  const hasFixed = m.some(r => r.evaluation_set === 'fixed');
  const es = (toggles.evalset === 'fixed' && hasFixed) ? 'fixed' : 'all';
  const sub = m.filter(r => r.evaluation_set === es);
  if(kind === 'ec') return sub.find(r => r.MAC == null) || sub[0];
  return sub.find(r => r.MAC === parseFloat(toggles.mac)) || sub[0];
}
function updateToggles(){
  const t = last && last.target;
  $('mac').querySelectorAll('button').forEach(b => b.disabled = !!(t && t.ref_kind === 'ec'));
  $('evalset').querySelectorAll('button').forEach(b => b.disabled = !!(t && !t.has_fixed));
}
function plot(id, data, layout){
  const el = $(id);
  if(el.querySelector('.ph')) el.innerHTML = '';
  Plotly.newPlot(id, data, layout, PCFG);
}
function placeholder(id, text){
  const el = $(id);
  if(!el.querySelector('.js-plotly-plot')) el.innerHTML = '<div class="ph">' + text + '</div>';
}

/* ---- tabs ------------------------------------------------------------------ */
$('tabs').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('tabs').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('on'));
  const pane = $('pane-' + b.dataset.tab);
  pane.classList.add('on');
  // plots drawn while a pane was hidden have a stale width — fix them on reveal.
  // Resize synchronously (rAF can be throttled in unfocused tabs) and again on
  // the next frame in case layout was still settling.
  const fix = () => pane.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
  fix(); requestAnimationFrame(fix);
});
function activeTab(){ return $('tabs').querySelector('button.on').dataset.tab; }
function switchTab(name){ $('tabs').querySelector(`button[data-tab="${name}"]`).click(); }

/* ---- presets --------------------------------------------------------------- */
function presetList(){
  const opts = ['<option value="">— configuration —</option>'];
  opts.push('<optgroup label="Built-in (setup matrix)">');
  BUILTIN_PRESETS.forEach((p, i) => opts.push(`<option value="b:${i}">${p.name}</option>`));
  opts.push('</optgroup>');
  const names = Object.keys(customPresets).sort();
  if(names.length){
    opts.push('<optgroup label="Custom">');
    names.forEach(n => opts.push(`<option value="c:${n}">${n}</option>`));
    opts.push('</optgroup>');
  }
  $('preset').innerHTML = opts.join('');
}
function currentPresetCfg(){
  return {...cfg(), kmode: $('kmode').value,
          k: $('kmode').value === 'manual' ? parseInt($('kval').value) : null};
}
function applyPreset(p){
  if(!ready || !p) return;
  $('cohort').value = p.cohort; $('cohort').onchange();
  if(p.cutoff && !$('cutoff').disabled) $('cutoff').value = p.cutoff;
  if(!$('selection_space').disabled) $('selection_space').value = p.selection_space || 'raw';
  $('spectra').value = p.spectra || 'raw';
  $('lot').value = p.lot || 'all';
  if([...$('target').options].some(o => o.value === (p.target || 'addis')))
    $('target').value = p.target || 'addis';
  syncEvalLot();
  if(!$('eval_lot').disabled &&
     [...$('eval_lot').options].some(o => o.value === (p.eval_lot || 'all')))
    $('eval_lot').value = p.eval_lot || 'all';
  $('mode').value = p.mode || 'site_heldout';
  $('kmode').value = p.kmode || 'auto'; $('kmode').onchange();
  if(p.kmode === 'manual' && p.k) $('kval').value = p.k;
  drawRanking();
  // .value assignments don't fire change events, so update/auto-run explicitly
  updateCfgSummary(); scheduleAutoRun();
}
$('preset').onchange = () => {
  const v = $('preset').value;
  $('preset_delete').disabled = !v.startsWith('c:');
  if(v.startsWith('b:')) applyPreset(BUILTIN_PRESETS[parseInt(v.slice(2))].cfg);
  else if(v.startsWith('c:')) applyPreset(customPresets[v.slice(2)]);
};
$('preset_save').onclick = () => {
  const name = prompt('Preset name:', $('preset').value.startsWith('c:') ? $('preset').value.slice(2) : '');
  if(!name) return;
  customPresets[name] = currentPresetCfg();
  localStorage.setItem('calib_explorer_presets', JSON.stringify(customPresets));
  presetList(); $('preset').value = 'c:' + name; $('preset_delete').disabled = false;
};
$('preset_delete').onclick = () => {
  const v = $('preset').value;
  if(!v.startsWith('c:')) return;
  delete customPresets[v.slice(2)];
  localStorage.setItem('calib_explorer_presets', JSON.stringify(customPresets));
  presetList();
};
/* the rarer preset actions live behind the ⋯ overflow menu */
$('preset_menu_btn').onclick = e => {
  e.stopPropagation();
  $('preset_menu').classList.toggle('open');
};
$('preset_menu').addEventListener('click', () => $('preset_menu').classList.remove('open'));
document.addEventListener('click', e => {
  if(!e.target.closest('.menuwrap')) $('preset_menu').classList.remove('open');
});
$('preset_export').onclick = () => {
  const blob = new Blob([JSON.stringify({app:'calibration_explorer', version:1, presets:customPresets}, null, 2)],
                       {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'explorer_presets.json'; a.click();
};
$('preset_import').onclick = () => $('preset_file').click();
$('preset_file').onchange = async () => {
  const f = $('preset_file').files[0];
  if(!f) return;
  try{
    const j = JSON.parse(await f.text());
    const incoming = j.presets || j;   // accept a bare {name: cfg} map too
    let n = 0;
    for(const [name, p] of Object.entries(incoming)){
      if(p && typeof p === 'object' && p.cohort){ customPresets[name] = p; n++; }
    }
    localStorage.setItem('calib_explorer_presets', JSON.stringify(customPresets));
    presetList();
    showError(null);
    if(!n) showError('No presets found in that file.');
  }catch(e){ showError('Import failed: ' + e.message); }
  $('preset_file').value = '';
};

/* ---- control wiring -------------------------------------------------------- */
function pillWire(id){
  $(id).querySelectorAll('button').forEach(b => b.onclick = () => {
    $(id).querySelectorAll('button').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); toggles[id] = b.dataset.v; redraw();
    if(optRows.length) renderOpt();   // optimizer scores follow the MAC · Fit toggles
    if(sitesRows.length) sitesRender();  // cross-site table follows them too
  });
}
['mac', 'est', 'evalset'].forEach(pillWire);

$('kmode').onchange = () => { $('kval').disabled = $('kmode').value !== 'manual'; };
$('cohort').onchange = () => {
  const c = $('cohort').value, ranked = RANKED.includes(c);
  $('cutoff').disabled = !ranked;
  $('cutoff').value = ranked ? defaults[c] : '';
  // selection-space toggle exists only where a spectra-based selection exists
  $('selection_space').disabled = !(c === 'eth_shaped' || c === 'analogs');
  if($('selection_space').disabled) $('selection_space').value = 'raw';
  drawRanking();
};
$('selection_space').onchange = () => { drawRanking(); drawOverlap(); };
$('cutoff').onchange = () => cutoffChanged(false);
$('lot').onchange = drawCohortInfo;
$('spectra').onchange = drawCohortInfo;

/* ---- startup --------------------------------------------------------------- */
placeholder('p_curve', 'CV curve appears here after a run.');
placeholder('p_cross', 'Addis crossplot appears here after a run.');
placeholder('p_sweep', 'Use “Sweep k” after a run to scan components from the rule choice to ~double.');
placeholder('p_spectra', 'Cohort-vs-Addis spectra appear here after a run.');
presetList();

async function poll(){
  const s = await (await fetch('/api/status')).json();
  if(s.error){ $('stats').innerHTML = '<span class="warn">Load failed: ' + s.error + '</span>'; return; }
  if(!s.ready){ $('stats').textContent = 'Loading data… ' + s.message; setTimeout(poll, 1500); return; }
  defaults = s.default_cutoff;
  $('cohort').innerHTML = Object.entries(s.cohorts).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  $('lot').innerHTML = '<option value="all">all</option>' +
    (s.lots || []).map(l => `<option value="${l}">${l}</option>`).join('');
  const curT = $('target').value;
  $('target').innerHTML = Object.entries(s.targets || {addis:'Addis (ETAD) — built-in'})
    .map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  if([...$('target').options].some(o => o.value === curT)) $('target').value = curT;
  $('eval_lot').innerHTML = '<option value="all">all lots</option>' +
    Object.entries(s.eval_lots || {}).filter(([l]) => l !== '?')
      .sort((a, b) => b[1] - a[1])
      .map(([l, n]) => `<option value="${l}">${l} (n=${n})</option>`).join('');
  syncEvalLot();
  ready = true;
  $('cohort').value = 'ocec'; $('cohort').onchange();
  $('checks').innerHTML = 'cohort checks: ' + s.checks.map(c =>
    `<span class="${c.ok ? 'ok' : 'warn'}" title="${c.name}: ${c.detail}">${c.ok ? '✓' : '✗ ' + c.name + ' (' + c.detail + ')'}</span>`).join(' ') +
    ' <button class="mini" id="benchcsv" title="Re-read the 725 MB pool CSV with polars and pandas in the server process and compare timings + values">time CSV engines</button> <span id="benchout"></span>';
  $('benchcsv').onclick = async () => {
    $('benchcsv').disabled = true;
    $('benchout').textContent = 'benchmarking — two full reads of the pool CSV, a minute or two…';
    try{
      const r = await (await fetch('/api/benchmark_pool_read', {method: 'POST'})).json();
      if(r.error){ $('benchout').innerHTML = '<span class="warn">' + r.error + '</span>'; }
      else {
        $('benchout').textContent =
          `polars ${r.polars_s}s vs pandas ${r.pandas_s}s (${r.speedup}×); ` +
          (r.cells_differing === 0 && r.index_equal
            ? 'values bit-identical'
            : `${r.cells_differing}/${r.cells_total} cells differ (max ${r.max_ulp_float32} float32 ulp)`);
      }
    } catch(e){ $('benchout').innerHTML = '<span class="warn">' + e + '</span>'; }
    $('benchcsv').disabled = false;
  };
  $('stats').textContent = 'Ready. Pick a preset (or configure by hand) and Run. (Entire-network cohort: first CV curve takes several minutes; cached afterwards.)';
  updateCfgSummary();
  drawPins();
  drawOverlap();
  batchTick(true);   // surface a batch already running (another tab, Colab, overnight)
}
poll();
placeholder('p_resid', 'Residuals appear here after a run.');
placeholder('p_overlap', 'The overlap matrix appears once data is loaded.');
placeholder('p_series', 'The dated EC series appears here after a run.');
placeholder('p_vsdeployed', 'The deployed-EC comparison appears here after a run.');
placeholder('p_composition', 'The composition ruler appears once data is loaded.');
$('target').onchange = () => { syncEvalLot(); drawCohortInfo(); };

/* ---- run ------------------------------------------------------------------- */
let runInFlight = false, rerunQueued = false;
async function run(){
  if(runInFlight){ rerunQueued = true; return; }
  runInFlight = true;
  busy(true); showError(null); $('run').disabled = true;
  const body = {...cfg(), k: $('kmode').value === 'manual' ? parseInt($('kval').value) : null};
  try{
    const j = await post('/api/run', body);
    if(j.error){ showError(j.error); return; }
    last = j; $('pin').disabled = false; $('sweep').disabled = false;
    $('kval').value = j.k;
    if(lastSweep && lastSweepCfg !== JSON.stringify(cfg())){
      lastSweep = null;
      placeholder('p_sweep', 'Use “Sweep k” after a run to scan components from the rule choice to ~double.');
    }
    redraw();
    drawSpectra();          // refresh side-by-side spectra for this config
  } finally {
    busy(false); $('run').disabled = false; runInFlight = false;
    updateCfgSummary();
    // a config change landed while this run was in flight — run again on it
    if(rerunQueued){ rerunQueued = false; run(); }
  }
}
$('run').onclick = run;

/* ---- auto-run: re-run shortly after any configuration change --------------- */
let autoRun = localStorage.getItem('calib_explorer_autorun') === '1';
let autoTimer = null;
function syncAutoRunUI(){
  $('autorun').classList.toggle('on', autoRun);
  $('autorun').setAttribute('aria-pressed', autoRun ? 'true' : 'false');
}
syncAutoRunUI();
$('autorun').onclick = () => {
  autoRun = !autoRun;
  localStorage.setItem('calib_explorer_autorun', autoRun ? '1' : '0');
  syncAutoRunUI();
  if(autoRun) scheduleAutoRun();      // give immediate feedback on enabling
};
function scheduleAutoRun(){
  if(!autoRun || !ready) return;
  clearTimeout(autoTimer);
  autoTimer = setTimeout(run, 600);   // debounced: rapid changes fold into one run
}
// native change events from every config control bubble up to the card
$('configcard').addEventListener('change', () => { updateCfgSummary(); scheduleAutoRun(); });

/* ---- config summary + phone-only collapse ---------------------------------- */
function updateCfgSummary(){
  if(!ready) return;
  const c = cfg();
  const name = $('cohort').selectedOptions[0] ? $('cohort').selectedOptions[0].textContent : c.cohort;
  $('cfgsummary').textContent =
    `— ${name}${c.cutoff ? ' (' + c.cutoff + ')' : ''} · cal:${SPECTRA_SHORT[c.spectra] || c.spectra}` +
    ` · ${MODE_SHORT[c.mode] || c.mode} · k:${$('kmode').value === 'manual' ? $('kval').value : 'auto'}` +
    (c.eval_lot !== 'all' ? ` · eval lot ${c.eval_lot}` : '');
}
let cfgCollapsed = localStorage.getItem('calib_explorer_cfgcollapsed') === '1';
function setCfgCollapsed(v){
  cfgCollapsed = v;
  $('configcard').classList.toggle('collapsed', v);
  $('cfgtoggle').querySelector('.chev').textContent = v ? '▸' : '▾';
  localStorage.setItem('calib_explorer_cfgcollapsed', v ? '1' : '0');
}
$('cfgtoggle').onclick = () => setCfgCollapsed(!cfgCollapsed);
setCfgCollapsed(cfgCollapsed);

$('sweep').onclick = async () => {
  if(!last) return;
  busy(true); showError(null); $('sweep').disabled = true;
  try{
    const a = last.auto_k;
    const hi = Math.min(Math.max(2 * a, a + 6), 20, parseInt(last.curve[last.curve.length - 1].n_components));
    const ks = [...new Set(Array.from({length: 8}, (_, i) => Math.round(a + i * (hi - a) / 7)))];
    const j = await post('/api/sweep', {...cfg(), ks});
    if(j.error){ showError(j.error); return; }
    lastSweep = j.rows; lastSweepCfg = JSON.stringify(cfg());
    drawSweep(lastSweep);
    if(activeTab() !== 'calibrate') switchTab('calibrate');
  } finally { busy(false); $('sweep').disabled = false; }
};

/* ---- drawing --------------------------------------------------------------- */
function redraw(){
  if(!last) return;
  updateToggles();
  drawStats(); drawCurve(); drawCross(); drawMetrics(); drawSeries(); drawPins();
  if(lastSweep) drawSweep(lastSweep);
}

function drawStats(){
  const j = last, c = j.config;
  const held = j.heldout ?
    `held-out TOR: R² <b>${j.heldout.R2.toFixed(2)}</b>, slope ${j.heldout.slope.toFixed(2)}, RMSE ${j.heldout.RMSE.toFixed(2)}` :
    '<span class="muted">held-out TOR: none (options B/B2 fit on all filters)</span>';
  $('stats').innerHTML = `
    <b>${j.cohort_label}</b><br>
    select on: <b>${c.selection_space === 'airspec' ? 'AIRSpec-corrected' : 'raw'}</b>
    · calibrate on: <b>${SPECTRA_LABEL[c.spectra] || c.spectra}</b><br>
    protocol: <b>${MODE_LABEL[c.mode] || c.mode}</b>
    · target: <b>${j.target ? j.target.label : 'Addis'}</b>${
      j.target && j.target.eval_lot && j.target.eval_lot !== 'all'
        ? ` · eval lot <b>${j.target.eval_lot}</b> (n=${j.target.n_eval})` : ''}${
      j.target && j.target.extrap_pct != null
        ? ` · extrapolation: <span class="${j.target.extrap_pct > 30 ? 'warn' : 'ok'}">${j.target.extrap_pct}%</span> of target beyond train p95` : ''}<br>
    n = <b>${j.n_cohort}</b> filters, ${j.n_train_sites} sites · n fitted = ${j.n_train}<br>
    k = <b>${j.k}</b> ${j.k === j.auto_k ? '(rule choice)' : '(manual; rule picks ' + j.auto_k + ')'}<br>
    RMSECV floor: ${j.rmsecv_floor} µg (${j.pct_rmsecv_floor}% of mean loading)<br>
    ${held}<br>
    ${j.eth_corr_coverage && c.selection_space === 'airspec' && c.cohort === 'eth_shaped' ? '<span class="muted">' + j.eth_corr_coverage + '</span><br>' : ''}
    ${j.analog_corr_coverage && c.selection_space === 'airspec' && c.cohort === 'analogs' ? '<span class="muted">' + j.analog_corr_coverage + '</span><br>' : ''}
    <span class="muted">${j.curve_cached ? 'curve from cache' : 'curve computed'} · ${j.elapsed_s}s</span>`;
}

function drawCurve(){
  const cv = last.curve, ks = cv.map(r => r.n_components), rm = cv.map(r => r.rmsecv);
  const data = [];
  const se = cv.map(r => r.rmse_se);
  if(se.some(v => v != null && isFinite(v))){
    data.push({x: ks.concat([...ks].reverse()),
      y: rm.map((v, i) => v + (se[i] || 0)).concat(rm.map((v, i) => v - (se[i] || 0)).reverse()),
      fill:'toself', fillcolor:'rgba(44,110,158,.13)', line:{width:0}, hoverinfo:'skip', showlegend:false, type:'scatter'});
  }
  data.push({x:ks, y:rm, mode:'lines+markers', line:{color:'#2C6E9E'}, marker:{size:5}, name:'RMSECV', type:'scatter'});
  data.push({x:[last.auto_k], y:[rm[ks.indexOf(last.auto_k)]], mode:'markers',
    marker:{size:13, symbol:'circle-open', color:'#22252A', line:{width:2}}, name:'rule choice', type:'scatter'});
  if(last.k !== last.auto_k)
    data.push({x:[last.k], y:[rm[ks.indexOf(last.k)]], mode:'markers',
      marker:{size:13, symbol:'diamond', color:'#B23327'}, name:'manual k', type:'scatter'});
  plot('p_curve', data, {
    xaxis:{title:'PLS components'}, yaxis:{title:'RMSECV (µg/filter)'},
    margin:{t:30, r:10, b:45, l:55}, legend:LEGEND_TOP});
  $('p_curve').on('plotly_click', ev => {
    $('kmode').value = 'manual'; $('kval').disabled = false; $('kval').value = ev.points[0].x;
    run();
  });
}

function evalContext(){
  const t = last.target, e = last.eval;
  const isFabs = t.ref_kind === 'fabs';
  const mac = isFabs ? parseFloat(toggles.mac) : 1;
  const useFixed = toggles.evalset === 'fixed' && t.has_fixed;
  const idx = e.ref.map((_, i) => i).filter(i => !useFixed || e.fixed[i]);
  const xDesc = isFabs ? `HIPS EC-equivalent, Fabs/${mac} (µg/m³)` : 'Reference EC (µg/m³)';
  const setDesc = (useFixed ? 'fixed cohort' : 'all pairs') +
    (t.eval_lot && t.eval_lot !== 'all' ? ` · lot ${t.eval_lot}` : '');
  return {t, e, isFabs, mac, useFixed, idx, xDesc, setDesc};
}
function groupTraces(idx, e, xOf, yOf){
  const byGroup = {};
  idx.forEach(i => { (byGroup[e.group[i]] = byGroup[e.group[i]] || []).push(i); });
  return Object.entries(byGroup).map(([g, ii], gi) => ({
    x: ii.map(xOf), y: ii.map(yOf), mode:'markers', name:g,
    marker:{size:6, color:groupColour(g, gi), opacity:.65}, type:'scatter'}));
}

function drawCross(){
  const {t, e, isFabs, mac, idx, xDesc, setDesc} = evalContext();
  $('cap_cross').textContent =
    `${t.label} crossplot — ${isFabs ? 'MAC ' + mac : 'EC reference'} · ${setDesc}`;
  const data = groupTraces(idx, e, i => e.ref[i] / mac, i => e.pred[i]);
  const xs = idx.map(i => e.ref[i] / mac), ys = idx.map(i => e.pred[i]);
  const hi = Math.max(...xs, ...ys) * 1.06, lo = Math.min(0, ...ys) * 1.05;
  data.push({x:[0, hi], y:[0, hi], mode:'lines', line:{dash:'dash', color:'#999', width:1}, name:'1:1', type:'scatter'});
  const m = metricRow(last.metrics);
  if(m){
    const sl = toggles.est === 'deming' ? m.deming_slope : m.ols_slope;
    const ic = toggles.est === 'deming' ? m.deming_intercept : m.ols_intercept;
    data.push({x:[0, hi], y:[ic, sl * hi + ic], mode:'lines', line:{color:'#22252A', width:2},
      name:`${toggles.est === 'deming' ? 'Deming' : 'OLS'}: y=${sl.toFixed(2)}x${ic >= 0 ? '+' : ''}${ic.toFixed(2)}`, type:'scatter'});
  }
  plot('p_cross', data, {
    // scaleanchor keeps µg/m³-per-pixel identical on both axes, so 1:1 is 45°
    xaxis:{title:xDesc, range:[0, hi], constrain:'domain'},
    yaxis:{title:'Predicted FTIR EC (µg/m³)', range:[lo, hi],
           scaleanchor:'x', scaleratio:1, constrain:'domain'},
    margin:{t:52, r:10, b:45, l:55}, legend:LEGEND_TOP});
  drawResiduals();
}

function drawResiduals(){
  // the meeting's residual check: does the correction remove the curve?
  const {e, mac, idx, xDesc, setDesc} = evalContext();
  $('cap_resid').textContent = `Residuals (predicted − reference) vs reference — ${setDesc}`;
  const data = groupTraces(idx, e, i => e.ref[i] / mac, i => e.pred[i] - e.ref[i] / mac);
  plot('p_resid', data, {
    xaxis:{title:xDesc},
    yaxis:{title:'residual (µg/m³)', zeroline:true, zerolinecolor:'#22252A', zerolinewidth:1.5},
    margin:{t:30, r:10, b:42, l:55}, legend:LEGEND_TOP});
}

function drawSeries(){
  const {e, t} = {e: last.eval, t: last.target};
  const dated = e.date ? e.date.map((d, i) => d ? {d: new Date(d), i} : null).filter(Boolean) : [];
  if(!dated.length){
    $('p_series').innerHTML = '<div class="ph">No dates for this target — add a Date column to reference.csv.</div>';
    $('seriesinfo').querySelector('tbody').innerHTML = '';
    $('p_vsdeployed').innerHTML = '<div class="ph">—</div>';
    return;
  }
  dated.sort((a, b) => a.d - b.d);
  $('cap_series').textContent = `${t.label} — predicted EC by date (this run)`;
  const data = groupTraces(dated.map(x => x.i), e, i => e.date[i], i => e.pred[i]);
  // 45-day rolling median (ftir_29's series view)
  const roll = dated.map(({d, i}) => {
    const win = dated.filter(o => Math.abs(o.d - d) <= 45 * 864e5).map(o => e.pred[o.i]).sort((x, y) => x - y);
    return {x: e.date[i], y: win[Math.floor(win.length / 2)]};
  });
  data.push({x: roll.map(r => r.x), y: roll.map(r => r.y), mode:'lines',
    line:{color:'#22252A', width:2}, name:'45-day rolling median', type:'scatter'});
  plot('p_series', data, {
    xaxis:{title:'sampling date'}, yaxis:{title:'predicted EC (µg/m³)',
    zeroline:true, zerolinecolor:'#B23327'},
    margin:{t:30, r:10, b:45, l:55}, legend:LEGEND_TOP});

  // plausibility card (ftir_29's checks, generalized)
  const preds = dated.map(x => e.pred[x.i]).sort((a, b) => a - b);
  const q = p => preds[Math.floor(p * (preds.length - 1))];
  const neg = preds.filter(v => v < 0).length, high = preds.filter(v => v > 8).length;
  const groups = {};
  dated.forEach(({i}) => { (groups[e.group[i]] = groups[e.group[i]] || []).push(e.pred[i]); });
  const gmed = Object.entries(groups).map(([g, v]) => {
    v.sort((a, b) => a - b); return `${g}: ${v[Math.floor(v.length / 2)].toFixed(2)}`; }).join(' · ');
  $('seriesinfo').querySelector('tbody').innerHTML = `
    <tr><td>n dated</td><td>${preds.length}</td></tr>
    <tr><td>median [IQR] (µg/m³)</td><td>${q(.5).toFixed(2)} [${q(.25).toFixed(2)}–${q(.75).toFixed(2)}]</td></tr>
    <tr><td>range</td><td>${preds[0].toFixed(2)} → ${preds[preds.length - 1].toFixed(2)}</td></tr>
    <tr><td>negative days</td><td class="${neg ? 'warn' : 'ok'}">${neg} (${(100 * neg / preds.length).toFixed(1)}%)</td></tr>
    <tr><td>days &gt; 8 µg/m³</td><td class="${high ? 'warn' : 'ok'}">${high}</td></tr>
    <tr><td>group medians</td><td>${gmed}</td></tr>`;

  // vs deployed (built-in Addis only — same filters)
  const dep = e.deployed;
  if(dep && dep.some(v => v != null)){
    const ii = e.pred.map((_, i) => i).filter(i => dep[i] != null);
    const hi2 = Math.max(...ii.map(i => Math.max(e.pred[i], dep[i]))) * 1.06;
    plot('p_vsdeployed', [
      ...groupTraces(ii, e, i => dep[i], i => e.pred[i]),
      {x:[0, hi2], y:[0, hi2], mode:'lines', line:{dash:'dash', color:'#999', width:1}, name:'1:1', type:'scatter'}],
      {xaxis:{title:'deployed SPARTAN EC (µg/m³)', range:[0, hi2], constrain:'domain'},
       yaxis:{title:'this run (µg/m³)', range:[0, hi2],
              scaleanchor:'x', scaleratio:1, constrain:'domain'},
       margin:{t:30, r:10, b:45, l:55}, legend:LEGEND_TOP});
    $('cap_vsdep').textContent = `vs deployed SPARTAN EC — same ${ii.length} filters`;
  }else{
    $('p_vsdeployed').innerHTML = '<div class="ph">No deployed-EC series for this target.</div>';
  }
}

function drawMetrics(){
  const active = metricRow(last.metrics);   // the row the MAC / Addis-set pills select
  $('metrics').querySelector('tbody').innerHTML = last.metrics.map(r => `<tr${r === active ? ' class="sel"' : ''}>
    <td>${r.evaluation_set === 'fixed' ? 'fixed cohort' : 'all pairs'}</td>
    <td>${r.MAC ?? '—'}</td><td>${r.n}</td>
    <td class="gs">${r.ols_slope.toFixed(2)}</td><td>${r.ols_intercept.toFixed(2)}</td>
    <td class="gs">${r.deming_slope.toFixed(2)}</td><td>${r.deming_intercept.toFixed(2)}</td>
    <td class="gs">${r.R2.toFixed(2)}</td><td>${r.RMSE.toFixed(2)}</td>
    <td class="gs">${(-r.ols_intercept / r.ols_slope).toFixed(2)}</td>
    <td>${(-r.deming_intercept / r.deming_slope).toFixed(2)}</td></tr>`).join('');
}

let rankView = 'rank';
let lastRanking = null;      // cached /api/ranking payload for the current selection
let sliderTimer = null;

function metricToRank(value){
  // invert the (ascending) metric curve: how many candidates fall at or below value
  if(!lastRanking) return null;
  const {rank, metric} = lastRanking;
  let lo = 0, hi = metric.length - 1;
  while(lo < hi){ const mid = (lo + hi) >> 1; metric[mid] < value ? lo = mid + 1 : hi = mid; }
  return Math.max(50, rank[lo] || 50);
}

function cutoffChanged(fromSlider){
  if(!fromSlider) $('cutslider').value = parseInt($('cutoff').value) || 0;
  $('cutlabel').textContent = autoRun
    ? `cutoff ${$('cutoff').value}`
    : `cutoff ${$('cutoff').value} — press Run to recalibrate`;
  renderRanking();
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(() => { drawOverlap(); drawCohortInfo(); }, 500);
  updateCfgSummary(); scheduleAutoRun();   // slider lives outside the config card
}

async function drawRanking(){
  const c = $('cohort').value;
  if(!ready) return;
  drawCohortInfo();
  if(!RANKED.includes(c)){
    lastRanking = null; lastRankingBoth = null;
    Plotly.purge('p_ranking');
    $('p_ranking').innerHTML = '<div class="ph">No cutoff for this cohort — membership is fixed.</div>';
    $('cutlabel').textContent = 'cohort membership is fixed';
    return;
  }
  if(rankView === 'both'){ await drawRankingBoth(); return; }
  const j = await post('/api/ranking', {cohort:c,
    selection_space: $('selection_space').disabled ? 'raw' : $('selection_space').value,
    cutoff: parseInt($('cutoff').value) || null});
  if(j.error){ lastRanking = null; $('p_ranking').innerHTML = '<div class="ph warn">' + j.error + '</div>'; return; }
  lastRanking = j;
  $('cutslider').min = 50; $('cutslider').max = j.n_total;
  $('cutslider').value = parseInt($('cutoff').value) || j.default_cutoff;
  $('cutlabel').textContent = `cutoff ${$('cutslider').value} of ${j.n_total}`;
  renderRanking();
}

/* raw vs corrected selection — Ann (Aug 19): does baselining change who gets
   picked, and do the metric curves change shape? Same committed metric, run in
   both spectra spaces, overlaid; the caption reports shared membership at the
   current cutoff from the overlap machinery. */
let lastRankingBoth = null;
async function drawRankingBoth(){
  const c = $('cohort').value;
  if(!(c === 'eth_shaped' || c === 'analogs')){
    lastRankingBoth = null;
    Plotly.purge('p_ranking');
    $('p_ranking').innerHTML = '<div class="ph">Raw-vs-corrected selection exists only for the spectra-based cohorts (Ethiopia-shaped, spectral analogs).</div>';
    return;
  }
  const cutoff = parseInt($('cutoff').value) || null;
  $('cap_ranking').textContent = 'Selection, raw vs corrected — loading both spaces…';
  const [raw, corr] = await Promise.all([
    post('/api/ranking', {cohort: c, selection_space: 'raw', cutoff}),
    post('/api/ranking', {cohort: c, selection_space: 'airspec', cutoff})]);
  if(raw.error || corr.error){
    lastRankingBoth = null;
    $('p_ranking').innerHTML = '<div class="ph warn">' + (raw.error || corr.error) + '</div>';
    return;
  }
  lastRankingBoth = {raw, corr};
  lastRanking = raw;                       // slider + click mapping follow raw space
  $('cutslider').min = 50; $('cutslider').max = raw.n_total;
  $('cutslider').value = parseInt($('cutoff').value) || raw.default_cutoff;
  $('cutlabel').textContent = `cutoff ${$('cutslider').value} of ${raw.n_total}`;
  renderRankingBoth();
  const body = {}; body[c === 'eth_shaped' ? 'eth_cutoff' : 'analog_cutoff'] = cutoff;
  const ov = await post('/api/overlap', body);
  if(!ov.error){
    const fam = c === 'eth_shaped' ? 'Ethiopia-shaped' : 'Spectral analogs';
    const row = ov.rows.find(r => r.a.startsWith(fam) && r.b.startsWith(fam) &&
      /raw/.test(r.a + r.b) && /corrected/.test(r.a + r.b));
    if(row) $('cap_ranking').textContent =
      `Selection, raw vs corrected — ${row.overlap} of ${Math.min(row.n_a, row.n_b)} filters shared at this cutoff (${Math.round(100 * row.overlap / Math.min(row.n_a, row.n_b))}%)`;
  }
}
function renderRankingBoth(){
  if(!lastRankingBoth) return;
  const {raw, corr} = lastRankingBoth;
  const cut = Math.min(parseInt($('cutoff').value) || raw.default_cutoff, raw.n_total);
  const all = raw.metric.concat(corr.metric);
  plot('p_ranking', [
    {x: raw.rank, y: raw.metric, mode:'lines', name:'selected on raw',
     line:{color:'#2C6E9E'}, type:'scatter'},
    {x: corr.rank, y: corr.metric, mode:'lines', name:'selected on AIRSpec-corrected',
     line:{color:'#7A4FA3'}, type:'scatter'},
    {x:[cut, cut], y:[Math.min(...all), Math.max(...all)], mode:'lines',
     name:`cutoff ${cut}`, line:{color:'#B23327', dash:'dash'}, type:'scatter'}],
    {xaxis:{title:'rank'}, yaxis:{title: raw.label, automargin:true},
     margin:{t:34, r:10, b:40, l:60}, legend: LEGEND_TOP});
  $('p_ranking').on('plotly_click', ev => {
    $('cutoff').value = Math.round(ev.points[0].x);
    cutoffChanged(false);
  });
}

function renderRanking(){
  if(rankView === 'both'){ renderRankingBoth(); return; }
  const j = lastRanking;
  if(!j) return;
  const cut = Math.min(parseInt($('cutoff').value) || j.default_cutoff, j.n_total);
  // metric value at the cutoff rank, from the cached curve
  let ci = 0; while(ci < j.rank.length - 1 && j.rank[ci] < cut) ci++;
  const cutMetric = j.metric[ci];
  if(rankView === 'rank'){
    const selIdx = j.rank.filter(r => r <= cut).length;
    plot('p_ranking', [
      {x:j.rank.slice(0, selIdx), y:j.metric.slice(0, selIdx), mode:'lines',
       line:{color:'#2C6E9E', width:2.5}, name:'selected', fill:'tozeroy',
       fillcolor:'rgba(44,110,158,.08)', type:'scatter'},
      {x:j.rank.slice(Math.max(0, selIdx - 1)), y:j.metric.slice(Math.max(0, selIdx - 1)),
       mode:'lines', line:{color:'#7A4FA3'}, name:'rest of pool', type:'scatter'},
      {x:[cut, cut], y:[Math.min(...j.metric), Math.max(...j.metric)], mode:'lines',
       line:{color:'#B23327', dash:'dash'}, name:`cutoff ${cut}`, type:'scatter'}],
      {xaxis:{title:'rank'}, yaxis:{title:j.label, automargin:true},
       margin:{t:12, r:10, b:40, l:60}, showlegend:false});
  }else{
    const h = j.hist;
    const colours = h.centers.map(c2 => c2 <= cutMetric ? '#2C6E9E' : '#c9c9c9');
    plot('p_ranking', [
      {x:h.centers, y:h.counts, type:'bar', marker:{color:colours},
       name:'IMPROVE candidates'},
      {x:[cutMetric, cutMetric], y:[0, Math.max(...h.counts)], mode:'lines',
       line:{color:'#B23327', dash:'dash'}, name:'cutoff', type:'scatter'}],
      {xaxis:{title:j.label + ' (clipped at p99)'}, yaxis:{title:'filters'},
       bargap:0, margin:{t:12, r:10, b:42, l:55}, showlegend:false});
  }
  $('p_ranking').on('plotly_click', ev => {
    const x = ev.points[0].x;
    $('cutoff').value = rankView === 'rank' ? Math.round(x) : metricToRank(x);
    cutoffChanged(false);
  });
}

$('rankview').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('rankview').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); rankView = b.dataset.v;
  if(rankView === 'both') drawRankingBoth();
  else {
    // restore the single-space caption the both-view may have replaced
    $('cap_ranking').textContent = 'Selection — click the plot or drag the slider to move the cutoff';
    renderRanking();
  }
});
$('cutslider').oninput = () => {
  $('cutoff').value = $('cutslider').value;
  cutoffChanged(true);
};

async function drawCohortInfo(){
  const j = await post('/api/cohort_info', cfg());
  const tb = $('cohortinfo').querySelector('tbody');
  if(j.error){ tb.innerHTML = `<tr><td>—</td><td class="warn">${j.error}</td></tr>`; return; }
  const s = v => v ? `${v.min} / ${v.median} / ${v.max}` : '—';
  tb.innerHTML = `
    <tr><td>cohort</td><td>${j.label} — n=${j.n}, ${j.n_sites} sites</td></tr>
    <tr><td>top sites</td><td>${j.top_sites.join(', ')}</td></tr>
    <tr><td>filter lots</td><td>${Object.entries(j.lots).map(([l, n]) => `${l}: ${n}`).join(' · ')}</td></tr>
    <tr><td>TOR EC (µg/filter)</td><td>${s(j.ec_loading_ug)} <span class="muted">(min / median / max)</span></td></tr>
    <tr><td>TOR EC (ng/m³)</td><td>${s(j.ec_ugm3)} <span class="muted">(local_db Value is ng/m³)</span></td></tr>
    <tr><td>OC/EC ratio</td><td>${s(j.ocec_ratio)}</td></tr>
    <tr><td>dates</td><td>${j.date_range ? j.date_range.join(' → ') : '—'}</td></tr>`;
  if(j.composition){
    const c = j.composition;
    const poolMax = Math.max(...c.pool), cohortMax = Math.max(...c.cohort) || 1;
    plot('p_composition', [
      {x:c.centers, y:c.pool.map(v => v / poolMax), type:'bar', name:'IMPROVE pool',
       marker:{color:'#d9d9d9'}},
      {x:c.centers, y:c.cohort.map(v => v / cohortMax), type:'bar', name:'this cohort',
       marker:{color:'rgba(44,110,158,.75)'}},
      {x:[c.addis_marker, c.addis_marker], y:[0, 1], mode:'lines', name:'Addis (FTIR-derived) 1.34',
       line:{color:'#B23327', dash:'dash', width:2}, type:'scatter'},
      {x:[c.pool_median, c.pool_median], y:[0, 1], mode:'lines', name:`pool median ${c.pool_median}`,
       line:{color:'#666', dash:'dot', width:1.5}, type:'scatter'}],
      {barmode:'overlay', bargap:0,
       xaxis:{title:'TOR OC/EC ratio (clipped at 25)'}, yaxis:{title:'relative freq.', showticklabels:false},
       margin:{t:26, r:8, b:40, l:40}, legend:LEGEND_TOP});
  }
}

const spectraBand = (wn, b, rgb, name, withBand=true) => {
  const traces = [];
  if(withBand) traces.push(
    {x: wn.concat([...wn].reverse()), y: b.q75.concat([...b.q25].reverse()),
     fill:'toself', fillcolor:`rgba(${rgb},.15)`, line:{width:0}, hoverinfo:'skip', showlegend:false, type:'scatter'});
  traces.push({x:wn, y:b.median, mode:'lines', line:{color:`rgb(${rgb})`}, name, type:'scatter'});
  return traces;
};

async function drawSpectra(){
  const body = {...cfg()};
  if(spectraMode === 'compare') body.compare = true;
  if(spectraMode === 'clusters') body.clusters = 3;
  const j = await post('/api/spectra', body);
  if(j.error){ $('p_spectra').innerHTML = '<div class="ph warn">' + j.error + '</div>'; return; }
  const spaceLabel = SPECTRA_LABEL[j.spectra] || j.spectra;
  const yTitle = j.spectra === 'deriv2' ? '2nd-derivative absorbance' : 'absorbance';
  let data;
  if(j.series){
    // multi-series: Ann's all-cohorts ask, or Satoshi's k-means sub-types
    $('cap_spectra').textContent = spectraMode === 'clusters'
      ? `Spectral sub-types within ${j.cohort_label || 'cohort'} (${spaceLabel}) — k-means medians, reference + IQR`
      : `Selection cohorts vs ${(j.reference_label || 'reference').split(' — ')[0]} (${spaceLabel}) — medians`;
    const COLOURS = ['122,79,163', '44,110,158', '143,140,132', '196,148,66', '84,140,102', '150,90,90'];
    data = j.series.flatMap((s, i) => spectraBand(j.wn, s, COLOURS[i % COLOURS.length], s.label, false))
      .concat(spectraBand(j.wn, j.reference, '178,51,39',
        (j.reference_label || 'reference').split(' — ')[0] + ' median'));
  }else{
    $('cap_spectra').textContent =
      `Spectra: cohort vs ${(j.reference_label || 'reference').split(' — ')[0]} (${spaceLabel}) — median + IQR`;
    data = [
      ...spectraBand(j.wn, j.cohort, '44,110,158', `cohort median (n=${j.n})`),
      ...spectraBand(j.wn, j.reference, '178,51,39',
        (j.reference_label || 'reference').split(' — ')[0] + ' median')];
  }
  plot('p_spectra', data,
    {xaxis:{title:'wavenumber (cm⁻¹)', autorange:'reversed'},
     yaxis:{title:yTitle}, margin:{t:46, r:10, b:40, l:55}, legend:LEGEND_TOP});
}

$('spectramode').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('spectramode').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); spectraMode = b.dataset.v; drawSpectra();
});

async function drawOverlap(){
  const c = cfg();
  const body = {};
  if(c.cohort === 'eth_shaped' && c.cutoff) body.eth_cutoff = c.cutoff;
  if(c.cohort === 'analogs' && c.cutoff) body.analog_cutoff = c.cutoff;
  if(c.cohort === 'ocec' && c.cutoff) body.ocec_cutoff = c.cutoff;
  const j = await post('/api/overlap', body);
  if(j.error) return;
  $('overlap').querySelector('tbody').innerHTML = j.rows.map(r => `<tr>
    <td>${r.a}</td><td>${r.b}</td><td>${r.n_a}</td><td>${r.n_b}</td>
    <td>${r.overlap}</td><td>${(100 * r.overlap / Math.min(r.n_a, r.n_b)).toFixed(0)}%</td></tr>`).join('');

  // pairwise matrix (lower triangle) — one sequential hue, every cell labelled
  const names = [];
  j.rows.forEach(r => { if(!names.includes(r.a)) names.push(r.a); if(!names.includes(r.b)) names.push(r.b); });
  const short = n => n.replace(/\s*\(\d+\)$/, '').replace('Ethiopia-shaped', 'Eth-shaped')
                      .replace('Spectral analogs', 'Analogs').replace('corrected', 'corr.');
  const byPair = {};
  j.rows.forEach(r => { byPair[r.a + '|' + r.b] = r; });
  const xN = names.slice(0, -1), yN = names.slice(1);
  const z = [], hover = [], ann = [];
  yN.forEach((yn, yi) => {
    const zr = [], hr = [];
    xN.forEach((xn, xi) => {
      const r = xi <= yi ? byPair[xn + '|' + yn] : null;
      if(!r){ zr.push(null); hr.push(''); return; }
      const pct = Math.round(100 * r.overlap / Math.min(r.n_a, r.n_b));
      zr.push(pct);
      hr.push(`${r.a} (n=${r.n_a})<br>${r.b} (n=${r.n_b})<br>shared: ${r.overlap} — ${pct}% of the smaller`);
      ann.push({x: short(xn), y: short(yn), xref:'x', yref:'y', showarrow:false,
        text:`<b>${pct}%</b><br><span style="font-size:10px">${r.overlap}</span>`,
        font:{size:11, color: pct > 55 ? '#fff' : '#22252A'}});
    });
    z.push(zr); hover.push(hr);
  });
  plot('p_overlap', [{
    type:'heatmap', x:xN.map(short), y:yN.map(short), z, zmin:0, zmax:100,
    colorscale:[[0, '#f2f6fa'], [1, '#2C6E9E']], xgap:2, ygap:2,
    hoverongaps:false, text:hover, hoverinfo:'text', showscale:false}],
    {xaxis:{tickangle:-28, tickfont:{size:11}, automargin:true},
     yaxis:{autorange:'reversed', tickfont:{size:11}, automargin:true},
     margin:{t:6, r:6, b:10, l:10}, annotations:ann});
}

function drawSweep(rows){
  const est = toggles.est;
  // older cached responses lack the MAC-6 fields — fall back to MAC 10 then
  const mac6 = toggles.mac === '6' && rows.some(r => r.ols_intercept_mac6 != null);
  const icOf = r => mac6 ? (est === 'deming' ? r.deming_intercept_mac6 : r.ols_intercept_mac6)
                         : (est === 'deming' ? r.deming_intercept : r.ols_intercept);
  $('cap_sweep').textContent =
    `k sweep — intercept (fixed 190, MAC ${mac6 ? 6 : 10}, ${est === 'deming' ? 'Deming' : 'OLS'}) & held-out R²`;
  plot('p_sweep', [
    {x: rows.map(r => r.k), y: rows.map(icOf),
     mode:'lines+markers', name:'Addis intercept', line:{color:'#2C6E9E'}, type:'scatter'},
    {x: rows.map(r => r.k), y: rows.map(r => r.heldout_R2), mode:'lines+markers',
     name:'held-out TOR R²', line:{color:'#B23327', dash:'dot'}, yaxis:'y2', type:'scatter'}],
    {xaxis:{title:'PLS components'},
     yaxis:{title:'intercept (µg/m³)', zeroline:true, zerolinecolor:'#22252A'},
     yaxis2:{title:'held-out R²', overlaying:'y', side:'right', range:[0, 1]},
     margin:{t:30, r:55, b:40, l:55}, legend:LEGEND_TOP});
}

/* ---- pinned runs ----------------------------------------------------------- */
$('pin').onclick = () => {
  if(!last) return;
  pins.push({label:last.cohort_label, n:last.n_cohort, lot:last.config.lot,
    target:last.target ? last.target.label : 'Addis', ref_kind:last.target ? last.target.ref_kind : 'fabs',
    selection_space:last.config.selection_space,
    spectra:last.config.spectra, mode:last.config.mode, k:last.k, auto_k:last.auto_k,
    heldout:last.heldout, metrics:last.metrics, when:new Date().toISOString().slice(0, 16)});
  localStorage.setItem('calib_explorer_pins', JSON.stringify(pins));
  drawPins();
};
$('clearpins').onclick = () => { pins = []; localStorage.setItem('calib_explorer_pins', '[]'); drawPins(); };

function pinRow(p){
  const m = metricRow(p.metrics, p.ref_kind) || {};
  const sl = toggles.est === 'deming' ? m.deming_slope : m.ols_slope;
  const ic = toggles.est === 'deming' ? m.deming_intercept : m.ols_intercept;
  return {sl, ic, r2: m.R2};
}

function drawPins(){
  $('pincount').textContent = pins.length ? `(${pins.length})` : '';
  const tb = $('pins').querySelector('tbody');
  tb.innerHTML = pins.map((p, i) => {
    const v = pinRow(p);
    return `<tr><td>${p.label}</td><td>${p.n}</td>
      <td>${p.selection_space === 'airspec' ? 'AIRSpec' : 'raw'}</td>
      <td>${SPECTRA_SHORT[p.spectra] || p.spectra}</td>
      <td>${MODE_SHORT[p.mode] || p.mode}</td>
      <td>${p.k}${p.k !== p.auto_k ? '*' : ''}</td>
      <td>${v.sl != null ? v.sl.toFixed(2) : '–'}</td><td>${v.ic != null ? v.ic.toFixed(2) : '–'}</td>
      <td>${v.r2 != null ? v.r2.toFixed(2) : '–'}</td>
      <td>${p.heldout ? p.heldout.R2.toFixed(2) : '–'}</td>
      <td><button class="gray" onclick="removePin(${i})">✕</button></td></tr>`;
  }).join('');
  // two-line tick labels: the config detail on its own line keeps the label
  // column narrow enough that the ladder still has room on small screens
  const labels = pins.map(p =>
    `${p.label}<br>sel:${p.selection_space === 'airspec' ? 'AIR' : 'raw'} · cal:${SPECTRA_SHORT[p.spectra] || p.spectra} · ${MODE_SHORT[p.mode] || p.mode} · k${p.k}`);
  const ic = pins.map(p => pinRow(p).ic);
  plot('p_ladder', [{x:ic, y:labels, mode:'markers', type:'scatter',
    marker:{size:11, color:'#2C6E9E'}}], {
    xaxis:{zeroline:true, zerolinecolor:'#22252A', zerolinewidth:2},
    yaxis:{automargin:true, tickfont:{size:10}},
    margin:{t:12, r:14, b:30, l:10}, showlegend:false});
}
window.removePin = i => { pins.splice(i, 1); localStorage.setItem('calib_explorer_pins', JSON.stringify(pins)); drawPins(); };

$('exportcsv').onclick = () => {
  const head = 'cohort,n,selection_space,calibration_spectra,protocol,k,auto_k,mac,evaluation_set,estimator,slope,intercept,R2,RMSE,heldout_TOR_R2\n';
  const lines = [];
  pins.forEach(p => p.metrics.forEach(m => {
    [['ols', m.ols_slope, m.ols_intercept], ['deming', m.deming_slope, m.deming_intercept]].forEach(([e, sl, ic]) => {
      lines.push([`"${p.label}"`, p.n, p.selection_space, p.spectra, p.mode, p.k, p.auto_k, m.MAC, m.evaluation_set, e, sl, ic, m.R2, m.RMSE,
        p.heldout ? p.heldout.R2 : ''].join(','));
    });
  }));
  const blob = new Blob([head + lines.join('\n')], {type:'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'explorer_pinned_runs.csv'; a.click();
};

/* ---- optimizer --------------------------------------------------------------
   Staged search over cohort × cutoff × selection space × calibration spectra ×
   protocol × k for a small |Addis intercept| and a slope near 1, driven entirely
   through /api/run and /api/sweep so every number is the shared-script math and
   cached configurations are effectively free.
     phase 1  screen every selected combination at its default cutoff, rule k
     phase 2  cutoff ladder (×0.5 / ×1.5 / ×2) around the leading ranked cohorts
     phase 3  k sweep on the overall leaders
   Scoring reads the MAC · Fit toggles; the held-out TOR R² floor keeps the
   search from walking into configurations that no longer generalize. */
let optRows = [];                 // one row per evaluated (configuration, k)
let optRunning = false, optStop = false, optSkipped = 0;

const optCfgKey = r => [r.cohort, r.cutoff, r.selection_space, r.spectra, r.mode,
                        r.lot || 'all', r.target, r.eval_lot || 'all'].join('|');
const optCfgOf = r => ({cohort: r.cohort, cutoff: r.cutoff, selection_space: r.selection_space,
                        spectra: r.spectra, mode: r.mode, lot: r.lot || 'all', target: r.target,
                        eval_lot: r.eval_lot || 'all'});
const optDescribe = c => `${c.cohort}${c.cutoff ? ' ' + c.cutoff : ''}` +
  ` · sel:${c.selection_space === 'airspec' ? 'AIR' : 'raw'} · cal:${SPECTRA_SHORT[c.spectra]}` +
  ` · ${MODE_SHORT[c.mode] || c.mode}`;

function optParts(r){   // slope/intercept at the current MAC · Fit · Addis-set toggles
  const mac6 = toggles.mac === '6';
  // all-pairs readout when the pill says so AND the row carries it (rows saved
  // before the full-readout upgrade only have the fixed set until backfilled)
  const useAll = toggles.evalset === 'all' && r.all_deming_slope != null;
  const P = useAll ? 'all_' : '';
  let slope = toggles.est === 'deming' ? r[P + 'deming_slope'] : r[P + 'ols_slope'];
  const ic = toggles.est === 'deming'
    ? (mac6 ? r[P + 'deming_intercept_mac6'] : r[P + 'deming_intercept'])
    : (mac6 ? r[P + 'ols_intercept_mac6'] : r[P + 'ols_intercept']);
  if(mac6 && slope != null) slope *= 0.6;   // slope scales by MAC (ftir_19)
  return {slope, ic, set: useAll ? 'all pairs' : 'fixed set'};
}
function optScore(r){
  const {slope, ic} = optParts(r);
  if(slope == null || ic == null) return Infinity;
  return Math.abs(ic) + (parseFloat($('opt_w').value) || 0) * Math.abs(slope - 1);
}
function optPasses(r){
  const min = parseFloat($('opt_minr2').value);
  if(r.heldout_R2 == null)
    return !$('opt_reqho').checked;   // B/B2: no held-out set — unfalsifiable
  return !isFinite(min) || r.heldout_R2 >= min;
}
function optStatus(t){ $('opt_status').textContent = t; }

function optAddRow(row){
  if(optRows.some(r => optCfgKey(r) === optCfgKey(row) && r.k === row.k)) return;
  optRows.push(row);
}
function optRowFromRun(c, j){
  const pick = (es, mac) => j.metrics.find(m => m.evaluation_set === es && m.MAC === mac)
            || j.metrics.find(m => m.MAC === mac) || j.metrics[0];
  const f10 = pick('fixed', 10), f6 = pick('fixed', 6);
  const a10 = pick('all', 10), a6 = pick('all', 6);
  return {...c, cohort_label: j.cohort_label, k: j.k, auto_k: j.auto_k,
    ols_slope: f10.ols_slope, ols_intercept: f10.ols_intercept,
    deming_slope: f10.deming_slope, deming_intercept: f10.deming_intercept,
    ols_intercept_mac6: f6.ols_intercept, deming_intercept_mac6: f6.deming_intercept,
    R2: f10.R2, RMSE: f10.RMSE,
    all_ols_slope: a10.ols_slope, all_ols_intercept: a10.ols_intercept,
    all_deming_slope: a10.deming_slope, all_deming_intercept: a10.deming_intercept,
    all_ols_intercept_mac6: a6.ols_intercept, all_deming_intercept_mac6: a6.deming_intercept,
    all_R2: a10.R2, all_RMSE: a10.RMSE,
    heldout_R2: j.heldout ? j.heldout.R2 : null};
}
function optTopConfigs(n, rankedOnly){
  const best = {};
  const consider = r => {
    if(rankedOnly && !RANKED.includes(r.cohort)) return;
    const key = optCfgKey(r), s = optScore(r);
    if(!(key in best) || s < best[key].s) best[key] = {s, c: optCfgOf(r), label: r.cohort_label};
  };
  optRows.filter(optPasses).forEach(consider);
  if(!Object.keys(best).length) optRows.forEach(consider);   // nothing passes: fall back
  return Object.values(best).sort((a, b) => a.s - b.s).slice(0, n);
}

async function optPhase(name, combos){
  let i = 0;
  for(const c of combos){
    if(optStop) return;
    i++;
    if(optRows.some(r => optCfgKey(r) === optCfgKey(c))) continue;   // already evaluated
    optStatus(`${name} ${i}/${combos.length} — ${optDescribe(c)}…`);
    const j = await post('/api/run', {...c, k: null});
    if(j.error){ optSkipped++; continue; }
    optAddRow(optRowFromRun(c, j));
    renderOpt();
  }
}

async function optimize(){
  if(optRunning || !ready) return;
  const cohorts = [...document.querySelectorAll('[data-opt-cohort]:checked')].map(x => x.dataset.optCohort);
  const spectraOpts = [...document.querySelectorAll('[data-opt-spectra]:checked')].map(x => x.dataset.optSpectra);
  const modes = [...document.querySelectorAll('[data-opt-mode]:checked')].map(x => x.dataset.optMode);
  if(!cohorts.length || !spectraOpts.length || !modes.length){
    optStatus('pick at least one cohort, one spectra space and one protocol'); return;
  }
  optRunning = true; optStop = false; optSkipped = 0;
  $('opt_start').disabled = true; $('opt_stop').disabled = false;
  try{
    const target = $('target').value || 'addis';
    const evalLot = $('eval_lot').disabled ? 'all' : ($('eval_lot').value || 'all');
    // phase 1 — screen at default cutoffs, rule k
    const combos = [];
    cohorts.forEach(co => {
      const spaces = ($('opt_corrsel').checked && (co === 'eth_shaped' || co === 'analogs'))
        ? ['raw', 'airspec'] : ['raw'];
      spaces.forEach(ss => spectraOpts.forEach(sp => modes.forEach(m =>
        combos.push({cohort: co, cutoff: RANKED.includes(co) ? defaults[co] : null,
                     selection_space: ss, spectra: sp, mode: m, lot: 'all', target,
                     eval_lot: evalLot}))));
    });
    await optPhase('screening', combos);
    if(optStop) return;
    // phase 2 — cutoff ladder around the leading ranked-cohort configurations
    const ladder = [];
    optTopConfigs(3, true).forEach(({c}) =>
      [0.5, 1.5, 2].forEach(f =>
        ladder.push({...c, cutoff: Math.max(50, Math.round(defaults[c.cohort] * f / 50) * 50)})));
    await optPhase('cutoff refine', ladder);
    if(optStop) return;
    // phase 3 — k sweep on the overall leaders
    const finalists = optTopConfigs(4, false);
    for(let f = 0; f < finalists.length; f++){
      if(optStop) return;
      const {c, label} = finalists[f];
      optStatus(`k sweep ${f + 1}/${finalists.length} — ${optDescribe(c)}…`);
      const base = optRows.find(r => optCfgKey(r) === optCfgKey({...c, target: c.target}));
      const a = base ? base.auto_k : 6;
      const hi = Math.min(Math.max(2 * a, a + 6), 20);
      const ks = [...new Set(Array.from({length: 8}, (_, i) => Math.round(a + i * (hi - a) / 7)))];
      const j = await post('/api/sweep', {...c, ks});
      if(j.error){ optSkipped++; continue; }
      j.rows.forEach(r => optAddRow({...c, cohort_label: label || (base && base.cohort_label) || c.cohort,
        k: r.k, auto_k: r.auto_k, ols_slope: r.ols_slope, ols_intercept: r.ols_intercept,
        deming_slope: r.deming_slope, deming_intercept: r.deming_intercept,
        ols_intercept_mac6: r.ols_intercept_mac6, deming_intercept_mac6: r.deming_intercept_mac6,
        R2: r.R2, heldout_R2: r.heldout_R2}));
      renderOpt();
    }
  } finally {
    optRunning = false;
    $('opt_start').disabled = false; $('opt_stop').disabled = true;
    $('opt_export').disabled = !optRows.length;
    renderOpt();
    optStatus((optStop ? 'stopped — results kept. ' : 'done. ') +
      `${optRows.length} runs evaluated${optSkipped ? `, ${optSkipped} skipped (error)` : ''}. ` +
      'Click a leaderboard row’s Load (or a point) to open that run in the explorer.');
  }
}
$('opt_start').onclick = optimize;
$('opt_stop').onclick = () => { optStop = true; optStatus('stopping after the current evaluation…'); };

/* ---- exhaustive server-side batch -------------------------------------------
   Same grid checkboxes, but the loop runs inside the Flask process: it survives
   closing this page, appends every scored (configuration, k) row to
   cache/batch_results.jsonl, and "Load saved results" merges those rows —
   whether computed here, overnight, or in Colab — into the same leaderboard. */
let batchPollTimer = null;
async function batchTick(auto){
  let s;
  try { s = await (await fetch('/api/batch_status')).json(); }
  catch(e){ return; }
  $('batch_stop').disabled = !s.running;
  $('batch_start').disabled = !!s.running;
  if(s.total){
    $('batch_status').textContent = (s.running
      ? `running ${s.done}/${s.total} — ${s.current}`
      : `finished ${s.done}/${s.total} · ${s.new_rows} new rows saved`)
      + (s.skipped ? ` · ${s.skipped} configs failed` : '');
  }
  if(s.running){
    batchPollTimer = setTimeout(() => batchTick(true), 3000);
  } else if(auto && batchPollTimer){
    batchPollTimer = null;
    batchLoad();                       // batch just finished — pull its rows in
  }
}
$('batch_start').onclick = async () => {
  const cohorts = [...document.querySelectorAll('[data-opt-cohort]:checked')].map(x => x.dataset.optCohort);
  const spectra = [...document.querySelectorAll('[data-opt-spectra]:checked')].map(x => x.dataset.optSpectra);
  const modes = [...document.querySelectorAll('[data-opt-mode]:checked')].map(x => x.dataset.optMode);
  if(!cohorts.length || !spectra.length || !modes.length){
    $('batch_status').textContent = 'pick at least one cohort, spectra space and protocol'; return;
  }
  const r = await post('/api/batch_start', {
    cohorts, spectra, modes,
    corrsel: $('opt_corrsel').checked,
    cutoff_ladder: $('batch_ladder').checked,
    sweep_k: $('batch_sweepk').checked,
    target: $('target').value || 'addis',
    eval_lot: $('eval_lot').disabled ? 'all' : ($('eval_lot').value || 'all'),
  });
  if(r.error){ $('batch_status').textContent = r.error; return; }
  $('batch_status').textContent = `started — ${r.total} configurations queued`;
  batchPollTimer = setTimeout(() => batchTick(true), 1500);
};
$('batch_stop').onclick = async () => {
  await post('/api/batch_stop', {});
  $('batch_status').textContent = 'stopping after the current configuration…';
};
async function batchLoad(){
  const r = await (await fetch('/api/batch_results')).json();
  const before = optRows.length;
  (r.rows || []).forEach(optAddRow);
  $('opt_export').disabled = !optRows.length;
  renderOpt();
  optStatus(`${optRows.length - before} saved batch rows merged (${optRows.length} total). ` +
            'The leaderboard and tradeoff view now cover them; click Load on a row to open it.');
}
$('batch_load').onclick = batchLoad;

function renderOpt(){
  $('optcount').textContent = optRows.length ? `(${optRows.length})` : '';
  const w = parseFloat($('opt_w').value) || 0;
  const estL = toggles.est === 'deming' ? 'Deming' : 'OLS';
  const setL = toggles.evalset === 'all' ? 'all pairs' : 'fixed set';
  $('cap_optboard').textContent =
    `Leaderboard — score = |intercept| + ${w}·|slope − 1| at MAC ${toggles.mac}, ${estL}, ${setL}` +
    (toggles.evalset === 'all' ? ' (rows without an all-pairs readout fall back to fixed until backfilled)' : '');
  const view = optRows.map((r, i) => ({r, i, s: optScore(r), pass: optPasses(r), ...optParts(r)}))
    .sort((a, b) => (b.pass - a.pass) || (a.s - b.s));
  const shown = view.slice(0, 20);
  $('optboard').querySelector('tbody').innerHTML = shown.map((v, rank) => `
    <tr${v.pass ? '' : ' class="fail" title="below the held-out R² floor"'}>
      <td>${rank + 1}</td><td>${v.r.cohort_label || v.r.cohort}</td>
      <td>${v.r.selection_space === 'airspec' ? 'AIRSpec' : 'raw'}</td>
      <td>${SPECTRA_SHORT[v.r.spectra] || v.r.spectra}</td>
      <td title="${MODE_LABEL[v.r.mode] || v.r.mode}">${{site_heldout:'A', app:'B', app_fmm:'B2'}[v.r.mode] || v.r.mode}</td>
      <td>${v.r.k}${v.r.k !== v.r.auto_k ? '*' : ''}</td>
      <td>${v.slope != null ? v.slope.toFixed(2) : '–'}</td>
      <td>${v.ic != null ? v.ic.toFixed(2) : '–'}</td>
      <td>${v.r.heldout_R2 != null ? v.r.heldout_R2.toFixed(2) : '—'}</td>
      <td><b>${isFinite(v.s) ? v.s.toFixed(2) : '–'}</b></td>
      <td><button class="gray" onclick="optApply(${v.i})">Load</button></td></tr>`).join('') +
    (view.length > 20 ? `<tr><td colspan="11" class="muted">… ${view.length - 20} more (Export CSV for all)</td></tr>` : '');
  drawPareto(view);
}

function drawPareto(view){
  if(!view.length){ placeholder('p_pareto', 'Start a search to populate the tradeoff view.'); return; }
  $('cap_pareto').textContent =
    `Intercept vs slope at MAC ${toggles.mac}, ${toggles.est === 'deming' ? 'Deming' : 'OLS'}, ` +
    `${toggles.evalset === 'all' ? 'all pairs' : 'fixed set'} — the frontier is the tradeoff`;
  const pts = view.filter(v => v.slope != null && v.ic != null);
  const pass = pts.filter(v => v.pass), fail = pts.filter(v => !v.pass);
  // Pareto front over the passing runs: sort by |slope−1|, keep strict |intercept| improvements
  const sorted = [...pass].sort((a, b) => Math.abs(a.slope - 1) - Math.abs(b.slope - 1));
  const front = []; let bestIc = Infinity;
  sorted.forEach(v => { if(Math.abs(v.ic) < bestIc - 1e-12){ front.push(v); bestIc = Math.abs(v.ic); } });
  const trace = (arr, name, marker) => ({
    x: arr.map(v => Math.abs(v.slope - 1)), y: arr.map(v => Math.abs(v.ic)),
    customdata: arr.map(v => v.i), mode: 'markers', name, marker, type: 'scatter',
    text: arr.map(v => `${v.r.cohort_label || v.r.cohort} · cal:${SPECTRA_SHORT[v.r.spectra]}` +
      ` · ${MODE_SHORT[v.r.mode]} · k${v.r.k}<br>slope ${v.slope.toFixed(2)}, intercept ${v.ic.toFixed(2)}` +
      `${v.r.heldout_R2 != null ? ', held-out R² ' + v.r.heldout_R2.toFixed(2) : ''}<br>click to load`),
    hoverinfo: 'text'});
  const data = [];
  if(fail.length) data.push(trace(fail, 'below R² floor', {size: 6, color: '#c9c9c9', opacity: .6}));
  data.push(trace(pass, 'candidates', {size: 7, color: '#2C6E9E', opacity: .65}));
  if(front.length){
    data.push({x: front.map(v => Math.abs(v.slope - 1)), y: front.map(v => Math.abs(v.ic)),
      mode: 'lines', name: 'Pareto frontier', line: {color: '#B23327', width: 1.5, dash: 'dot'},
      hoverinfo: 'skip', type: 'scatter'});
    data.push({...trace(front, 'frontier runs', {size: 10, color: '#B23327', symbol: 'diamond'})});
  }
  plot('p_pareto', data, {
    xaxis: {title: '|slope − 1|', rangemode: 'tozero'},
    yaxis: {title: '|intercept| (µg/m³)', rangemode: 'tozero'},
    margin: {t: 30, r: 10, b: 45, l: 55}, legend: LEGEND_TOP});
  $('p_pareto').on('plotly_click', ev => {
    const i = ev.points[0].customdata;
    if(i != null) optApply(i);
  });
}

window.optApply = i => {
  const r = optRows[i];
  if(!r) return;
  applyPreset({cohort: r.cohort, cutoff: r.cutoff, selection_space: r.selection_space,
               spectra: r.spectra, mode: r.mode, target: r.target, lot: r.lot || 'all',
               eval_lot: r.eval_lot || 'all', kmode: 'manual', k: r.k});
  switchTab('calibrate');
  run();
};

$('opt_export').onclick = () => {
  const head = 'cohort,cutoff,selection_space,calibration_spectra,protocol,k,auto_k,' +
    'ols_slope_mac10,ols_intercept_mac10,deming_slope_mac10,deming_intercept_mac10,' +
    'ols_intercept_mac6,deming_intercept_mac6,R2,heldout_TOR_R2\n';
  const lines = optRows.map(r => [`"${r.cohort_label || r.cohort}"`, r.cutoff ?? '', r.selection_space,
    r.spectra, r.mode, r.k, r.auto_k, r.ols_slope, r.ols_intercept, r.deming_slope,
    r.deming_intercept, r.ols_intercept_mac6, r.deming_intercept_mac6, r.R2, r.heldout_R2 ?? ''].join(','));
  const blob = new Blob([head + lines.join('\n')], {type: 'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'explorer_optimizer_runs.csv'; a.click();
};
['opt_w', 'opt_minr2', 'opt_reqho'].forEach(id => $(id).onchange = renderOpt);
placeholder('p_pareto', 'Start a search to populate the tradeoff view.');

/* ---- analog lab -------------------------------------------------------------
   Compares the committed spectral-analog selection with literature similarity
   metrics (SAM/cosine, LOCAL's Pearson, normalized Euclidean, NN-cosine, and
   Reggente-2016's Mahalanobis-in-score-space as a selector), in raw, AIRSpec
   or 2nd-derivative space. The server sends full per-filter rank arrays once
   per space, so moving the cutoff recomputes everything client-side. */
let alData = {};
let alSpace = 'raw';
const AL_SHORT = {committed:'committed', cosine_median:'cos/SAM', corr_median:'Pearson',
  eucl_norm_median:'Eucl-n', nearest_cosine:'NN-cos', mahalanobis_pca:'Mahal-PCA'};
const AL_SPACE_LABEL = {raw:'raw spectra', airspec:'AIRSpec-corrected', deriv2:'SG 2nd derivative'};

async function alLoad(){
  if(alData[alSpace]){ alRender(); return; }
  $('cap_al').textContent = `Analog lab — computing similarity metrics on ${AL_SPACE_LABEL[alSpace]} (first time per space takes ~10-30s)…`;
  placeholder('p_al_overlap', 'computing…'); placeholder('p_al_rank', 'computing…'); placeholder('p_al_pca', 'computing…');
  const j = await post('/api/analog_lab', {space: alSpace});
  if(j.error){ $('cap_al').textContent = 'Analog lab — ' + j.error; return; }
  alData[alSpace] = j;
  $('al_cut').max = j.n;
  if(!$('al_metric').options.length)
    $('al_metric').innerHTML = j.metrics.filter(m => m !== 'committed')
      .map(m => `<option value="${m}"${m === 'corr_median' ? ' selected' : ''}>${j.labels[m]}</option>`).join('');
  alRender();
  alSpectra();
}

function alRender(){
  const j = alData[alSpace];
  if(!j) return;
  const cutoff = Math.min(parseInt($('al_cut').value) || 500, j.n);
  $('al_cutlabel').textContent = `cutoff ${cutoff} of ${j.n} eligible filters`;
  $('cap_al').textContent = `Analog lab — committed selection vs literature metrics (${AL_SPACE_LABEL[alSpace]})`;
  const ms = j.metrics;
  const mem = {};
  ms.forEach(m => {
    const r = j.ranks[m], s = new Uint8Array(j.n);
    for(let i = 0; i < j.n; i++) s[i] = r[i] < cutoff ? 1 : 0;
    mem[m] = s;
  });
  const z = [], ann = [];
  ms.forEach(a => {
    const row = [];
    ms.forEach(b => {
      let inter = 0;
      for(let i = 0; i < j.n; i++) if(mem[a][i] && mem[b][i]) inter++;
      const pct = Math.round(100 * inter / cutoff);
      row.push(pct);
      if(a !== b) ann.push({x: AL_SHORT[b], y: AL_SHORT[a], text: `${pct}%`, showarrow: false,
        font: {size: 10, color: pct > 55 ? '#fff' : '#22252A'}});
    });
    z.push(row);
  });
  plot('p_al_overlap', [{type:'heatmap', x: ms.map(m => AL_SHORT[m]), y: ms.map(m => AL_SHORT[m]),
    z, zmin: 0, zmax: 100, colorscale: [[0, '#f2f6fa'], [1, '#2C6E9E']], xgap: 2, ygap: 2, showscale: false}],
    {xaxis:{tickangle:-25, tickfont:{size:10}, automargin:true},
     yaxis:{autorange:'reversed', tickfont:{size:10}, automargin:true},
     margin:{t:6, r:6, b:10, l:10}, annotations: ann});
  $('al_agree').querySelector('tbody').innerHTML = Object.entries(j.agreement)
    .sort((a, b) => b[1] - a[1])
    .map(([m, rho]) => `<tr><td>${j.labels[m]}</td><td>ρ = ${rho.toFixed(3)}</td></tr>`).join('');

  const alt = $('al_metric').value || 'corr_median';
  const step = Math.max(1, Math.floor(j.n / 3000));
  const xs = [], ys = [], cols = [];
  for(let i = 0; i < j.n; i += step){
    xs.push(j.ranks.committed[i]); ys.push(j.ranks[alt][i]);
    const inC = j.ranks.committed[i] < cutoff, inA = j.ranks[alt][i] < cutoff;
    cols.push(inC && inA ? '#2C6E9E' : inC ? '#B23327' : inA ? '#548C66' : '#dcdcdc');
  }
  $('cap_al_rank').textContent =
    `Committed rank vs ${j.labels[alt]} — blue: both select · red: committed only · green: alternative only`;
  plot('p_al_rank', [
    {x: xs, y: ys, mode:'markers', marker:{size:3, color:cols}, type:'scattergl', hoverinfo:'skip'},
    {x:[cutoff, cutoff], y:[0, j.n], mode:'lines', line:{color:'#B23327', dash:'dot', width:1}, hoverinfo:'skip'},
    {x:[0, j.n], y:[cutoff, cutoff], mode:'lines', line:{color:'#548C66', dash:'dot', width:1}, hoverinfo:'skip'}],
    {xaxis:{title:'committed rank'}, yaxis:{title:'alternative rank'},
     showlegend:false, margin:{t:10, r:10, b:42, l:55}});

  const px = [], py = [], pc = [];
  j.sample_idx.forEach((idx, k) => {
    px.push(j.pool_xy[k][0]); py.push(j.pool_xy[k][1]);
    const inC = j.ranks.committed[idx] < cutoff, inA = j.ranks[alt][idx] < cutoff;
    pc.push(inC && inA ? '#2C6E9E' : inC ? '#B23327' : inA ? '#548C66' : '#e3e3e3');
  });
  const ax = j.addis_xy.map(p => p[0]), ay = j.addis_xy.map(p => p[1]);
  let layoutRanges = {}, excludedTrace = null;
  if($('al_focus').checked){
    // robust view: axes trimmed to the central 1-99% cloud; points outside are
    // CLAMPED to the edge as orange x (hover shows their true coordinates)
    const qb = arr => {
      const s = [...arr].sort((a, b) => a - b);
      const q = p => s[Math.floor(p * (s.length - 1))];
      const lo = q(0.01), hi = q(0.99), pad = (hi - lo) * 0.15 || 1e-6;
      return [lo - pad, hi + pad];
    };
    const bx = qb(px.concat(ax)), by = qb(py.concat(ay));
    const clamp = (v, b) => Math.min(Math.max(v, b[0]), b[1]);
    const ex = [], ey = [], etext = [];
    const sweep = (xs, ys, label) => {
      for(let i = 0; i < xs.length; i++){
        if(xs[i] < bx[0] || xs[i] > bx[1] || ys[i] < by[0] || ys[i] > by[1]){
          ex.push(clamp(xs[i], bx)); ey.push(clamp(ys[i], by));
          etext.push(`${label} — true PC1 ${xs[i].toFixed(3)}, PC2 ${ys[i].toFixed(3)}`);
        }
      }
    };
    sweep(px, py, 'pool'); sweep(ax, ay, 'Addis');
    layoutRanges = {xr: bx, yr: by};
    if(ex.length) excludedTrace = {x: ex, y: ey, mode:'markers',
      marker:{size:7, symbol:'x', color:'#C4652F'},
      name:`excluded from view (${ex.length}) — clamped to edge`,
      text: etext, hoverinfo:'text', type:'scatter'};
  }
  const data = [
    {x: px, y: py, mode:'markers', marker:{size:4, color:pc, opacity:.8},
     name:'pool (same colours as left)', type:'scattergl', hoverinfo:'skip'},
    {x: ax, y: ay, mode:'markers',
     marker:{size:6, symbol:'star', color:'#22252A'}, name:'Addis (ETAD)', type:'scatter'}];
  if(excludedTrace) data.push(excludedTrace);
  plot('p_al_pca', data,
    {xaxis:{title:`PC1 (${Math.round(j.explained[0]*100)}%)`,
            ...(layoutRanges.xr ? {range: layoutRanges.xr} : {})},
     yaxis:{title:`PC2 (${Math.round(j.explained[1]*100)}%)`,
            ...(layoutRanges.yr ? {range: layoutRanges.yr} : {})},
     legend: LEGEND_TOP, margin:{t:30, r:10, b:42, l:50}});
}

/* comparative spectra: the alt-metric cohort vs the committed cohort vs Addis,
   drawn in the lab's spectra space; fetched on slider release / metric change */
let alSpecKey = null;
async function alSpectra(){
  const j = alData[alSpace];
  if(!j) return;
  const cutoff = Math.min(parseInt($('al_cut').value) || 500, j.n);
  const alt = $('al_metric').value || 'corr_median';
  const key = `${alSpace}|${alt}|${cutoff}`;
  if(key === alSpecKey) return;
  alSpecKey = key;
  $('cap_al_spec').textContent = 'loading spectra…';
  const r = await post('/api/analog_lab_spectra', {space: alSpace, metric: alt, cutoff});
  if(r.error){ $('cap_al_spec').textContent = r.error; return; }
  if(alSpecKey !== key) return;                    // stale response
  $('cap_al_spec').textContent =
    `top ${r.n} under ${r.metric_label} (green) vs committed top ${r.n} (grey, median only) vs Addis (red) — ${AL_SPACE_LABEL[alSpace]}`;
  const yTitle = alSpace === 'deriv2' ? '2nd-derivative absorbance' : 'absorbance';
  plot('p_al_spectra', [
    ...spectraBand(r.wn, r.alt, '84,140,102', `selected by ${AL_SHORT[alt]} (n=${r.n})`),
    ...spectraBand(r.wn, r.committed, '143,140,132', 'committed cohort median', false),
    ...spectraBand(r.wn, r.addis, '178,51,39', 'Addis (ETAD) median')],
    {xaxis:{title:'wavenumber (cm⁻¹)', autorange:'reversed'},
     yaxis:{title:yTitle}, margin:{t:46, r:10, b:40, l:55}, legend:LEGEND_TOP});
}

$('al_space').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('al_space').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); alSpace = b.dataset.v; alLoad();
});
$('al_metric').onchange = () => { alRender(); alSpectra(); };
$('al_cut').oninput = alRender;
$('al_cut').onchange = alSpectra;      // fetch spectra on slider release only
$('al_focus').onchange = alRender;
document.querySelector('[data-tab="analoglab"]').addEventListener('click', () => {
  if(ready && !alData[alSpace]) alLoad();
});

/* ---- cutoff refinement (server-side hill-climb, +-10 steps) ----------------- */
$('refine_start').onclick = async () => {
  const cohorts = [...document.querySelectorAll('[data-opt-cohort]:checked')].map(x => x.dataset.optCohort);
  const spectra = [...document.querySelectorAll('[data-opt-spectra]:checked')].map(x => x.dataset.optSpectra);
  const modes = [...document.querySelectorAll('[data-opt-mode]:checked')].map(x => x.dataset.optMode);
  if(!cohorts.length || !spectra.length || !modes.length){
    $('batch_status').textContent = 'pick at least one cohort, spectra space and protocol'; return;
  }
  const r = await post('/api/refine_start', {cohorts, spectra, modes,
    corrsel: $('opt_corrsel').checked,
    w: parseFloat($('opt_w').value) || 5,
    min_r2: parseFloat($('opt_minr2').value),
    target: $('target').value || 'addis',
    eval_lot: $('eval_lot').disabled ? 'all' : ($('eval_lot').value || 'all')});
  if(r.error){ $('batch_status').textContent = r.error; return; }
  $('batch_status').textContent = `refining cutoffs for ${r.bases} base configurations in ±10 steps…`;
  batchPollTimer = setTimeout(() => batchTick(true), 1500);
};

/* ---- cross-site tab ---------------------------------------------------------
   One click: the current configuration evaluated against every target, with
   the Reggente-2016 extrapolation diagnostic flagging out-of-domain rows. */
let sitesRows = [];
function sitesPick(metrics){       // metrics row per the MAC/est/evalset toggles
  const hasFixed = metrics.some(r => r.evaluation_set === 'fixed');
  const es = (toggles.evalset === 'fixed' && hasFixed) ? 'fixed' : 'all';
  const sub = metrics.filter(r => r.evaluation_set === es);
  return sub.find(r => r.MAC === parseFloat(toggles.mac)) || sub[0] || metrics[0];
}
function sitesRender(){
  if(!sitesRows.length) return;
  const estL = toggles.est === 'deming' ? 'Deming' : 'OLS';
  $('cap_sites').textContent =
    `Cross-site — MAC ${toggles.mac}, ${estL}, ${toggles.evalset === 'all' ? 'all pairs' : 'fixed set'}`;
  const view = sitesRows.map(r => {
    if(r.error) return {site: r.site, error: r.error};
    const m = sitesPick(r.metrics);
    return {site: r.site, label: r.label, n: r.n, k: r.k,
      slope: toggles.est === 'deming' ? m.deming_slope : m.ols_slope,
      ic: toggles.est === 'deming' ? m.deming_intercept : m.ols_intercept,
      R2: m.R2, extrap: r.extrap_pct};
  });
  $('sitestbl').querySelector('tbody').innerHTML = view.map(v => v.error
    ? `<tr><td>${v.site}</td><td colspan="6" class="warn">${v.error}</td></tr>`
    : `<tr>
        <td title="${v.label}">${v.site}</td><td>${v.n}</td><td>${v.k}</td>
        <td>${v.slope.toFixed(2)}</td><td>${v.ic.toFixed(2)}</td><td>${v.R2.toFixed(2)}</td>
        <td class="${v.extrap != null && v.extrap > 30 ? 'warn' : ''}">${v.extrap != null ? v.extrap.toFixed(0) + '%' : '—'}</td>
      </tr>`).join('');
  const ok = view.filter(v => !v.error);
  plot('p_sites', [
    {x: ok.map(v => v.ic), y: ok.map(v => v.site), mode: 'markers+text',
     text: ok.map(v => `${v.slope.toFixed(2)}x` + (v.extrap != null && v.extrap > 30 ? ' ⚠' : '')),
     textposition: 'middle right', textfont: {size: 11},
     marker: {size: 12, color: ok.map(v => v.extrap != null && v.extrap > 30 ? '#C4652F' : '#2C6E9E')},
     type: 'scatter', hoverinfo: 'skip'}],
    {xaxis: {title: 'intercept (µg/m³)', zeroline: true, zerolinecolor: '#22252A', zerolinewidth: 2},
     yaxis: {automargin: true}, showlegend: false, margin: {t: 12, r: 60, b: 40, l: 10}});
}
$('sites_run').onclick = async () => {
  if(!ready) return;
  busy(true); $('sites_run').disabled = true;
  $('cap_sites').textContent = 'Cross-site — evaluating on every target (uncached sites fit fresh)…';
  try{
    const j = await post('/api/cross_site',
      {...cfg(), k: $('kmode').value === 'manual' ? parseInt($('kval').value) : null});
    if(j.error){ $('cap_sites').textContent = 'Cross-site — ' + j.error; return; }
    sitesRows = j.rows;
    sitesRender();
  } finally { busy(false); $('sites_run').disabled = false; }
};

/* ---- cross-site target spectra (Sites tab) -----------------------------------
   Same targets the Sites table evaluates, drawn as median+IQR in a chosen
   baseline space. The baseline pill is the point: APRLssb anchors segment 2 at
   the minimum over 1520–1600 cm⁻¹, so a ~1617 band is suppressed under
   "AIRSpec" and visible under "neutral". */
let ssSpace = 'raw', ssRange = 'full', ssCache = {};
const SS_COLOURS = ['#B23327','#C4652F','#2C6E9E','#7A4FA3','#8F8C84','#548C66'];
// 1500-1800 not 1400-1800: below ~1450 the neutral baseline rides the PTFE-mask
// edge, and that rise dominates the y-scale and hides the 1617 feature.
const SS_RANGES = {full:[400,4000], '1400_1800':[1500,1800], '2700_3600':[2700,3600]};

async function drawSiteSpectra(){
  if(!ready) return;
  if(!ssCache[ssSpace]){
    $('cap_sitespec').textContent = `Target spectra by site — computing ${ssSpace} baseline…`;
    placeholder('p_sitespec', 'computing…');
    const j = await post('/api/site_spectra', {space: ssSpace});
    if(j.error){ $('cap_sitespec').textContent = 'Target spectra — ' + j.error; return; }
    ssCache[ssSpace] = j;
  }
  const j = ssCache[ssSpace];
  const [lo, hi] = SS_RANGES[ssRange];
  const showIqr = $('ss_iqr').checked;
  const data = [];
  j.series.forEach((s, i) => {
    const col = SS_COLOURS[i % SS_COLOURS.length];
    const idx = s.wn.map((v, k) => k).filter(k => s.wn[k] >= lo && s.wn[k] <= hi);
    const wn = idx.map(k => s.wn[k]);
    if(showIqr){
      data.push({x: wn.concat([...wn].reverse()),
        y: idx.map(k => s.q75[k]).concat(idx.map(k => s.q25[k]).reverse()),
        fill:'toself', fillcolor: col.replace('#','rgba(').length ? col + '25' : col,
        line:{width:0}, hoverinfo:'skip', showlegend:false, type:'scatter'});
    }
    data.push({x: wn, y: idx.map(k => s.median[k]), mode:'lines',
      line:{color: col, width:1.7}, name:`${s.label} (n=${s.n})`, type:'scatter'});
  });
  const note = ssSpace === 'airspec'
    ? ' — NOTE: this baseline anchors at 1520–1600 cm⁻¹ and suppresses ~1617 features'
    : (ssSpace === 'neutral' ? ' — independent pspline_arpls, no anchor window' : '');
  $('cap_sitespec').textContent = `Target spectra by site — median${showIqr ? ' + IQR' : ''}, ${ssSpace} baseline${note}`;
  plot('p_sitespec', data, {
    xaxis:{title:'wavenumber (cm⁻¹)', autorange:'reversed'},
    yaxis:{title: ssSpace === 'raw' ? 'absorbance' : 'baseline-corrected absorbance'},
    margin:{t:34, r:10, b:44, l:60}, legend: LEGEND_TOP,
    shapes: ssRange === '1400_1800' ? [{type:'line', x0:1617, x1:1617, yref:'paper', y0:0, y1:1,
      line:{color:'#22252A', dash:'dot', width:1}}] : []});
}
$('ss_space').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('ss_space').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); ssSpace = b.dataset.v; drawSiteSpectra();
});
$('ss_range').querySelectorAll('button').forEach(b => b.onclick = () => {
  $('ss_range').querySelectorAll('button').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); ssRange = b.dataset.v; drawSiteSpectra();
});
$('ss_iqr').onchange = drawSiteSpectra;
document.querySelector('[data-tab="sites"]').addEventListener('click', () => {
  if(ready && !document.querySelector('#p_sitespec .js-plotly-plot')) drawSiteSpectra();
});

/* ---- HIPS lab tab -----------------------------------------------------------
   Per-filter weighted York/EIV fits (replacing pooled-lambda Deming) under
   three blank-line variants, plus the blank ledger. The point of the table:
   an intercept that moves between blank lines is calibration-sensitive; one
   that holds still belongs to the aerosol. */
let hipsRows = null, hipsBlanksLoaded = false;
const HIPS_KINDS = [['deployed', '#2C6E9E'], ['lot_lin', '#7A4FA3'], ['lot_quad', '#B23327']];

function hipsFitCell(f){
  if(!f || f.error) return `<td class="warn">${f ? f.error : '—'}</td>`;
  return `<td title="MSWD-inflated SEs; median recomputed Fabs ${f.median_fabs} Mm⁻¹">` +
         `${f.slope.toFixed(2)}x ${f.intercept >= 0 ? '+' : '−'}${Math.abs(f.intercept).toFixed(2)}` +
         `<span class="muted"> ±${f.intercept_se.toFixed(2)}</span></td>`;
}

function hipsRender(){
  if(!hipsRows) return;
  $('hipstbl').querySelector('tbody').innerHTML = hipsRows.map(r => r.error
    ? `<tr><td>${r.site}</td><td colspan="6" class="warn">${r.error}</td></tr>`
    : `<tr><td title="${r.label}">${r.site}</td>` +
      `<td title="${r.n_matched}/${r.n} filters matched in the batch export">${r.n_matched}</td>` +
      `<td class="${r.fits.deployed && r.fits.deployed.kappa < 0.8 ? 'warn' : ''}">` +
        `${r.fits.deployed ? r.fits.deployed.kappa.toFixed(2) : '—'}</td>` +
      hipsFitCell(r.fits.deployed) + hipsFitCell(r.fits.lot_lin) + hipsFitCell(r.fits.lot_quad) +
      `<td class="${r.frac_below_blank_r1 > 0.2 ? 'warn' : ''}">` +
        `${(r.frac_below_blank_r1 * 100).toFixed(0)}%</td></tr>`).join('');

  const ok = hipsRows.filter(r => !r.error);
  const traces = HIPS_KINDS.map(([kind, color]) => ({
    x: ok.map(r => r.fits[kind] && !r.fits[kind].error ? r.fits[kind].intercept : null),
    y: ok.map(r => r.site),
    error_x: {type: 'data',
      array: ok.map(r => r.fits[kind] && !r.fits[kind].error ? r.fits[kind].intercept_se : null),
      color, thickness: 1},
    name: {deployed: 'deployed line', lot_lin: 'lot-common linear', lot_quad: 'lot quadratic'}[kind],
    mode: 'markers', type: 'scatter', marker: {size: 10, color}}));
  plot('p_hips', traces,
    {xaxis: {title: 'intercept (µg/m³)', zeroline: true, zerolinecolor: '#22252A', zerolinewidth: 2},
     yaxis: {automargin: true},
     legend: {orientation: 'h', y: -0.25}, margin: {t: 12, r: 20, b: 40, l: 10}});
}

async function hipsBlanks(){
  if(hipsBlanksLoaded) return;
  const j = await fetch('/api/hips_blanks').then(r => r.json());
  if(j.error) return;
  hipsBlanksLoaded = true;
  const lots = Object.keys(j.lots).sort();
  $('hipsblanks').querySelector('tbody').innerHTML = lots.map(lot => {
    const L = j.lots[lot];
    const curved = L.rms_lin > 1.2 * L.rms_quad;
    return `<tr><td>${lot}</td><td>${L.n}</td>` +
      `<td class="${curved ? 'warn' : ''}">${L.rms_lin.toFixed(1)}</td>` +
      `<td>${L.rms_quad.toFixed(1)}</td>` +
      `<td>${L.r1_min.toFixed(0)}–${L.r1_max.toFixed(0)}</td>` +
      `<td>${L.tau0_mean != null ? (L.tau0_mean >= 0 ? '+' : '−') + Math.abs(L.tau0_mean).toFixed(4) + ' ±' + L.tau0_sd.toFixed(4) : '—'}</td></tr>`;
  }).join('');
}

$('hips_run').onclick = async () => {
  if(!ready) return;
  busy(true); $('hips_run').disabled = true;
  $('cap_hips').textContent = 'HIPS lab — fitting every SPARTAN target (uncached calibrations fit fresh)…';
  try{
    const j = await post('/api/hips_york',
      {...cfg(), k: $('kmode').value === 'manual' ? parseInt($('kval').value) : null});
    if(j.error){ $('cap_hips').textContent = 'HIPS lab — ' + j.error; return; }
    hipsRows = j.rows;
    $('cap_hips').textContent =
      'HIPS lab — York/EIV fits with per-filter Fabs uncertainties, under three blank-line variants';
    hipsRender();
  } finally { busy(false); $('hips_run').disabled = false; }
};
document.querySelector('[data-tab="hips"]').addEventListener('click', () => { if(ready) hipsBlanks(); });
