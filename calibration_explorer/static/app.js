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
let spectraMode = 'single';

let last = null;
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
  };
}
function metricRow(m){
  if(!m) return null;
  return m.find(r => r.evaluation_set === toggles.evalset && String(r.MAC) == (toggles.mac === '10' ? '10' : '6'));
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
  // plots drawn while a pane was hidden have zero width — fix them on reveal
  requestAnimationFrame(() =>
    pane.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p)));
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
  $('mode').value = p.mode || 'site_heldout';
  $('kmode').value = p.kmode || 'auto'; $('kmode').onchange();
  if(p.kmode === 'manual' && p.k) $('kval').value = p.k;
  drawRanking();
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
  ready = true;
  $('cohort').value = 'ocec'; $('cohort').onchange();
  $('checks').innerHTML = 'cohort checks: ' + s.checks.map(c =>
    `<span class="${c.ok ? 'ok' : 'warn'}" title="${c.name}: ${c.detail}">${c.ok ? '✓' : '✗ ' + c.name + ' (' + c.detail + ')'}</span>`).join(' ');
  $('stats').textContent = 'Ready. Pick a preset (or configure by hand) and Run. (Entire-network cohort: first CV curve takes several minutes; cached afterwards.)';
  drawPins();
  drawOverlap();
}
poll();
placeholder('p_resid', 'Residuals appear here after a run.');

/* ---- run ------------------------------------------------------------------- */
async function run(){
  busy(true); showError(null); $('run').disabled = true;
  const body = {...cfg(), k: $('kmode').value === 'manual' ? parseInt($('kval').value) : null};
  try{
    const j = await post('/api/run', body);
    if(j.error){ showError(j.error); return; }
    last = j; $('pin').disabled = false; $('sweep').disabled = false;
    $('kval').value = j.k;
    redraw();
    drawSpectra();          // refresh side-by-side spectra for this config
  } finally { busy(false); $('run').disabled = false; }
}
$('run').onclick = run;

$('sweep').onclick = async () => {
  if(!last) return;
  busy(true); showError(null); $('sweep').disabled = true;
  try{
    const a = last.auto_k;
    const hi = Math.min(Math.max(2 * a, a + 6), 20, parseInt(last.curve[last.curve.length - 1].n_components));
    const ks = [...new Set(Array.from({length: 8}, (_, i) => Math.round(a + i * (hi - a) / 7)))];
    const j = await post('/api/sweep', {...cfg(), ks});
    if(j.error){ showError(j.error); return; }
    drawSweep(j.rows);
    if(activeTab() !== 'calibrate') switchTab('calibrate');
  } finally { busy(false); $('sweep').disabled = false; }
};

/* ---- drawing --------------------------------------------------------------- */
function redraw(){
  if(!last) return;
  drawStats(); drawCurve(); drawCross(); drawMetrics(); drawPins();
}

function drawStats(){
  const j = last, c = j.config;
  const held = j.heldout ?
    `held-out TOR: R² <b>${j.heldout.R2.toFixed(2)}</b>, slope ${j.heldout.slope.toFixed(2)}, RMSE ${j.heldout.RMSE.toFixed(2)}` :
    '<span class="muted">held-out TOR: none (Calibration-app protocol fits on all filters)</span>';
  $('stats').innerHTML = `
    <b>${j.cohort_label}</b><br>
    select on: <b>${c.selection_space === 'airspec' ? 'AIRSpec-corrected' : 'raw'}</b>
    · calibrate on: <b>${SPECTRA_LABEL[c.spectra] || c.spectra}</b><br>
    protocol: <b>${c.mode === 'app' ? 'Calibration app' : 'Site-held-out'}</b><br>
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

function drawCross(){
  const a = last.addis, mac = parseFloat(toggles.mac);
  const useFixed = toggles.evalset === 'fixed';
  $('cap_cross').textContent = `Addis crossplot — MAC ${mac} · ${useFixed ? 'fixed 190' : 'all pairs'}`;
  const idx = a.fabs.map((_, i) => i).filter(i => !useFixed || a.fixed[i]);
  const bySeason = {};
  idx.forEach(i => { (bySeason[a.season[i]] = bySeason[a.season[i]] || []).push(i); });
  const data = Object.entries(bySeason).map(([s, ii]) => ({
    x: ii.map(i => a.fabs[i] / mac), y: ii.map(i => a.pred[i]), mode:'markers', name:s,
    marker:{size:6, color:seasonColour(s), opacity:.65}, type:'scatter'}));
  const xs = idx.map(i => a.fabs[i] / mac), ys = idx.map(i => a.pred[i]);
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
    xaxis:{title:`HIPS EC-equivalent, Fabs/${mac} (µg/m³)`, range:[0, hi]},
    yaxis:{title:'Predicted FTIR EC (µg/m³)', range:[lo, hi]},
    margin:{t:52, r:10, b:45, l:55}, legend:LEGEND_TOP});
  drawResiduals(idx, mac);
}

function drawResiduals(idx, mac){
  // the meeting's residual check: does the correction remove the curve?
  const a = last.addis;
  $('cap_resid').textContent =
    `Residuals (predicted − Fabs/${mac}) vs HIPS — ${toggles.evalset === 'fixed' ? 'fixed 190' : 'all pairs'}`;
  const bySeason = {};
  idx.forEach(i => { (bySeason[a.season[i]] = bySeason[a.season[i]] || []).push(i); });
  const data = Object.entries(bySeason).map(([s, ii]) => ({
    x: ii.map(i => a.fabs[i] / mac), y: ii.map(i => a.pred[i] - a.fabs[i] / mac),
    mode:'markers', name:s, marker:{size:6, color:seasonColour(s), opacity:.65}, type:'scatter'}));
  plot('p_resid', data, {
    xaxis:{title:`HIPS EC-equivalent, Fabs/${mac} (µg/m³)`},
    yaxis:{title:'residual (µg/m³)', zeroline:true, zerolinecolor:'#22252A', zerolinewidth:1.5},
    margin:{t:30, r:10, b:42, l:55}, legend:LEGEND_TOP});
}

function drawMetrics(){
  $('metrics').querySelector('tbody').innerHTML = last.metrics.map(r => `<tr>
    <td>${r.evaluation_set === 'fixed' ? 'fixed 190' : 'all pairs'}</td><td>${r.MAC}</td><td>${r.n}</td>
    <td>${r.ols_slope.toFixed(2)}</td><td>${r.ols_intercept.toFixed(2)}</td>
    <td>${r.deming_slope.toFixed(2)}</td><td>${r.deming_intercept.toFixed(2)}</td>
    <td>${r.R2.toFixed(2)}</td><td>${r.RMSE.toFixed(2)}</td></tr>`).join('');
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
  $('cutlabel').textContent = `cutoff ${$('cutoff').value} — press Run to recalibrate`;
  renderRanking();
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(() => { drawOverlap(); drawCohortInfo(); }, 500);
}

async function drawRanking(){
  const c = $('cohort').value;
  if(!ready) return;
  drawCohortInfo();
  if(!RANKED.includes(c)){
    lastRanking = null;
    Plotly.purge('p_ranking');
    $('p_ranking').innerHTML = '<div class="ph">No cutoff for this cohort — membership is fixed.</div>';
    $('cutlabel').textContent = 'cohort membership is fixed';
    return;
  }
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

function renderRanking(){
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
  b.classList.add('on'); rankView = b.dataset.v; renderRanking();
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
      ? `Spectral sub-types within ${j.cohort_label || 'cohort'} (${spaceLabel}) — k-means medians, Addis + IQR`
      : `Selection cohorts vs Addis (${spaceLabel}) — medians, Addis + IQR`;
    const COLOURS = ['122,79,163', '44,110,158', '143,140,132', '196,148,66', '84,140,102', '150,90,90'];
    data = j.series.flatMap((s, i) => spectraBand(j.wn, s, COLOURS[i % COLOURS.length], s.label, false))
      .concat(spectraBand(j.wn, j.addis, '178,51,39', 'Addis median'));
  }else{
    $('cap_spectra').textContent = `Spectra: cohort vs Addis (${spaceLabel}) — median + IQR`;
    data = [
      ...spectraBand(j.wn, j.cohort, '44,110,158', `cohort median (n=${j.n})`),
      ...spectraBand(j.wn, j.addis, '178,51,39', 'Addis median')];
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
}

function drawSweep(rows){
  const est = toggles.est;
  $('cap_sweep').textContent =
    `k sweep — intercept (fixed 190, MAC 10, ${est === 'deming' ? 'Deming' : 'OLS'}) & held-out R²`;
  plot('p_sweep', [
    {x: rows.map(r => r.k), y: rows.map(r => est === 'deming' ? r.deming_intercept : r.ols_intercept),
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
    selection_space:last.config.selection_space,
    spectra:last.config.spectra, mode:last.config.mode, k:last.k, auto_k:last.auto_k,
    heldout:last.heldout, metrics:last.metrics, when:new Date().toISOString().slice(0, 16)});
  localStorage.setItem('calib_explorer_pins', JSON.stringify(pins));
  drawPins();
};
$('clearpins').onclick = () => { pins = []; localStorage.setItem('calib_explorer_pins', '[]'); drawPins(); };

function pinRow(p){
  const m = metricRow(p.metrics) || {};
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
      <td>${p.mode === 'app' ? 'Calib. app' : 'Site-held-out'}</td>
      <td>${p.k}${p.k !== p.auto_k ? '*' : ''}</td>
      <td>${v.sl != null ? v.sl.toFixed(2) : '–'}</td><td>${v.ic != null ? v.ic.toFixed(2) : '–'}</td>
      <td>${v.r2 != null ? v.r2.toFixed(2) : '–'}</td>
      <td>${p.heldout ? p.heldout.R2.toFixed(2) : '–'}</td>
      <td><button class="gray" onclick="removePin(${i})">✕</button></td></tr>`;
  }).join('');
  const labels = pins.map(p =>
    `${p.label} · sel:${p.selection_space === 'airspec' ? 'AIR' : 'raw'} · cal:${SPECTRA_SHORT[p.spectra] || p.spectra} · ${p.mode === 'app' ? 'app' : 's-h-o'} · k${p.k}`);
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
