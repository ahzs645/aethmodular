"""Build a standalone, OFFLINE HTML recreation of the tool's Calibrate-tab interface.

Vanilla JS + Plotly (inlined locally — no CDN, no React/Babel, works offline by double-click).
Layout/plots matched to the tool's captured specs: top nav, species tabs, sidebar (Refine +
filter detail on click), 2x2 plots (RMSEP CV+AdjCV, measured-vs-predicted colored by site with
dotted 1:1, residuals-by-site box, residuals-over-time), stats panel, Full/Cleaned toggle.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE/"data/recreation_ec_data.json").read_text())
PLOTLY = (HERE/"data/plotly.min.js").read_text()

APP = r"""
const D = window.__DATA__;
const SPECIES = ["OC","EC","S","Fe","Si","Sulfate","fAbs","PM2.5","OPTR","OPTT","EC1"];
let view = "full";

function hsl(h,l,s){function f(n){const k=(n+h*12)%12;const a=s*Math.min(l,1-l);return l-a*Math.max(-1,Math.min(k-3,9-k,1));}
  return `rgb(${Math.round(f(0)*255)},${Math.round(f(8)*255)},${Math.round(f(4)*255)})`;}
const SITES=[...new Set(D.full.rows.map(r=>r.site))].sort();
const PAL={}; SITES.forEach((s,i)=>PAL[s]=hsl(i/Math.max(SITES.length,1),0.62,0.45));
function bySite(rows){const m={};rows.forEach(r=>{(m[r.site]=m[r.site]||[]).push(r);});return m;}
const NB={displayModeBar:false,responsive:true};

function draw(){
  const cur=D[view], c=cur.rmsep;
  Plotly.react('p_rmsep',[
    {x:c.k,y:c.cv,mode:"lines",name:"CV",line:{color:"rgba(31,119,180,1)"}},
    {x:c.k,y:c.adjcv,mode:"lines",name:"Adjusted CV",line:{color:"rgba(255,127,14,1)"}}],
    {title:{text:D.species,font:{size:14}},xaxis:{title:"Components",range:[0,80]},yaxis:{title:"RMSEP"},
     margin:{t:28,r:10,b:40,l:50},legend:{x:1,y:1,xanchor:"right",font:{size:10}}},NB);

  const g=bySite(cur.rows),ct=[];
  SITES.forEach(s=>{const rs=g[s]; if(!rs)return;
    ct.push({x:rs.map(r=>r.measured),y:rs.map(r=>r.predicted),mode:"markers",name:s,type:"scatter",
      marker:{color:PAL[s],size:7,line:{width:0.4,color:"#fff"}},
      text:rs.map(r=>`${r.site}|${r.date}|${r.resid}`),
      hovertemplate:"%{text}<br>meas %{x} pred %{y}<extra></extra>"});});
  const mx=Math.max(...cur.rows.map(r=>Math.max(r.measured,r.predicted)))*1.05;
  ct.push({x:[0,mx],y:[0,mx],mode:"lines",line:{color:"rgb(10,10,10)",dash:"dot"},showlegend:false,hoverinfo:"skip"});
  Plotly.react('p_cross',ct,{title:{text:D.K+" components",font:{size:14}},xaxis:{title:"measured"},yaxis:{title:"predicted"},
    margin:{t:28,r:10,b:40,l:55},showlegend:true,legend:{font:{size:8}}},NB);
  document.getElementById('p_cross').on('plotly_click',e=>{const t=(e.points[0].text||"").split("|");
    document.getElementById('det').innerHTML=`<div><b>Site:</b> ${t[0]}</div><div><b>Sample Date:</b> ${t[1]}</div>`+
    `<div><b>measured:</b> ${e.points[0].x} µg</div><div><b>predicted:</b> ${e.points[0].y} µg</div><div><b>resid:</b> ${t[2]} µg</div>`;});

  const bt=SITES.filter(s=>g[s]).map(s=>({y:g[s].map(r=>r.resid),name:s,type:"box",marker:{color:PAL[s]},boxpoints:"outliers"}));
  Plotly.react('p_box',bt,{title:{text:"residuals by site",font:{size:14}},xaxis:{title:"Site"},yaxis:{title:"resid"},
    margin:{t:28,r:10,b:55,l:50},showlegend:false},NB);

  const tt=SITES.filter(s=>g[s]).map(s=>{const rs=g[s].slice().sort((a,b)=>a.date<b.date?-1:1);
    return {x:rs.map(r=>r.date),y:rs.map(r=>r.resid),mode:"lines+markers",name:s,line:{color:PAL[s]},marker:{size:3}};});
  Plotly.react('p_ts',tt,{title:{text:"residuals over time",font:{size:14}},xaxis:{title:"SampleDate"},yaxis:{title:"resid"},
    margin:{t:28,r:10,b:40,l:50},showlegend:false},NB);

  const s=cur.stats;
  document.getElementById('stats').innerHTML=
    `<div><b>R2:</b> ${s.r2}</div><div><b>RMSE:</b> ${s.rmse}</div><div><b>Bias:</b> ${s.bias} (µg)</div>`+
    `<div><b>Bias (%):</b> ${s.biaspct}</div><div><b>Error:</b> ${s.error} (µg)</div><div><b>Error (%):</b> ${s.errorpct}</div>`+
    `<div style="margin-top:8px;color:#888">⬇ Export Coefficients<br>⬇ Export Model</div>`;
  document.getElementById('nsel').textContent=s.n;
  document.querySelectorAll('.tg').forEach(b=>b.classList.toggle('on',b.dataset.v===view));
}
function setView(v){view=v;draw();}
window.addEventListener('load',()=>{
  document.querySelectorAll('.tg').forEach(b=>b.onclick=()=>setView(b.dataset.v));
  draw();
});
"""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>FTIR Calibration Tools (recreation)</title>
<style>
 body{margin:0;font-family:'Helvetica Neue',Arial,sans-serif;color:#333}
 .topbar{background:#2b2b2b;color:#cfcfcf;padding:10px 16px;font-size:15px}
 .topbar .brand{color:#fff;margin-right:18px}.topbar a{color:#bdbdbd;margin-right:14px;text-decoration:none}.topbar a.active{color:#fff;font-weight:600}
 .tabs{border-bottom:1px solid #ddd;padding:8px 16px}.tabs a{color:#2986cc;margin-right:16px;font-size:14px;text-decoration:none}
 .tabs a.active{color:#222;font-weight:600;border-bottom:2px solid #2986cc;padding-bottom:6px}
 .tg{border:1px solid #2986cc;background:#fff;color:#2986cc;padding:5px 12px;cursor:pointer}.tg.on{background:#2986cc;color:#fff}
 .wrap{display:grid;grid-template-columns:225px 1fr;gap:8px;padding:10px 16px}
 .side{font-size:13px}.side .btn{display:block;border:1px solid #ccc;background:#f7f7f7;border-radius:4px;padding:6px 10px;margin:6px 0;width:175px;text-align:left}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
 .crosswrap{position:relative}.statbox{position:absolute;right:6px;top:6px;font-size:12.5px;line-height:1.5}.statbox b{color:#111}
 .muted{color:#888} .toolbar{padding:6px 16px}
 h-plot{display:block;height:330px}
</style></head><body>
 <div class="topbar"><span class="brand">FTIR Calibration Tools</span><a>Select</a><a class="active">Calibrate</a></div>
 <div class="tabs">__TABS__</div>
 <div class="toolbar">
   <button class="tg" data-v="full">Before (full, n=__NFULL__)</button>
   <button class="tg" data-v="clean">After (cleaned, n=__NCLEAN__)</button>
   <span class="muted" style="margin-left:10px">K=__K__ · __NREM__ removed by 3&sigma; · click a cross-plot point to inspect</span>
 </div>
 <div class="wrap">
   <div class="side">
     <div><b id="nsel"></b> FTIR analyses selected</div>
     <button class="btn">Refine Sample List</button><button class="btn">View spectrum</button>
     <div id="stats" style="margin-top:12px;font-size:13px;line-height:1.65;border-top:1px solid #eee;padding-top:8px"></div>
     <div id="det" class="muted" style="margin-top:10px;border-top:1px solid #eee;padding-top:8px">Click a point in the cross-plot to inspect a filter.</div>
   </div>
   <div class="grid">
     <div id="p_rmsep" style="height:330px"></div>
     <div class="crosswrap"><div id="p_cross" style="height:430px"></div></div>
     <div id="p_box" style="height:330px"></div>
     <div id="p_ts" style="height:330px"></div>
   </div>
 </div>
 <script>__PLOTLY__</script>
 <script>window.__DATA__=__DATA_JSON__;</script>
 <script>__APP__</script>
</body></html>"""

tabs = "".join(f'<a class="{"active" if sp==DATA["species"] else ""}">{sp}</a>' for sp in
               ["OC","EC","S","Fe","Si","Sulfate","fAbs","PM2.5","OPTR","OPTT","EC1"])
html = (PAGE.replace("__TABS__",tabs).replace("__NFULL__",str(DATA["full"]["stats"]["n"]))
        .replace("__NCLEAN__",str(DATA["clean"]["stats"]["n"])).replace("__K__",str(DATA["K"]))
        .replace("__NREM__",str(len(DATA["removed"])))
        .replace("__PLOTLY__",PLOTLY).replace("__DATA_JSON__",json.dumps(DATA)).replace("__APP__",APP))
(HERE/"recreation_app.html").write_text(html)
print("wrote recreation_app.html  (", len(html)//1024, "KB )")
