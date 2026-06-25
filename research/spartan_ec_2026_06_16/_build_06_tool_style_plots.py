"""Builds 06_tool_style_plots.ipynb — interactive plotly recreations of the tool's
Calibrate-tab graphs for EC, matched to the tool's ACTUAL specs (extracted live from the
running app via gd.data/gd.layout). Each plot separate, BEFORE (K=31, no cleaning) vs AFTER
(cleaned).

Tool specs we matched (captured 2026-06-25 from the live cross/box/ts/rmsep .js-plotly-plot):
  * RMSEP : single blue line (name "CV"), title = species ("EC"), x="Components" (0-80), y="RMSEP"
  * Cross : measured vs predicted, points COLORED BY SITE (legend), black DOTTED 1:1 line,
            title = "<K> components", x="measured", y="predicted"
  * Box   : boxplot of residuals per Site,  x="Site", y="resid"
  * TS    : residuals over time, one line per Site, x="SampleDate", y="resid"
  * palette: ggplotly-style qualitative hue by site (pastel rainbow)

Run:  python _build_06_tool_style_plots.py
      jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 06_tool_style_plots.ipynb --output 06_tool_style_plots.ipynb
"""
from pathlib import Path
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# EC calibration — tool-matched interactive plots (before vs after cleaning)

These reproduce the tool's Calibrate-tab graphs using the tool's **actual** plot definitions,
extracted live from the running app (`gd.data` / `gd.layout` of each `.js-plotly-plot`):

- **RMSEP** — single blue "CV" line; title = species; x = *Components* (0–80); y = *RMSEP*
- **Cross** — measured vs predicted, **colored by Site** (legend), **black dotted** 1:1 line; title = *"{K} components"*
- **Box** — **residuals per Site** (x = *Site*, y = *resid*)
- **Time series** — **residuals over SampleDate**, one line per Site

Shown **before** (your exported K=31, no cleaning) vs **after** (3σ outlier removal + K re-pick).
Interactive plotly (hover/zoom/pan) — the same library the tool uses.""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd, colorsys
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook"
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

def find_repo_root(s=None):
    p=Path(s or Path.cwd()).resolve()
    for c in [p,*p.parents]:
        if (c/"research").exists() and (c/"AGENTS.md").exists(): return c
    return Path.cwd()
DATA=find_repo_root()/"research/spartan_ec_2026_06_16/data"

X=pd.read_csv(DATA/"rds_EC_X.csv").drop(columns=["id"]).to_numpy(float)
y=pd.read_csv(DATA/"rds_EC_Ymeasured.csv")["Y_measured"].to_numpy(float)
meta=pd.read_csv(DATA/"rds_EC_metadata.csv"); meta["date"]=pd.to_datetime(meta.SampleDate)
sites=meta.Site.to_numpy()

def rmsep_curve(Xv,yv,ks,cv=10,seed=0):
    kf=KFold(cv,shuffle=True,random_state=seed); o={}
    for k in ks:
        if k>=min(Xv.shape): continue
        p=cross_val_predict(PLSRegression(n_components=k,scale=False),Xv,yv,cv=kf).ravel()
        o[k]=float(np.sqrt(np.mean((yv-p)**2)))
    return o
def fit_pred(Xv,yv,k): return PLSRegression(n_components=k,scale=False).fit(Xv,yv).predict(Xv).ravel()
def stats(yv,pv):
    r=yv-pv; return dict(r2=1-np.sum(r**2)/np.sum((yv-yv.mean())**2),rmse=float(np.sqrt(np.mean(r**2))))

# pastel hue palette by site (ggplotly-like), assigned over sorted unique sites
def site_palette(site_list):
    us=sorted(pd.unique(site_list)); n=len(us); cols={}
    for i,s in enumerate(us):
        r,g,b=colorsys.hls_to_rgb(i/max(n,1), 0.62, 0.45)   # pastel, mid-light, moderate sat
        cols[s]=f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return cols
print("X",X.shape,"| sites",meta.Site.nunique())""")

md(r"""## Build the two calibration states (BEFORE K=31 / AFTER cleaned)""")
code(r"""KS=list(range(1,41))
K0=31
pred0=fit_pred(X,y,K0); s0=stats(y,pred0); curve0=rmsep_curve(X,y,KS)
keep=y>=0
for _ in range(6):
    p=fit_pred(X[keep],y[keep],K0); r=y[keep]-p; d=np.abs(r)>3*r.std()
    if not d.any(): break
    keep[np.where(keep)[0][d]]=False
curveF=rmsep_curve(X[keep],y[keep],KS); KF=min(curveF,key=curveF.get)
predF=fit_pred(X[keep],y[keep],KF); sF=stats(y[keep],predF)
print(f"BEFORE n=906 K={K0} R2={s0['r2']:.3f} RMSE={s0['rmse']:.2f}")
print(f"AFTER  n={keep.sum()} K={KF} R2={sF['r2']:.3f} RMSE={sF['rmse']:.2f} (removed {int((~keep).sum())})")
PAL=site_palette(sites)""")

# tool-matched figure builders
code(r"""def fig_rmsep(curve, K, species="EC"):
    f=go.Figure()
    f.add_scatter(x=list(curve), y=list(curve.values()), mode="lines",
                  line=dict(color="rgba(31,119,180,1)"), name="CV")
    f.add_vline(x=K, line=dict(color="grey", dash="dash"))
    f.update_layout(template="plotly_white", title=species, xaxis_title="Components",
                    yaxis_title="RMSEP", width=620, height=420,
                    xaxis=dict(range=[0,80]), showlegend=True)
    return f

def _by_site(yv,pv,m):
    res=yv-pv; df=pd.DataFrame({"measured":yv,"predicted":pv,"resid":res,
                                "Site":m.Site.values,"date":m.date.values})
    return df

def fig_cross(yv,pv,m,K):
    df=_by_site(yv,pv,m); f=go.Figure()
    for s,g in df.groupby("Site"):
        f.add_scatter(x=g.measured, y=g.predicted, mode="markers", name=s,
            marker=dict(color=PAL[s], size=7, line=dict(width=0.4,color="white")),
            customdata=np.stack([g.Site, g.date.astype(str), g.resid.round(2)],axis=1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>meas %{x:.2f} pred %{y:.2f}"
                          "<br>resid %{customdata[2]}<extra></extra>")
    lim=float(max(yv.max(),pv.max()))*1.05
    f.add_scatter(x=[0,lim],y=[0,lim],mode="lines",line=dict(color="rgb(10,10,10)",dash="dot"),
                  showlegend=False,hoverinfo="skip")
    f.update_layout(template="plotly_white", title=f"{K} components",
                    xaxis_title="measured", yaxis_title="predicted", width=720, height=560)
    return f

def fig_box(yv,pv,m):
    df=_by_site(yv,pv,m); f=go.Figure()
    for s,g in df.groupby("Site"):
        f.add_box(y=g.resid, name=s, marker_color=PAL[s], boxpoints="outliers")
    f.update_layout(template="plotly_white", title="residuals by site",
                    xaxis_title="Site", yaxis_title="resid", width=900, height=440, showlegend=False)
    return f

def fig_ts(yv,pv,m):
    df=_by_site(yv,pv,m).sort_values("date"); f=go.Figure()
    for s,g in df.groupby("Site"):
        f.add_scatter(x=g.date, y=g.resid, mode="lines+markers", name=s,
                      line=dict(color=PAL[s]), marker=dict(size=4))
    f.update_layout(template="plotly_white", title="residuals over time",
                    xaxis_title="SampleDate", yaxis_title="resid", width=900, height=440)
    return f
print("tool-matched builders ready")""")

md(r"""## 1. RMSEP — BEFORE""")
code(r"""fig_rmsep(curve0, K0)""")
md(r"""## 1. RMSEP — AFTER""")
code(r"""fig_rmsep(curveF, KF)""")
md(r"""## 2. Measured vs Predicted (colored by site) — BEFORE""")
code(r"""fig_cross(y, pred0, meta, K0)""")
md(r"""## 2. Measured vs Predicted — AFTER""")
code(r"""fig_cross(y[keep], predF, meta[keep].reset_index(drop=True), KF)""")
md(r"""## 3. Residuals by site (box) — BEFORE""")
code(r"""fig_box(y, pred0, meta)""")
md(r"""## 3. Residuals by site (box) — AFTER""")
code(r"""fig_box(y[keep], predF, meta[keep].reset_index(drop=True))""")
md(r"""## 4. Residuals over time — BEFORE""")
code(r"""fig_ts(y, pred0, meta)""")
md(r"""## 4. Residuals over time — AFTER""")
code(r"""fig_ts(y[keep], predF, meta[keep].reset_index(drop=True))""")

md(r"""## Notes
- Plot **definitions** (traces, axes, 1:1 dotted line, color-by-site, titles) match the tool's
  live `.js-plotly-plot` specs captured 2026-06-25. The site palette is a ggplotly-style pastel
  hue (the tool interpolates a qualitative scale over sites); exact per-site RGBs differ slightly.
- The tool's plotly is v4.12; this is v6.3 — minor default-styling differences remain.
- Full **click-to-remove-and-refit** reactivity still needs a small Dash/Shiny-for-Python app.""")

nb["cells"]=cells
nb["metadata"]={"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}}
with open(Path(__file__).resolve().parent/"06_tool_style_plots.ipynb","w") as fh: nbf.write(nb,fh)
print("wrote 06_tool_style_plots.ipynb")
