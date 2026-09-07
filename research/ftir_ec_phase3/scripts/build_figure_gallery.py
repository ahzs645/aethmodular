"""Compile the figure output tree into one self-contained HTML gallery.

Every figure is base64-embedded, so the result is a single file that opens
anywhere with no server and no image directory beside it. Provenance is read
from source rather than asserted: the script greps ``scripts/*.py`` for
``output/plots/<dir>`` and reports which builder writes each directory, so a new
figure set documents itself the first time it is generated.

Captions are authored only where a caption is actually known (the 2026-08-27
stability and pathway sets). Everything else gets its filename and its builder,
which is honest rather than invented.

    # the session's own figures, small enough to publish as an artifact
    python3 build_figure_gallery.py --sets stability,pathways --out gallery.html

    # everything in the tree; too large for an artifact, fine locally
    python3 build_figure_gallery.py --all --out gallery_all.html

    python3 build_figure_gallery.py --list        # what is available

Artifacts cap at 16 MB and base64 inflates by 4/3, so the script prints the
finished size and says plainly whether it will publish.
"""
from __future__ import annotations

import argparse
import base64
import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE3 = HERE.parent
PLOTS = PHASE3 / "output/plots"
ARTIFACT_LIMIT_MB = 16.0

# Captions for the figures whose claim is known. Anything absent here is listed
# with its filename and builder instead of a made-up description.
CAPTIONS = {
    "pathways/01_specification_curve.png": (
        "Every Addis iteration, all 20,869 of them",
        "Each specification's intercept, sorted, with the analytic choices that "
        "produced it marked beneath. Raw baselines block the bad end and AIRSpec "
        "fills the good end, which is the visual form of baselining being 49% of "
        "the variance. 355 clear every guardrail and none reaches zero."),
    "stability/01_ocec_mismatch_law.png": (
        "The OC/EC mismatch law",
        "FTIR under-predicts by 0.303 ± 0.026 µg/m³ per unit of cohort-to-target "
        "OC/EC mismatch. At a perfectly matched cohort, −1.63 ± 0.10 remains."),
    "stability/02_variance_decomposition.png": (
        "One knob matters, six are decoys",
        "Baselining is 49% of intercept variance and 56% of slope variance. CV "
        "protocol and training lot are 0.1% each."),
    "stability/03_cross_site_sign_flip.png": (
        "The regional sign flip",
        "Both Ethiopian sites read low under every spectral representation; the "
        "other three read high under all four. Scale clipped at ±2 so Delhi "
        "saturates rather than flattening the contrast."),
    "stability/04_ensemble_reproducibility.png": (
        "Ensembling buys reproducibility, not score",
        "Ten individual configurations fan out by 0.70 µg/m³ between blind halves; "
        "their ensemble moves 0.04."),
    "stability/05_r2_partition_artifact.png": (
        "The held-out R² step function is an artifact",
        "Every discontinuity coincides with the train/test split being redrawn, "
        "because the site-grouped folds are recomputed from whichever sites the "
        "cohort happens to contain."),
    "stability/06_cohort_neighbourhood.png": (
        "Cohort choice swamps the ranking",
        "Below ~900 filters a ±50-filter change moves the objective further than "
        "the whole distance from rank 1 to rank 10."),
    "stability/07_blind_half_rank.png": (
        "The vetted winner survives",
        "Top ten on the early half cluster in the corner on the late half. This "
        "corrects the earlier claim that the leaderboard top was noise."),
    "stability/08_band_vs_offset.png": (
        "The 1617 cm⁻¹ band does not track the offset",
        "Pasadena has the strongest normalised band of any site and a positive "
        "offset; Bishoftu the weakest and a negative one."),
    "pathways/02_loading_within_addis.png": (
        "Loading is not the mechanism",
        "Within Addis, r = −0.10 against filter darkness, and the lightest "
        "quartile already carries −1.61 of the −1.7."),
    "pathways/03_additive_not_multiplicative.png": (
        "The offset is additive",
        "Residual against concentration is flat, slope −0.088. A wrong MAC or any "
        "proportional error would tilt this line."),
    "pathways/04_no_temporal_drift.png": (
        "No drift, no processing step",
        "+0.016 µg/m³ per year across three years and multiple analysis batches."),
    "pathways/05_domain_diagnostic.png": (
        "Addis is in-domain and still wrong",
        "Weakley's diagnostic puts Addis at 0.4% out-of-domain, the lowest of the "
        "five sites. Beijing flags at 32% and reads high."),
    "pathways/06_implied_mac_by_site.png": (
        "No single MAC reconciles Addis",
        "Bar length is the gap between the MAC that would zero the bias and the "
        "MAC that would make the slope 1. Beijing is consistent at ~6.9; Addis "
        "needs 15.4 and 10.0 at once."),
    "pathways/07_mac_invariance.png": (
        "What MAC moves, and what it does not",
        "The Deming intercept is identical at MAC 6, 10 and 17. The mean residual "
        "swings from −4.98 to +0.29."),
    "pathways/08_residual_is_spectrally_visible.png": (
        "The residual's variation is spectrally visible",
        "A model predicts the FTIR−HIPS residual from the spectra at CV R² "
        "0.32–0.40 under strict schemes, about two thirds the predictability of "
        "the reference itself."),
    "pathways/09_offset_under_every_slice.png": (
        "Negative under every slice",
        "Season, PMF source class, filter lot and blind half. The dry season at "
        "−0.26 is the shallowest and matches the committed seasonal table."),
    # ---- crossplots: the result, and the same fit read out on each view ----
    "crossplots/01_result_crossplot_all.png": (
        "The result",
        "lowest-OC/EC 450, AIRSpec, k=9, all 239 pairs: Deming 0.93x −1.42, R² 0.71. "
        "Reproduces the group deck's winner slide, with season colouring added."),
    "crossplots/02_result_by_blind_half_all.png": (
        "The same fit on each blind half",
        "Early 0.88x −1.20 (R² 0.60) against late 0.98x −1.62 (R² 0.82). Both below "
        "the 1:1 line; the offset is not a property of which half you look at."),
    "crossplots/03_result_by_season_all.png": (
        "The same fit by season",
        "Belg, Kiremt and Dry read out separately from one fitted model."),
    "crossplots/04_result_by_pmf_class_all.png": (
        "The same fit by PMF source class",
        "Marine (n=44) against Combustion (n=58) from Navid's apportionment."),
    "crossplots/05_same_calibration_every_site_all.png": (
        "One calibration, every site",
        "Addis and Bishoftu sit below the 1:1 line; Beijing, Delhi and Pasadena "
        "above it. Each panel keeps its own scale because site loadings differ 50×."),
    # ---- decisions: what is feasible and what the objective selects ----
    "decisions/01_feasible_region.png": (
        "The feasible region does not contain the target",
        "All 20,869 configurations in the (slope, intercept) plane fall on one "
        "diagonal band. The target (1, 0) is off it. The 907 near-zero intercepts "
        "are all bought with a slope near 0.3."),
    "decisions/02_winner_vs_weight.png": (
        "Three winners, depending on an unjustified weight",
        "score = |intercept| + w·|slope − 1|. The shipped w = 5 picks deriv2 k=20, "
        "the family that did not transfer to other sites; w ≥ 10 picks AIRSpec."),
    "decisions/03_guardrail_funnel.png": (
        "One guardrail does 95% of the eliminating",
        "The held-out R² floor removes 19,389 of 20,514 eliminated rows, and it "
        "acts on the least stable quantity in the system."),
    "decisions/04_r2_per_10_filter_step.png": (
        "How far held-out R² moves per 10-filter cohort step",
        "8,699 adjacent steps across the dense series: median 0.007, but 28% move "
        "by more than 0.05 and the p90 is 0.22. Plateaus are rare; the earlier "
        "'plateaus of three' claim was one window, not the system."),
    "decisions/05_season_cross_application.png": (
        "Season winners read out on every season",
        "The all-year winner ranks #318 of 345 on Kiremt. The Kiremt winner beats "
        "it on three of four readouts, so optimising on the wet season generalises "
        "better than optimising on everything."),
    "decisions/06_season_winner_by_site.png": (
        "Each season's winner at every site",
        "Colour is |slope − 1|. The all-year winner (deriv2 k=20) gives Beijing "
        "9.78x and Delhi −26.9; the Kiremt winner (ocec 800 AIRSpec k=21) transfers, "
        "and is the repo's own terminal candidate rediscovered."),
}


def builders_by_dir() -> dict[str, list[str]]:
    """Which script writes each plot directory, read out of the sources."""
    out: dict[str, set[str]] = {}
    for script in sorted(HERE.glob("*.py")):
        if script.name == Path(__file__).name:
            continue
        try:
            text = script.read_text(errors="ignore")
        except OSError:
            continue
        for name in re.findall(r"output/plots/([A-Za-z0-9_]+)", text):
            out.setdefault(name, set()).add(script.name)
    return {k: sorted(v) for k, v in out.items()}


def title_from(stem: str) -> str:
    """A readable title from a filename, without inventing meaning."""
    s = re.sub(r"^\d+[_-]", "", stem)
    return s.replace("_", " ").replace("-", " ").strip() or stem


def collect(sets: list[str] | None):
    """Group every PNG under PLOTS by its directory, at any depth.

    Nested sets are real: deck/by_protocol/ holds two more, and a couple of
    figures sit loose at the top. A non-recursive walk silently dropped 20 of
    144, which is exactly the kind of quiet omission a gallery must not have.
    """
    buckets: dict[str, list[Path]] = {}
    for png in sorted(PLOTS.rglob("*.png")):
        rel = png.parent.relative_to(PLOTS)
        buckets.setdefault(str(rel) if str(rel) != "." else "(top level)",
                           []).append(png)
    if sets:
        want = set(sets)
        missing = want - set(buckets)
        if missing:
            sys.exit(f"no such figure set: {', '.join(sorted(missing))}")
        buckets = {k: v for k, v in buckets.items() if k in want}
    return sorted(buckets.items())


def build(groups, builders, out_path: Path, title: str):
    parts, total_src, n = [], 0, 0
    for name, pngs in groups:
        who = builders.get(name.split("/")[0])
        prov = (f"generated by <code>{html.escape(', '.join(who))}</code>"
                if who else "builder not identified in <code>scripts/</code>")
        cards = []
        for p in pngs:
            rel = f"{name}/{p.name}"
            raw = p.read_bytes()
            total_src += len(raw)
            n += 1
            uri = "data:image/png;base64," + base64.b64encode(raw).decode()
            cap = CAPTIONS.get(rel)
            head = html.escape(cap[0]) if cap else html.escape(title_from(p.stem))
            body = (html.escape(cap[1]) if cap
                    else f'<span class="unc">{html.escape(p.name)}</span>')
            cards.append(
                f'<figure class="fig" data-q="{html.escape(rel.lower())} '
                f'{html.escape(head.lower())}">'
                f'<img src="{uri}" alt="{head}" loading="lazy">'
                f'<figcaption><b>{head}</b>{body}</figcaption></figure>')
        parts.append(
            f'<section id="s-{html.escape(name.replace("/", "-"))}">'
            f'<h2>{html.escape(name)}'
            f'<span class="cnt">{len(pngs)}</span></h2>'
            f'<p class="prov">{prov}</p><div class="grid">'
            + "".join(cards) + "</div></section>")

    nav = "".join(f'<a href="#s-{html.escape(nm.replace("/", "-"))}">{html.escape(nm)}'
                  f'<span>{len(ps)}</span></a>' for nm, ps in groups)
    doc = TEMPLATE.replace("{{TITLE}}", html.escape(title)) \
                  .replace("{{NAV}}", nav) \
                  .replace("{{BODY}}", "".join(parts)) \
                  .replace("{{N}}", str(n)) \
                  .replace("{{SETS}}", str(len(groups)))
    out_path.write_text(doc)
    mb = out_path.stat().st_size / 1e6
    print(f"  {n} figures from {len(groups)} sets -> {out_path}")
    print(f"  source {total_src/1e6:.1f} MB, page {mb:.1f} MB")
    if mb > ARTIFACT_LIMIT_MB:
        print(f"  NOTE: over the {ARTIFACT_LIMIT_MB:.0f} MB artifact limit; "
              f"opens locally but will not publish. Use --sets for a subset.")
    else:
        print(f"  fits the {ARTIFACT_LIMIT_MB:.0f} MB artifact limit.")


TEMPLATE = """<title>{{TITLE}}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap">
<style>
:root{--ground:#F8F8F6;--surface:#FFF;--sunk:#F1F1EE;--ink:#22252A;--body:#3B4048;
--muted:#6E7178;--hair:#E3E2DE;--blue:#2C6E9E;
--serif:"IBM Plex Serif",Georgia,serif;--sans:"IBM Plex Sans",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--ground:#16181B;--surface:#1D2024;--sunk:#212429;--ink:#E9E9E6;--body:#C3C6CB;
--muted:#8E9299;--hair:#2E3238;--blue:#6BA6D0}}
:root[data-theme="dark"]{--ground:#16181B;--surface:#1D2024;--sunk:#212429;
--ink:#E9E9E6;--body:#C3C6CB;--muted:#8E9299;--hair:#2E3238;--blue:#6BA6D0}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--body);font-family:var(--sans);
font-size:16px;line-height:1.6;margin:0;padding:0 clamp(16px,3vw,34px) 80px}
.top{max-width:1400px;margin:0 auto;padding:clamp(30px,5vw,54px) 0 8px;
display:flex;flex-direction:column;gap:14px}
h1{font-family:var(--serif);font-size:clamp(1.8rem,4vw,2.5rem);color:var(--ink);
margin:0;letter-spacing:-.015em;line-height:1.12}
.sub{color:var(--muted);font-size:14.5px}
.tools{display:flex;flex-wrap:wrap;gap:9px;align-items:center;
position:sticky;top:0;background:var(--ground);padding:11px 0;z-index:5;
border-bottom:1px solid var(--hair)}
input[type=search]{font-family:var(--sans);font-size:14px;padding:7px 12px;
border:1px solid var(--hair);border-radius:4px;background:var(--surface);
color:var(--ink);min-width:230px}
input[type=search]:focus{outline:2px solid var(--blue);outline-offset:1px}
.nav{display:flex;flex-wrap:wrap;gap:5px}
.nav a{font-family:var(--mono);font-size:11px;text-decoration:none;color:var(--muted);
border:1px solid var(--hair);border-radius:3px;padding:3px 7px;background:var(--surface)}
.nav a:hover,.nav a:focus{color:var(--ink);border-color:var(--blue);outline:none}
.nav a span{color:var(--blue);margin-left:5px}
main{max-width:1400px;margin:0 auto;display:flex;flex-direction:column;gap:38px;
padding-top:26px}
section{display:flex;flex-direction:column;gap:11px;scroll-margin-top:70px}
h2{font-family:var(--mono);font-size:12.5px;letter-spacing:.1em;text-transform:uppercase;
color:var(--ink);margin:0;padding-bottom:7px;border-bottom:1px solid var(--hair);
display:flex;gap:10px;align-items:baseline}
h2 .cnt{color:var(--blue);font-size:11px}
.prov{margin:0;font-size:12.5px;color:var(--muted)}
code{font-family:var(--mono);font-size:.86em;background:var(--sunk);padding:1px 5px;
border-radius:3px;color:var(--ink)}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}
.fig{margin:0;background:var(--surface);border:1px solid var(--hair);border-radius:5px;
padding:12px;display:flex;flex-direction:column;gap:10px}
.fig img{width:100%;height:auto;display:block;border-radius:3px;background:#fff}
figcaption{font-size:13.2px;color:var(--muted)}
figcaption b{display:block;font-family:var(--serif);font-size:14.4px;color:var(--ink);
margin-bottom:3px;font-weight:600}
.unc{font-family:var(--mono);font-size:12px}
.empty{color:var(--muted);font-size:14px;padding:20px 0}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="top">
  <h1>{{TITLE}}</h1>
  <div class="sub">{{N}} figures across {{SETS}} sets, each embedded in this file.
  Captions are authored where the claim is known; the rest carry their filename and
  the script that produced them.</div>
</div>
<div class="tools">
  <input type="search" id="q" placeholder="filter figures…" aria-label="Filter figures">
  <nav class="nav">{{NAV}}</nav>
</div>
<main id="main">{{BODY}}</main>
<script>
const q=document.getElementById('q');
q.addEventListener('input',()=>{
  const t=q.value.trim().toLowerCase();
  document.querySelectorAll('section').forEach(sec=>{
    let shown=0;
    sec.querySelectorAll('.fig').forEach(f=>{
      const hit=!t||f.dataset.q.includes(t);
      f.style.display=hit?'':'none'; if(hit)shown++;
    });
    sec.style.display=shown?'':'none';
  });
});
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", help="comma-separated figure directories")
    ap.add_argument("--all", action="store_true", help="every directory")
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--out", default="gallery.html")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    builders = builders_by_dir()
    if a.list:
        for name, pngs in collect(None):
            who = ", ".join(builders.get(name, [])) or "unknown builder"
            capped = sum(1 for p in pngs if f"{name}/{p.name}" in CAPTIONS)
            print(f"  {name:24s} {len(pngs):3d} figures  "
                  f"{capped:3d} captioned   {who}")
        return

    if not a.all and not a.sets:
        sys.exit("pass --sets a,b or --all (or --list to see what exists)")
    sets = None if a.all else [s.strip() for s in a.sets.split(",") if s.strip()]
    groups = collect(sets)
    if not groups:
        sys.exit("no figures found")
    title = a.title or ("Phase-3 figure archive" if a.all
                        else "Addis calibration figures")
    build(groups, builders, Path(a.out), title)


if __name__ == "__main__":
    main()
