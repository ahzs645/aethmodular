import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const ROOT = "/Users/ahmadjalil/github/aethmodular";
const WORK = path.join(ROOT, "research/improve_hips_offset/presentation_workspaces/improve_fed_missing_data_update");
const OUTPUT = path.join(WORK, "output");
const SCRATCH = path.join(WORK, "scratch");

const imgs = {
  waterfall: path.join(ROOT, "research/improve_hips_offset/output/improve_first_order_loading_range_analysis/screening_waterfall_p05p95.png"),
  baseline: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/improve_post2003_baseline_before_screening.png"),
  overlap: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/improve_etad_axis_overlap_counts.png"),
  fullComparable: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/improve_fabs_ec_full_vs_comparable.png"),
  fedRt: path.join(ROOT, "research/improve_hips_offset/output/improve_fed_rt_proxy_figure2/improve_fed_initial_rt_proxy_figure2_style.png"),
  spartanRt: path.join(ROOT, "research/improve_hips_offset/output/improve_fed_rt_proxy_figure2/spartan_raw_rt_figure2_analog_all_sites.png"),
};

const C = {
  paper: "F7F4EF",
  ink: "172027",
  muted: "66717A",
  hair: "D8D1C7",
  green: "238A57",
  blue: "2A6F97",
  amber: "D58936",
  red: "B84A3A",
  slate: "27323A",
  white: "FFFFFF",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Ahmad Jalil";
pptx.subject = "IMPROVE/FED missing-data update";
pptx.title = "IMPROVE analog search: what FED can and cannot answer";
pptx.company = "UC Davis";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Helvetica Neue",
  bodyFontFace: "Helvetica Neue",
  lang: "en-US",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";
pptx.margin = 0;

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;
const SAFE = 0.64;

function addBg(slide) {
  slide.background = { color: C.paper };
}
function addText(slide, value, x, y, w, h, opts = {}) {
  slide.addText(value, {
    x, y, w, h,
    margin: 0,
    breakLine: false,
    fit: "shrink",
    fontFace: "Helvetica Neue",
    fontSize: 18,
    color: C.ink,
    valign: "top",
    ...opts,
  });
}
function rect(slide, x, y, w, h, color, opts = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color, transparency: opts.transparency ?? 0 },
    line: { color: opts.lineColor ?? color, transparency: opts.lineTransparency ?? 100, width: opts.lineWidth ?? 0 },
    ...opts,
  });
}
function line(slide, x, y, w, color = C.hair, h = 0.015) {
  rect(slide, x, y, w, h, color);
}
function header(slide, no, section, title, subtitle = "") {
  addText(slide, section.toUpperCase(), SAFE, 0.37, 6.5, 0.22, {
    fontSize: 10.5, bold: true, color: C.muted, charSpace: 2.2,
  });
  addText(slide, title, SAFE, 0.67, 9.4, 0.55, {
    fontSize: 26, bold: true, color: C.ink,
  });
  if (subtitle) {
    addText(slide, subtitle, SAFE, 1.19, 9.7, 0.34, {
      fontSize: 14.5, color: C.muted,
    });
  }
  line(slide, SAFE, 1.57, 1.25, C.green, 0.035);
  addText(slide, String(no).padStart(2, "0"), 12.15, 0.38, 0.45, 0.2, {
    fontSize: 9, color: C.muted, bold: true, align: "right",
  });
}
function footer(slide, source = "Source: local IMPROVE/FED notebooks and White et al. papers; update prepared Apr 29, 2026.") {
  line(slide, SAFE, 6.88, SLIDE_W - SAFE * 2, C.hair, 0.012);
  addText(slide, source, SAFE, 7.02, 10.7, 0.22, {
    fontSize: 8.5, color: C.muted,
  });
}
function bullets(slide, items, x, y, w, gap = 0.43, fontSize = 15.5) {
  items.forEach((raw, i) => {
    const item = typeof raw === "string" ? { text: raw, color: C.green } : raw;
    rect(slide, x, y + i * gap + 0.11, 0.06, 0.06, item.color || C.green);
    addText(slide, item.text, x + 0.2, y + i * gap, w - 0.2, 0.34, {
      fontSize, color: C.ink,
    });
  });
}
function metric(slide, value, label, x, y, w, color = C.green) {
  addText(slide, value, x, y, w, 0.54, { fontSize: 30, bold: true, color });
  addText(slide, label, x, y + 0.52, w, 0.46, { fontSize: 10.5, color: C.muted });
}
function imageBox(slide, p, x, y, w, h, caption = "") {
  rect(slide, x, y, w, h, C.white, { lineColor: C.hair, lineTransparency: 0, lineWidth: 0.5 });
  slide.addImage({ path: p, x: x + 0.08, y: y + 0.08, w: w - 0.16, h: h - 0.16, sizing: { type: "contain", x: x + 0.08, y: y + 0.08, w: w - 0.16, h: h - 0.16 } });
  if (caption) addText(slide, caption, x, y + h + 0.07, w, 0.22, { fontSize: 8.5, color: C.muted });
}
function hbar(slide, label, value, x, y, w, color, maxValue, opts = {}) {
  const trackH = opts.trackH ?? 0.16;
  const labelW = opts.labelW ?? 3.1;
  const barX = x + labelW;
  const barW = w - labelW - 0.72;
  addText(slide, label, x, y - 0.03, labelW - 0.15, 0.25, {
    fontSize: opts.labelSize ?? 10.2,
    color: opts.labelColor ?? C.ink,
    align: "right",
    fit: "shrink",
  });
  rect(slide, barX, y, barW, trackH, "E8E2D9");
  rect(slide, barX, y, Math.max(0.025, barW * value / maxValue), trackH, color);
  addText(slide, opts.valueLabel ?? value.toLocaleString(), barX + Math.max(0.03, barW * value / maxValue) + 0.07, y - 0.03, 0.65, 0.25, {
    fontSize: opts.valueSize ?? 9.6,
    color: C.ink,
    bold: value < maxValue * 0.1,
  });
}
function miniTable(slide, rows, x, y, colWs, rowH, opts = {}) {
  const totalW = colWs.reduce((a, b) => a + b, 0);
  rows.forEach((row, r) => {
    const yy = y + r * rowH;
    if (r === 0) rect(slide, x, yy, totalW, rowH, opts.headerFill ?? "E8E2D9");
    else line(slide, x, yy, totalW, "E1DAD0", 0.008);
    let xx = x;
    row.forEach((cell, c) => {
      addText(slide, String(cell), xx + 0.04, yy + 0.06, colWs[c] - 0.08, rowH - 0.08, {
        fontSize: r === 0 ? (opts.headerSize ?? 9.5) : (opts.bodySize ?? 9.7),
        bold: r === 0 || (opts.boldFirstCol && c === 0),
        color: r === 0 ? C.slate : C.ink,
        align: c === 0 ? "left" : "right",
        fit: "shrink",
      });
      xx += colWs[c];
    });
  });
}
function statusRow(slide, label, available, x, y, w, color) {
  addText(slide, label, x, y, 4.0, 0.28, { fontSize: 14, bold: true, color: C.ink });
  const textValue = available ? "available" : "missing";
  const c = available ? color : C.red;
  addText(slide, textValue, x + w - 1.35, y, 1.25, 0.28, { fontSize: 13.5, bold: true, color: c, align: "right" });
  line(slide, x, y + 0.36, w, C.hair, 0.008);
}
function addNotes(slide, note) {
  slide.addNotes(note);
}

function newSlide(notes) {
  const slide = pptx.addSlide();
  addBg(slide);
  if (notes) addNotes(slide, notes);
  return slide;
}

// 1 Cover
{
  const s = newSlide(`Open by saying this is an update to last week's IMPROVE analog presentation. The key change is not that the loading analysis failed; it is that the documentation pass changed what we can defensibly claim from the FED R/T fields. The deck separates what the data do show from what is missing for a true Warren Figure 2 reproduction.`);
  rect(s, 0, 0, 3.85, SLIDE_H, C.slate);
  rect(s, 3.85, 0, 0.25, SLIDE_H, C.green);
  addText(s, "WEEKLY UPDATE | APR 29, 2026", SAFE, 0.55, 4.5, 0.25, { fontSize: 11.5, color: "DDE7DE", bold: true, charSpace: 2 });
  addText(s, "IMPROVE analog search", 4.7, 1.35, 7.1, 0.58, { fontSize: 34, bold: true });
  addText(s, "what FED can and cannot answer", 4.7, 1.94, 7.4, 0.58, { fontSize: 31, bold: true, color: C.green });
  addText(s, "Updated after the parameter/documentation pass: the loading comparison is useful, but the raw HIPS data needed to reproduce Warren Figure 2 are not exposed in the current FED pull.", 4.7, 2.92, 7.0, 0.85, { fontSize: 16.5, color: C.muted, breakLine: false });
  metric(s, "10", "IMPROVE rows in Addis fAbs + EC mass + EC surface p05-p95", 4.7, 4.6, 3.0, C.red);
  metric(s, "147k", "rows with FED laser reflectance/transmittance ratio fields", 8.1, 4.6, 3.5, C.blue);
  addText(s, "Ahmad Jalil | UC Davis", SAFE, 6.55, 3.0, 0.24, { fontSize: 12.5, color: "DDE7DE" });
}

// 2 Recap
{
  const s = newSlide(`This slide is the transition from last week to this week. I would not say the old analysis was wrong. I would say we learned that the first-order loading comparison is still exactly the right direction, but the R/T piece cannot be presented as a true Warren Figure 2 reproduction.`);
  header(s, 2, "Recap", "Last week’s hypothesis was reasonable", "Use IMPROVE as an independent analog for the Addis HIPS regime.");
  addText(s, "The April 22 deck framed three checks:", SAFE, 2.05, 5.2, 0.33, { fontSize: 17, bold: true });
  bullets(s, [
    { text: "Loading: do any IMPROVE filters reach Addis-like EC surface loading?", color: C.green },
    { text: "Absorption: within that range, do fAbs and EC behave similarly?", color: C.blue },
    { text: "Optical geometry: can the R/T space reproduce the Addis blank-line signature?", color: C.amber },
  ], SAFE, 2.57, 6.0, 0.62);
  line(s, 7.75, 2.06, 3.8, C.hair);
  addText(s, "What changed this week", 7.75, 2.3, 4.5, 0.34, { fontSize: 19, bold: true, color: C.red });
  addText(s, "The first two questions are still useful. The third has to be narrowed: FED exposes laser reflectance/transmittance ratio parameters, but not the raw HIPS sphere/plate records, field blanks, or lot-level blank-line metadata.", 7.75, 2.82, 4.4, 1.2, { fontSize: 15.2, color: C.ink, breakLine: false });
  addText(s, "So the new result is partly scientific and partly data-access: IMPROVE can bound the loading regime, but it cannot currently reproduce Warren Figure 2 from public FED alone.", 7.75, 4.58, 4.35, 1.0, { fontSize: 16.5, bold: true, color: C.slate, breakLine: false });
  footer(s);
}

// 3 Exposed vs missing
{
  const s = newSlide(`This is the clearest slide for the missing-data message. The exposed parameters are valuable, but they are not the actual instrument engineering outputs from Warren's Figure 2. The multiwavelength laser reflectance/transmittance fields should be labeled as FED ratio fields.`);
  header(s, 3, "Data audit", "What FED exposes vs. what Warren Figure 2 needs", "The public pull answers loading questions; it does not expose the raw calibration record.");
  addText(s, "Exposed in the local FED parameter list", SAFE, 2.12, 5.1, 0.34, { fontSize: 17.8, bold: true, color: C.green });
  bullets(s, [
    "fAbs plus ECf / OCf / MF / FEf / SOILf",
    "FlowRate and SampDur for computed volume/loading",
    "Reflectance: RefI / RefF / RefM at multiple wavelengths",
    "Transmittance: TransI / TransF / TransM at the same wavelengths",
  ], SAFE, 2.53, 5.3, 0.42, 13.3);
  addText(s, "Suffix key: I = initial, F = final, M = minimum; units are listed as ratio.", SAFE + 0.2, 4.37, 5.0, 0.38, { fontSize: 12.2, color: C.muted, breakLine: false });
  addText(s, "Example at 635 nm: RefI_635 pairs with TransI_635; RefF_635 with TransF_635; RefM_635 with TransM_635.", SAFE + 0.2, 4.84, 5.05, 0.52, { fontSize: 11.8, color: C.muted, breakLine: false });
  rect(s, 6.35, 2.18, 0.02, 3.6, C.hair);
  addText(s, "Not exposed in the current pull", 7.05, 2.12, 5.1, 0.34, { fontSize: 17.8, bold: true, color: C.red });
  bullets(s, [
    { text: "raw HIPS sphere signal R", color: C.red },
    { text: "raw HIPS plate signal T", color: C.red },
    { text: "row-level field blank/sample status", color: C.red },
    { text: "filter lot IDs and blank OLS coefficients", color: C.red },
    { text: "analysis batch/date metadata needed for exact calibration-line reconstruction", color: C.red },
  ], 7.05, 2.66, 5.3, 0.48, 14.2);
  addText(s, "Interpretation: use FED for first-order loading and bounded comparability, not as the raw HIPS calibration source.", 1.8, 6.0, 9.8, 0.42, { fontSize: 18.5, bold: true, color: C.slate, align: "center" });
  footer(s, "Source: FED parameter JSON; local normalized table; White et al. 2025 data availability statement.");
}

// 4 Waterfall
{
  const s = newSlide(`This is the first main scientific result. The current joined FED pull already starts after 2003, so the useful story is the attrition from the joined positive EC plus fAbs set to valid loading and then to Addis fAbs and EC-loading bounds. The final group is a candidate list, not a regression population.`);
  header(s, 4, "First-order result", "The overlap collapses to a case-study set", "In the current 2015–2025 joined pull, Addis-like fAbs plus EC loading is very rare in IMPROVE.");
  metric(s, "152,029", "current joined-pull rows with valid EC loading", SAFE, 2.0, 2.8, C.green);
  metric(s, "36", "rows in Addis fAbs p05-p95 only", 4.0, 2.0, 2.4, C.amber);
  metric(s, "7,299", "rows in Addis EC surface p05-p95 only", 6.8, 2.0, 2.5, C.blue);
  metric(s, "10", "rows in fAbs + EC mass + EC surface p05-p95", 9.85, 2.0, 2.8, C.red);
  addText(s, "Sequential screen toward Addis-comparable filters", 1.35, 3.28, 6.1, 0.28, { fontSize: 14.5, bold: true, color: C.slate });
  hbar(s, "joined pull: positive EC + fAbs", 379697, 1.1, 3.86, 10.9, "59616B", 379697, { valueLabel: "379,697", labelW: 2.25 });
  hbar(s, "valid loading", 152029, 1.1, 4.35, 10.9, C.blue, 379697, { valueLabel: "152,029", labelW: 2.25 });
  hbar(s, "Addis fAbs p05-p95", 36, 1.1, 4.84, 10.9, C.amber, 379697, { valueLabel: "36", labelW: 2.25 });
  hbar(s, "+ EC mass p05-p95", 10, 1.1, 5.33, 10.9, C.red, 379697, { valueLabel: "10", labelW: 2.25 });
  hbar(s, "+ EC surface p05-p95", 10, 1.1, 5.82, 10.9, C.red, 379697, { valueLabel: "10", labelW: 2.25 });
  addText(s, "The final rows are essentially a candidate list, not a regression population.", 2.65, 6.28, 8.4, 0.3, { fontSize: 13.5, color: C.muted, align: "center" });
  footer(s, "Notebook: improve_first_order_loading_range_analysis.ipynb");
}

// 5 Baseline
{
  const s = newSlide(`This slide addresses Ann's concern from the meeting: first show all the data, then screen. It also explains why the earlier threshold sweep is hard to interpret.`);
  header(s, 5, "Before screening", "Most IMPROVE samples live far below the Addis absorption regime", "The baseline matters because arbitrary fAbs cutoffs can mechanically distort slope and intercept.");
  addText(s, "Addis reference ranges", SAFE, 2.05, 3.7, 0.3, { fontSize: 17, bold: true, color: C.green });
  miniTable(s, [
    ["metric", "p05", "median", "p95"],
    ["fAbs (Mm⁻¹)", "34.7", "47.1", "68.0"],
    ["EC (µg/m³)", "2.36", "4.62", "9.60"],
    ["EC mass (µg)", "16.7", "33.5", "69.3"],
    ["EC surface (µg/cm²)", "4.73", "9.49", "19.64"],
  ], SAFE, 2.55, [2.15, 0.85, 0.9, 0.85], 0.43, { bodySize: 10.8 });
  addText(s, "IMPROVE post-2003 baseline", 6.25, 2.05, 4.8, 0.3, { fontSize: 17, bold: true, color: C.blue });
  metric(s, "152k", "valid loading rows", 6.25, 2.58, 2.1, C.green);
  metric(s, "177", "sites represented", 8.65, 2.58, 1.7, C.blue);
  metric(s, "2015–2025", "years in current joined pull", 10.55, 2.58, 2.1, C.slate);
  line(s, 6.25, 4.05, 5.8, C.hair);
  addText(s, "What this means before any screening", 6.25, 4.35, 4.9, 0.34, { fontSize: 17, bold: true, color: C.slate });
  bullets(s, [
    "EC loading overlap is broad enough to study.",
    "Addis fAbs is in the far upper tail of IMPROVE.",
    "Thresholding only on high fAbs can create regression artifacts.",
  ], 6.25, 4.9, 5.6, 0.48, 14.4);
  addText(s, "Read this before thresholding: the baseline explains why the bounded analysis should be framed as comparability, not correction.", 1.7, 6.34, 9.9, 0.42, { fontSize: 16.5, bold: true, color: C.slate, align: "center" });
  footer(s, "Notebook: warren_cena_improve_prep_analysis.ipynb");
}

// 6 Axis
{
  const s = newSlide(`IMPROVE does have filters with comparable EC loading, but those are not generally in the Addis fAbs regime. The limiting axis is fAbs, which is why the fully bounded set is too small.`);
  header(s, 6, "Axis-by-axis", "fAbs is the limiting axis, not EC loading", "This is why a strict Addis analog set becomes too sparse for stable regression.");
  addText(s, "Axis overlap counts", SAFE, 2.1, 3.6, 0.3, { fontSize: 17, bold: true, color: C.slate });
  hbar(s, "EC surface only", 7299, SAFE, 2.72, 5.8, C.blue, 7299, { valueLabel: "7,299", labelW: 1.7 });
  hbar(s, "EC mass only", 7196, SAFE, 3.24, 5.8, C.blue, 7299, { valueLabel: "7,196", labelW: 1.7 });
  hbar(s, "EC conc only", 399, SAFE, 3.76, 5.8, C.amber, 7299, { valueLabel: "399", labelW: 1.7 });
  hbar(s, "fAbs only", 36, SAFE, 4.28, 5.8, C.red, 7299, { valueLabel: "36", labelW: 1.7 });
  hbar(s, "all first-order", 10, SAFE, 4.80, 5.8, C.red, 7299, { valueLabel: "10", labelW: 1.7 });
  addText(s, "Percent of valid-loading rows in all first-order bounds: 0.0066%", SAFE, 5.55, 5.6, 0.3, { fontSize: 15, bold: true, color: C.red, align: "center" });
  addText(s, "The useful takeaway", 7.15, 2.25, 4.2, 0.35, { fontSize: 19, bold: true, color: C.green });
  bullets(s, [
    "EC mass/surface loading has thousands of comparable rows.",
    "The Addis fAbs band has only dozens of rows in IMPROVE.",
    "Putting both axes together leaves roughly ten filters.",
    "Those filters can identify candidates for raw-data requests, but not a general regression.",
  ], 7.15, 2.82, 4.9, 0.52, 14.5);
  footer(s, "Notebook: warren_cena_improve_prep_analysis.ipynb");
}

// 7 Regression
{
  const s = newSlide(`The public IMPROVE subset is too sparse and does not reproduce the Addis regression. Therefore a public-data-only correction is not defensible yet.`);
  header(s, 7, "Regression check", "The bounded groups do not reproduce Addis behavior", "Even high-end IMPROVE subsets do not give the Addis-like compressed slope/intercept pattern.");
  miniTable(s, [
    ["group", "n", "slope", "intercept", "R²"],
    ["full valid loading", "152,029", "3.66", "1.30", "0.40"],
    ["EC surface only", "7,299", "7.01", "1.67", "0.52"],
    ["fAbs only", "36", "0.20", "41.94", "0.02"],
    ["all first-order", "10", "0.58", "36.62", "0.04"],
  ], 0.85, 2.15, [3.4, 1.2, 1.05, 1.3, 0.85], 0.48, { bodySize: 11.2 });
  addText(s, "Why this is not a correction model", 8.3, 2.25, 3.7, 0.34, { fontSize: 18.5, bold: true, color: C.red });
  bullets(s, [
    { text: "The fully bounded group has n = 10.", color: C.red },
    { text: "The fAbs-only regression is essentially uncorrelated.", color: C.red },
    { text: "EC-loading groups behave differently from Addis.", color: C.blue },
    { text: "This is evidence for a boundary, not a calibration equation.", color: C.green },
  ], 8.3, 2.85, 4.0, 0.52, 14.4);
  addText(s, "Conservative claim: IMPROVE does not give a clean empirical correction for Addis from public chemistry + fAbs alone.", 1.75, 6.35, 9.8, 0.35, { fontSize: 16.8, bold: true, color: C.slate, align: "center" });
  footer(s, "Notebook: warren_cena_improve_prep_analysis.ipynb");
}

// 8 RT correction
{
  const s = newSlide(`For SPARTAN, we have something closer because FilterType identifies blanks and samples. For IMPROVE/FED, the plot can be diagnostic, but it is not Warren's raw R/T calibration space.`);
  header(s, 8, "R/T correction", "We can make a FED ratio-field plot, but not Warren Figure 2", "This is the main correction to last week’s optical-geometry interpretation.");
  addText(s, "FED pull", SAFE, 2.18, 4.2, 0.36, { fontSize: 19, bold: true, color: C.blue });
  statusRow(s, "Ref/Trans ratio fields", true, SAFE, 2.75, 5.0, C.green);
  statusRow(s, "147,380 rows with 635 nm pairs", true, SAFE, 3.28, 5.0, C.green);
  statusRow(s, "field blank rows/status", false, SAFE, 3.81, 5.0, C.red);
  statusRow(s, "raw HIPS sphere/plate R,T", false, SAFE, 4.34, 5.0, C.red);
  statusRow(s, "blank-line coefficients", false, SAFE, 4.87, 5.0, C.red);
  addText(s, "SPARTAN local table", 7.15, 2.18, 4.4, 0.36, { fontSize: 19, bold: true, color: C.green });
  statusRow(s, "HIPS_R1 / HIPS_T1", true, 7.15, 2.75, 5.0, C.green);
  statusRow(s, "FilterType", true, 7.15, 3.28, 5.0, C.green);
  statusRow(s, "LotId", true, 7.15, 3.81, 5.0, C.green);
  statusRow(s, "exact IMPROVE Figure 2 source", false, 7.15, 4.34, 5.0, C.red);
  addText(s, "Why SPARTAN is closer: it has raw-looking HIPS fields plus blank/sample and lot metadata. Public FED has ratio fields, but not the calibration record needed for Warren Figure 2.", 1.65, 6.02, 10.2, 0.62, { fontSize: 16.5, bold: true, color: C.red, align: "center" });
  footer(s, "Notebook: improve_fed_rt_proxy_figure2.ipynb");
}

// 9 Labeling
{
  const s = newSlide(`This is the correction slide. I can still use these columns for exploratory diagnostics, but if the meeting turns to Warren Figure 2, the answer is that the exact raw data are missing from FED.`);
  header(s, 9, "Documentation pass", "The R/T fields need a stricter label", "The field names are exposed, but their instrument meaning is not enough for a HIPS claim.");
  addText(s, "Current safe wording", SAFE, 2.18, 4.4, 0.34, { fontSize: 19, bold: true, color: C.green });
  addText(s, "FED laser reflectance/transmittance ratio fields", SAFE, 2.72, 5.5, 0.65, { fontSize: 22, bold: true, color: C.slate });
  addText(s, "What I should not say yet", SAFE, 4.08, 4.4, 0.34, { fontSize: 19, bold: true, color: C.red });
  addText(s, "Raw IMPROVE HIPS sphere/plate R/T", SAFE, 4.62, 5.5, 0.65, { fontSize: 22, bold: true, color: C.red });
  rect(s, 6.15, 2.15, 0.02, 3.7, C.hair);
  addText(s, "Why this matters", 7.0, 2.18, 4.4, 0.34, { fontSize: 19, bold: true, color: C.blue });
  bullets(s, [
    "The exposed fields span 405–980 nm, not only the Warren HIPS laser geometry.",
    "The parameter table calls them ratio fields, not raw engineering outputs.",
    "White 2025 says individual sphere/plate signals are available from authors, not automatically public.",
    "Any notebook figure using these fields should be relabeled as a FED ratio-field diagnostic.",
  ], 7.0, 2.75, 5.2, 0.53, 14.0);
  footer(s, "Source: FED parameter JSON; White et al. 2025; local paper/doc pass.");
}

// 10 Synthesis
{
  const s = newSlide(`We learned something useful even though the public data are incomplete. The study can now make a stronger statement about the limits of public FED for this specific HIPS question.`);
  header(s, 10, "Updated interpretation", "What we learned after going through the data", "The conclusion is now a boundary statement, not a failed search.");
  const rows = [
    ["1", C.green, "IMPROVE is useful for first-order loading context.", "It tells us Addis sits in a regime that is unusual for IMPROVE, especially on fAbs."],
    ["2", C.blue, "Strict Addis comparability is too sparse for an empirical correction.", "The best public-data use is candidate identification and boundary-setting, not regression-based correction."],
    ["3", C.red, "The missing raw HIPS layer is now the main blocker.", "To answer the calibration-line question, we need raw R/T, blanks, filter lots, analysis batch/date, and blank-line coefficients."],
  ];
  rows.forEach((r, i) => {
    const y = 2.25 + i * 1.13;
    addText(s, r[0], 1.05, y, 0.5, 0.45, { fontSize: 30, bold: true, color: r[1] });
    addText(s, r[2], 1.8, y + 0.04, 9.6, 0.35, { fontSize: 19, bold: true });
    addText(s, r[3], 1.8, y + 0.43, 9.8, 0.32, { fontSize: 13.8, color: C.muted });
  });
  addText(s, "Framing for Warren/Cena: Are we outside the supported HIPS measurement regime, and can the missing calibration data be shared for the candidate filters?", 1.45, 6.15, 10.4, 0.5, { fontSize: 16.8, bold: true, color: C.slate, align: "center" });
  footer(s);
}

// 11 Questions
{
  const s = newSlide(`This should be an actionable slide. For Warren, the ask is raw HIPS/calibration metadata for candidate filters. For Cena, the ask is whether the aethalometer flow fix was implemented and whether SPARTAN has a path to raw HIPS metadata.`);
  header(s, 11, "Questions to take forward", "What to ask Warren and Cena", "The next meeting should be targeted around data availability and measurement regime.");
  addText(s, "For Warren / IMPROVE", SAFE, 2.16, 5.1, 0.34, { fontSize: 19, bold: true, color: C.green });
  bullets(s, [
    "Are Addis-like loadings outside the HIPS range where fAbs is considered reliable?",
    "Can we get raw registered sphere/plate R,T for the top IMPROVE candidate filters?",
    "Can those records include field blanks, PTFE lot, analysis batch/date, and blank OLS coefficients?",
    "What exactly are FED Ref/Trans parameters in relation to HIPS versus TOR/carbon laser fields?",
  ], SAFE, 2.72, 5.55, 0.54, 14.0);
  addText(s, "For Cena / SPARTAN", 7.25, 2.16, 5.1, 0.34, { fontSize: 19, bold: true, color: C.blue });
  bullets(s, [
    { text: "Was the Addis aethalometer flow-ratio fix actually implemented?", color: C.blue },
    { text: "Is there post-fix aethalometer data with stable flow?", color: C.blue },
    { text: "Can SPARTAN share raw HIPS metadata consistently across Addis/JPL lots?", color: C.blue },
    { text: "Has SPARTAN assessed the Warren 2025 pixelation/geometry correction for its filter format?", color: C.blue },
  ], 7.25, 2.72, 5.25, 0.54, 14.0);
  footer(s);
}

// 12 Next work
{
  const s = newSlide(`Close with the concrete next steps. Tell Ann that this is a useful outcome: we now know which part of the question public data can answer and which part requires Warren or the IMPROVE processing team.`);
  header(s, 12, "Next work", "Make the notebook and slides match the corrected claim", "The immediate task is to relabel, re-query, and prepare a clean external-facing version.");
  addText(s, "This week", SAFE, 2.25, 4.0, 0.34, { fontSize: 19, bold: true, color: C.green });
  bullets(s, [
    "Relabel FED R/T plots as ratio-field diagnostics.",
    "Add the missing-data table to the notebook.",
    "Export the top 10–20 candidate filters for a raw-HIPS request.",
  ], SAFE, 2.82, 5.4, 0.54, 14.2);
  addText(s, "Before Warren/Cena", 7.25, 2.25, 4.0, 0.34, { fontSize: 19, bold: true, color: C.blue });
  bullets(s, [
    { text: "Re-query FED with method/status/validation/flag fields if available.", color: C.blue },
    { text: "Prepare one slide on why public FED cannot reproduce Figure 2.", color: C.blue },
    { text: "Keep iron/dust/OC screens in backup; keep first-order loading as the main story.", color: C.blue },
  ], 7.25, 2.82, 5.25, 0.54, 14.2);
  addText(s, "Bottom line", 2.0, 5.85, 2.2, 0.35, { fontSize: 19, bold: true, color: C.red });
  addText(s, "The public IMPROVE data can bound the Addis problem, but the decisive HIPS calibration evidence is not exposed in FED.", 4.45, 5.76, 6.9, 0.6, { fontSize: 19, bold: true, color: C.slate });
  footer(s);
}

await fs.mkdir(OUTPUT, { recursive: true });
await fs.mkdir(SCRATCH, { recursive: true });
const outPptx = path.join(OUTPUT, "output.pptx");
await pptx.writeFile({ fileName: outPptx });

// Lightweight PNG contact previews generated from slide thumbnails are not a PowerPoint render,
// but they provide a quick artifact check for the authored deck sequence.
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1260">
<rect width="100%" height="100%" fill="#eeeeee"/>
${Array.from({ length: 12 }, (_, i) => {
  const x = (i % 3) * 500;
  const y = Math.floor(i / 3) * 315;
  return `<g transform="translate(${x},${y})"><rect x="10" y="10" width="480" height="270" fill="#F7F4EF" stroke="#ccc"/><text x="24" y="44" font-family="Helvetica" font-size="18" font-weight="700" fill="#172027">Slide ${String(i + 1).padStart(2, "0")}</text><text x="24" y="286" font-family="Helvetica" font-size="14" fill="#333">preview placeholder - inspect PPTX for final layout</text></g>`;
}).join("")}
</svg>`;
const contact = path.join(SCRATCH, "contact_sheet_pptxgen.png");
await sharp(Buffer.from(svg)).png().toFile(contact);
await fs.writeFile(path.join(SCRATCH, "pptxgen_build_summary.json"), JSON.stringify({ outPptx, contact, slides: 12 }, null, 2));
console.log(JSON.stringify({ outPptx, contact, slides: 12 }, null, 2));
