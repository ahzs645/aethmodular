import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  layers,
  text,
  image,
  shape,
  fill,
} from "@oai/artifact-tool";

const ROOT = "/Users/ahmadjalil/github/aethmodular";
const OUT = path.join(
  ROOT,
  "research/improve_hips_offset/presentation_workspaces/improve_fed_missing_data_update",
);
const SCRATCH = path.join(OUT, "scratch");
const OUTPUT = path.join(OUT, "output");

const img = {
  waterfall: path.join(ROOT, "research/improve_hips_offset/output/improve_first_order_loading_range_analysis/screening_waterfall_p05p95.png"),
  baseline: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/improve_post2003_baseline_before_screening.png"),
  overlap: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/improve_etad_axis_overlap_counts.png"),
  fullComparable: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/improve_fabs_ec_full_vs_comparable.png"),
  rtCaveat: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/improve_rt_proxy_caveat.png"),
  spartanRt: path.join(ROOT, "research/improve_hips_offset/output/improve_fed_rt_proxy_figure2/spartan_raw_rt_figure2_analog_all_sites.png"),
  fedRt: path.join(ROOT, "research/improve_hips_offset/output/improve_fed_rt_proxy_figure2/improve_fed_initial_rt_proxy_figure2_style.png"),
  spartanSites: path.join(ROOT, "research/improve_hips_offset/output/warren_cena_improve_prep_analysis/spartan_four_site_hips_fabs_vs_ec.png"),
};

const C = {
  paper: "#F7F4EF",
  ink: "#172027",
  muted: "#66717A",
  hair: "#D8D1C7",
  green: "#238A57",
  blue: "#2A6F97",
  amber: "#D58936",
  red: "#B84A3A",
  slate: "#27323A",
  white: "#FFFFFF",
};

const W = 1920;
const H = 1080;
const SAFE = 92;

function t(value, x, y, w, h, style = {}, name = "text") {
  return text(value, {
    name,
    x,
    y,
    width: w,
    height: h,
    style: {
      fontFace: "Helvetica Neue",
      color: C.ink,
      fontSize: 28,
      ...style,
    },
  });
}

function rect(x, y, w, h, fillColor, name = "rect", extra = {}) {
  return shape({
    name,
    shape: "rect",
    x,
    y,
    width: w,
    height: h,
    fill: fillColor,
    line: { color: fillColor, transparency: 100 },
    ...extra,
  });
}

function line(x, y, w, color = C.hair, weight = 3, name = "rule") {
  return rect(x, y, w, weight, color, name);
}

function title(slideNo, section, titleText, subtitleText = "") {
  const elems = [
    t(section.toUpperCase(), SAFE, 54, 900, 28, {
      fontSize: 18,
      color: C.muted,
      bold: true,
      charSpacing: 2.5,
    }, "section-label"),
    t(titleText, SAFE, 92, 1280, 92, {
      fontSize: 48,
      bold: true,
      color: C.ink,
    }, "slide-title"),
  ];
  if (subtitleText) {
    elems.push(t(subtitleText, SAFE, 170, 1300, 56, {
      fontSize: 24,
      color: C.muted,
    }, "slide-subtitle"));
  }
  elems.push(
    line(SAFE, 224, 188, C.green, 5, "title-accent"),
    t(String(slideNo).padStart(2, "0"), 1760, 56, 70, 28, {
      fontSize: 16,
      color: C.muted,
      bold: true,
      align: "right",
    }, "slide-number"),
  );
  return elems;
}

function footer(source = "Source: local IMPROVE/FED notebooks and White et al. papers; update prepared Apr 29, 2026.") {
  return [
    line(SAFE, 1000, W - SAFE * 2, C.hair, 2, "footer-rule"),
    t(source, SAFE, 1018, 1420, 28, {
      fontSize: 15,
      color: C.muted,
    }, "source"),
  ];
}

function bullet(lines, x, y, w, gap = 52, style = {}) {
  return lines.flatMap((item, i) => [
    rect(x, y + i * gap + 12, 9, 9, item.color || C.green, `bullet-dot-${i}`),
    t(item.text || item, x + 28, y + i * gap, w - 28, 42, {
      fontSize: 25,
      color: C.ink,
      ...style,
    }, `bullet-${i}`),
  ]);
}

function metric(value, label, x, y, w, accent = C.green) {
  return [
    t(value, x, y, w, 74, {
      fontSize: 56,
      bold: true,
      color: accent,
    }, `metric-${label}`),
    t(label, x, y + 70, w, 58, {
      fontSize: 19,
      color: C.muted,
    }, `metric-label-${label}`),
  ];
}

function framedImage(pathname, x, y, w, h, name, caption = "") {
  const elems = [
    rect(x, y, w, h, C.white, `${name}-paper`, { line: { color: C.hair, width: 1 } }),
    image({ name, path: pathname, x: x + 18, y: y + 18, width: w - 36, height: h - 36, fit: "contain", alt: name }),
  ];
  if (caption) {
    elems.push(t(caption, x, y + h + 10, w, 26, { fontSize: 16, color: C.muted }, `${name}-caption`));
  }
  return elems;
}

function slideFrom(elements, notes) {
  const s = deck.slides.add();
  s.background.color = C.paper;
  s.compose(layers({ name: "slide-root", width: fill, height: fill }, elements), {
    frame: { left: 0, top: 0, width: W, height: H },
    baseUnit: 8,
  });
  s.speakerNotes.setText(notes);
  return s;
}

const deck = Presentation.create({ slideSize: { width: W, height: H } });

// Slide 1
slideFrom([
  rect(0, 0, W, H, C.paper, "cover-bg"),
  rect(0, 0, 560, H, C.slate, "cover-spine"),
  rect(560, 0, 38, H, C.green, "cover-green-cut"),
  t("WEEKLY UPDATE | APR 29, 2026", SAFE, 82, 700, 32, { fontSize: 20, color: "#DDE7DE", bold: true, charSpacing: 2.2 }, "cover-eyebrow"),
  t("IMPROVE analog search", 680, 200, 980, 88, { fontSize: 62, bold: true, color: C.ink }, "cover-title-1"),
  t("what FED can and cannot answer", 680, 286, 1040, 88, { fontSize: 58, bold: true, color: C.green }, "cover-title-2"),
  t("Updated after the parameter/documentation pass: the loading comparison is useful, but the raw HIPS data needed to reproduce Warren Figure 2 are not exposed in the current FED pull.", 680, 420, 1040, 118, { fontSize: 29, color: C.muted }, "cover-promise"),
  ...metric("10", "IMPROVE rows in Addis fAbs + EC mass + EC surface p05-p95", 680, 666, 460, C.red),
  ...metric("147k", "rows with FED laser reflectance/transmittance ratio fields", 1180, 666, 500, C.blue),
  t("Ahmad Jalil | UC Davis", SAFE, 930, 360, 36, { fontSize: 22, color: "#DDE7DE" }, "cover-author"),
], `Open by saying this is an update to last week's IMPROVE analog presentation. The key change is not that the loading analysis failed; it is that the documentation pass changed what we can defensibly claim from the FED R/T fields. The deck separates what the data do show from what is missing for a true Warren Figure 2 reproduction.`);

// Slide 2
slideFrom([
  ...title(2, "Recap", "Last week’s hypothesis was reasonable", "Use IMPROVE as an independent analog for the Addis HIPS regime."),
  t("The April 22 deck framed three checks:", SAFE, 290, 760, 40, { fontSize: 30, bold: true }, "setup"),
  ...bullet([
    { text: "Loading: do any IMPROVE filters reach Addis-like EC surface loading?", color: C.green },
    { text: "Absorption: within that range, do fAbs and EC behave similarly?", color: C.blue },
    { text: "Optical geometry: can the R/T space reproduce the Addis blank-line signature?", color: C.amber },
  ], SAFE, 360, 920, 70),
  rect(1120, 294, 560, 4, C.hair, "right-rule"),
  t("What changed this week", 1120, 326, 620, 44, { fontSize: 34, bold: true, color: C.red }, "changed-title"),
  t("The first two questions are still useful. The third has to be narrowed: FED exposes laser reflectance/transmittance ratio parameters, but not the raw HIPS sphere/plate records, field blanks, or lot-level blank-line metadata.", 1120, 388, 620, 172, { fontSize: 27, color: C.ink }, "changed-body"),
  t("So the new result is partly scientific and partly data-access: IMPROVE can bound the loading regime, but it cannot currently reproduce Warren Figure 2 from public FED alone.", 1120, 628, 620, 150, { fontSize: 29, bold: true, color: C.slate }, "changed-claim"),
  ...footer(),
], `This slide is the transition from last week to this week. I would not say the old analysis was wrong. I would say we learned that the first-order loading comparison is still exactly the right direction, but the R/T piece cannot be presented as a true Warren Figure 2 reproduction. That distinction matters for Warren and Cena because it tells them exactly what we need from them.`);

// Slide 3
slideFrom([
  ...title(3, "Data audit", "What FED exposes vs. what Warren Figure 2 needs", "The public pull answers loading questions; it does not expose the raw calibration record."),
  t("Exposed in the local FED parameter list", SAFE, 300, 700, 38, { fontSize: 31, bold: true, color: C.green }, "left-heading"),
  ...bullet([
    "fAbs plus ECf / OCf / MF / FEf / SOILf",
    "FlowRate and SampDur for computed volume/loading",
    "RefI/RefF/RefM and TransI/TransF/TransM at multiple wavelengths",
    "Units are listed as ratio for those laser fields",
  ], SAFE, 366, 730, 58),
  t("Not exposed in the current pull", 1050, 300, 740, 38, { fontSize: 31, bold: true, color: C.red }, "right-heading"),
  ...bullet([
    { text: "raw HIPS sphere signal R", color: C.red },
    { text: "raw HIPS plate signal T", color: C.red },
    { text: "row-level field blank/sample status", color: C.red },
    { text: "filter lot IDs and blank OLS coefficients", color: C.red },
    { text: "analysis batch/date metadata needed for exact calibration-line reconstruction", color: C.red },
  ], 1050, 366, 760, 56),
  rect(780, 312, 4, 520, C.hair, "divider"),
  t("Interpretation: use FED for first-order loading and bounded comparability, not as the raw HIPS calibration source.", 245, 870, 1420, 50, { fontSize: 32, bold: true, color: C.slate, align: "center" }, "bottom-claim"),
  ...footer("Source: FED parameter JSON; local normalized table; White et al. 2025 data availability statement."),
], `This is the clearest slide for the missing-data message. The exposed parameters are valuable, but they are not the actual instrument engineering outputs from Warren's Figure 2. The multiwavelength laser reflectance/transmittance fields should be labeled as FED ratio fields. I should avoid saying they are raw HIPS R and T unless Warren or the documentation confirms that.`);

// Slide 4
slideFrom([
  ...title(4, "First-order result", "The overlap collapses to a case-study set", "After post-2003/stable-period filtering, Addis-like fAbs plus EC loading is very rare in IMPROVE."),
  ...metric("152,029", "valid post-2003 IMPROVE rows with loading", SAFE, 292, 430, C.green),
  ...metric("36", "rows in Addis fAbs p05-p95 only", 565, 292, 360, C.amber),
  ...metric("7,299", "rows in Addis EC surface p05-p95 only", 965, 292, 390, C.blue),
  ...metric("10", "rows in fAbs + EC mass + EC surface p05-p95", 1395, 292, 410, C.red),
  ...framedImage(img.waterfall, 205, 510, 1510, 390, "screening-waterfall", "Sequential p05-p95 screening from the first-order notebook."),
  ...footer("Notebook: improve_first_order_loading_range_analysis.ipynb"),
], `This is the first main scientific result. The key phrase is: EC loading overlap exists, but the combined overlap with Addis fAbs is tiny. That means we should not over-interpret regressions inside the fully bounded set. It is a small case-study set, not a robust analog population.`);

// Slide 5
slideFrom([
  ...title(5, "Before screening", "Most IMPROVE samples live far below the Addis absorption regime", "The baseline matters because arbitrary fAbs cutoffs can mechanically distort slope and intercept."),
  ...framedImage(img.baseline, 130, 290, 1660, 560, "baseline-distributions", "Post-2003 baseline distributions before any Addis-style screen."),
  t("Read this before thresholding: the Addis fAbs band sits in the far upper tail of IMPROVE, while EC loading overlap is much broader.", 230, 890, 1460, 56, { fontSize: 30, bold: true, color: C.slate, align: "center" }, "baseline-claim"),
  ...footer("Notebook: warren_cena_improve_prep_analysis.ipynb"),
], `This slide addresses Ann's concern from the meeting: first show all the data, then screen. It also explains why the earlier threshold sweep is hard to interpret: once we select only high fAbs rows, we can create intercept behavior mechanically.`);

// Slide 6
slideFrom([
  ...title(6, "Axis-by-axis", "fAbs is the limiting axis, not EC loading", "This is why a strict Addis analog set becomes too sparse for stable regression."),
  ...framedImage(img.overlap, 130, 290, 780, 550, "axis-overlap", "Axis overlap counts relative to Addis p05-p95 ranges."),
  t("The useful takeaway", 1030, 318, 600, 40, { fontSize: 34, bold: true, color: C.green }, "takeaway-heading"),
  ...bullet([
    "EC mass/surface loading has thousands of comparable rows.",
    "The Addis fAbs band has only dozens of rows in IMPROVE.",
    "Putting both axes together leaves roughly ten filters.",
    "Those filters can identify candidates for raw-data requests, but not a general regression.",
  ], 1030, 388, 690, 60),
  ...footer("Notebook: warren_cena_improve_prep_analysis.ipynb"),
], `Emphasize that this slide changes the framing. IMPROVE does have filters with comparable EC loading, but those are not generally in the Addis fAbs regime. That means the problem is not just mass loading; it may be loading plus chemistry, optical regime, site/instrument differences, or a HIPS correction question.`);

// Slide 7
slideFrom([
  ...title(7, "Regression check", "The bounded groups do not reproduce Addis behavior", "Even high-end IMPROVE subsets do not give the Addis-like compressed slope/intercept pattern."),
  ...framedImage(img.fullComparable, 120, 285, 1680, 610, "full-vs-comparable", "Full IMPROVE cloud versus bounded Addis-comparable subsets."),
  t("This supports a conservative claim: IMPROVE does not give a clean empirical correction for Addis from public chemistry + fAbs alone.", 260, 920, 1400, 48, { fontSize: 29, bold: true, color: C.slate, align: "center" }, "regression-claim"),
  ...footer("Notebook: warren_cena_improve_prep_analysis.ipynb"),
], `Here I would be careful. The result does not prove Addis is uniquely wrong; it shows that the public IMPROVE subset is too sparse and does not reproduce the Addis regression. Therefore a public-data-only correction is not defensible yet.`);

// Slide 8
slideFrom([
  ...title(8, "R/T correction", "We can make a FED ratio-field plot, but not Warren Figure 2", "This is the main correction to last week’s optical-geometry interpretation."),
  ...framedImage(img.fedRt, 110, 300, 620, 520, "fed-rt", "FED RefI_635 / TransI_635 ratio-field plot."),
  ...framedImage(img.spartanRt, 800, 300, 900, 520, "spartan-rt", "SPARTAN raw-looking HIPS_R1 / HIPS_T1 analog with field blanks."),
  t("Why the right panel is closer: SPARTAN has FilterType and lot IDs. The FED pull does not expose field blanks, raw R/T, or blank-line coefficients.", 260, 880, 1390, 60, { fontSize: 29, bold: true, color: C.red, align: "center" }, "rt-claim"),
  ...footer("Notebook: improve_fed_rt_proxy_figure2.ipynb"),
], `This slide directly answers whether we can replot Warren Figure 2. For SPARTAN, we have something closer because FilterType identifies blanks and samples. For IMPROVE/FED, the plot can be diagnostic, but it is not Warren's raw R/T calibration space. I should say this plainly.`);

// Slide 9
slideFrom([
  ...title(9, "Documentation pass", "The R/T fields need a stricter label", "The field names are exposed, but their instrument meaning is not enough for a HIPS claim."),
  t("Current safe wording", SAFE, 310, 620, 38, { fontSize: 34, bold: true, color: C.green }, "safe-heading"),
  t("FED laser reflectance/transmittance ratio fields", SAFE, 370, 760, 70, { fontSize: 38, bold: true, color: C.slate }, "safe-wording"),
  t("What I should not say yet", SAFE, 560, 620, 38, { fontSize: 34, bold: true, color: C.red }, "unsafe-heading"),
  t("Raw IMPROVE HIPS sphere/plate R/T", SAFE, 620, 760, 70, { fontSize: 38, bold: true, color: C.red }, "unsafe-wording"),
  rect(925, 300, 4, 500, C.hair, "divider"),
  t("Why this matters", 1020, 310, 620, 40, { fontSize: 34, bold: true, color: C.blue }, "why-heading"),
  ...bullet([
    "The exposed fields span 405–980 nm, not only the Warren HIPS laser geometry.",
    "The parameter table calls them ratio fields, not raw engineering outputs.",
    "White 2025 says individual sphere/plate signals are available from authors, not automatically public.",
    "Any notebook figure using these fields should be relabeled as a FED ratio-field diagnostic.",
  ], 1020, 380, 760, 58),
  ...footer("Source: FED parameter JSON; White et al. 2025; local paper/doc pass."),
], `This is the correction slide. It is important because it protects the analysis from overclaiming. I can still use these columns for exploratory diagnostics, but if the meeting turns to Warren Figure 2, the answer is that the exact raw data are missing from FED.`);

// Slide 10
slideFrom([
  ...title(10, "Updated interpretation", "What we learned after going through the data", "The conclusion is now a boundary statement, not a failed search."),
  t("1", 160, 315, 70, 70, { fontSize: 54, bold: true, color: C.green }, "num1"),
  t("IMPROVE is useful for first-order loading context.", 250, 320, 1320, 48, { fontSize: 34, bold: true }, "learn1"),
  t("It tells us Addis sits in a regime that is unusual for IMPROVE, especially on fAbs.", 250, 372, 1320, 38, { fontSize: 25, color: C.muted }, "learn1b"),
  t("2", 160, 480, 70, 70, { fontSize: 54, bold: true, color: C.blue }, "num2"),
  t("Strict Addis comparability is too sparse for an empirical correction.", 250, 485, 1320, 48, { fontSize: 34, bold: true }, "learn2"),
  t("The best public-data use is candidate identification and boundary-setting, not regression-based correction.", 250, 537, 1320, 38, { fontSize: 25, color: C.muted }, "learn2b"),
  t("3", 160, 645, 70, 70, { fontSize: 54, bold: true, color: C.red }, "num3"),
  t("The missing raw HIPS layer is now the main blocker.", 250, 650, 1320, 48, { fontSize: 34, bold: true }, "learn3"),
  t("To answer the calibration-line question, we need raw R/T, blanks, filter lots, analysis batch/date, and blank-line coefficients.", 250, 702, 1320, 38, { fontSize: 25, color: C.muted }, "learn3b"),
  t("Framing for Warren/Cena: Are we outside the supported HIPS measurement regime, and can the missing calibration data be shared for the candidate filters?", 205, 850, 1510, 60, { fontSize: 30, bold: true, color: C.slate, align: "center" }, "framing"),
  ...footer(),
], `This is the synthesis slide. I would say: we learned something useful even though the public data are incomplete. The study can now make a stronger statement about the limits of public FED for this specific HIPS question, and it sets up a precise data request rather than vague troubleshooting.`);

// Slide 11
slideFrom([
  ...title(11, "Questions to take forward", "What to ask Warren and Cena", "The next meeting should be targeted around data availability and measurement regime."),
  t("For Warren / IMPROVE", SAFE, 300, 760, 42, { fontSize: 34, bold: true, color: C.green }, "warren-heading"),
  ...bullet([
    "Are Addis-like loadings outside the HIPS range where fAbs is considered reliable?",
    "Can we get raw registered sphere/plate R,T for the top IMPROVE candidate filters?",
    "Can those records include field blanks, PTFE lot, analysis batch/date, and blank OLS coefficients?",
    "What exactly are FED Ref/Trans parameters in relation to HIPS versus TOR/carbon laser fields?",
  ], SAFE, 365, 780, 58),
  t("For Cena / SPARTAN", 1050, 300, 760, 42, { fontSize: 34, bold: true, color: C.blue }, "cena-heading"),
  ...bullet([
    { text: "Was the Addis aethalometer flow-ratio fix actually implemented?", color: C.blue },
    { text: "Is there post-fix aethalometer data with stable flow?", color: C.blue },
    { text: "Can SPARTAN share raw HIPS metadata consistently across Addis/JPL lots?", color: C.blue },
    { text: "Has SPARTAN assessed the Warren 2025 pixelation/geometry correction for its filter format?", color: C.blue },
  ], 1050, 365, 780, 58),
  ...footer(),
], `This should be an actionable slide. The point is not just to show what is missing, but to turn it into a request list. For Warren, the ask is raw HIPS/calibration metadata for candidate filters. For Cena, the ask is whether the aethalometer flow fix was implemented and whether SPARTAN has a path to raw HIPS metadata.`);

// Slide 12
slideFrom([
  ...title(12, "Next work", "Make the notebook and slides match the corrected claim", "The immediate task is to relabel, re-query, and prepare a clean external-facing version."),
  t("This week", SAFE, 315, 420, 38, { fontSize: 34, bold: true, color: C.green }, "this-week"),
  ...bullet([
    "Relabel FED R/T plots as ratio-field diagnostics.",
    "Add the missing-data table to the notebook.",
    "Export the top 10–20 candidate filters for a raw-HIPS request.",
  ], SAFE, 380, 760, 58),
  t("Before Warren/Cena", 1050, 315, 560, 38, { fontSize: 34, bold: true, color: C.blue }, "before-meeting"),
  ...bullet([
    { text: "Re-query FED with method/status/validation/flag fields if available.", color: C.blue },
    { text: "Prepare one slide on why public FED cannot reproduce Figure 2.", color: C.blue },
    { text: "Keep iron/dust/OC screens in backup; keep first-order loading as the main story.", color: C.blue },
  ], 1050, 380, 760, 58),
  t("Bottom line", 315, 800, 340, 40, { fontSize: 34, bold: true, color: C.red }, "bottom-heading"),
  t("The public IMPROVE data can bound the Addis problem, but the decisive HIPS calibration evidence is not exposed in FED.", 650, 795, 980, 70, { fontSize: 34, bold: true, color: C.slate }, "bottom-line"),
  ...footer(),
], `Close with the concrete next steps. I would tell Ann that this is a useful outcome: we now know which part of the question public data can answer and which part requires Warren or the IMPROVE processing team. The revised notebook should not present the ratio fields as raw HIPS data.`);

await fs.mkdir(SCRATCH, { recursive: true });
await fs.mkdir(OUTPUT, { recursive: true });

const pptxBlob = await PresentationFile.exportPptx(deck);
const pptxPath = path.join(OUTPUT, "improve_fed_missing_data_update_apr29_2026.pptx");
await pptxBlob.save(pptxPath);
await pptxBlob.save(path.join(OUTPUT, "output.pptx"));

const previewPaths = [];
const slideCount = deck.slides.count;
for (let i = 0; i < slideCount; i += 1) {
  const slide = deck.slides.getItem(i);
  const png = await slide.export({ format: "png" });
  const outPath = path.join(SCRATCH, `slide_${String(i + 1).padStart(2, "0")}.png`);
  await fs.writeFile(outPath, Buffer.from(await png.arrayBuffer()));
  previewPaths.push(outPath);
}

await fs.writeFile(
  path.join(SCRATCH, "build_summary.json"),
  JSON.stringify({ pptxPath, slideCount, previewPaths }, null, 2),
);

console.log(JSON.stringify({ pptxPath, slideCount, previewPaths }, null, 2));
process.exit(0);
