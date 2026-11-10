/*
 * Investigation status deck (9 Nov 2026), for the supervisor's end-of-December check-in.
 *
 * Reads the same bundle as the gallery's status page, so slides and gallery agree:
 *   gallery/app/public/data/meetings/status-20261109.json  (uv run python gallery/data/export_meeting_20261109.py)
 *
 * Build (pptxgenjs is not a repo dependency; install it anywhere and point NODE_PATH at it):
 *   NODE_PATH=<dir>/node_modules node deliverables/status_2026-11-09/build_deck.cjs
 * Optional: APPLY_THEME=<path to the pptx skill's scripts/apply_theme.js> writes the theme colours into the file.
 */
const path = require('path')
const fs = require('fs')
const pptxgen = require('pptxgenjs')

const ROOT = path.resolve(__dirname, '../..')
const BUNDLE = path.join(ROOT, 'gallery/app/public/data/meetings/status-20261109.json')
const OUT = path.join(__dirname, 'investigation_status_2026-11-09.pptx')
const bundle = JSON.parse(fs.readFileSync(BUNDLE, 'utf8'))

/** a bundle table as an array of row objects */
const rows = (name) => {
  const t = bundle.tables[name]
  const keys = Object.keys(t)
  return t[keys[0]].map((_, i) => Object.fromEntries(keys.map((k) => [k, t[k][i]])))
}

// ---------------------------------------------------------------- theme
const THEME = {
  name: 'Addis carbon',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '22252A', lt1: 'FFFFFF', dk2: '3A3F45', lt2: 'F3F4F5',
    accent1: 'C49442', accent2: '2C6E9E', accent3: 'B23327', accent4: '8A8F98', accent5: '7A4FA3', accent6: '27AE60',
    hlink: '2C6E9E', folHlink: '7A4FA3',
  },
}
const HEX = THEME.colors
const MUTED = '5B5852'
const GRID = 'E3E5E8'
// gallery candidate colours (bundle.colors), without '#'
const colorOf = (name) => (bundle.colors[name] || '#8A8F98').replace('#', '').toUpperCase()

const pres = new pptxgen()
pres.layout = 'LAYOUT_WIDE' // 13.333 x 7.5 in
pres.title = 'FTIR EC calibration at Addis Ababa: investigation status'
pres.subject = 'Status for the end-of-December check-in'
pres.author = 'Ahmad Jalil'
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace }
const C = pres.SchemeColor

const W = 13.333
const FOOT = 'aethmodular investigation status · 9 Nov 2026'

pres.defineSlideMaster({
  title: 'Title dark',
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.2, w: 11.7, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: 'bottom', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.3, w: 11.7, h: 1.2, fontSize: 20, color: C.accent1, valign: 'top', margin: 0 }, text: '' } },
  ],
})
pres.defineSlideMaster({
  title: 'Section dark',
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 3.0, w: 11.7, h: 1.2, fontSize: 40, bold: true, color: C.background1, valign: 'top', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 2.3, w: 11.7, h: 0.6, fontSize: 18, color: C.accent1, valign: 'bottom', margin: 0 }, text: '' } },
  ],
  slideNumber: { x: 12.4, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent4, align: 'right' },
})
pres.defineSlideMaster({
  title: 'Content',
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.35, w: 12.1, h: 0.75, fontSize: 32, bold: true, color: C.text1, valign: 'top', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.6, y: 1.12, w: 12.1, h: 0.55, fontSize: 16, color: C.text2, valign: 'top', margin: 0 }, text: '' } },
    { text: { text: FOOT, options: { x: 0.6, y: 6.95, w: 8, h: 0.3, fontSize: 10, color: HEX.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.4, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent4, align: 'right' },
})

// ---------------------------------------------------------------- helpers
let section = ''
const addSection = (title) => { section = title; pres.addSection({ title }) }
const content = (title, kicker) => {
  const s = pres.addSlide({ masterName: 'Content', sectionTitle: section })
  s.addText(title, { placeholder: 'title' })
  if (kicker) s.addText(kicker, { placeholder: 'body' })
  return s
}
const chartText = (extra = {}) => ({
  catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt', dataLabelFontFace: '+mn-lt', legendFontFace: '+mn-lt',
  titleFontFace: '+mn-lt', valAxisTitleFontFace: '+mn-lt', catAxisTitleFontFace: '+mn-lt',
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, dataLabelColor: HEX.dk1,
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12,
  valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: 'none' },
  catAxisLineShow: false, valAxisLineShow: false,
  ...extra,
})
/** a panel of commentary: a heading and short paragraphs, on a tinted card */
const commentary = (s, { x, y, w, h, heading, paras, name }) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.08, objectName: `${name} card` })
  const runs = [{ text: heading, options: { bold: true, fontSize: 18, color: C.text1, breakLine: true, paraSpaceAfter: 8 } }]
  paras.forEach((p, i) => runs.push({ text: p, options: { fontSize: 14, color: C.text1, bullet: true, paraSpaceAfter: 6, breakLine: i < paras.length - 1 } }))
  s.addText(runs, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: h - 0.4, valign: 'top', margin: 0, isTextBox: true, objectName: `${name} text` })
}
const stat = (s, { x, y, w, value, label, color, name }) => {
  s.addText(value, { x, y, w, h: 0.85, fontSize: 44, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: `${name} value` })
  s.addText(label, { x, y: y + 0.85, w, h: 0.75, fontSize: 13, color: C.text2, margin: 0, valign: 'top', isTextBox: true, objectName: `${name} label` })
}
const f2 = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2)

// ---------------------------------------------------------------- data
const ecShare = rows('ec_share')
const loading = rows('loading')
const addis = rows('addis')
const cals = rows('cals')
const scores = rows('scores')
const heavy = rows('heavy')
const commits = rows('commits')
const milestones = rows('milestones')
const claims = rows('claims')
const HIPS = 'HIPS optical BC (Fabs / MAC 10)'
const AETH = 'Aethalometer IR BC'
const conc = loading[0], mass = loading[1]
const hips = addis.filter((r) => r.comparator === HIPS)
const aeth = addis.filter((r) => r.comparator === AETH)
const byName = (list, name) => list.find((r) => r.candidate === name)
const peak = commits.reduce((a, b) => (b.commits > a.commits ? b : a))

// ================================================================ 1. title
addSection('Status')
{
  const s = pres.addSlide({ masterName: 'Title dark', sectionTitle: section })
  s.addText('FTIR EC calibration at Addis Ababa: where the investigation stands', { placeholder: 'title' })
  s.addText([
    { text: 'Status for the end-of-December check-in', options: { breakLine: true } },
    { text: 'Ahmad Jalil · 9 November 2026', options: { color: 'C9CCD1', fontSize: 16 } },
  ], { placeholder: 'body' })
  s.addNotes('Status update after the 7 November closeout of the existing-data investigation. The same charts are in the aethmodular gallery: Investigations, Meetings, Nov 9, 2026.')
}

// ================================================================ 2. bottom line
{
  const s = content('The bottom line', 'The analysis is finished; the writing that counts as finished has not started')
  const cards = [
    { head: 'Done', color: HEX.accent6, big: '7 Nov', body: 'Existing-data investigation closed out: six preprocessing pipelines on 12,386 IMPROVE filters, a reproducible, preserved release' },
    { head: 'Behind', color: HEX.accent3, big: '0 drafts', body: 'No paper or chapter draft yet. The supervisor’s test for finished is the chapter written and the paper ready' },
    { head: 'Next', color: HEX.accent2, big: '~18 Dec', body: 'Paper 1 and the chapter to the supervisor, before the end-of-December status and the January decision' },
  ]
  cards.forEach((c, i) => {
    const x = 0.6 + i * 4.1, y = 2.0, w = 3.8, h = 3.9
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.1, objectName: `${c.head} card` })
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.35, w: 0.32, h: 0.32, fill: { color: c.color }, line: { color: c.color }, objectName: `${c.head} marker` })
    s.addText(c.head, { x: x + 0.75, y: y + 0.27, w: w - 1, h: 0.5, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `${c.head} heading` })
    s.addText(c.big, { x: x + 0.3, y: y + 1.0, w: w - 0.6, h: 1.0, fontSize: 44, bold: true, color: c.color, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: `${c.head} figure` })
    s.addText(c.body, { x: x + 0.3, y: y + 2.15, w: w - 0.6, h: 2.1, fontSize: 15, color: C.text1, valign: 'top', margin: 0, isTextBox: true, objectName: `${c.head} body` })
  })
  s.addNotes('One message: the analysis question has a defensible answer with existing data, but the deliverable the supervisor asked for (chapter + paper) does not exist yet. The rest of the deck is the evidence, then the plan.')
}

// ================================================================ 3. commitments
{
  const s = content('What this update answers to', 'Commitments from the 1 Sep committee meeting and the 7 Oct supervisory check-in')
  const head = ['From', 'Commitment', 'Due'].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: HEX.dk1 } } }))
  const body = [
    ['Committee, 1 Sep', 'Go/no-go on the Addis work: a defensible pathway, or stop', 'Dec–Jan'],
    ['Committee, 1 Sep', 'Prince George / PurpleAir results so Peter can contribute', 'Open (on hold)'],
    ['Supervisor, 7 Oct', 'Status at the end of the fall semester; finished = chapter written and paper ready', 'End of Dec'],
    ['Supervisor, 7 Oct', 'Choose the next objective (international collaborations, Brian’s project, or another component)', 'Jan at the latest'],
    ['Supervisor, 7 Oct', 'Next committee meeting', 'Within 6 months (~Apr)'],
  ].map((r) => r.map((t) => ({ text: t })))
  s.addTable([head, ...body], { x: 0.6, y: 1.95, w: 12.1, colW: [2.4, 7.3, 2.4], fontSize: 15, color: HEX.dk1, border: { type: 'solid', pt: 0.75, color: GRID }, rowH: 0.62, valign: 'middle', margin: 0.08, objectName: 'commitments table' })
  s.addNotes('These are the commitments the December status is measured against. The PurpleAir item is the one the committee may raise: Peter said he cannot contribute until there is Prince George material.')
}

// ================================================================ section: findings
addSection('What we learned')
{
  const s = pres.addSlide({ masterName: 'Section dark', sectionTitle: section })
  s.addText('Section 1', { placeholder: 'body' })
  s.addText('What the existing data showed', { placeholder: 'title' })
}

// ================================================================ 4. domain
{
  const s = content('Addis is outside the IMPROVE library', 'Composition and loading both sit beyond what the calibration was trained on')
  const share = ecShare
  s.addChart(pres.charts.BAR, [{ name: 'EC share of PM2.5 (%)', labels: share.map((r) => r.population), values: share.map((r) => r.ec_share) }], chartText({
    x: 0.6, y: 1.9, w: 7.4, h: 4.8, barDir: 'bar', catAxisOrientation: 'maxMin',
    chartColors: share.map((r) => colorOf(r.network)),
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0"%"',
    valAxisMinVal: 0, valAxisMaxVal: 30, valAxisLabelFormatCode: '0"%"',
    showTitle: true, title: 'EC share of PM2.5 (median)', titleFontSize: 14, titleColor: HEX.dk1,
    showLegend: false, objectName: 'EC share chart',
  }))
  stat(s, { x: 8.5, y: 2.0, w: 4.2, value: `${Math.round(conc['Addis median'] / conc['IMPROVE median'])}×`, label: 'Addis median vs IMPROVE median, µg/m³ (HIPS Fabs / MAC 10)', name: 'concentration ratio' })
  stat(s, { x: 8.5, y: 3.65, w: 4.2, value: `${Math.round(mass['Addis median'] / mass['IMPROVE median'])}×`, label: 'the same per filter, which is what FTIR sees', name: 'mass ratio' })
  stat(s, { x: 8.5, y: 5.3, w: 4.2, value: `${Math.round(mass['% of Addis above IMPROVE p99'])}%`, label: 'of Addis filters above the IMPROVE 99th percentile per filter', name: 'above p99', color: C.accent3 })
  s.addNotes('Addis EC is about a fifth of PM2.5, against about 4 percent across IMPROVE, and Addis has the second-lowest OC/EC in either network. Per filter, most Addis filters are heavier than almost every IMPROVE filter; the only IMPROVE filters in that range are wildfire and other unusual days. Sources: 28 Oct speciation table and 9 Oct loading summary.')
}

// ================================================================ 5. baselining vs HIPS
{
  const sp = byName(hips, 'Spline baseline'), pub = byName(hips, 'SPARTAN published')
  const s = content('Baselining is the biggest single gain', `Against HIPS, the intercept moves from ${f2(pub.intercept)} (published) to ${f2(sp.intercept)} µg/m³ (spline), but the slope is ${sp.slope.toFixed(2)}`)
  s.addChart(pres.charts.BAR, [{ name: 'Addis intercept vs HIPS (µg/m³)', labels: hips.map((r) => r.candidate), values: hips.map((r) => r.intercept) }], chartText({
    x: 0.6, y: 1.9, w: 7.6, h: 4.8, barDir: 'bar', catAxisOrientation: 'maxMin',
    chartColors: hips.map((r) => colorOf(r.candidate)),
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00',
    valAxisMinVal: -7, valAxisMaxVal: 0, valAxisLabelFormatCode: '0',
    showTitle: true, title: 'Addis intercept against HIPS optical BC, µg/m³ (target 0)', titleFontSize: 14, titleColor: HEX.dk1,
    showLegend: false, catAxisLabelPos: 'low', objectName: 'HIPS intercept chart',
  }))
  commentary(s, { x: 8.6, y: 1.9, w: 4.1, h: 4.8, name: 'baselining', heading: 'What it means', paras: [
    'Removing the spectral baseline closes most of the intercept gap',
    `Slopes still range ${Math.min(...hips.map((r) => r.slope)).toFixed(2)}–${Math.max(...hips.map((r) => r.slope)).toFixed(2)}; none is 1:1`,
    'The slope scales with the assumed MAC (10 m²/g, a convention); the intercept does not',
    `Published product on ${pub.n} filters, recalibrations on ${sp.n}`,
  ] })
  s.addNotes('Deming FTIR EC against HIPS Fabs / MAC 10, k 18, from the closeout saved fits. Raw spectra and VIBES without baselining reproduce the published offset; the spline baseline is what moves it.')
}

// ================================================================ 6. aethalometer
{
  const hipsSelf = byName(aeth, 'HIPS optical BC itself')
  const s = content('The aethalometer does not pick a winner', `Most candidates are already near zero against it; HIPS itself sits at ${f2(hipsSelf.intercept)} µg/m³`)
  s.addChart(pres.charts.BAR, [{ name: 'Addis intercept vs aethalometer (µg/m³)', labels: aeth.map((r) => r.candidate), values: aeth.map((r) => r.intercept) }], chartText({
    x: 0.6, y: 1.9, w: 7.6, h: 4.8, barDir: 'bar', catAxisOrientation: 'maxMin',
    chartColors: aeth.map((r) => colorOf(r.candidate)),
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '+0.00;-0.00;0.00',
    valAxisMinVal: -1.5, valAxisMaxVal: 3.5, valAxisLabelFormatCode: '0.0',
    showTitle: true, title: `Intercept against aethalometer IR BC, µg/m³ (n = ${hipsSelf.n})`, titleFontSize: 14, titleColor: HEX.dk1,
    showLegend: false, catAxisLabelPos: 'low', objectName: 'aethalometer intercept chart',
  }))
  commentary(s, { x: 8.6, y: 1.9, w: 4.1, h: 4.8, name: 'aethalometer', heading: 'Read with care', paras: [
    'The large negative FTIR intercept appears against HIPS, not against the aethalometer',
    'So which measurement carries the offset is still open',
    'Agreement between two optical methods is a diagnostic, not proof of correct EC',
    'No Addis filter has thermal (TOR) EC',
  ] })
  s.addNotes('This is the slide to be careful on. PTFE-anchored VIBES gives +0.10 against the aethalometer, but so does the published product (+0.08). The aethalometer comparison therefore cannot select a method. HIPS against the same aethalometer has intercept +2.84.')
}

// ================================================================ 7. seasons
{
  const order = ['all IMPROVE (Figure 7)', 'spectral look-alikes (Figure 9)', '500 lowest TOR OC/EC', 'spectral top 2,000, lowest OC/EC', 'urban IMPROVE only']
  const seasons = ['Dry', 'Belg', 'Kiremt']
  const data = order.map((cal) => ({ name: cal, labels: seasons, values: seasons.map((se) => cals.find((r) => r.calibration === cal && r.season === se).slope) }))
  const kir = cals.find((r) => r.calibration === 'spectral look-alikes (Figure 9)' && r.season === 'Kiremt')
  const s = content('The seasons fail differently', `Season-specific look-alikes lift Kiremt’s slope to ${kir.slope.toFixed(2)}; Dry and Belg stay below 0.75`)
  s.addChart(pres.charts.BAR, data, chartText({
    x: 0.6, y: 1.9, w: 7.9, h: 4.8, barDir: 'col', barGapWidthPct: 60,
    chartColors: order.map(colorOf),
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelFontSize: 9,
    valAxisMinVal: 0, valAxisMaxVal: 1.2, valAxisLabelFormatCode: '0.0',
    showTitle: true, title: 'Addis slope by season and calibration filters (target 1)', titleFontSize: 14, titleColor: HEX.dk1,
    showLegend: true, legendPos: 'b', objectName: 'seasonal slope chart',
  }))
  commentary(s, { x: 8.9, y: 1.9, w: 3.8, h: 4.8, name: 'seasons', heading: 'Composition, not concentration', paras: [
    'Per µg of EC, Addis has about a tenth of the organic absorbance of its IMPROVE look-alikes',
    'Least organic signal in Dry (diesel-like), most in Kiremt',
    'Choosing filters by OC/EC or urban sites does not rescue Dry or Belg',
  ] })
  s.addNotes('Baseline-corrected spectra, Deming FTIR EC vs HIPS Fabs / MAC 10 on the 189 poster filters (28 Oct meeting). The organic-signal ratios (Dry 0.07–0.12, Belg 0.13–0.22, Kiremt 0.18–0.28) are from the 21 Oct notes to Ann.')
}

// ================================================================ 8. scoreboard
{
  const s = content('No universal winner on IMPROVE', 'Six pipelines, the same 12,386 filters, 18 PLS components, five site-grouped folds')
  const metrics = [['MedAE', 'Median abs. error'], ['MAE', 'Mean abs. error'], ['RMSE', 'RMSE']]
  s.addChart(pres.charts.BAR, metrics.map(([k, label]) => ({ name: label, labels: scores.map((r) => r.candidate), values: scores.map((r) => r[k]) })), chartText({
    x: 0.6, y: 1.9, w: 7.9, h: 4.8, barDir: 'col', barGapWidthPct: 50,
    chartColors: [HEX.accent4, HEX.accent1, HEX.dk2],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelFontSize: 9, catAxisLabelFontSize: 11,
    valAxisMinVal: 0, valAxisMaxVal: 5, valAxisMajorUnit: 1, valAxisLabelFormatCode: '0',
    showTitle: true, title: 'Out-of-fold error against IMPROVE TOR EC, µg per filter (lower is better)', titleFontSize: 14, titleColor: HEX.dk1,
    showLegend: true, legendPos: 'b', objectName: 'scoreboard chart',
  }))
  const best = (k) => scores.reduce((a, b) => (b[k] < a[k] ? b : a)).candidate
  commentary(s, { x: 8.9, y: 1.9, w: 3.8, h: 4.8, name: 'scoreboard', heading: 'Best by metric', paras: [
    `Median abs. error: ${best('MedAE')}`,
    `Mean abs. error and 95th percentile: ${best('MAE')}`,
    `RMSE: ${best('RMSE')}`,
    'Descriptive differences, no significance test; a retrospective comparison',
  ] })
  s.addNotes('From the 7 Nov closeout scoreboard (gallery/review/closeout_20261107). Every candidate predicts every eligible filter, so the comparison is on identical physical filters.')
}

// ================================================================ 9. heavy filters
{
  const epo = byName(heavy, 'Anchored + EPO'), anch = byName(heavy, 'PTFE-anchored VIBES')
  const s = content('EPO helps the Addis-like filters, at a cost', `Median FTIR/TOR on the 163 heavy, low-OC/EC IMPROVE filters: ${anch.ratio.toFixed(2)} → ${epo.ratio.toFixed(2)} with EPO`)
  s.addChart([
    { type: pres.charts.BAR, data: [{ name: 'Median FTIR / TOR', labels: heavy.map((r) => r.candidate), values: heavy.map((r) => r.ratio) }],
      options: { barDir: 'col', chartColors: heavy.map((r) => colorOf(r.candidate)), showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00' } },
    { type: pres.charts.LINE, data: [{ name: '1:1 target', labels: heavy.map((r) => r.candidate), values: heavy.map(() => 1) }],
      options: { chartColors: [HEX.accent3], lineSize: 2, lineDash: 'dash', lineDataSymbol: 'none', showValue: false } },
  ], chartText({
    x: 0.6, y: 1.9, w: 7.9, h: 4.8, catAxisLabelFontSize: 11,
    valAxisMinVal: 0, valAxisMaxVal: 1.2, valAxisLabelFormatCode: '0.0',
    showTitle: true, title: 'Heavy, low-OC/EC IMPROVE filters: median FTIR / TOR (dashed = 1:1)', titleFontSize: 14, titleColor: HEX.dk1,
    showLegend: false, objectName: 'heavy filter chart',
  }))
  const sc = (n) => scores.find((r) => r.candidate === n)
  commentary(s, { x: 8.9, y: 1.9, w: 3.8, h: 4.8, name: 'heavy', heading: 'The trade', paras: [
    'These 163 filters (15–150 µg EC, OC/EC < 2.5) are the IMPROVE filters most like Addis',
    `EPO raises their ratio toward 1 but overall MAE rises ${sc('PTFE-anchored VIBES').MAE.toFixed(2)} → ${sc('Anchored + EPO').MAE.toFixed(2)} µg/filter`,
    'A composition-dependent trade-off, not a preferred general EC product',
  ] })
  s.addNotes('The matched-resolution control (same 1,361 channels, with and without EPO) shows the same trade, so it is not a resolution artefact.')
}

// ================================================================ 10. limits
{
  const s = content('What existing data cannot answer', 'Absolute EC accuracy at Addis needs thermal EC on Addis filters')
  s.addTable([
    [{ text: 'Do not claim', options: { bold: true, color: C.background1, fill: { color: HEX.dk1 } } }, { text: 'Why', options: { bold: true, color: C.background1, fill: { color: HEX.dk1 } } }],
    ...claims.map((r) => [{ text: r['do not claim'], options: { bold: true } }, { text: r.why }]),
  ], { x: 0.6, y: 1.9, w: 8.2, colW: [3.4, 4.8], fontSize: 13, color: HEX.dk1, border: { type: 'solid', pt: 0.75, color: GRID }, rowH: 0.6, valign: 'middle', margin: 0.07, objectName: 'claims table' })
  commentary(s, { x: 9.2, y: 1.9, w: 3.5, h: 4.8, name: 'validation', heading: 'The validation step', paras: [
    'About 10 paired FTIR + TOR filters from Addis city are expected through Ann',
    'Funding for a year of collocated filters was lost',
    'Agree what counts as agreement before they arrive',
  ] })
  s.addNotes('These limits belong in the paper as stated limitations, not as reasons to delay writing.')
}

// ================================================================ section: plan
addSection('Plan')
{
  const s = pres.addSlide({ masterName: 'Section dark', sectionTitle: section })
  s.addText('Section 2', { placeholder: 'body' })
  s.addText('Schedule and decisions', { placeholder: 'title' })
}

// ================================================================ 11. time
{
  const s = content('Where the time went', `${peak.commits} commits in the week of ${peak.week}; no manuscript draft yet`)
  s.addChart(pres.charts.BAR, [{ name: 'Commits', labels: commits.map((r) => r.week), values: commits.map((r) => r.commits) }], chartText({
    x: 0.6, y: 1.9, w: 8.2, h: 4.8, barDir: 'col', barGapWidthPct: 40,
    chartColors: [HEX.accent2],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0',
    valAxisMinVal: 0, valAxisLabelFormatCode: '0',
    showTitle: true, title: 'Commits to aethmodular per week (week starting)', titleFontSize: 14, titleColor: HEX.dk1,
    showLegend: false, objectName: 'commit chart',
  }))
  commentary(s, { x: 9.2, y: 1.9, w: 3.5, h: 4.8, name: 'time', heading: 'What changes', paras: [
    'Most of this was analysis, gallery and release work',
    'Valuable, but the supervisor counts chapters and papers',
    'Analysis is frozen from now until the draft exists',
  ] })
  s.addNotes('Commits are a proxy for where effort went, not a measure of progress. The point is the shift from building to writing.')
}

// ================================================================ 12. timeline
{
  const s = content('Path to the January decision', 'Writing first; open analysis waits until the draft exists')
  const steps = [
    ['7 Nov', 'Closeout done', HEX.accent6],
    ['~20 Nov', 'Paper 1 outline to Ann', HEX.accent2],
    ['Early Dec', 'Full draft to Ann', HEX.accent2],
    ['~18 Dec', 'Paper + chapter to supervisor', HEX.accent2],
    ['End Dec', 'Semester status', HEX.accent1],
    ['January', 'Go/no-go and next objective', HEX.accent1],
    ['~April', 'Committee meeting', HEX.accent4],
  ]
  const x0 = 1.5, x1 = 11.8, y = 3.4
  s.addShape(pres.shapes.LINE, { x: x0, y, w: x1 - x0, h: 0, line: { color: HEX.accent4, width: 2 }, objectName: 'timeline axis' })
  const step = (x1 - x0) / (steps.length - 1)
  steps.forEach(([when, what, color], i) => {
    const cx = x0 + i * step
    s.addShape(pres.shapes.OVAL, { x: cx - 0.2, y: y - 0.2, w: 0.4, h: 0.4, fill: { color }, line: { color: HEX.lt1, width: 2 }, objectName: `step ${i + 1} marker` })
    s.addText(when, { x: cx - 0.85, y: y - 1.05, w: 1.7, h: 0.5, fontSize: 16, bold: true, color: C.text1, align: 'center', margin: 0, isTextBox: true, objectName: `step ${i + 1} date` })
    s.addText(what, { x: cx - 0.85, y: y + 0.4, w: 1.7, h: 1.0, fontSize: 13, color: C.text1, align: 'center', valign: 'top', margin: 0, isTextBox: true, objectName: `step ${i + 1} label` })
  })
  const waiting = milestones.filter((m) => m.status === 'Waiting on others').map((m) => `${m.milestone} (${m.due})`)
  s.addText([{ text: 'Waiting on others: ', options: { bold: true } }, { text: waiting.join('; ') }],
    { x: 0.6, y: 5.2, w: 12.1, h: 0.9, fontSize: 14, color: C.text2, margin: 0, valign: 'top', isTextBox: true, objectName: 'waiting note' })
  s.addNotes('Proposed dates. The end-of-December status and January decision are the supervisor’s; the writing dates are mine to commit to.')
}

// ================================================================ 13. decisions
{
  const s = content('Decisions to make', 'Four questions for the supervisor and Ann')
  const items = [
    ['Go/no-go on Addis', 'Proposed: a defensible pathway exists. Write the existing-data result now; the TOR filters validate it later'],
    ['Next objective (January)', 'Brian’s glacier black carbon project, international collaborations across SPARTAN/MAIA, or restarting PurpleAir'],
    ['Paper 1 target journal', 'To agree with Ann and the supervisor once the outline exists'],
    ['TOR acceptance criteria', 'Eligibility, uncertainty and agreement limits, set before the Addis city filters arrive'],
  ]
  items.forEach(([head, body], i) => {
    const col = i % 2, row = Math.floor(i / 2)
    const x = 0.6 + col * 6.15, y = 1.95 + row * 2.4, w = 5.95, h = 2.2
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.1, objectName: `decision ${i + 1} card` })
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 }, objectName: `decision ${i + 1} badge` })
    s.addText(String(i + 1), { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.background1, align: 'center', valign: 'middle', margin: 0, isTextBox: true, objectName: `decision ${i + 1} number` })
    s.addText(head, { x: x + 1.05, y: y + 0.3, w: w - 1.3, h: 0.55, fontSize: 19, bold: true, color: C.text1, valign: 'middle', margin: 0, isTextBox: true, objectName: `decision ${i + 1} heading` })
    s.addText(body, { x: x + 1.05, y: y + 0.95, w: w - 1.3, h: 1.1, fontSize: 14, color: C.text1, valign: 'top', margin: 0, isTextBox: true, objectName: `decision ${i + 1} body` })
  })
  s.addNotes('Decision 1 is the committee’s exit criterion from 1 Sep, made concrete. Decision 2 is the supervisor’s January decision.')
}

// ================================================================ 14. close
{
  const s = pres.addSlide({ masterName: 'Title dark', sectionTitle: section })
  s.addText('Next two weeks', { placeholder: 'title' })
  s.addText([
    { text: 'Send this status to the supervisor', options: { bullet: true, breakLine: true } },
    { text: 'Paper 1 outline to Ann by ~20 November', options: { bullet: true, breakLine: true } },
    { text: 'Close the October loose ends: progress report additions, Brian, the rejected paper, Mariam, Satoshi', options: { bullet: true, breakLine: true } },
    { text: 'Gallery: Investigations → Meetings → Nov 9, 2026', options: { bullet: true, color: 'C9CCD1' } },
  ], { x: 0.8, y: 4.3, w: 11.7, h: 2.4, fontSize: 20, color: HEX.accent1, paraSpaceAfter: 8, valign: 'top', margin: 0, isTextBox: true, objectName: 'next steps' })
  s.addNotes('The gallery page has the same charts, interactive, with the metric and comparator pickers.')
}

;(async () => {
  await pres.writeFile({ fileName: OUT })
  if (process.env.APPLY_THEME) {
    const { applyTheme } = require(process.env.APPLY_THEME)
    await applyTheme(OUT, THEME)
  }
  console.log('wrote', path.relative(ROOT, OUT))
})()
