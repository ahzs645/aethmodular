import { DOT_OPACITY, INK_DARK, INK_LIGHT, onPaintPair, paintPairs } from './theme.ts'

/**
 * The dark theme as one stylesheet, so a theme switch recolours the page without redrawing a chart.
 *
 * Marks are drawn in light colours (lib/theme INK, BASELINE_COLOR, themed()). For every paint pair
 * (a light colour with a dark twin) this adds a CSS variable that holds the twin on the dark page
 * (html[data-theme="dark"]) and in a chart set to dark (.panel[data-chart-theme="dark"], components/
 * DisplayControls), and rules that show it:
 *   - an SVG fill, stroke or stop-color attribute of the light hex (React writes them as given);
 *   - an inline HTML color / background of it (React sets el.style, which the browser serialises as
 *     "color: rgb(…)"), so colours handed to legends, labels and tables follow too.
 * The SVG rules sit in :where() so any rule in styles.css still wins; the inline ones need !important
 * to beat the inline style. Scatter dots tagged data-dot get their dark opacity step the same way.
 * Nothing inside [data-print] (the PDF export's off-screen copy, which svg2pdf reads computed styles
 * from) is touched, and a figure saved as SVG or PNG carries no stylesheet, so exports stay light.
 */
const DARK = ':root[data-theme="dark"]'
const CHART_DARK = '.panel[data-chart-theme="dark"]'
const CHART_LIGHT = `${DARK} .panel[data-chart-theme="light"]`
const OUTSIDE_PRINT = ':not([data-print] *)'

const rgb = (hex: string) => `rgb(${[1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(', ')})`
const inline = (prop: string, value: string) => [`[style^="${prop}: ${value}"]`, `[style*="; ${prop}: ${value}"]`]

export function pairRules(light: string, dark: string): string[] {
  if (!/^#[0-9a-f]{6}$/.test(light) || !/^#[0-9a-f]{6}$/.test(dark)) return []
  const v = `--p-${light.slice(1)}`
  const paint = `var(${v}, ${light})`
  const html = (props: string[]) => props.flatMap((p) => inline(p, rgb(light))).map((s) => `${s}${OUTSIDE_PRINT}`).join(',')
  return [
    `${DARK},${CHART_DARK}{${v}:${dark}}`,
    `${CHART_LIGHT}{${v}:${light}}`,
    ...['fill', 'stroke', 'stop-color'].map((a) => `:where(svg [${a}="${light}" i]${OUTSIDE_PRINT}){${a}:${paint}}`),
    `${html(['color'])}{color:${paint}!important}`,
    `${html(['background', 'background-color'])}{background-color:${paint}!important}`,
  ]
}

export function dotRules(): string[] {
  return Object.entries(DOT_OPACITY).filter(([, o]) => o.light !== o.dark).flatMap(([level, o]) => {
    const dot = `[data-dot="${level}"]${OUTSIDE_PRINT}`
    return [`${DARK} ${dot},${CHART_DARK} ${dot}{fill-opacity:${o.dark}}`, `${CHART_LIGHT} ${dot}{fill-opacity:${o.light}}`]
  })
}

/**
 * SVG text with no fill of its own, and none inherited from a [fill] ancestor, is drawn black by the
 * browser: on the dark page that is unreadable. It takes the text ink instead (INK_DARK.text, and the
 * light step in a chart set to light), so a label or value written as a bare <text> follows the theme.
 * Everything sits in :where(), so any fill a chart or styles.css gives the text still wins, and the
 * light-chart rule comes second to override at the same (zero) specificity. Exports stay as drawn.
 */
export function textRules(): string[] {
  const bare = (scope: string) => `${scope} svg text:not([fill], [fill] *, [style*="fill"], [data-print] *)`
  return [
    `:where(${bare(DARK)}, ${bare(CHART_DARK)}){fill:${INK_DARK.text}}`,
    `:where(${bare(CHART_LIGHT)}){fill:${INK_LIGHT.text}}`,
  ]
}

/** the whole stylesheet for the pairs registered so far (node tests read it) */
export const themeCss = () => [...[...paintPairs()].flatMap(([l, d]) => pairRules(l, d)), ...dotRules(), ...textRules()].join('\n')

if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.id = 'theme-paint'
  style.textContent = themeCss()
  document.head.appendChild(style)
  // themed() pairs register when a chart first renders: add their rules then
  onPaintPair((l, d) => { style.textContent += `\n${pairRules(l, d).join('\n')}` })
}
