import { DOT_OPACITY, onPaintPair, paintPairs } from './theme.ts'

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

/** the whole stylesheet for the pairs registered so far (node tests read it) */
export const themeCss = () => [...[...paintPairs()].flatMap(([l, d]) => pairRules(l, d)), ...dotRules()].join('\n')

if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.id = 'theme-paint'
  style.textContent = themeCss()
  document.head.appendChild(style)
  // themed() pairs register when a chart first renders: add their rules then
  onPaintPair((l, d) => { style.textContent += `\n${pairRules(l, d).join('\n')}` })
}
