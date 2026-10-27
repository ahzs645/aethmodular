import { INK_DARK, INK_LIGHT, type Ink } from './theme'

// Both source palettes are matched: one chart may follow a different theme from the page.
// Only display ink changes; the site and season palettes remain those supplied by the data.
export const CHART_DISPLAY_CSS = (Object.keys(INK_LIGHT) as (keyof Ink)[]).flatMap(key =>
  ['fill', 'stroke'].map(attr => {
    const selectors = [...new Set([INK_LIGHT[key], INK_DARK[key]])].map(color => `.panel[data-chart-theme] svg [${attr}="${color}" i]`).join(',')
    return `${selectors}{${attr}:var(--chart-${key})}`
  }),
).join('\n')

export function chartInkStyle(theme: 'light' | 'dark') {
  const ink = theme === 'dark' ? INK_DARK : INK_LIGHT
  return {
    ...Object.fromEntries(Object.entries(ink).flatMap(([key, color]) => [[`--chart-${key}`, color], [`--ink-${key}`, color]])),
    '--accent': ink.accent, '--accent-soft': ink.tint, '--danger': ink.negative, '--good': ink.good,
    '--ink-hover': ink.empty, '--ink-faint': ink.identity, '--ink-strong': ink.fit,
    '--danger-soft': theme === 'dark' ? '#3a1f24' : '#fbe9ec',
    colorScheme: theme,
  }
}
