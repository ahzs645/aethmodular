import type { IUIThemeConfig, VegaGlobalConfig } from '@kanaries/graphic-walker'
import { FONT, INK_DARK, INK_LIGHT, type Ink } from '@/lib/theme'

const uiColours = (ink: Ink): IUIThemeConfig['light'] => ({
  background: ink.panel, foreground: ink.text, primary: ink.accent, 'primary-foreground': ink.panel,
  muted: ink.empty, 'muted-foreground': ink.muted, border: ink.border, ring: ink.accent,
  accent: ink.tint, 'accent-foreground': ink.text, dimension: ink.accent, measure: ink.deming,
})
export const WALKER_UI_THEME: IUIThemeConfig = { light: uiColours(INK_LIGHT), dark: uiColours(INK_DARK) }
const plotColours = (ink: Ink) => ({
  background: 'transparent', font: FONT.family,
  axis: { labelColor: ink.axis, titleColor: ink.text, domainColor: ink.axis, tickColor: ink.axis, gridColor: ink.grid,
    labelFontSize: 11, titleFontSize: 12 },
  legend: { labelColor: ink.text, titleColor: ink.text, labelFontSize: 11, titleFontSize: 12 },
  title: { color: ink.text }, view: { stroke: ink.border },
} satisfies VegaGlobalConfig)
export const WALKER_PLOT_THEME = { light: plotColours(INK_LIGHT), dark: plotColours(INK_DARK) }

/** The only styling tied to upstream DOM structure. Checked by check-chart-builder.mjs.
 * Keep the shadow boundary: gallery styles must not alter menus, drag handles or plot marks.
 */
export const WALKER_SHADOW_CSS = `
  :host { display: block; }
  .App, .App .font-sans { font-family: var(--font, ${FONT.family}); }
  .App [role="tabpanel"][data-state="active"] > div.border {
    margin: 0; padding: 0; border: 0; border-radius: 0;
  }
  .App [role="tabpanel"][data-state="active"] > div:first-child:not(.border) {
    margin: 0 0 6px; padding: 0;
  }
  .App [data-rbd-draggable-id] { font-size: calc(12px * var(--text-scale, 1)); }
  @media (max-width: 639px) {
    .App div:has(> [data-rbd-droppable-id="dimensions"]) {
      display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 6px;
    }
    .App div:has(> [data-rbd-droppable-id="dimensions"]) > h4 { grid-column: 1 / -1; }
    .App [data-rbd-droppable-id="dimensions"], .App [data-rbd-droppable-id="measures"] {
      height: 180px; min-height: 0; overflow-y: auto;
    }
  }
`
