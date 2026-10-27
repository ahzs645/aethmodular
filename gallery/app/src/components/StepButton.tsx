/**
 * The gallery's one previous / next button: a chevron, square, exactly as tall as the Combobox it
 * usually sits beside (styles.css .step-btn takes the trigger's font, line height, padding and border).
 * Used wherever something steps: the single-chart view's charts, a chart's own steps (Connected
 * scatter's months or years). Text arrows (← →) and taller touch buttons beside a picker are not used.
 */
export function StepButton({ dir, label, title, disabled, onClick }: {
  dir: 'prev' | 'next'
  /** what it steps to, for assistive tech: "Previous chart", "Next step" */
  label: string
  title?: string
  disabled?: boolean
  onClick: () => void
}) {
  return <button type="button" className="btn step-btn" aria-label={label} title={title ?? label} disabled={disabled} onClick={onClick}>
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d={dir === 'prev' ? 'M10 3.5 5.5 8l4.5 4.5' : 'M6 3.5 10.5 8 6 12.5'} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  </button>
}
