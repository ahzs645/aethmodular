import { ChartFrame } from '@/components/ChartFrame'
import { useSearchParam } from '@/lib/url'
import { useFitHeight } from '@/lib/chartFit'

/** Plain text deliberately stays plain text: citations cannot inject markup or scripts. */
export function NoteTile() {
  const [text, setText] = useSearchParam<string>('note.text', '')
  const height = useFitHeight()
  return <ChartFrame id="research-note" title="Research note" exportable={false}>
    <textarea aria-label="Research note" className="dashboard-note" placeholder="Hypothesis, cohort notes, source citation or caveat…" value={text}
      maxLength={12000} onChange={e => setText(e.target.value)} style={{ height: height ?? 220 }} />
  </ChartFrame>
}
