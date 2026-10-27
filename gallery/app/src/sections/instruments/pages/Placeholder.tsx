import { ChartFrame } from '@/components/ChartFrame'

/** A sub-page not built yet: what it will hold and the notebooks it comes from. */
export function Placeholder({ title, notebooks, plan }: { title: string; notebooks: string[]; plan: string }) {
  return (
    <ChartFrame id={`aeth-${title}`} title={title} subtitle={plan} exportable={false}>
      <ul className="subtitle">{notebooks.map((n) => <li key={n}><code>{n}</code></li>)}</ul>
    </ChartFrame>
  )
}
