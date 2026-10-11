import { ChartFrame, Empty } from '@/components/ChartFrame'

/**
 * The panel a page shows while its export loads, or when it cannot be read:
 * the same frame on every tab, with the command that regenerates the file.
 */
export function ExportState({ title, what, error, command }: {
  /** panel title, usually the page or section name */
  title: string
  /** what is loading, for the sentence ("the meeting follow-up export") */
  what: string
  error?: string
  /** shell command that rebuilds the export */
  command?: string
}) {
  return <ChartFrame title={title} exportable={false}>
    <Empty>{error
      ? <>Could not load {what}: {error}.{command && <> Run <code>{command}</code> and reload.</>}</>
      : <>Loading {what}…</>}</Empty>
  </ChartFrame>
}
