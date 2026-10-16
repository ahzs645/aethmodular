/** The header's small line icons (16 px, currentColor). */
const Svg = ({ children }: { children: React.ReactNode }) => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>{children}</svg>
)
export const SearchIcon = () => <Svg><circle cx="7" cy="7" r="4.5" /><path d="m10.4 10.4 3.6 3.6" /></Svg>
export const BookIcon = () => <Svg><path d="M2.5 3.2C4.3 2.4 6.2 2.5 8 3.6c1.8-1.1 3.7-1.2 5.5-.4v9.6c-1.8-.8-3.7-.7-5.5.4-1.8-1.1-3.7-1.2-5.5-.4z" /><path d="M8 3.6v9.6" /></Svg>
export const ViewsIcon = () => <Svg><path d="M4.5 2.5h7v11l-3.5-2.6-3.5 2.6z" /></Svg>
export const SidebarIcon = () => <Svg><rect x="2" y="2.5" width="12" height="11" rx="1.5" /><path d="M10 2.5v11" /></Svg>
export const TextSizeIcon = () => <Svg><path d="M2 12.5 5 4l3 8.5M3.1 9.5h3.8" /><path d="M9.5 12.5 11.6 7l2.1 5.5M10.2 10.7h2.8" /></Svg>
export const ExplainIcon = () => <Svg><circle cx="8" cy="8" r="6" /><path d="M8 7.2v4" /><path d="M8 4.8v.01" /></Svg>
