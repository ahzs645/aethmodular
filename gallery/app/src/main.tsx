import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { isEmbed, restoreSharedView, watchSharedViews } from './lib/views'
import { useTheme } from './lib/colorTheme'
import './styles.css'

// an iframe embed shows the charts only (no header, no filter panel)
if (isEmbed()) document.documentElement.dataset.embed = '1'

/** the whole app redraws under a new theme: chart colours are read while rendering (lib/colorTheme) */
function Themed() {
  return <App key={useTheme()} />
}

watchSharedViews()
// a shared `#v=<token>` link is expanded into the plain URL the app reads before it renders
restoreSharedView().finally(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <Themed />
    </React.StrictMode>
  )
})
