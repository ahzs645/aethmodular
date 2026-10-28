import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { isEmbed, restoreSharedView, watchSharedViews } from './lib/views'
import { installUndo } from './lib/undo'
import './lib/colorTheme'
import './lib/themeCss'
import './styles.css'

// an iframe embed shows the charts only (no header, no filter panel)
if (isEmbed()) document.documentElement.dataset.embed = '1'

watchSharedViews()
// a shared `#v=<token>` link is expanded into the plain URL the app reads before it renders
restoreSharedView().finally(() => {
  // ⌘Z / Ctrl+Z over every setting in the URL, from the view the link opened on
  installUndo()
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  )
})
