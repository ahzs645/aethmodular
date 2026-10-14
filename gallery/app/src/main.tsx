import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { isEmbed, restoreSharedView, watchSharedViews } from './lib/views'
import './styles.css'

// an iframe embed shows the charts only (no header, no filter panel)
if (isEmbed()) document.documentElement.dataset.embed = '1'

watchSharedViews()
// a shared `#v=<token>` link is expanded into the plain URL the app reads before it renders
restoreSharedView().finally(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  )
})
