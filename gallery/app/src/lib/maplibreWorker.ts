/**
 * Locate MapLibre 6's tile worker for Vite.
 *
 * MapLibre 6 finds its worker at run time with a URL Vite cannot see, so without this the worker 404s, no
 * source loads and a map never goes idle. `?url` makes Vite emit the worker file and hand back its address.
 * components/GeoMap imports this before anything else, so it runs in the lazy map chunk (the main bundle
 * stays free of MapLibre) and before the first map exists; geo-toolkit shares this one MapLibre module
 * (vite.config.ts dedupe), so the setting reaches its maps too. The same recipe as geo-toolkit's example
 * consumer (examples/toolkit-consumer/src/maplibreWorker.ts).
 */
import { prewarm, setWorkerUrl } from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url'

setWorkerUrl(workerUrl)
prewarm()
