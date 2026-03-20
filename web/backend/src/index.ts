import { Elysia } from "elysia";
import { staticPlugin } from '@elysiajs/static'
import { html } from '@elysiajs/html'
import { cors } from '@elysiajs/cors'
import { getUSBCameras, handleListStreams, handleGetStream, handleCreateStream, handleUpdateStream, handleDeleteStream, handleStopStream, handleStartStream, getModels, getModelTags, getModelClasses, getModelStatus, prepareModel, buildTrtModel, getCachedModels, deleteCachedModel, clearAllCache, getStreamBackendStatus } from './cameras.js'
import { ironflock, getIronFlockConfig } from './ironflock.js'
import { stat } from "node:fs/promises";
import { join } from "node:path";
console.log('CURRENT', process.cwd())
const app = new Elysia();
const frontendDist = Bun.env.FRONTEND_DIST || join(process.cwd(), "frontend", "dist");
let hasFrontendDist = false;
try {
  const stats = await stat(frontendDist);
  hasFrontendDist = stats.isDirectory();
} catch {
  hasFrontendDist = false;
}

if (hasFrontendDist) {
  app.use(staticPlugin({
    assets: frontendDist,
    prefix: "/"
  }))
}
app.use(html())
app.use(cors())
// ── IronFlock config for frontend SDK ──────────────────────────────────────
app.get('/api/ironflock-config', () => getIronFlockConfig())
// ── Stream CRUD ────────────────────────────────────────────────────────────
app.get('/streams', handleListStreams)
app.get('/streams/:camStream', handleGetStream)
app.post('/streams', handleCreateStream)
app.put('/streams/:camStream', handleUpdateStream)
app.delete('/streams/:camStream', handleDeleteStream)
app.post('/streams/:camStream/stop', handleStopStream)
app.post('/streams/:camStream/start', handleStartStream)
app.get('/streams/:camStream/backend', getStreamBackendStatus)
// ── Camera discovery ───────────────────────────────────────────────────────
app.get('/cameras', getUSBCameras)
// ── Model catalog ──────────────────────────────────────────────────────────
app.get('/cameras/models', getModels)
app.get('/cameras/models/tags', getModelTags)
app.get('/cameras/models/cache', getCachedModels)
app.delete('/cameras/models/cache', clearAllCache)
app.delete('/cameras/models/:modelId/cache', deleteCachedModel)
app.get('/cameras/models/:modelId/classes', getModelClasses)
app.get('/cameras/models/*/classes', getModelClasses)
app.get('/cameras/models/:modelId/status', getModelStatus)
app.get('/cameras/models/*/status', getModelStatus)
app.post('/cameras/models/prepare', prepareModel)
app.post('/cameras/models/build-trt', buildTrtModel)
app.get('/', async () => {
  if (hasFrontendDist) {
    return Bun.file(join(frontendDist, "index.html"))
  }

  return {
    ok: false,
    message: "Frontend dist not found. Use the Vite dev server at http://localhost:5173."
  }
})
app.listen(1100);

console.log(
  `🦊 Elysia is running at ${app.server?.hostname}:${app.server?.port}`
);
