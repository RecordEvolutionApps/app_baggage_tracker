import { Elysia } from "elysia";
import { staticPlugin } from '@elysiajs/static'
import { html } from '@elysiajs/html'
import { cors } from '@elysiajs/cors'
import { getUSBCameras, getStreamSetup, selectCamera, listStreams, createStream, deleteStream, getModels, getModelTags, getModelClasses, getModelStatus, prepareModel, updateStreamModel, updateStreamSahi, updateStreamSmoothing, updateStreamConfidence, updateStreamFrameBuffer, updateStreamNmsIou, updateStreamSahiIou, updateStreamOverlapRatio, updateStreamClassList, updateStreamClassNames, getStreamBackendStatus } from './cameras.js'
import { getMask, saveMask } from "./mask.js";
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
app.get('/mask', getMask)
app.post('/mask/save', saveMask)
app.get('/cameras', getUSBCameras)
app.get('/cameras/setup', getStreamSetup)
app.post('/cameras/select', selectCamera)
app.get('/cameras/streams', listStreams)
app.post('/cameras/streams', createStream)
app.delete('/cameras/streams/*', deleteStream)
app.get('/cameras/models', getModels)
app.get('/cameras/models/tags', getModelTags)
app.get('/cameras/models/:modelId/classes', getModelClasses)
app.get('/cameras/models/*/classes', getModelClasses)
app.get('/cameras/models/:modelId/status', getModelStatus)
app.get('/cameras/models/*/status', getModelStatus)
app.post('/cameras/models/prepare', prepareModel)
app.post('/cameras/model', updateStreamModel)
app.post('/cameras/sahi', updateStreamSahi)
app.post('/cameras/smoothing', updateStreamSmoothing)
app.post('/cameras/confidence', updateStreamConfidence)
app.post('/cameras/frameBuffer', updateStreamFrameBuffer)
app.post('/cameras/nmsIou', updateStreamNmsIou)
app.post('/cameras/sahiIou', updateStreamSahiIou)
app.post('/cameras/overlapRatio', updateStreamOverlapRatio)
app.post('/cameras/classList', updateStreamClassList)
app.post('/cameras/classNames', updateStreamClassNames)
app.get('/cameras/streams/:camStream/backend', getStreamBackendStatus)
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
