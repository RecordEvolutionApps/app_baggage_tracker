import { Elysia } from "elysia";
import { staticPlugin } from '@elysiajs/static'
import { html } from '@elysiajs/html'
import { cors } from '@elysiajs/cors'
import { getUSBCameras, handleListStreams, handleGetStream, handleCreateStream, handleUpdateStream, handleDeleteStream, handleStopStream, handleStartStream, getStreamBackendStatus } from './cameras.js'
import { getModels, getModelTags, getModelClasses, getModelStatus, prepareModel, buildTrtModel, getCachedModels, deleteCachedModel, clearAllCache } from './models.js'
import { ironflock, getIronFlockConfig } from './ironflock.js'
import { stat } from "node:fs/promises";
import { join } from "node:path";
console.log('CURRENT', process.cwd())

// ── WebSocket proxy for mediasoup signaling ────────────────────────────────
// In production the browser can't reach port 1200 directly through the FRP
// tunnel, so we proxy the WebSocket through the web server on port 1100.
const videoApiUrl = new URL(Bun.env.VIDEO_API || 'http://mediasoup:8000');
const MEDIASOUP_WS = `ws://${videoApiUrl.hostname}:1200`;

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
// ── Model catalog (DEV mode only — in production the frontend calls the ────
// ── video service directly via WAMP RPCs)                                ────
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
// ── mediasoup WebSocket proxy (browser ↔ mediasoup signaling) ──────────────
// Use Bun's raw WebSocket server to avoid Elysia's auto JSON parsing.
const upstreamMap = new WeakMap<object, WebSocket>();
const bufferMap = new WeakMap<object, string[]>();

Bun.serve({
  port: 1100,
  websocket: {
    open(ws) {
      console.log(`[ws-proxy] Browser connected, opening upstream to ${MEDIASOUP_WS}`);
      const upstream = new WebSocket(MEDIASOUP_WS);
      upstreamMap.set(ws, upstream);
      const buffer: string[] = [];
      bufferMap.set(ws, buffer);

      upstream.addEventListener('open', () => {
        console.log(`[ws-proxy] Upstream to mediasoup opened (buffered ${buffer.length} msgs)`);
        for (const msg of buffer) upstream.send(msg);
        buffer.length = 0;
      });
      upstream.addEventListener('message', (ev: MessageEvent) => {
        const text = typeof ev.data === 'string' ? ev.data : JSON.stringify(ev.data);
        try { const m = JSON.parse(text); if (m.type !== 'keyframeRequested') console.log(`[ws-proxy] ← mediasoup:`, m.type || m.error || '(unknown)', m.camId || ''); } catch {}
        ws.sendText(text);
      });
      upstream.addEventListener('close', (ev) => {
        console.log(`[ws-proxy] Upstream closed (code=${(ev as CloseEvent).code}, reason=${(ev as CloseEvent).reason})`);
        upstreamMap.delete(ws);
        bufferMap.delete(ws);
        try { ws.close(); } catch {}
      });
      upstream.addEventListener('error', (ev) => {
        console.error(`[ws-proxy] Upstream error:`, ev);
        upstreamMap.delete(ws);
        bufferMap.delete(ws);
        try { ws.close(); } catch {}
      });
    },
    message(ws, message) {
      const str = typeof message === 'string' ? message : message.toString();
      try { const m = JSON.parse(str); if (m.type !== 'requestKeyFrame') console.log(`[ws-proxy] → mediasoup:`, m.type || '(unknown)', m.camId || ''); } catch {}
      const upstream = upstreamMap.get(ws);
      const buffer = bufferMap.get(ws);
      if (upstream && upstream.readyState === WebSocket.OPEN) {
        upstream.send(str);
      } else if (buffer) {
        console.log(`[ws-proxy] Upstream not ready, buffering message`);
        buffer.push(str);
      } else {
        console.warn(`[ws-proxy] No upstream and no buffer — dropping message`);
      }
    },
    close(ws, code, reason) {
      console.log(`[ws-proxy] Browser disconnected (code=${code}, reason=${reason})`);
      const upstream = upstreamMap.get(ws);
      if (upstream) {
        upstream.close();
        upstreamMap.delete(ws);
      }
      bufferMap.delete(ws);
    },
  },
  fetch(req, server) {
    const url = new URL(req.url);
    if (url.pathname === '/ws') {
      console.log(`[ws-proxy] WS upgrade request from ${req.headers.get('x-forwarded-for') || 'direct'}`);
      if (server.upgrade(req)) return undefined;
      console.error(`[ws-proxy] WS upgrade FAILED`);
      return new Response('WebSocket upgrade failed', { status: 400 });
    }
    return app.fetch(req);
  },
});

console.log(`🦊 Elysia is running at 0.0.0.0:1100`);
