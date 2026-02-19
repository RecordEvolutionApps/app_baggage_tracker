import { createWorker } from 'mediasoup';
import { WebSocketServer } from 'ws';
import http from 'node:http';
import os from 'node:os';

// ── Configuration ──────────────────────────────────────────────────────────
const WS_PORT       = parseInt(process.env.WS_PORT || '1200', 10);
const LISTEN_IP     = process.env.LISTEN_IP || '0.0.0.0';
const RTP_PORT_MIN  = parseInt(process.env.RTP_PORT_MIN || '2000', 10);
const RTP_PORT_MAX  = parseInt(process.env.RTP_PORT_MAX || '2999', 10);

// Auto-detect host IP if ANNOUNCED_IP is not explicitly set.
// This finds the first non-internal IPv4 address (e.g. 192.168.x.x).
function getHostIp() {
  for (const ifaces of Object.values(os.networkInterfaces())) {
    for (const iface of ifaces) {
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return '127.0.0.1';
}

const ANNOUNCED_IP = process.env.ANNOUNCED_IP || getHostIp();
const ANNOUNCED_IP_FALLBACK = process.env.ANNOUNCED_IP_FALLBACK || null;
console.log(`[mediasoup] ANNOUNCED_IP=${ANNOUNCED_IP}`);
if (ANNOUNCED_IP_FALLBACK) {
  console.log(`[mediasoup] ANNOUNCED_IP_FALLBACK=${ANNOUNCED_IP_FALLBACK}`);
}

// Build listenInfos array: primary IP first, optional fallback second.
// Each entry produces an ICE candidate; the browser tries all and uses
// whichever is reachable (e.g. reverse-proxy URL in prod, LAN IP as fallback).
function buildListenInfos() {
  const infos = [
    { protocol: 'udp', ip: LISTEN_IP, announcedAddress: ANNOUNCED_IP },
    { protocol: 'tcp', ip: LISTEN_IP, announcedAddress: ANNOUNCED_IP },
  ];
  if (ANNOUNCED_IP_FALLBACK) {
    infos.push(
      { protocol: 'udp', ip: LISTEN_IP, announcedAddress: ANNOUNCED_IP_FALLBACK },
      { protocol: 'tcp', ip: LISTEN_IP, announcedAddress: ANNOUNCED_IP_FALLBACK },
    );
  }
  return infos;
}

// No pre-defined camera list — ingests are created on demand via POST /ingest

// ── State ──────────────────────────────────────────────────────────────────
/** @type {import('mediasoup').types.Worker} */
let worker;
/** @type {import('mediasoup').types.Router} */
let router;

// Per-camera: { producer, plainTransport }
const cameras = new Map();

// Track all connected WebSocket clients so we can broadcast notifications
/** @type {Set<import('ws').WebSocket>} */
const wsClients = new Set();

// ── mediasoup setup ────────────────────────────────────────────────────────
async function startMediasoup() {
  worker = await createWorker({
    logLevel: 'warn',
    rtcMinPort: RTP_PORT_MIN,
    rtcMaxPort: RTP_PORT_MAX,
  });

  worker.on('died', () => {
    console.error('mediasoup Worker died, exiting…');
    process.exit(1);
  });

  const mediaCodecs = [
    {
      kind:      'video',
      mimeType:  'video/H264',
      clockRate: 90000,
      parameters: {
        'packetization-mode':      1,
        'profile-level-id':        '42e01f',
        'level-asymmetry-allowed': 1,
      },
      rtcpFeedback: [
        { type: 'nack' },
        { type: 'nack', parameter: 'pli' },
        { type: 'ccm', parameter: 'fir' },
      ],
    },
  ];

  router = await worker.createRouter({ mediaCodecs });
  console.log(`[mediasoup] Router created (id=${router.id})`);
}

async function createCameraIngest(camId, rtpPort) {
  const plainTransport = await router.createPlainTransport({
    // Listen on localhost since video container shares network namespace
    listenInfo: {
      protocol: 'udp',
      ip: '127.0.0.1',
      ...(rtpPort !== 0 && { port: rtpPort })
    },
    rtcpMux:  true,
    comedia:  true,  // auto-detect sender from first incoming RTP packet
  });

  const assignedPort = plainTransport.tuple.localPort;

  const producer = await plainTransport.produce({
    kind: 'video',
    rtpParameters: {
      codecs: [
        {
          mimeType:    'video/H264',
          clockRate:   90000,
          payloadType: 96,
          parameters: {
            'packetization-mode': 1,
            'profile-level-id':  '42e01f',
          },
          rtcpFeedback: [
            { type: 'nack' },
            { type: 'nack', parameter: 'pli' },
            { type: 'ccm', parameter: 'fir' },
          ],
        },
      ],
      encodings: [{ ssrc: 11111111 }],  // Must specify SSRC for PlainTransport produce
    },
  });

  // Log PlainTransport events only at debug level
  plainTransport.on('tuple', (tuple) => {
    console.debug(`[mediasoup] PlainTransport ${camId} tuple: ${JSON.stringify(tuple)}`);
  });

  cameras.set(camId, { plainTransport, producer });
  console.log(`[mediasoup] Ingest ready: ${camId} ← UDP port ${assignedPort} (producer ${producer.id})`);
}

// ── WebSocket signaling for browsers ───────────────────────────────────────
function startSignaling() {
  const httpServer = http.createServer(async (_req, res) => {
    const url = new URL(_req.url, `http://localhost:${WS_PORT}`);

    // ── Health / list active ingests ──────────────────────────────────────
    if (_req.method === 'GET' && url.pathname === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      const camData = {};
      for (const [id, cam] of cameras.entries()) {
        camData[id] = cam.plainTransport.tuple.localPort;
      }
      res.end(JSON.stringify({ ok: true, cameras: [...cameras.keys()], ports: camData }));
      return;
    }

    // ── Create ingest on demand ───────────────────────────────────────────
    if (_req.method === 'POST' && url.pathname === '/ingest') {
      let body = '';
      _req.on('data', (chunk) => { body += chunk; });
      _req.on('end', async () => {
        try {
          const { streamId } = JSON.parse(body);
          if (!streamId) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'streamId is required' }));
            return;
          }
          if (cameras.has(streamId)) {
            // Already exists — return existing port
            const port = cameras.get(streamId).plainTransport.tuple.localPort;
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ streamId, port }));
            return;
          }
          await createCameraIngest(streamId, 0);
          const port = cameras.get(streamId).plainTransport.tuple.localPort;
          res.writeHead(201, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ streamId, port }));
        } catch (err) {
          console.error('[mediasoup] POST /ingest error:', err);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: err.message }));
        }
      });
      return;
    }

    // ── Delete ingest ─────────────────────────────────────────────────────
    const deleteMatch = _req.method === 'DELETE' && url.pathname.startsWith('/ingest/');
    if (deleteMatch) {
      const streamId = decodeURIComponent(url.pathname.slice('/ingest/'.length));
      const cam = cameras.get(streamId);
      if (!cam) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: `No ingest for ${streamId}` }));
        return;
      }
      try {
        cam.producer.close();
        cam.plainTransport.close();
        cameras.delete(streamId);
        console.log(`[mediasoup] Ingest removed: ${streamId}`);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'removed', streamId }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    res.writeHead(404);
    res.end();
  });

  const wss = new WebSocketServer({ server: httpServer });

  wss.on('connection', (ws) => {
    console.log('[signaling] New browser connection');
    wsClients.add(ws);

    // Per-connection state
    const consumerTransports = new Map(); // camId → WebRtcTransport
    const consumers          = new Map(); // camId → Consumer

    ws.on('message', async (raw) => {
      let msg;
      try { msg = JSON.parse(raw); } catch { return; }

      try {
        const reply = await handleMessage(msg, ws, consumerTransports, consumers);
        if (reply) {
          ws.send(JSON.stringify({ id: msg.id, ...reply }));
        }
      } catch (err) {
        console.error(`[signaling] Error handling "${msg.type}":`, err);
        ws.send(JSON.stringify({ id: msg.id, error: err.message }));
      }
    });

    ws.on('close', () => {
      console.log('[signaling] Browser disconnected, cleaning up');
      wsClients.delete(ws);
      for (const consumer of consumers.values()) consumer.close();
      for (const transport of consumerTransports.values()) transport.close();
    });
  });

  httpServer.listen(WS_PORT, () => {
    console.log(`[signaling] WebSocket server listening on port ${WS_PORT}`);
  });
}

async function handleMessage(msg, ws, consumerTransports, consumers) {
  switch (msg.type) {

    // ── Step 1: Browser requests router RTP capabilities ───────────────
    case 'getRouterRtpCapabilities': {
      return { type: 'routerRtpCapabilities', data: router.rtpCapabilities };
    }

    // ── Step 2: Browser requests a WebRTC transport for consuming ──────
    case 'createConsumerTransport': {
      const camId = msg.camId || 'frontCam';

      // Don't create a transport if the camera hasn't registered yet
      if (!cameras.has(camId)) {
        throw new Error(`Camera "${camId}" not found`);
      }

      // Close any existing transport for this camera to prevent leaks
      const existing = consumerTransports.get(camId);
      if (existing) {
        existing.close();
        consumerTransports.delete(camId);
      }

      const transport = await router.createWebRtcTransport({
        listenInfos: buildListenInfos(),
        enableUdp: true,
        enableTcp: true,
        preferUdp: true,
      });

      transport.on('icestatechange', (iceState) => {
        console.log(`[mediasoup] WebRTC transport ${camId} ICE state: ${iceState}`);
      });
      transport.on('dtlsstatechange', (dtlsState) => {
        console.log(`[mediasoup] WebRTC transport ${camId} DTLS state: ${dtlsState}`);
      });

      consumerTransports.set(camId, transport);

      console.log(`[mediasoup] WebRTC transport ${camId} created`);

      return {
        type: 'consumerTransportCreated',
        data: {
          id:             transport.id,
          iceParameters:  transport.iceParameters,
          iceCandidates:  transport.iceCandidates,
          dtlsParameters: transport.dtlsParameters,
        },
        camId,
      };
    }

    // ── Step 3: Browser connects the transport (DTLS handshake) ────────
    case 'connectConsumerTransport': {
      const camId = msg.camId || 'frontCam';
      console.log(`[mediasoup] connectConsumerTransport called for ${camId}`);
      const transport = consumerTransports.get(camId);
      if (!transport) throw new Error(`No transport for ${camId}`);
      await transport.connect({ dtlsParameters: msg.dtlsParameters });
      return { type: 'consumerTransportConnected', camId };
    }

    // ── Step 4: Browser requests to consume a camera's video ───────────
    case 'consume': {
      const camId = msg.camId || 'frontCam';
      const camera = cameras.get(camId);
      if (!camera) throw new Error(`Camera "${camId}" not found`);

      const transport = consumerTransports.get(camId);
      if (!transport) throw new Error(`No transport for ${camId}`);

      if (!router.canConsume({ producerId: camera.producer.id, rtpCapabilities: msg.rtpCapabilities })) {
        throw new Error(`Cannot consume ${camId}`);
      }

      const consumer = await transport.consume({
        producerId:      camera.producer.id,
        rtpCapabilities: msg.rtpCapabilities,
        paused:          false,
      });

      consumer.on('producerpause', () => {
        console.log(`[mediasoup] Consumer ${camId}: producer paused`);
      });
      consumer.on('producerresume', () => {
        console.log(`[mediasoup] Consumer ${camId}: producer resumed`);
      });
      consumer.on('producerclose', () => {
        console.log(`[mediasoup] Consumer ${camId}: producer closed — notifying client`);
        consumers.delete(camId);
        try {
          ws.send(JSON.stringify({ type: 'producerClosed', camId }));
        } catch {}
      });

      consumers.set(camId, consumer);

      console.log(`[mediasoup] Consumer created for ${camId}`);

      return {
        type: 'consumed',
        data: {
          id:            consumer.id,
          producerId:    camera.producer.id,
          kind:          consumer.kind,
          rtpParameters: consumer.rtpParameters,
        },
        camId,
      };
    }

    // ── Step 5: Browser resumes consumer ───────────────────────────────
    case 'resumeConsumer': {
      const camId = msg.camId || 'frontCam';
      const consumer = consumers.get(camId);
      if (consumer) await consumer.resume();
      return { type: 'consumerResumed', camId };
    }

    // ── Request a keyframe from the producer ──────────────────────────
    case 'requestKeyFrame': {
      const camId = msg.camId || 'frontCam';
      const consumer = consumers.get(camId);
      if (!consumer) throw new Error(`No consumer for ${camId}`);
      const camera = cameras.get(camId);

      if (typeof consumer.requestKeyFrame !== 'function') {
        throw new Error(`requestKeyFrame not supported for ${camId}`);
      }
      await consumer.requestKeyFrame();
      return { type: 'keyframeRequested', camId };
    }

    // ── List available cameras ─────────────────────────────────────────
    case 'getAvailableCameras': {
      return {
        type: 'availableCameras',
        data: [...cameras.keys()],
      };
    }

    default:
      console.warn(`[signaling] Unknown message type: ${msg.type}`);
      return null;
  }
}

// ── Boot ───────────────────────────────────────────────────────────────────
await startMediasoup();
startSignaling();

// Graceful shutdown on SIGTERM/SIGINT (Docker stop)
for (const sig of ['SIGTERM', 'SIGINT']) {
  process.on(sig, () => {
    console.log(`[mediasoup] Received ${sig}, shutting down…`);
    try { worker?.close(); } catch {}
    // Force exit after 500ms if worker.close() hangs
    setTimeout(() => process.exit(0), 500).unref();
    process.exit(0);
  });
}
