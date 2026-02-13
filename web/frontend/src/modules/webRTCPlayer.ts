import { Device } from 'mediasoup-client';
import type { Transport, Consumer } from 'mediasoup-client/lib/types';

const SIGNALING_PORT = '1200';

interface VideoPlayers {
  [camId: string]: HTMLVideoElement | undefined;
}

let device: Device;
let ws: WebSocket;
let initializing = false;
let sessionId = 0;

const MAX_SUBSCRIBE_ATTEMPTS = 180; // ~3 minutes at 1s intervals
const RETRY_DELAY_MS = 1000;

// Per-camera state
const transports = new Map<string, Transport>();
const consumers  = new Map<string, Consumer>();
const keyframeTimers = new Map<string, number>();
const resubscribeTimers = new Map<string, number>();
const subscribeTokens = new Map<string, number>();
const noFramesTimers = new Map<string, number>();

function cleanup() {
  for (const consumer of consumers.values()) {
    try { consumer.close(); } catch {}
  }
  consumers.clear();
  for (const timerId of keyframeTimers.values()) {
    clearInterval(timerId);
  }
  keyframeTimers.clear();
  for (const timerId of resubscribeTimers.values()) {
    clearTimeout(timerId);
  }
  resubscribeTimers.clear();
  for (const timerId of noFramesTimers.values()) {
    clearTimeout(timerId);
  }
  noFramesTimers.clear();
  for (const transport of transports.values()) {
    try { transport.close(); } catch {}
  }
  transports.clear();
}

// ── Signaling helpers ──────────────────────────────────────────────────────

function getSignalingUrl(): string {
  // In production (ironflock tunnel): replace port segment in hostname
  // e.g. device-baggagetracker-1100.app.ironflock.com → device-baggagetracker-1200.app.ironflock.com
  const pa = location.host.split('-');
  if (pa.length >= 3) {
    const jns = pa[2]?.split('.') ?? [];
    jns[0] = SIGNALING_PORT;
    pa[2] = jns.join('.');
    const host = pa.join('-');
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${host}`;
  }
  // Local dev: connect to mediasoup on port 1200
  return `ws://${location.hostname}:${SIGNALING_PORT}`;
}

let msgCounter = 0;
const pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();

function sendRequest(msg: Record<string, any>): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      reject(new Error('WebSocket not open'));
      return;
    }
    const id = ++msgCounter;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ ...msg, id }));
  });
}

function onWsMessage(event: MessageEvent): void {
  const msg = JSON.parse(event.data);
  const cb = pending.get(msg.id);
  if (cb) {
    pending.delete(msg.id);
    if (msg.error) {
      cb.reject(new Error(msg.error));
    } else {
      cb.resolve(msg);
    }
  }
}

// ── Main initialization ────────────────────────────────────────────────────

async function initMediasoup(videoPlayers: VideoPlayers): Promise<void> {
  if (initializing) return;
  initializing = true;
  const currentSessionId = ++sessionId;


  // Clean up any previous session
  cleanup();

  try {
    const signalingUrl = getSignalingUrl();
    console.log('[mediasoup] connecting websocket', signalingUrl);
    ws = new WebSocket(signalingUrl);
    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = (e) => reject(e);
    });
    ws.onmessage = onWsMessage;

    // 1. Get router RTP capabilities
    const { data: routerRtpCapabilities } = await sendRequest({
      type: 'getRouterRtpCapabilities',
    });

    // 2. Create mediasoup-client Device and load capabilities
    device = new Device();
    await device.load({ routerRtpCapabilities });

    // 3. For each camera that has a video element, subscribe (with retry)
    for (const [camId, videoEl] of Object.entries(videoPlayers)) {
      if (!videoEl) continue;
      void subscribeUntilReady(camId, videoEl, currentSessionId);
    }

    // Handle reconnection
    ws.onclose = () => {
      console.warn('[mediasoup] WebSocket closed, will retry in 3s…');
      initializing = false;
      cleanup();
      setTimeout(() => initMediasoup(videoPlayers), 3000);
    };

  } catch (err) {
    console.error('[mediasoup] Init failed:', err);
    initializing = false;
    cleanup();
    setTimeout(() => initMediasoup(videoPlayers), 3000);
  }
}

async function subscribeUntilReady(
  camId: string,
  videoEl: HTMLVideoElement,
  forSessionId: number,
): Promise<void> {
  const token = (subscribeTokens.get(camId) ?? 0) + 1;
  subscribeTokens.set(camId, token);
  for (let attempt = 1; attempt <= MAX_SUBSCRIBE_ATTEMPTS; attempt++) {
    if (forSessionId !== sessionId) return;
    if (subscribeTokens.get(camId) !== token) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      await subscribeToCamera(camId, videoEl, forSessionId, token);
      return; // success
    } catch (err: any) {
      const isNotFound = err?.message?.includes('not found');
      const isWsClosed = err?.message?.includes('WebSocket not open');
      if ((isNotFound || isWsClosed) && attempt < MAX_SUBSCRIBE_ATTEMPTS) {
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
      } else {
        console.warn('[mediasoup] subscribe failed for', camId, err);
        return;
      }
    }
  }
}

async function subscribeToCamera(
  camId: string,
  videoEl: HTMLVideoElement,
  forSessionId: number,
  token: number,
): Promise<void> {
  if (subscribeTokens.get(camId) !== token) return;
  // 1. Create a consumer transport on the server
  const transportReply = await sendRequest({
    type: 'createConsumerTransport',
    camId,
  });
  const transportData = transportReply.data;
  if (subscribeTokens.get(camId) !== token) return;

  // 2. Create the local RecvTransport
  const transport = device.createRecvTransport({
    id:             transportData.id,
    iceParameters:  transportData.iceParameters,
    iceCandidates:  transportData.iceCandidates,
    dtlsParameters: transportData.dtlsParameters,
  });
  transports.set(camId, transport);

  // 3. Handle 'connect' event (DTLS)
  transport.on('connect', async ({ dtlsParameters }, callback, errback) => {
    try {
      await sendRequest({
        type: 'connectConsumerTransport',
        camId,
        dtlsParameters,
      });
      callback();
    } catch (err: any) {
      errback(err);
    }
  });

  transport.on('connectionstatechange', (state: string) => {
    if (state === 'failed' || state === 'disconnected') {
      console.warn(`[mediasoup] Transport ${camId} ${state}`);
      scheduleResubscribe(camId, videoEl, forSessionId);
    }
  });

  // 4. Request to consume the camera's producer
  const consumeReply = await sendRequest({
    type: 'consume',
    camId,
    rtpCapabilities: device.rtpCapabilities,
  });
  const consumeData = consumeReply.data;
  if (subscribeTokens.get(camId) !== token) {
    try { transport.close(); } catch {}
    transports.delete(camId);
    return;
  }

  // 5. Create the Consumer on the local transport
  const consumer = await transport.consume({
    id:            consumeData.id,
    producerId:    consumeData.producerId,
    kind:          consumeData.kind,
    rtpParameters: consumeData.rtpParameters,
  });
  consumers.set(camId, consumer);
  if (subscribeTokens.get(camId) !== token) {
    try { consumer.close(); } catch {}
    consumers.delete(camId);
    try { transport.close(); } catch {}
    transports.delete(camId);
    return;
  }

  consumer.on('trackended', () => {
    scheduleResubscribe(camId, videoEl, forSessionId);
  });

  consumer.track.onunmute = () => {
    void sendRequest({ type: 'requestKeyFrame', camId }).catch(() => {});
  };

  consumer.on('transportclose', () => {
    scheduleResubscribe(camId, videoEl, forSessionId);
  });

  // 6. Attach to the video element
  const stream = new MediaStream([consumer.track]);
  videoEl.srcObject = stream;

  videoEl.onloadedmetadata = () => {
    const noFramesId = noFramesTimers.get(camId);
    if (noFramesId) {
      clearTimeout(noFramesId);
      noFramesTimers.delete(camId);
    }
    videoEl.play().catch(() => {});
  };

  // 7. Tell server we're ready to receive
  await sendRequest({ type: 'resumeConsumer', camId });

  videoEl.play().catch(() => {});

  // Request keyframes until the video element has dimensions or we resubscribe.
  if (keyframeTimers.has(camId)) {
    clearInterval(keyframeTimers.get(camId)!);
  }
  const timerId = window.setInterval(() => {
    if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
      clearInterval(timerId);
      keyframeTimers.delete(camId);
      const noFramesId = noFramesTimers.get(camId);
      if (noFramesId) {
        clearTimeout(noFramesId);
        noFramesTimers.delete(camId);
      }
      return;
    }
    void sendRequest({ type: 'requestKeyFrame', camId }).catch(() => {});
  }, 1000);
  keyframeTimers.set(camId, timerId);

  // If no frames arrive shortly after subscribe, resubscribe.
  const noFramesId = window.setTimeout(() => {
    if (forSessionId !== sessionId) return;
    if (videoEl.videoWidth > 0 || videoEl.videoHeight > 0) return;

    scheduleResubscribe(camId, videoEl, forSessionId);
  }, 15000);
  noFramesTimers.set(camId, noFramesId);

  // resumeConsumer already sent above
}

function scheduleResubscribe(
  camId: string,
  videoEl: HTMLVideoElement,
  forSessionId: number,
): void {
  if (forSessionId !== sessionId) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (resubscribeTimers.has(camId)) {
    return;
  }
  cleanupCamera(camId);
  const timerId = window.setTimeout(() => {
    resubscribeTimers.delete(camId);
    void subscribeUntilReady(camId, videoEl, forSessionId);
  }, RETRY_DELAY_MS);
  resubscribeTimers.set(camId, timerId);
}

function cleanupCamera(camId: string): void {
  const consumer = consumers.get(camId);
  if (consumer) {
    try { consumer.close(); } catch {}
    consumers.delete(camId);
  }
  const timerId = keyframeTimers.get(camId);
  if (timerId) {
    clearInterval(timerId);
    keyframeTimers.delete(camId);
  }
  const resubscribeId = resubscribeTimers.get(camId);
  if (resubscribeId) {
    clearTimeout(resubscribeId);
    resubscribeTimers.delete(camId);
  }
  const noFramesId = noFramesTimers.get(camId);
  if (noFramesId) {
    clearTimeout(noFramesId);
    noFramesTimers.delete(camId);
  }
  const transport = transports.get(camId);
  if (transport) {
    try { transport.close(); } catch {}
    transports.delete(camId);
  }
}

export { initMediasoup }
