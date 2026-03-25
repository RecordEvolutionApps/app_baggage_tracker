import { LitElement, html, css } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import './video-canvas.js';
import { mainStyles, CamSetup } from './utils.js';
import { readStream, subscribeStreams } from './streams-sdk.js';

@customElement('camera-player')
export class CameraPlayer extends LitElement {
  @property({ type: String }) declare label: string;
  @property({ type: String }) declare id: string;
  @property({ type: Boolean }) declare stopped: boolean;
  basepath: string;

  // @property so that stream-editor (or any parent) can push a freshly-received
  // full StreamConfig directly into this component without waiting for the
  // WAMP subscription to fire again.
  @property({ type: Object })
  declare camSetup?: CamSetup;

  animation_handle: any;
  canvasElement?: HTMLCanvasElement;

  constructor() {
    super();
    this.basepath = window.location.protocol + '//' + window.location.host;
    this.label = 'Front';
    this.id = 'frontCam';
  }

  get videoElement() {
     return this.shadowRoot?.getElementById('video') as HTMLVideoElement;
  }

  async getCameraMetadata() {
    try {
      const config = await readStream(this.id);
      if (config) {
        this.camSetup = {
          ...config,
          source: {
            ...config.source,
            width: config.source?.width ?? 640,
            height: config.source?.height ?? 480,
          },
        } as CamSetup;
      }
      console.log('got Camera Setup', this.camSetup)
    } catch (error) {
      console.error('Failed to get cameras', error);
    } finally {
      this.requestUpdate();
    }
  }

protected async firstUpdated() {
  const videoElement = this.shadowRoot?.getElementById('video') as HTMLVideoElement;

  this.getCameraMetadata();

  // Subscribe to WAMP stream-config updates for this specific stream.
  // The Python publisher calls publish_stream() which writes a new row to the
  // IronFlock 'streams' table; that triggers this callback on all subscribers.
  // This ensures camSetup (source type, path, and — critically — actual
  // capture resolution written by setVideoSource) stays in sync without a
  // browser refresh, even when the publisher row arrives after firstUpdated.
  subscribeStreams((config) => {
    if (config.camStream !== this.id) return;
    if ((config as any).deleted) return;
    // Accept every non-deleted row for this stream — source, inference, and
    // processing sections all need to flow downstream through Lit bindings:
    //   camera-player → video-canvas → canvas-toolbox → camera-dialog
    //                                                  → inference-setup
    // Only skip a row that carries no meaningful data at all (e.g. the bare
    // shell written when a new stream is first created with no source yet).
    const hasContent = !!(config.source?.path || config.source?.type ||
                          config.inference?.model || config.name);
    if (!hasContent) return;
    this.camSetup = {
      ...config,
      source: {
        ...config.source,
        // Keep the last known measured resolution when the incoming row omits it,
        // so the canvas never reverts to 640×480 mid-stream.
        width: config.source?.width ?? this.camSetup?.source?.width ?? 640,
        height: config.source?.height ?? this.camSetup?.source?.height ?? 480,
      },
    } as CamSetup;
  });

  // Force a re-render so video-canvas gets the real <video> element
  // (on first render, shadow DOM was empty so this.videoElement was null)
  this.requestUpdate();

  if (videoElement) {
    this.dispatchEvent(new CustomEvent('video-ready', {
      detail: { videoElement: videoElement }
    }));
  }
}

  static styles = [
    mainStyles,
    css`
      :host {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
        box-sizing: border-box;
        background: #fff;
      }

      h3 {
        display: none;
        align-self: baseline;
        margin: 0 0 5px 0;
      }

      @media only screen and (max-width: 600px) {
        h3 {
          display: block;
        }
        :host {
          padding: 5px 16px 0 16px;
        }
      }
    `,
  ];

  render() {
    return html`
      <video-canvas
        .video=${this.videoElement}
        .camSetup=${this.camSetup}
        .width=${this.camSetup?.source?.width ?? 1280}
        .height=${this.camSetup?.source?.height ?? 720}
        .camStream=${this.id}
        .stopped=${this.stopped}
      ></video-canvas>
      <video id="video" autoplay controls muted playsinline hidden></video>
    `;
  }
}
