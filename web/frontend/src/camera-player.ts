import { LitElement, html, css } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import './video-canvas.js';
import { mainStyles, CamSetup } from './utils.js';

@customElement('camera-player')
export class CameraPlayer extends LitElement {
  @property({ type: String }) declare label: string;
  @property({ type: String }) declare id: string;
  basepath: string;

  @state()
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
      this.camSetup = await fetch(
        `${this.basepath}/cameras/setup?camStream=${this.id}`,
        {
          method: 'GET',
          headers: {
            Accept: 'application/json',
          },
        },
      ).then(res => res.json());
      console.log('got Camera Setup', this.camSetup)
    } catch (error) {
      console.error('Failed to get cameras', error);
    } finally {
      this.requestUpdate();
    }
  }

protected async firstUpdated() {
  const videoElement = this.shadowRoot?.getElementById('video') as HTMLVideoElement;
  // this.videoElement = videoElement; // Still set the property internally

  this.getCameraMetadata();

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
        .width=${this.camSetup?.width ?? 1280}
        .height=${this.camSetup?.height ?? 720}
        .camStream=${this.id}
      ></video-canvas>
      <video id="video" autoplay controls muted playsinline hidden></video>
    `;
  }
}
