import { LitElement, html, css } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import './video-canvas.js';
import { mainStyles, CamSetup } from './utils.js';

@customElement('camera-player')
export class CameraPlayer extends LitElement {
  @property({ type: String }) label = 'Front';
  @property({ type: String }) id = 'frontCam';
  basepath: string;

  @state()
  camSetup?: CamSetup;

  animation_handle: any;
  videoElement?: HTMLVideoElement;
  canvasElement?: HTMLCanvasElement;

  constructor() {
    super();
    this.basepath = window.location.protocol + '//' + window.location.host;
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
    this.videoElement = this.shadowRoot?.getElementById(
      'video',
    ) as HTMLVideoElement;

    this.getCameraMetadata();

    this.dispatchEvent(new CustomEvent('video-ready'));
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
        margin: 0 auto;
        padding: 23px 16px;
        box-sizing: border-box;
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
      <h3>Vehicle Counter</h3>
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
