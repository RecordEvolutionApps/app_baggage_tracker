import { LitElement, html, css } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import './camera-selector.js'
import './video-canvas.js'

@customElement('camera-player')
export class CameraPlayer extends LitElement {

  @property({ type: String }) label = 'Front';
  @property({ type: String }) id = 'frontCam';
  basepath: string

  @state()
  camera: any;

  animation_handle: any;
  width: number
  height: number
  videoElement?: HTMLVideoElement
  canvasElement?: HTMLCanvasElement;

  constructor() {
    super()
    this.basepath = window.location.protocol + '//' + window.location.host
    this.width = 1280
    this.height = 720
  }

  async getCameraMetadata() {
    try {
      const selected = await fetch(`${this.basepath}/cameras/setup?cam=${this.id}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      }).then(res => res.json())

      this.width = selected.width
      this.height = selected.height
      this.camera = selected

    } catch (error) {
      console.error("Failed to get cameras", error)
    } finally {
      this.requestUpdate()
    }
  }

  protected async firstUpdated() {
    this.videoElement = this.shadowRoot?.getElementById('video') as HTMLVideoElement

    this.getCameraMetadata()

    this.dispatchEvent(new CustomEvent('video-ready'))
  }

  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      align-items: center;
      flex: 1;
      position: relative;
    }

    nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      width: 100%;
      padding: 8px 16px;
      box-sizing: border-box;
      font-weight: 600;
      font-size: 24px;
    }
  `;

  render() {
    return html`
      <nav>
        <div>${this.label}</div>
        <camera-selector .camera=${this.camera} .id=${this.id} label="Choose Camera"></camera-selector>
      </nav>
      <video-canvas .video=${this.videoElement} .width=${this.width} .height=${this.height}></video-canvas>
      <video id="video" autoplay controls muted playsinline hidden></video>
    `;
  }
}
