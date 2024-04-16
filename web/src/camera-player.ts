import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';
import './camera-player.js'
import './camera-selector.js'

@customElement('camera-player')
export class CameraPlayer extends LitElement {
  
  @property({ type: String }) label = 'Front';
  @property({ type: String }) id = 'frontCam';
  basepath: string

  width: string
  height: string
  videoElement?: HTMLVideoElement

  constructor() {
    super()
    this.basepath = window.location.protocol + '//' + window.location.host
    this.width = "1920"
    this.height = "1080"
  }

  protected async firstUpdated() {
    this.videoElement = this.shadowRoot?.getElementById('video') as HTMLVideoElement
    
    try {
      const { width, height } = await fetch(`${this.basepath}/cameras/setup?cam=${this.id}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      }).then(res => res.json())

      this.width = width
      this.height = height
    } catch (error) {
      console.error("Failed to get cameras", error)
    } finally {
      this.requestUpdate()
    }

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
        <camera-selector .id=${this.id} label="Choose Camera"></camera-selector>
      </nav>
      <video id="video" autoplay controls muted playsinline width="${this.width}" height="${this.height}"></video>
    `;
  }
}
