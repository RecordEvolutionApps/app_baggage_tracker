import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';
import './camera-player.js'
import './camera-selector.js'

@customElement('camera-player')
export class CameraPlayer extends LitElement {
  
  @property({ type: String }) label = 'Front';
  @property({ type: String }) id = 'frontCam';

  videoElement?: HTMLVideoElement

  protected firstUpdated(): void {
      this.videoElement = this.shadowRoot?.getElementById('video') as HTMLVideoElement
  }

  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      align-items: center;
      flex: 1;
      position: relative;
    }

    video {
        width: 100%;
        height: auto;
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
      <video id="video" autoplay controls muted playsinline width="1920" height="1080"></video>
    `;
  }
}
