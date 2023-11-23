import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';

@customElement('camera-player')
export class CameraPlayer extends LitElement {
  @property({ type: String }) header = 'My app';

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
      position: relative;
    }

    video {
        /* Set video element to fill the container */
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0;
        left: 0;
    }
    
  `;

  render() {
    return html`
      <video id="remoteVideo" autoplay controls muted playsinline width="1920" height="1080"></video>
    `;
  }
}
