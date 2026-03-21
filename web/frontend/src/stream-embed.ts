import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { initMediasoup, stopMediasoup } from './modules/webRTCPlayer.js';

@customElement('stream-embed')
export class StreamEmbed extends LitElement {
  @property({ type: String }) declare camStream: string;

  constructor() {
    super();
    this.camStream = '';
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    stopMediasoup();
  }

  protected async firstUpdated() {
    await this.updateComplete;
    const video = this.shadowRoot?.getElementById('video') as HTMLVideoElement;
    if (video && this.camStream) {
      initMediasoup({ [this.camStream]: video });
    }
  }

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
      background: #000;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
  `;

  render() {
    return html`<video id="video" autoplay muted playsinline></video>`;
  }
}
