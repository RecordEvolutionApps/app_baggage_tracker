import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import './stream-gallery.js';
import './stream-editor.js';
import './stream-embed.js';
import { stopMediasoup } from './modules/webRTCPlayer.js';
import { mainStyles } from './utils.js';

@customElement('camera-shell')
export class CameraShell extends LitElement {

  @state() declare private view: 'gallery' | 'editor' | 'embed';
  @state() declare private editCamStream: string;
  @state() declare private embedCamStream: string;

  private boundHashChange = this.onHashChange.bind(this);

  constructor() {
    super();
    this.view = 'gallery';
    this.editCamStream = '';
    this.embedCamStream = '';
  }

  connectedCallback() {
    super.connectedCallback();
    window.addEventListener('hashchange', this.boundHashChange);
    this.parseHash();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener('hashchange', this.boundHashChange);
  }

  private onHashChange() {
    this.parseHash();
  }

  private parseHash() {
    const hash = window.location.hash;
    if (hash.startsWith('#edit/')) {
      const camStream = decodeURIComponent(hash.slice('#edit/'.length));
      if (camStream) {
        stopMediasoup();
        this.editCamStream = camStream;
        this.view = 'editor';
        return;
      }
    }
    if (hash.startsWith('#view/')) {
      const camStream = decodeURIComponent(hash.slice('#view/'.length));
      if (camStream) {
        stopMediasoup();
        this.embedCamStream = camStream;
        this.view = 'embed';
        return;
      }
    }
    // Default: gallery
    stopMediasoup();
    this.view = 'gallery';
    this.editCamStream = '';
    this.embedCamStream = '';
  }

  private onEditStream(e: CustomEvent) {
    const camStream = e.detail?.camStream;
    if (camStream) {
      window.location.hash = `#edit/${encodeURIComponent(camStream)}`;
    }
  }

  private onCloseEditor() {
    window.location.hash = '';
  }

  static styles = [
    mainStyles,
    css`
      :host {
        height: 100%;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        background-color: #fbfcff;
      }
    `,
  ];

  render() {
    if (this.view === 'embed' && this.embedCamStream) {
      return html`
        <stream-embed
          .camStream=${this.embedCamStream}
        ></stream-embed>
      `;
    }

    if (this.view === 'editor' && this.editCamStream) {
      return html`
        <stream-editor
          .camStream=${this.editCamStream}
          @close-editor=${this.onCloseEditor}
        ></stream-editor>
      `;
    }

    return html`
      <stream-gallery
        @edit-stream=${this.onEditStream}
      ></stream-gallery>
    `;
  }
}
