import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { mainStyles, Camera } from './utils.js';
import { initMediasoup, stopMediasoup } from './modules/webRTCPlayer.js';

import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/dialog/dialog.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/icon/icon.js';
import '@material/web/iconbutton/icon-button.js';

@customElement('stream-gallery')
export class StreamGallery extends LitElement {
  @state() declare streams: Camera[];
  @state() declare showAddDialog: boolean;
  @state() declare newStreamName: string;
  @state() declare loadingStreams: Set<string>;

  constructor() {
    super();
    this.streams = [];
    this.showAddDialog = false;
    this.newStreamName = '';
    this.loadingStreams = new Set();
  }

  private basepath = window.location.protocol + '//' + window.location.host;

  connectedCallback() {
    super.connectedCallback();
    this.loadStreams();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    stopMediasoup();
  }

  async loadStreams() {
    try {
      const res = await fetch(`${this.basepath}/cameras/streams`);
      this.streams = await res.json();
    } catch (err) {
      console.error('Failed to load streams', err);
      this.streams = [];
    }

    // Mark all streams as loading
    this.loadingStreams = new Set(this.streams.map((s) => s.camStream));

    // Wait for render, then init mediasoup with all video elements
    await this.updateComplete;
    this.attachVideoListeners();
    this.initVideoPlayers();
  }

  private attachVideoListeners() {
    this.shadowRoot?.querySelectorAll<HTMLVideoElement>('video[data-cam]').forEach((v) => {
      const camId = v.dataset.cam;
      if (!camId) return;
      const onPlaying = () => {
        if (this.loadingStreams.has(camId)) {
          const next = new Set(this.loadingStreams);
          next.delete(camId);
          this.loadingStreams = next;
        }
        v.removeEventListener('playing', onPlaying);
      };
      v.addEventListener('playing', onPlaying);
    });
  }

  private initVideoPlayers() {
    const videoPlayers: Record<string, HTMLVideoElement | undefined> = {};
    this.shadowRoot?.querySelectorAll<HTMLVideoElement>('video[data-cam]').forEach((v) => {
      const camId = v.dataset.cam;
      if (camId) videoPlayers[camId] = v;
    });
    if (Object.keys(videoPlayers).length > 0) {
      initMediasoup(videoPlayers);
    }
  }

  private openAddDialog() {
    this.newStreamName = '';
    this.showAddDialog = true;
    this.updateComplete.then(() => {
      const dialog = this.shadowRoot?.querySelector('md-dialog') as any;
      dialog?.show();
    });
  }

  private async confirmAdd() {
    const name = this.newStreamName.trim();
    if (!name) return;

    // Sanitize camStream id: lowercase, replace spaces with underscores
    const camStream = name.replace(/[^a-zA-Z0-9_-]/g, '_').replace(/_{2,}/g, '_');

    try {
      const res = await fetch(`${this.basepath}/cameras/streams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, camStream }),
      });
      if (!res.ok) {
        const err = await res.json();
        console.error('Failed to create stream', err);
        return;
      }
    } catch (err) {
      console.error('Failed to create stream', err);
      return;
    }

    this.showAddDialog = false;
    // Navigate to editor for the new stream
    this.dispatchEvent(new CustomEvent('edit-stream', {
      detail: { camStream },
      bubbles: true,
      composed: true,
    }));
  }

  private cancelAdd() {
    this.showAddDialog = false;
  }

  private onTileClick(camStream: string) {
    this.dispatchEvent(new CustomEvent('edit-stream', {
      detail: { camStream },
      bubbles: true,
      composed: true,
    }));
  }

  static styles = [
    mainStyles,
    css`
      :host {
        display: block;
        width: 100%;
        padding: 24px;
        box-sizing: border-box;
      }

      .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
      }

      .header h2 {
        margin: 0;
        color: #002e6a;
        font-family: sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
      }

      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 20px;
      }

      .tile {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        cursor: pointer;
        background: #1a1a2e;
        aspect-ratio: 16 / 9;
        transition: box-shadow 0.2s, transform 0.15s;
      }

      .tile:hover {
        box-shadow: 0 4px 20px rgba(0, 46, 106, 0.25);
        transform: translateY(-2px);
      }

      .tile video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
      }

      .tile-loading {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 2;
        gap: 12px;
        background: #1a1a2e;
      }

      .tile-loading[hidden] {
        display: none;
      }

      .tile-loading span {
        font-family: sans-serif;
        font-size: 0.8rem;
        color: #8a8b9e;
      }

      .spinner {
        width: 28px;
        height: 28px;
        border: 3px solid #2a2a4e;
        border-top-color: #5e7ce0;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }

      .tile-label {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 8px 14px;
        background: linear-gradient(transparent, rgba(0,0,0,0.7));
        color: #fff;
        font-family: sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
      }

      .tile-placeholder {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
        color: #556;
        font-family: sans-serif;
        font-size: 0.85rem;
      }

      .add-tile {
        border-radius: 12px;
        border: 2px dashed #788894;
        aspect-ratio: 16 / 9;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: border-color 0.2s, background 0.2s;
        background: transparent;
        gap: 8px;
      }

      .add-tile:hover {
        border-color: #002e6a;
        background: rgba(0, 46, 106, 0.04);
      }

      .add-tile md-icon {
        font-size: 40px;
        color: #788894;
      }

      .add-tile:hover md-icon {
        color: #002e6a;
      }

      .add-tile span {
        color: #788894;
        font-family: sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
      }

      .add-tile:hover span {
        color: #002e6a;
      }

      md-dialog {
        --md-dialog-container-color: #fbfcff;
      }

      md-outlined-text-field {
        width: 100%;
        margin-top: 8px;
      }
    `,
  ];

  render() {
    return html`
      <div class="header">
        <h2>Video Streams</h2>
      </div>

      <div class="grid">
        ${repeat(
          this.streams,
          (s) => s.camStream,
          (s) => html`
            <div class="tile" @click=${() => this.onTileClick(s.camStream)}>
              <div class="tile-loading" ?hidden=${!this.loadingStreams.has(s.camStream)}>
                <div class="spinner"></div>
                <span>Connecting…</span>
              </div>
              <video data-cam=${s.camStream} autoplay muted playsinline></video>
              <div class="tile-label">${s.name || s.camStream}</div>
            </div>
          `,
        )}

        <div class="add-tile" @click=${this.openAddDialog}>
          <md-icon>add_circle</md-icon>
          <span>Add Stream</span>
        </div>
      </div>

      <md-dialog
        .open=${this.showAddDialog}
        @close=${this.cancelAdd}
      >
        <div slot="headline">New Video Stream</div>
        <form slot="content" id="add-form" method="dialog">
          <p style="margin:0 0 8px; color:#5e5f61; font-family:sans-serif; font-size:0.9rem;">
            Enter a name for the new video stream.
          </p>
          <md-outlined-text-field
            label="Stream Name"
            .value=${this.newStreamName}
            @input=${(e: any) => (this.newStreamName = e.target.value)}
            @keydown=${(e: KeyboardEvent) => { if (e.key === 'Enter') { e.preventDefault(); this.confirmAdd(); }}}
          ></md-outlined-text-field>
        </form>
        <div slot="actions">
          <md-text-button @click=${this.cancelAdd}>Cancel</md-text-button>
          <md-filled-button @click=${this.confirmAdd}>Create</md-filled-button>
        </div>
      </md-dialog>
    `;
  }
}
