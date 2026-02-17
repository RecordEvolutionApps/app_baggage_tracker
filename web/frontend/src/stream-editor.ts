import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { mainStyles } from './utils.js';
import './camera-player.js';
import { CameraPlayer } from './camera-player.js';
import { initMediasoup, stopMediasoup } from './modules/webRTCPlayer.js';

import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/dialog/dialog.js';
import '@material/web/icon/icon.js';
import '@material/web/iconbutton/icon-button.js';

@customElement('stream-editor')
export class StreamEditor extends LitElement {
  @property({ type: String }) declare camStream: string;

  @state() declare showDeleteDialog: boolean;
  @state() declare stopped: boolean;
  @state() declare toggling: boolean;

  constructor() {
    super();
    this.showDeleteDialog = false;
    this.stopped = false;
    this.toggling = false;
  }

  private basepath = window.location.protocol + '//' + window.location.host;

  connectedCallback() {
    super.connectedCallback();
    this.loadStoppedState();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    stopMediasoup();
  }

  private async loadStoppedState() {
    try {
      const res = await fetch(`${this.basepath}/cameras/setup?camStream=${this.camStream}`);
      const data = await res.json();
      this.stopped = !!data.camera?.stopped;
    } catch (err) {
      console.error('Failed to load stream state', err);
    }
  }

  private onVideoReady(_event: CustomEvent) {
    if (this.stopped) return;          // don't init WebRTC if stopped
    const videoPlayers: Record<string, HTMLVideoElement | undefined> = {};
    const player = this.shadowRoot?.querySelector('camera-player') as CameraPlayer;
    if (player?.id && player.videoElement) {
      videoPlayers[player.id] = player.videoElement;
    }
    initMediasoup(videoPlayers);
  }

  private async toggleStream() {
    if (this.toggling) return;
    this.toggling = true;
    try {
      const action = this.stopped ? 'start' : 'stop';
      const res = await fetch(
        `${this.basepath}/cameras/streams/${encodeURIComponent(this.camStream)}/${action}`,
        { method: 'POST' },
      );
      if (!res.ok) {
        console.error(`Failed to ${action} stream`, await res.text());
        return;
      }
      this.stopped = !this.stopped;

      if (this.stopped) {
        // Tear down WebRTC player
        stopMediasoup();
      } else {
        // Re-init WebRTC after the video process comes back
        await this.updateComplete;
        // Small delay to let the video process start and begin producing RTP
        setTimeout(() => {
          const videoPlayers: Record<string, HTMLVideoElement | undefined> = {};
          const player = this.shadowRoot?.querySelector('camera-player') as CameraPlayer;
          if (player?.id && player.videoElement) {
            videoPlayers[player.id] = player.videoElement;
          }
          initMediasoup(videoPlayers);
        }, 500);
      }
    } catch (err) {
      console.error('Error toggling stream', err);
    } finally {
      this.toggling = false;
    }
  }

  private goBack() {
    this.dispatchEvent(new CustomEvent('close-editor', {
      bubbles: true,
      composed: true,
    }));
  }

  private openDeleteDialog() {
    this.showDeleteDialog = true;
  }

  private cancelDelete() {
    this.showDeleteDialog = false;
  }

  private async confirmDelete() {
    try {
      await fetch(`${this.basepath}/cameras/streams/${encodeURIComponent(this.camStream)}`, {
        method: 'DELETE',
      });
    } catch (err) {
      console.error('Failed to delete stream', err);
    }
    this.showDeleteDialog = false;
    this.goBack();
  }

  static styles = [
    mainStyles,
    css`
      :host {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
      }

      .top-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 20px;
        background: #e9eaf2;
        border-bottom: 1px solid #d0d2db;
        flex-shrink: 0;
      }

      .top-bar h3 {
        margin: 0;
        flex: 1;
        color: #002e6a;
        font-family: sans-serif;
        font-weight: 600;
        font-size: 1.15rem;
      }

      .editor-body {
        flex: 1;
        overflow: auto;
        display: flex;
        flex-direction: column;
      }

      .editor-body camera-player {
        flex: 1;
        height: 100%;
        overflow: hidden;
      }

      md-dialog {
        --md-dialog-container-color: #fbfcff;
      }

      .top-bar md-filled-button,
      .top-bar md-outlined-button,
      .top-bar md-text-button {
        --md-filled-button-label-text-font: sans-serif;
        --md-outlined-button-label-text-font: sans-serif;
        --md-text-button-label-text-font: sans-serif;
      }

      md-filled-button {
        --md-filled-button-container-color: #002e6a;
      }

      .delete-btn {
        --md-outlined-button-outline-color: #b3261e;
        --md-outlined-button-label-text-color: #b3261e;
        --md-outlined-button-hover-label-text-color: #fff;
        --md-outlined-button-hover-state-layer-color: #b3261e;
        --md-outlined-button-pressed-label-text-color: #fff;
        --md-outlined-button-pressed-state-layer-color: #b3261e;
      }

      .delete-btn md-icon {
        color: #b3261e;
      }

      .back-btn {
        --md-icon-button-icon-color: #002e6a;
      }

      .stop-btn {
        --md-outlined-button-outline-color: #c77900;
        --md-outlined-button-label-text-color: #c77900;
        --md-outlined-button-hover-label-text-color: #fff;
        --md-outlined-button-hover-state-layer-color: #c77900;
        --md-outlined-button-pressed-label-text-color: #fff;
        --md-outlined-button-pressed-state-layer-color: #c77900;
      }

      .stop-btn md-icon {
        color: #c77900;
      }

      .play-btn {
        --md-filled-button-container-color: #1a7d2f;
      }

      .stopped-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #1a1a2eee;
        z-index: 10;
        gap: 16px;
      }

      .stopped-overlay md-icon {
        font-size: 64px;
        color: #5e7ce0;
      }

      .stopped-overlay span {
        font-family: sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #ccd0e0;
      }

      .editor-body {
        position: relative;
      }
    `,
  ];

  render() {
    return html`
      <div class="top-bar">
        <md-icon-button class="back-btn" @click=${this.goBack}>
          <md-icon>arrow_back</md-icon>
        </md-icon-button>
        <h3>${this.camStream}</h3>

        ${this.stopped ? html`
          <md-filled-button class="play-btn" @click=${this.toggleStream} ?disabled=${this.toggling}>
            <md-icon slot="icon">play_arrow</md-icon>
            Start
          </md-filled-button>
        ` : html`
          <md-outlined-button class="stop-btn" @click=${this.toggleStream} ?disabled=${this.toggling}>
            <md-icon slot="icon">stop</md-icon>
            Stop
          </md-outlined-button>
        `}

        <md-outlined-button class="delete-btn" @click=${this.openDeleteDialog}>
          <md-icon slot="icon">delete</md-icon>
          Delete
        </md-outlined-button>

      </div>

      <div class="editor-body">
        ${this.stopped ? html`
          <div class="stopped-overlay">
            <md-icon>videocam_off</md-icon>
            <span>Stream stopped</span>
          </div>
        ` : html``}
        <camera-player
          id=${this.camStream}
          label=${this.camStream}
          @video-ready=${this.onVideoReady}
        ></camera-player>
      </div>

      <md-dialog
        .open=${this.showDeleteDialog}
        @close=${this.cancelDelete}
      >
        <div slot="headline">Delete Stream</div>
        <div slot="content">
          <p style="margin:0; color:#5e5f61; font-family:sans-serif; font-size:0.9rem;">
            Are you sure you want to delete <strong>${this.camStream}</strong>?
            This will stop the video stream and remove its configuration.
          </p>
        </div>
        <div slot="actions">
          <md-text-button @click=${this.cancelDelete}>Cancel</md-text-button>
          <md-filled-button style="--md-filled-button-container-color:#b3261e" @click=${this.confirmDelete}>
            Delete
          </md-filled-button>
        </div>
      </md-dialog>
    `;
  }
}
