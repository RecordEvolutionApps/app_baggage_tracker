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

  constructor() {
    super();
    this.showDeleteDialog = false;
  }

  private basepath = window.location.protocol + '//' + window.location.host;

  disconnectedCallback() {
    super.disconnectedCallback();
    stopMediasoup();
  }

  private onVideoReady(_event: CustomEvent) {
    const videoPlayers: Record<string, HTMLVideoElement | undefined> = {};
    const player = this.shadowRoot?.querySelector('camera-player') as CameraPlayer;
    if (player?.id && player.videoElement) {
      videoPlayers[player.id] = player.videoElement;
    }
    initMediasoup(videoPlayers);
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
    `,
  ];

  render() {
    return html`
      <div class="top-bar">
        <md-icon-button class="back-btn" @click=${this.goBack}>
          <md-icon>arrow_back</md-icon>
        </md-icon-button>
        <h3>${this.camStream}</h3>
        <md-outlined-button class="delete-btn" @click=${this.openDeleteDialog}>
          <md-icon slot="icon">delete</md-icon>
          Delete
        </md-outlined-button>

      </div>

      <div class="editor-body">
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
