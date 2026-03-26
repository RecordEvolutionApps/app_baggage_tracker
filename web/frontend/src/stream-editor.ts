import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { mainStyles } from './utils.js';
import './camera-player.js';
import { CameraPlayer } from './camera-player.js';
import { initMediasoup, stopMediasoup } from './modules/webRTCPlayer.js';
import { readStream, writeStream, deleteStream, subscribeStreams } from './streams-sdk.js';
import { ironflock, ironflockReady, deviceKey } from './ironflock.js';

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
  @state() declare streamName: string;
  @state() declare editingName: boolean;
  @state() declare fullConfig: Record<string, any> | null;

  constructor() {
    super();
    this.showDeleteDialog = false;
    this.stopped = false;
    this.toggling = false;
    this.streamName = '';
    this.editingName = false;
    this.fullConfig = null;
  }

  connectedCallback() {
    super.connectedCallback();
    this.loadStoppedState();
    this._subscribeToChanges();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    stopMediasoup();
  }

  private _subscribeToChanges() {
    subscribeStreams((config) => {
      if (config.camStream !== this.camStream) return;

      // Stream was deleted by another user — go back
      if ((config as any).deleted) {
        this.goBack();
        return;
      }

      // Update name if changed externally
      if (config.name && config.name !== this.streamName && !this.editingName) {
        this.streamName = config.name;
      }

      // Update stopped state and handle WebRTC accordingly
      const wasStopped = this.stopped;
      const nowStopped = !!config.stopped;
      if (nowStopped !== wasStopped && !this.toggling) {
        this.stopped = nowStopped;
        if (nowStopped) {
          stopMediasoup();
        } else {
          this.updateComplete.then(() => {
            setTimeout(() => {
              const videoPlayers: Record<string, HTMLVideoElement | undefined> = {};
              const player = this.shadowRoot?.querySelector('camera-player') as CameraPlayer;
              if (player?.id && player.videoElement) {
                videoPlayers[player.id] = player.videoElement;
              }
              initMediasoup(videoPlayers);
            }, 500);
          });
        }
      }

      // Update full config (used when saving name or other top-level edits)
      this.fullConfig = config;

      // Push the full config directly into camera-player so all downstream
      // components (video-canvas, canvas-toolbox, inference-setup, camera-dialog)
      // update immediately via Lit property bindings, without waiting for the
      // player's own subscription callback to fire.
      const player = this.shadowRoot?.querySelector('camera-player') as CameraPlayer;
      if (player && config.source?.path) {
        (player as any).camSetup = {
          ...config,
          source: {
            ...config.source,
            width: config.source.width ?? (player as any).camSetup?.source?.width ?? 640,
            height: config.source.height ?? (player as any).camSetup?.source?.height ?? 480,
          },
        };
      }
    });
  }

  private async loadStoppedState() {
    try {
      const data = await readStream(this.camStream);
      if (data) {
        this.stopped = !!data.stopped;
        this.streamName = data.name ?? '';
        this.fullConfig = data;
      }
    } catch (err) {
      console.error('Failed to load stream state', err);
    }
  }

  private startEditName() {
    this.editingName = true;
  }

  private cancelEditName() {
    this.editingName = false;
  }

  private async saveName(value: string) {
    const trimmed = value.trim();
    this.editingName = false;
    this.streamName = trimmed;
    if (!this.fullConfig) return;
    try {
      const updated = { ...this.fullConfig, name: trimmed } as any;
      await writeStream(this.camStream, updated);
      this.fullConfig = updated;
    } catch (err) {
      console.error('Error renaming stream', err);
    }
  }

  override updated(changed: Map<string, unknown>) {
    super.updated(changed);
    if (changed.has('editingName') && this.editingName) {
      const input = this.shadowRoot?.querySelector<HTMLInputElement>('.name-input');
      if (input) { input.focus(); input.select(); }
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
      await ironflockReady;
      if (this.stopped) {
        // Start the stream subprocess
        await ironflock.callDeviceFunction(deviceKey, 'startStream', [this.camStream, this.fullConfig?.source ?? {}]);
        await writeStream(this.camStream, { stopped: false } as any, 'started');
      } else {
        // Stop the stream subprocess
        await ironflock.callDeviceFunction(deviceKey, 'stopStream', [this.camStream]);
        await writeStream(this.camStream, { stopped: true } as any, 'stopped');
      }
      // State will be updated by the subscription callback.
      // Set it locally too for immediate feedback.
      this.stopped = !this.stopped;

      if (this.stopped) {
        stopMediasoup();
      } else {
        await this.updateComplete;
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
      await ironflockReady;
      // Kill the subprocess before removing the config
      await ironflock.callDeviceFunction(deviceKey, 'deleteStream', [this.camStream]).catch(() => {});
      await deleteStream(this.camStream);
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

      .stream-title {
        cursor: pointer;
        border-radius: 4px;
        padding: 2px 4px;
        transition: background 0.15s;
      }

      .stream-title:hover {
        background: rgba(0, 46, 106, 0.08);
        text-decoration: underline dotted;
      }

      .name-input {
        flex: 1;
        margin: 0;
        padding: 2px 6px;
        background: #fff;
        border: 2px solid #002e6a;
        border-radius: 4px;
        color: #002e6a;
        font-family: sans-serif;
        font-weight: 600;
        font-size: 1.15rem;
        outline: none;
        min-width: 0;
      }

      .editor-body {
        flex: 1;
        overflow: auto;
        display: flex;
        flex-direction: column;
        position: relative;
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


    `,
  ];

  render() {
    return html`
      <div class="top-bar">
        <md-icon-button class="back-btn" @click=${this.goBack}>
          <md-icon>arrow_back</md-icon>
        </md-icon-button>
        ${this.editingName ? html`
          <input
            class="name-input"
            .value=${this.streamName}
            @keydown=${(e: KeyboardEvent) => {
              if (e.key === 'Enter') this.saveName((e.target as HTMLInputElement).value);
              if (e.key === 'Escape') this.cancelEditName();
            }}
            @blur=${(e: FocusEvent) => this.saveName((e.target as HTMLInputElement).value)}
          />
        ` : html`
          <h3 class="stream-title" @click=${this.startEditName} title="Click to rename">${this.streamName || this.camStream}</h3>
        `}

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
        <camera-player
          id=${this.camStream}
          label=${this.camStream}
          ?stopped=${this.stopped}
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
