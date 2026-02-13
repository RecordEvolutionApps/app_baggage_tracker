import { LitElement, html, css } from 'lit';
import { customElement } from 'lit/decorators.js';
import './camera-player.js';
import { initMediasoup } from './modules/webRTCPlayer.js';
import { CameraPlayer } from './camera-player.js';
import { mainStyles } from './utils.js';

@customElement('camera-shell')
export class CameraShell extends LitElement {

  protected startStream(event: CustomEvent): void {
    const target = event.target as CameraPlayer;
    const videoElement = event.detail?.videoElement as HTMLVideoElement | undefined;
    if (!videoElement || !target?.id) {
      console.error('[camera-shell] videoElement or camId missing from event');
      return;
    }

    // Build videoPlayers from all rendered camera-player elements
    const videoPlayers: Record<string, HTMLVideoElement | undefined> = {};
    this.shadowRoot?.querySelectorAll('camera-player').forEach((el) => {
      const player = el as CameraPlayer;
      if (player.id && player.videoElement) {
        videoPlayers[player.id] = player.videoElement;
      }
    });
    console.log('[camera-shell] videoPlayers', videoPlayers);

    initMediasoup(videoPlayers);
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

      .cam-container {
        display: flex;
        width: 100%;
        gap: 8px;
        justify-content: space-around;
      }
    `,
  ];

  render() {
    return html`
      <div class="cam-container">
        <camera-player
          id="frontCam"
          label="Front"
          @video-ready=${this.startStream}
        ></camera-player>
      </div>
      <!-- <div class="cam-container">
      <camera-player id="leftCam" label="Left"></camera-player>
      <camera-player id="backCam" label="Back"></camera-player>
      <camera-player id="rightCam" label="Right"></camera-player>
    </div> -->
    `;
  }
}
