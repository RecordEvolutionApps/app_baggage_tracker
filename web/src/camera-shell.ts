import { LitElement, html, css } from 'lit';
import { customElement } from 'lit/decorators.js';
import './camera-player.js'
import { initJanus } from './modules/webRTCPlayer'
import { CameraPlayer } from './camera-player.js';

@customElement('camera-shell')
export class CameraShell extends LitElement {


  protected startJanus(): void {
    const frontCam = this.shadowRoot?.getElementById('frontCam') as CameraPlayer
    const leftCam = this.shadowRoot?.getElementById('leftCam') as CameraPlayer
    const backCam = this.shadowRoot?.getElementById('backCam') as CameraPlayer
    const rightCam = this.shadowRoot?.getElementById('rightCam') as CameraPlayer

    const videoPlayers = {
      frontCam: frontCam?.videoElement,
      leftCam: leftCam?.videoElement,
      backCam: backCam?.videoElement,
      rightCam: rightCam?.videoElement,
    }
    console.log('videoPlayers', frontCam, videoPlayers)

    initJanus(videoPlayers)
  }

  static styles = css`
    :host {
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
    }

    .cam-container {
      display: flex;
      width: 100%;
      gap: 8px;
      justify-content: space-around;
    }

  `;

  render() {
    return html`
    <div class="cam-container">
      <camera-player id="frontCam" label="Front" @video-ready=${this.startJanus}></camera-player>
    </div>
    <!-- <div class="cam-container">
      <camera-player id="leftCam" label="Left"></camera-player>
      <camera-player id="backCam" label="Back"></camera-player>
      <camera-player id="rightCam" label="Right"></camera-player>
    </div> -->
    


    `;
  }
}
