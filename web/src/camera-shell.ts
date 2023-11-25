import { LitElement, html, css } from 'lit';
import { customElement } from 'lit/decorators.js';
import './camera-player.js'
import { initJanus } from './modules/webRTCPlayer.js'

@customElement('camera-shell')
export class CameraShell extends LitElement {


  protected firstUpdated(): void {
      const videoPlayers = {
        frontCam: this.shadowRoot?.getElementById('frontCam'),
        leftCam: this.shadowRoot?.getElementById('leftCam'),
        backCam: this.shadowRoot?.getElementById('backCam'),
        rightCam: this.shadowRoot?.getElementById('rightCam'),
      }
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

    camera-player {
      border: 2px solid #fff;
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
      <camera-player id="frontCam" label="Front"></camera-player>
    </div>
    <!-- <div class="cam-container">
      <camera-player id="leftCam" label="Left"></camera-player>
      <camera-player id="backCam" label="Back"></camera-player>
      <camera-player id="rightCam" label="Right"></camera-player>
    </div> -->
    


    `;
  }
}
