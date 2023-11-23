import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';
import './camera-player.js'
import './camera-selector.js'

@customElement('camera-shell')
export class CameraShell extends LitElement {
  @property({ type: String }) header = 'My app';

  static styles = css`
    :host {
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
    }
  `;

  render() {
    return html`
      <div> Choose the Camera </div>
      <camera-selector></camera-selector>
      <camera-player></camera-player>        
    `;
  }
}
