import { LitElement, html, css } from 'lit';
import { state, customElement } from 'lit/decorators.js';

@customElement('camera-selector')
export class CameraSelector extends LitElement {

  @state()
  private camList: any[] = [];

  firstUpdated() {
      fetch('http://localhost:1100/cameras', {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      }).then(res => res.json())
        .then(res => console.log(res))
  }

  static styles = css`
    :host {
      display: block;
    }
  `;

  render() {
    return html`
      
    `;
  }
}
