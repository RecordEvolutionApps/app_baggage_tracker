import { LitElement, html, css } from 'lit';
import { state, customElement } from 'lit/decorators.js';
import {repeat} from 'lit/directives/repeat.js';

import '@material/web/select/outlined-select.js'
import '@material/web/select/select-option.js'
import { MdOutlinedSelect } from '@material/web/select/outlined-select.js';

@customElement('camera-selector')
export class CameraSelector extends LitElement {

  @state()
  private camList: any[] = [];
  basepath: string
  selector?: MdOutlinedSelect
  constructor() {
    super()
    this.basepath = window.location.protocol + '//' + window.location.host 
  }
  async firstUpdated() {
      
      this.selector = this.shadowRoot?.getElementById('selector') as MdOutlinedSelect

      await this.getCameras()

      const selected = await fetch(`${this.basepath}/cameras/setup?cam=${this.id}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      }).then(res => res.json())
      console.log('selected', selected)
      this.selector.select(selected.device)
  }

async getCameras() {
      this.camList = await fetch(`${this.basepath}/cameras`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      }).then(res => res.json())

      console.log('CAMLIST', this.camList)
      await this.updateComplete
}
  async selectCamera() {
    const value = this.selector?.value
    console.log('selected', value, this.id)
    const payload = {
      device: value,
      cam: this.id
    }
    await fetch(`${this.basepath}/cameras/select`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload), 
    })
  }

  static styles = css`
    :host {
      display: block;
    }
    :root {
      --md-filled-select-text-field-container-shape: 0px;
      --md-filled-select-text-field-container-color: #f7faf9;
      --md-filled-select-text-field-input-text-color: #005353;
      --md-filled-select-text-field-input-text-font: system-ui;
    }

    md-filled-select::part(menu) {
      --md-menu-container-color: #f4fbfa;
      --md-menu-container-shape: 0px;
    }
  `;

  render() {
    return html`
      <md-outlined-select id="selector" @change=${this.selectCamera} @opening=${this.getCameras}>
        ${repeat(this.camList, c => c.path, c => html`
        <md-select-option value="${c.path}">
          <div slot="headline">${c.path}</div>
        </md-select-option>
        `)}
      </md-outlined-select>
    `;
  }
}
