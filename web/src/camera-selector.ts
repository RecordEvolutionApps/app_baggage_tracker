import { LitElement, html, css } from 'lit';
import { state, property, customElement } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';

import '@material/web/select/outlined-select.js';
import '@material/web/select/select-option.js';
import { MdOutlinedSelect } from '@material/web/select/outlined-select.js';
import { mainStyles } from './utils';

@customElement('camera-selector')
export class CameraSelector extends LitElement {
  @property({ type: Object }) camera: any;

  @state()
  private camList: any[] = [];
  basepath: string;
  selector?: MdOutlinedSelect;
  constructor() {
    super();
    this.basepath = window.location.protocol + '//' + window.location.host;
  }

  async firstUpdated() {
    this.selector = this.shadowRoot?.getElementById(
      'selector',
    ) as MdOutlinedSelect;

    await this.getCameras();

    this.selector.select(this.camera?.device?.id);
  }

  async getCameras() {
    this.camList = await fetch(`${this.basepath}/cameras`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    }).then(res => res.json());

    console.log('CAMLIST', this.camList);

    await this.updateComplete;
  }
  async selectCamera() {
    const value = this.selector?.value;
    console.log('selected', value, this.id);
    const payload = {
      id: value,
      deviceName: this.id,
    };

    await fetch(`${this.basepath}/cameras/select`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
    });
  }

  static styles = [
    mainStyles,
    css`
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
    `,
  ];

  render() {
    return html`
      <md-outlined-select id="selector" @change=${this.selectCamera}>
        ${repeat(
          this.camList,
          c => c.id,
          c => html`
            <md-select-option value="${c.id}">
              <div slot="headline">${c.name} (${c.id})</div>
            </md-select-option>
          `,
        )}
      </md-outlined-select>
    `;
  }
}
