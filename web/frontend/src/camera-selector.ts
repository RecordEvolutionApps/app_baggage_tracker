import { LitElement, html, css } from 'lit';
import { state, property, customElement } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';

import '@material/web/select/outlined-select.js';
import '@material/web/select/select-option.js';
import { MdOutlinedSelect } from '@material/web/select/outlined-select.js';
import { mainStyles, Camera, CamSetup } from './utils';

@customElement('camera-selector')
export class CameraSelector extends LitElement {

  @property({ type: String}) 
  camStream: string = 'frontCam'

  @property({ type: Object })
  camSetup?: CamSetup

  @state()
  private camList: Camera[] = [];
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
    this.selector.select(this.camSetup?.camera?.id ?? '');
  }

  update(changedProps: any) {
    if (changedProps.has('camSetup')) {
      this.selector?.select(this.camSetup?.camera?.id ?? '')
    }
    super.update(changedProps)
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
    console.log('selected', value, this.camStream);
    const payload = {
      id: value,
      type: 'USB',
      camStream: this.camStream,
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

      md-outlined-select {
        min-width: 242px;
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
      <md-outlined-select
        id="selector"
        label="Select your device"
        @change=${this.selectCamera}
      >
        ${repeat(
          this.camList,
          (c: Camera) => c.id,
          (c: Camera) => html`
            <md-select-option value="${c.id ?? ''}">
              <div slot="headline">${c.name} (${c.id})</div>
            </md-select-option>
          `,
        )}
      </md-outlined-select>
    `;
  }
}
