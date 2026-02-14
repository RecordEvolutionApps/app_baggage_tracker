import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { mainStyles, CamSetup } from './utils.js';
import './camera-selector.js';
import './ip-camera-dialog.js';
import { IpCameraDialog } from './ip-camera-dialog.js';

import '@material/web/button/elevated-button.js';
import '@material/web/radio/radio.js';

@customElement('camera-setup')
export class CameraSetup extends LitElement {
  @property({ type: String })
  declare camStream: string;

  @property({ type: Object })
  declare camSetup?: CamSetup;

  @state()
  declare selectedCamType: 'USB' | 'IP';

  cameraDialog?: IpCameraDialog;

  constructor() {
    super();
    this.camStream = 'frontCam';
    this.selectedCamType = 'USB';
  }

  static styles = [
    mainStyles,
    css`
      :host {
        --md-sys-color-primary: #002e6a;
        --md-sys-color-on-primary: #ffffff;
        --md-sys-color-primary-container: #2986cc;
        --md-sys-color-on-primary-container: #002020;
      }

      .control {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #aaa;
      }

      md-elevated-button {
        width: 100%;
        --md-elevated-button-container-color: #eceff1;
        --md-elevated-button-label-text-color: #5e5f61;
      }

      .mb16 {
        margin-bottom: 16px;
      }

      .paging:not([active]) {
        display: none !important;
      }

      .column {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .column > div {
        overflow: hidden;
        text-overflow: ellipsis;
      }
    `,
  ];

  protected firstUpdated(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ): void {
    this.cameraDialog = this.shadowRoot?.getElementById(
      'camera-dialog',
    ) as IpCameraDialog;
  }

  update(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ) {
    if (_changedProperties.has('camSetup') && this.camSetup) {
      this.selectedCamType = this.camSetup?.camera?.type ?? 'USB';
    }
    super.update(_changedProperties);
  }

  selectCameraType(_type: 'USB' | 'IP') {
    this.selectedCamType = this.selectedCamType === 'USB' ? 'IP' : 'USB';
    if (this.selectedCamType !== 'IP') return;

    if (this.camSetup?.camera?.path && this.camSetup?.camera.type === 'IP') {
      this.cameraDialog?.submitIPCamera();
    } else {
      this.cameraDialog?.show();
    }
  }

  onCreateIPClick() {
    this.cameraDialog?.show();
  }

  onSetIPCamera(ev: CustomEvent) {
    const camSetup = {
      camera: ev.detail,
      width: 1270,
      height: 720,
    } as CamSetup;
    this.camSetup = camSetup;
    this.dispatchEvent(
      new CustomEvent('camera-setup-changed', {
        detail: camSetup,
        bubbles: true,
        composed: true,
      }),
    );
  }

  onDialogCancel() {
    if (this.cameraDialog) {
      this.cameraDialog.onDialogCancel();
    }
  }

  render() {
    return html`
      <div class="control">
        <div>
          <div class="mb16">
            <md-radio
              id="usb"
              value="USB"
              aria-label="USB"
              @change=${() => this.selectCameraType('USB')}
              .checked=${this.selectedCamType === 'USB'}
            ></md-radio>
            <label for="usb">USB</label>
            <md-radio
              id="ip"
              value="IP"
              aria-label="IP"
              @change=${() => this.selectCameraType('IP')}
              .checked=${this.selectedCamType === 'IP'}
            ></md-radio>
            <label for="ip">IP Camera</label>
          </div>
        </div>
        <div class="paging" ?active=${this.selectedCamType === 'USB'}>
          <camera-selector
            .camStream=${this.camStream}
            .camSetup=${this.camSetup}
          ></camera-selector>
        </div>

        <div class="column paging" ?active=${this.selectedCamType === 'IP'}>
          <div>
            ${this.camSetup?.camera?.type === 'IP'
              ? this.camSetup?.camera?.path ?? ''
              : ''}
          </div>
          <md-elevated-button @click=${this.onCreateIPClick}>
            Setup IP Camera
            <md-icon slot="icon">edit</md-icon>
          </md-elevated-button>
        </div>
      </div>

      <ip-camera-dialog
        id="camera-dialog"
        .camStream=${this.camStream}
        .camSetup=${this.camSetup}
        @camera-selected=${this.onSetIPCamera}
      >
      </ip-camera-dialog>
    `;
  }
}
