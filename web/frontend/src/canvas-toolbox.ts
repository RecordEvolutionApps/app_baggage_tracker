import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';
import { mainStyles, CamSetup, Camera } from './utils.js';
import './camera-selector.js';
import './ip-camera-dialog.js';
import '@material/web/button/elevated-button.js';
import '@material/web/button/text-button.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/radio/radio.js';
import '@material/web/dialog/dialog.js';
import { MdDialog } from '@material/web/dialog/dialog.js';
import { IpCameraDialog } from './ip-camera-dialog.js';

@customElement('canvas-toolbox')
export class CanvasToolbox extends LitElement {
  @property({ type: Object })
  canvas?: HTMLCanvasElement;

  @property({ type: Object })
  polygonManager?: PolygonManager;

  @property({ type: String })
  camStream: string = 'frontCam'

  @property({ type: Object })
  camSetup?: CamSetup

  @state()
  polygons: Polygon[] = [];

  @state()
  selectedPolygon: Polygon | null = null;

  @state()
  mask_name: string = '';

  @state()
  selectedCamType: 'USB' | 'IP' = 'USB'

  dialog?: MdDialog;
  cameraDialog?: IpCameraDialog;

  static styles = [
    mainStyles,
    css`
      :host {
        min-width: 64px;
        --md-sys-color-primary: #002e6a;
        --md-sys-color-on-primary: #FFFFFF;
        --md-sys-color-primary-container: #2986cc;
        --md-sys-color-on-primary-container: #002020;
      }
      
      .primary {
        background: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
      }

      .paging:not([active]) { display: none !important; }

      ul {
        list-style-type: none;
        margin-block-start: 0;
        margin-block-end: 0;
        margin-inline-start: 0px;
        margin-inline-end: 0px;
        padding-inline-start: 0;
      }

      md-elevated-button, md-text-button {
        width: 100%;
        --md-elevated-button-container-color: #eceff1;
        --md-elevated-button-label-text-color: #5e5f61;
      }

      .mb16 {
        margin-bottom: 16px;
      }

      h4 {
        color: #5e5f61;
        margin: 11px 0;
      }

      #dialog {
        --md-dialog-container-color: #fff;
        --md-dialog-headline-color: #5e5f61;
        --md-dialog-supporting-text-color: #5e5f61;
      }
      h3 {
        margin: 0;
        color: #5e5f61;
        font-size: 20px;
      }

      .column {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      form {
        color: #5e5f61;
      }

      @media only screen and (max-width: 600px) {
        h3 {
          display: none;
        }
      }
    `,
  ];

  protected firstUpdated(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ): void {
    if (!this.polygonManager) throw new Error('polygon manager not defined');

    this.dialog = this.shadowRoot?.getElementById('dialog') as MdDialog;
    this.cameraDialog = this.shadowRoot?.getElementById('camera-dialog') as IpCameraDialog;

    this.polygonManager.addEventListener('update', (ev: any) => {
      const { polygons, selectedPolygon } = ev.detail;
      this.polygons = polygons;
      this.selectedPolygon = selectedPolygon;

      this.requestUpdate();
    });

    this.polygons = this.polygonManager?.getAll();
    this.requestUpdate();
  }

  update(_changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>) {
    if (_changedProperties.has('camSetup') && this.camSetup)
    this.selectedCamType = this.camSetup?.camera?.type ?? 'USB'
    super.update(_changedProperties)
  }

  handleMaskNameInput(ev: { target: { value: string } }) {
    if (ev.target) {
      this.mask_name = ev.target.value;
    }
  }

  handleNameInputKeypress(ev: any) {
    if (ev.key === 'Enter' && this.mask_name.length) {
      this.dialog?.close('create');
    }
  }

  createPolygon() {
    console.log(this.dialog?.returnValue);
    if (this.dialog?.returnValue === 'create') {
      this.polygonManager?.create(this.mask_name);
    }

    this.mask_name = '';
  }

  onCreateClick() {
    this.dialog?.show();
  }

  onCreateIPClick() {
    this.cameraDialog?.show();
  }

  onDialogCancel() {
    if (this.dialog) {
      this.dialog.returnValue = 'cancel';
    }
    if (this.cameraDialog) {
      this.cameraDialog.onDialogCancel();
    }
  }

  undoLastLine() {
    if (this.selectedPolygon) {
      this.selectedPolygon.undo();

      // Since a Polygon has some state of it's own, but no event emitter, we need to ask the manager for an update instead
      this.polygonManager?.update();
    }
  }

  commitPolygon() {
    if (this.selectedPolygon) {
      this.selectedPolygon.commit();

      // Since a Polygon has some state of it's own, but no event emitter, we need to ask the manager for an update instead
      this.polygonManager?.update();
    }
  }

  selectCameraType(type: 'USB' | 'IP') {
    this.selectedCamType = this.selectedCamType === 'USB' ? 'IP' : 'USB'
    if (this.selectedCamType !== 'IP') return
    
    if (this.camSetup?.camera?.path && this.camSetup?.camera.type === 'IP') {
      this.cameraDialog?.submitIPCamera();
    } else {
      this.cameraDialog?.show();
    }
  }

  onSetIPCamera(ev: CustomEvent) {
    this.camSetup = {camera: ev.detail, width: 1270, height: 720} as CamSetup
  }

  render() {
    return html`<div>
        <h3>Vehicle Counter</h3>
        <ul>
          <li>
            <h4>Camera</h4>
          </li>
          <li>
            <div class="mb16">
              <md-radio id="usb" value="USB" aria-label="USB" 
                @change=${ () => this.selectCameraType('USB')}
                .checked=${ this.selectedCamType === 'USB'}
                ></md-radio>
              <label for="usb">USB</label>
              <md-radio id="ip" value="IP" aria-label="IP" 
                @change=${ () => this.selectCameraType('IP')}
                .checked=${ this.selectedCamType === 'IP'}
                ></md-radio>
              <label for="ip">IP Camera</label>
            </div>
          </li>
          <li class="paging" ?active=${this.selectedCamType === 'USB'}>
            <camera-selector
              .camStream=${this.camStream}
              .camSetup=${this.camSetup}
            ></camera-selector>
          </li>

          <li class="column paging" ?active=${this.selectedCamType === 'IP'}>
            <div>${this.camSetup?.camera.type === 'IP' ? this.camSetup?.camera?.path ?? '' : ''}</div>
            <md-elevated-button @click=${this.onCreateIPClick}>
              Setup IP Camera
              <md-icon slot="icon">edit</md-icon>
            </md-elevated-button>
          </li>
          <li>
            <h4>Detection Zones</h4>
          </li>
          <li class="mb16">
            <md-elevated-button @click=${this.onCreateClick}>
              Create
              <md-icon slot="icon">add_circle</md-icon>
            </md-elevated-button>
          </li>
          <li class="mb16">
            <md-elevated-button @click=${this.undoLastLine}>
              Undo
              <md-icon slot="icon">undo</md-icon>
            </md-elevated-button>
          </li>
          <li>
            <md-elevated-button
              .disabled=${this.selectedPolygon?.committed || !this.selectedPolygon?.isCommitable}
              @click=${this.commitPolygon}
            >
              Commit
              <md-icon slot="icon">close_fullscreen</md-icon>
            </md-elevated-button>
          </li>
        </ul>
      </div>

      <ip-camera-dialog
        id="camera-dialog"
        .camStream=${this.camStream}
        .camSetup=${this.camSetup}
        @camera-selected=${ this.onSetIPCamera}
        >

      </ip-camera-dialog>

      <md-dialog
        @cancel=${this.onDialogCancel}
        @close=${this.createPolygon}
        id="dialog"
      >
        <div slot="headline">Zone name</div>
        <form slot="content" id="create-mask-form" method="dialog">
          <div style="display: flex; flex-direction: column;">
            <p>Enter a name for your zone</p>
            <md-outlined-text-field
              label="Name"
              autofocus
              maxlength="18"
              @keyup=${this.handleNameInputKeypress}
              @input=${this.handleMaskNameInput}
              .value=${this.mask_name}
              required
            >
            </md-outlined-text-field>
          </div>
        </form>
        <div slot="actions">
          <md-text-button @click=${() => this.dialog?.close('cancel')}>Cancel</md-text-button>
          <md-text-button
            form="create-mask-form"
            .disabled=${this.mask_name.length === 0}
            value="create"
            >Create</md-text-button
          >
        </div>
      </md-dialog> `;
  }
}
