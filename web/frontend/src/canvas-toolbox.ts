import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';
import { mainStyles, CamSetup } from './utils.js';
import './camera-selector.js';

import '@material/web/button/elevated-button.js';
import '@material/web/button/text-button.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/dialog/dialog.js';
import { MdDialog } from '@material/web/dialog/dialog.js';

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
  mask_name = '';

  dialog?: MdDialog;

  initialized = false;

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

    this.polygonManager.addEventListener('update', (ev: any) => {
      const { polygons, selectedPolygon } = ev.detail;
      this.polygons = polygons;
      this.selectedPolygon = selectedPolygon;

      this.requestUpdate();
    });

    this.polygons = this.polygonManager?.getAll();
    this.requestUpdate();
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

  onDialogCancel() {
    if (this.dialog) {
      this.dialog.returnValue = 'cancel';
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

  render() {
    return html`<div>
        <h3>Vehicle Counter</h3>
        <ul>
          <li>
            <h4>Camera</h4>
          </li>
          <li>
            <camera-selector
              .camStream=${this.camStream}
              .camSetup=${this.camSetup}
            ></camera-selector>
          </li>
          <li>
            <h4>Detection Zone</h4>
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
              .disabled=${this.selectedPolygon?.committed ||
      !this.selectedPolygon?.isCommitable}
              @click=${this.commitPolygon}
            >
              Commit
              <md-icon slot="icon">close_fullscreen</md-icon>
            </md-elevated-button>
          </li>
        </ul>
      </div>

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
