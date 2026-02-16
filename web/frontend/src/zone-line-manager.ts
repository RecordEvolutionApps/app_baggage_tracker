import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';
import { mainStyles } from './utils.js';

import '@material/web/button/elevated-button.js';
import '@material/web/button/text-button.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/dialog/dialog.js';
import { MdDialog } from '@material/web/dialog/dialog.js';

@customElement('zone-line-manager')
export class ZoneLineManager extends LitElement {
  @property({ type: Object })
  declare polygonManager?: PolygonManager;

  @property({ type: Object })
  declare selectedPolygon: Polygon | null;

  @state()
  declare mask_name: string;

  zoneDialog?: MdDialog;
  lineDialog?: MdDialog;
  private polygonUpdateHandler = () => {
    this.requestUpdate();
  };
  private lastPolygonManager?: PolygonManager;

  constructor() {
    super();
    this.selectedPolygon = null;
    this.mask_name = '';
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

      md-elevated-button,
      md-text-button {
        width: 100%;
        --md-elevated-button-container-color: #eceff1;
        --md-elevated-button-label-text-color: #5e5f61;
      }

      .mb16 {
        margin-bottom: 16px;
      }

      .button-row {
        display: flex;
        gap: 8px;
      }

      .dialog {
        --md-dialog-container-color: #fff;
        --md-dialog-headline-color: #5e5f61;
        --md-dialog-supporting-text-color: #5e5f61;
      }

      form {
        color: #5e5f61;
      }
    `,
  ];

  protected firstUpdated(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ): void {
    this.zoneDialog = this.shadowRoot?.getElementById(
      'zonedialog',
    ) as MdDialog;
    this.lineDialog = this.shadowRoot?.getElementById(
      'linedialog',
    ) as MdDialog;
  }

  protected updated(
    changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ): void {
    if (changedProperties.has('polygonManager')) {
      if (this.lastPolygonManager) {
        this.lastPolygonManager.removeEventListener(
          'update',
          this.polygonUpdateHandler,
        );
      }
      if (this.polygonManager) {
        this.polygonManager.addEventListener(
          'update',
          this.polygonUpdateHandler,
        );
      }
      this.lastPolygonManager = this.polygonManager;
    }
    super.updated(changedProperties);
  }

  handleMaskNameInput(ev: { target: { value: string } }) {
    if (ev.target) {
      this.mask_name = ev.target.value;
    }
  }

  handleZoneNameInputKeypress(ev: any) {
    if (ev.key === 'Enter' && this.mask_name.length) {
      this.zoneDialog?.close('create');
    }
  }

  handleLineNameInputKeypress(ev: any) {
    if (ev.key === 'Enter' && this.mask_name.length) {
      this.lineDialog?.close('create');
    }
  }

  createZone() {
    console.log(this.zoneDialog?.returnValue);
    if (this.zoneDialog?.returnValue === 'create') {
      this.polygonManager?.create(this.mask_name, 'ZONE');
    }
    this.mask_name = '';
  }

  createLine() {
    console.log(this.lineDialog?.returnValue);
    if (this.lineDialog?.returnValue === 'create') {
      this.polygonManager?.create(this.mask_name, 'LINE');
    }
    this.mask_name = '';
  }

  onCreateZoneClick() {
    this.zoneDialog?.show();
  }

  onCreateLineClick() {
    this.lineDialog?.show();
  }

  onDialogCancel() {
    if (this.zoneDialog) {
      this.zoneDialog.returnValue = 'cancel';
    }
  }

  undoLastLine() {
    if (this.selectedPolygon) {
      this.selectedPolygon.undo();
      this.polygonManager?.update();
    }
  }

  commitPolygon() {
    if (this.selectedPolygon) {
      this.selectedPolygon.commit();
      this.polygonManager?.update();
    }
  }

  render() {
    return html`
      <div class="control">
        <div class="mb16">
          <div class="button-row">
            <md-elevated-button @click=${this.onCreateZoneClick}>
              Zone
              <md-icon slot="icon">add_circle</md-icon>
            </md-elevated-button>
            <md-elevated-button
              .disabled=${this.selectedPolygon?.committed ||
              !this.selectedPolygon?.isCommitable}
              @click=${this.commitPolygon}
            >
              Close Zone
              <md-icon slot="icon">close_fullscreen</md-icon>
            </md-elevated-button>
          </div>
        </div>
        <div>
          <md-elevated-button @click=${this.onCreateLineClick}>
            Line
            <md-icon slot="icon">add_circle</md-icon>
          </md-elevated-button>
        </div>
      </div>

      <md-dialog
        @cancel=${this.onDialogCancel}
        @close=${this.createLine}
        id="linedialog"
        class="dialog"
      >
        <div slot="headline">Counter-Line Name</div>
        <form slot="content" id="create-line-form" method="dialog">
          <div style="display: flex; flex-direction: column;">
            <p>
              Enter a name for your counter line. After you clicked create, this
              panel closes and you can start setting the start and end points of
              your line. After you set the end point the line will automatically
              be submitted for counting.
            </p>
            <md-outlined-text-field
              label="Name"
              autofocus
              maxlength="18"
              @keyup=${this.handleLineNameInputKeypress}
              @input=${this.handleMaskNameInput}
              .value=${this.mask_name}
              required
            >
            </md-outlined-text-field>
          </div>
        </form>
        <div slot="actions">
          <md-text-button @click=${() => this.lineDialog?.close('cancel')}
            >Cancel</md-text-button
          >
          <md-text-button
            form="create-line-form"
            .disabled=${this.mask_name.length === 0}
            value="create"
            >Create</md-text-button
          >
        </div>
      </md-dialog>

      <md-dialog
        @cancel=${this.onDialogCancel}
        @close=${this.createZone}
        id="zonedialog"
        class="dialog"
      >
        <div slot="headline">Zone Name</div>
        <form slot="content" id="create-zone-form" method="dialog">
          <div style="display: flex; flex-direction: column;">
            <p>
              Enter a name for your zone. After you clicked create, this panel
              closes and you can start setting the corner points of your zone.
              When you click the "Close Zone" Button, the last point will
              automatically be connected to the first point to close the zone and
              the zone will be submitted for counting.
            </p>
            <md-outlined-text-field
              label="Name"
              autofocus
              maxlength="18"
              @keyup=${this.handleZoneNameInputKeypress}
              @input=${this.handleMaskNameInput}
              .value=${this.mask_name}
              required
            >
            </md-outlined-text-field>
          </div>
        </form>
        <div slot="actions">
          <md-text-button @click=${() => this.zoneDialog?.close('cancel')}
            >Cancel</md-text-button
          >
          <md-text-button
            form="create-zone-form"
            .disabled=${this.mask_name.length === 0}
            value="create"
            >Create</md-text-button
          >
        </div>
      </md-dialog>
    `;
  }
}
