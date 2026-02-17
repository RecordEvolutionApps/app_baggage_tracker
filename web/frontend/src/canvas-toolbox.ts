import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';
import { CamSetup } from './utils.js';
import './camera-dialog.js';
import { CameraDialog } from './camera-dialog.js';
import './inference-setup.js';
import './zone-line-manager.js';

import '@material/web/button/outlined-button.js';
import '@material/web/icon/icon.js';

@customElement('canvas-toolbox')
export class CanvasToolbox extends LitElement {
  @property({ type: Object })
  declare canvas?: HTMLCanvasElement;

  @property({ type: Object })
  declare polygonManager?: PolygonManager;

  @property({ type: String })
  declare camStream: string;

  @property({ type: Object })
  declare camSetup?: CamSetup;

  @state()
  declare polygons: Polygon[];

  @state()
  declare selectedPolygon: Polygon | null;

  constructor() {
    super();
    this.camStream = 'frontCam';
    this.polygons = [];
    this.selectedPolygon = null;
  }

  static styles = [
    css`
      .wrapper {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      h4 {
        color: #5e5f61;
        margin: 16px 0px 0px;
      }

      h3 {
        margin: 0;
        color: #5e5f61;
        font-size: 20px;
      }

      .camera-button {
        width: 100%;
        --md-outlined-button-container-color: #eceff1;
        --md-outlined-button-label-text-color: #5e5f61;
      }

      .source-label {
        font-family: sans-serif;
        font-size: 0.8rem;
        color: #888;
        margin: 4px 0 0;
        word-break: break-all;
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

    this.polygonManager.addEventListener('update', (ev: any) => {
      const { polygons, selectedPolygon } = ev.detail;
      this.polygons = polygons;
      this.selectedPolygon = selectedPolygon;
      this.requestUpdate();
    });

    // Sync current state (importFromRemote may have already completed)
    this.polygons = this.polygonManager?.getAll();
    this.selectedPolygon = this.polygonManager?.getSelected() ?? null;
    this.requestUpdate();

    // Re-fetch in case data wasn't loaded yet when the manager was created
    this.polygonManager.importFromRemote();
  }

  private onCameraSetupChanged(ev: CustomEvent) {
    this.camSetup = ev.detail as CamSetup;
  }

  private openCameraDialog() {
    const dialog = this.shadowRoot?.querySelector('camera-dialog') as CameraDialog;
    dialog?.show(this.camStream, this.camSetup);
  }

  private get currentSourceLabel(): string {
    const cam = this.camSetup?.camera;
    if (!cam) return 'No camera configured';
    switch (cam.type) {
      case 'USB': {
        const parts = [cam.name, cam.id, cam.path].filter(Boolean);
        return `USB: ${parts.join(' · ') || 'Unknown'}`;
      }
      case 'Demo':
        return 'Demo Video';
      case 'YouTube':
        return `YouTube: ${cam.path ?? ''}`;
      case 'IP':
        if (cam.path === 'demoVideo') return 'Demo Video';
        return `IP: ${cam.path ?? ''}`;
      default:
        return cam.path ?? 'Unknown';
    }
  }

  render() {
    return html`
      <div class="wrapper">
        <h4>Camera Setup</h4>
        <md-outlined-button class="camera-button" @click=${this.openCameraDialog}>
          Configure Camera Source
        </md-outlined-button>
        <div class="source-label">${this.currentSourceLabel}</div>

        <camera-dialog
          .camStream=${this.camStream}
          .camSetup=${this.camSetup}
          @camera-setup-changed=${this.onCameraSetupChanged}
        ></camera-dialog>

        <h4>Inference Setup</h4>
        <inference-setup
          .camStream=${this.camStream}
          .camSetup=${this.camSetup}
        ></inference-setup>

        <h4>Zones &amp; Lines</h4>
        <zone-line-manager
          .polygonManager=${this.polygonManager}
          .selectedPolygon=${this.selectedPolygon}
        ></zone-line-manager>
      </div>
    `;
  }
}
