import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';
import { CamSetup } from './utils.js';
import './camera-setup.js';
import './inference-setup.js';
import './zone-line-manager.js';

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

  render() {
    return html`
      <div class="wrapper">
        <h4>Camera Setup</h4>
        <camera-setup
          .camStream=${this.camStream}
          .camSetup=${this.camSetup}
          @camera-setup-changed=${this.onCameraSetupChanged}
        ></camera-setup>

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
