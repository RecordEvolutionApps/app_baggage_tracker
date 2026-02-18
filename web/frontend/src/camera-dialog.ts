import { LitElement, html, css, nothing, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { mainStyles, Camera, CamSetup, DeviceCameraInfo } from './utils.js';

import '@material/web/dialog/dialog.js';
import { MdDialog } from '@material/web/dialog/dialog.js';
import '@material/web/button/text-button.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/select/outlined-select.js';
import '@material/web/select/select-option.js';
import { MdOutlinedSelect } from '@material/web/select/outlined-select.js';
import '@material/web/textfield/outlined-text-field.js';
import { MdOutlinedTextField } from '@material/web/textfield/outlined-text-field.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

type SourceTab = 'Local' | 'Demo' | 'YouTube' | 'IP' | 'Image';

@customElement('camera-dialog')
export class CameraDialog extends LitElement {
  @property({ type: String })
  declare camStream: string;

  @property({ type: Object })
  declare camSetup?: CamSetup;

  @state() declare activeTab: SourceTab;
  @state() declare localCameras: DeviceCameraInfo[];
  @state() declare selectedLocalId: string;
  @state() declare selectedResolution: string; // "WxH"
  @state() declare ipPath: string;
  @state() declare ipUsername: string;
  @state() declare ipPassword: string;
  @state() declare youtubeUrl: string;
  @state() declare imageUrl: string;
  @state() declare loading: boolean;

  private dialog?: MdDialog;
  private basepath = window.location.protocol + '//' + window.location.host;

  constructor() {
    super();
    this.camStream = 'frontCam';
    this.activeTab = 'Local';
    this.localCameras = [];
    this.selectedLocalId = '';
    this.selectedResolution = '';
    this.ipPath = '';
    this.ipUsername = '';
    this.ipPassword = '';
    this.youtubeUrl = '';
    this.imageUrl = '';
    this.loading = false;
  }

  static styles = [
    mainStyles,
    css`
      :host {
        --md-sys-color-primary: #002e6a;
        --md-sys-color-on-primary: #ffffff;
      }

      .dialog {
        --md-dialog-container-color: #fff;
        --md-dialog-headline-color: #5e5f61;
        --md-dialog-supporting-text-color: #5e5f61;
        min-width: min(600px, 90vw) !important;
        max-width: 90vw !important;
      }

      .tab-content {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 16px 0 8px;
        min-height: 200px;
      }

      .field-row {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .resolution-row {
        display: flex;
        gap: 12px;
        align-items: flex-start;
      }

      .resolution-row md-outlined-select {
        flex: 1;
      }

      md-outlined-select {
        width: 100%;
      }

      md-outlined-text-field {
        width: 100%;
      }

      .demo-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        padding: 24px 16px;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        background: #f7faf9;
        text-align: center;
      }

      .demo-card .demo-icon {
        font-size: 48px;
        color: #5e5f61;
      }

      .demo-card p {
        margin: 0;
        color: #5e5f61;
        font-family: sans-serif;
        font-size: 0.9rem;
      }

      .source-label {
        font-family: sans-serif;
        font-size: 0.85rem;
        color: #5e5f61;
        margin: 0;
      }

      .current-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        color: #2d5a2d;
        background: #eef6ee;
        border: 1px solid #8bc48b;
        border-radius: 3px;
        padding: 1px 6px;
        margin-left: 6px;
      }

      .interface-badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border-radius: 3px;
        padding: 1px 5px;
        margin-left: 6px;
      }
      .interface-badge.usb {
        color: #1a5276;
        background: #d6eaf8;
        border: 1px solid #85c1e9;
      }
      .interface-badge.csi {
        color: #6c3483;
        background: #e8daef;
        border: 1px solid #bb8fce;
      }
      .interface-badge.gmsl {
        color: #784212;
        background: #fdebd0;
        border: 1px solid #f0b27a;
      }
      .interface-badge.other {
        color: #5e5f61;
        background: #eee;
        border: 1px solid #ccc;
      }

      md-tabs {
        --md-primary-tab-container-color: transparent;
      }

      .hint {
        font-size: 0.8rem;
        color: #888;
        font-family: sans-serif;
        margin: 0;
      }
    `,
  ];

  protected firstUpdated(_: PropertyValueMap<any> | Map<PropertyKey, unknown>) {
    this.dialog = this.shadowRoot?.getElementById('camera-dialog') as MdDialog;
  }

  /** Open the dialog, pre-populated with current camera setup */
  show(camStream?: string, currentSetup?: CamSetup) {
    if (camStream) this.camStream = camStream;
    if (currentSetup) this.camSetup = currentSetup;
    this.populateFromSetup();
    this.fetchLocalCameras();
    this.dialog?.show();
  }

  private populateFromSetup() {
    const cam = this.camSetup?.camera;
    if (!cam) {
      this.activeTab = 'Local';
      return;
    }

    switch (cam.type) {
      case 'USB':
        this.activeTab = 'Local';
        this.selectedLocalId = cam.id ?? '';
        if (cam.width && cam.height) {
          this.selectedResolution = `${cam.width}x${cam.height}`;
        }
        break;
      case 'Demo':
        this.activeTab = 'Demo';
        break;
      case 'YouTube':
        this.activeTab = 'YouTube';
        this.youtubeUrl = cam.path ?? '';
        break;
      case 'Image':
        this.activeTab = 'Image';
        this.imageUrl = cam.path ?? '';
        break;
      case 'IP':
      default:
        // Legacy: YouTube URLs stored as IP type
        if (cam.path?.startsWith('https://youtu') || cam.path?.startsWith('https://www.youtube.com')) {
          this.activeTab = 'YouTube';
          this.youtubeUrl = cam.path ?? '';
        } else if (cam.path === 'demoVideo') {
          this.activeTab = 'Demo';
        } else {
          this.activeTab = 'IP';
          this.ipPath = cam.path ?? '';
          this.ipUsername = cam.username ?? '';
          this.ipPassword = cam.password ?? '';
        }
        break;
    }
  }

  private async fetchLocalCameras() {
    try {
      this.loading = true;
      const res = await fetch(`${this.basepath}/cameras`);
      if (res.ok) {
        this.localCameras = await res.json();
      }
    } catch (err) {
      console.error('Failed to fetch cameras:', err);
    } finally {
      this.loading = false;
    }
  }

  private get selectedCamera(): DeviceCameraInfo | undefined {
    return this.localCameras.find(c => c.id === this.selectedLocalId);
  }

  private get resolutionsForSelected(): { width: number; height: number }[] {
    return this.selectedCamera?.resolutions ?? [];
  }

  private onTabChange(ev: Event) {
    const tabs = ev.target as any;
    const index = tabs.activeTabIndex;
    const tabMap: SourceTab[] = ['Local', 'Demo', 'YouTube', 'IP', 'Image'];
    this.activeTab = tabMap[index] ?? 'Local';
  }

  private onLocalCameraChange(ev: Event) {
    const select = ev.target as MdOutlinedSelect;
    this.selectedLocalId = select.value;
    // Reset resolution when camera changes
    this.selectedResolution = '';
  }

  private onResolutionChange(ev: Event) {
    const select = ev.target as MdOutlinedSelect;
    this.selectedResolution = select.value;
  }

  private onDialogClose() {
    // No-op: apply is handled directly by the Apply button click handler
  }

  private onDialogCancel() {
    if (this.dialog) {
      this.dialog.returnValue = 'cancel';
    }
  }

  private onApplyClick() {
    this.applySelection();
    this.dialog?.close('apply');
  }

  private async applySelection() {
    let camera: Camera;
    let width: number | undefined;
    let height: number | undefined;

    switch (this.activeTab) {
      case 'Local': {
        const localCam = this.selectedCamera;
        if (!localCam) return;
        if (this.selectedResolution) {
          const [w, h] = this.selectedResolution.split('x').map(Number);
          width = w;
          height = h;
        }
        camera = {
          id: localCam.id,
          type: 'USB',
          name: localCam.name,
          path: localCam.path,
          camStream: this.camStream,
          width,
          height,
        };
        break;
      }
      case 'Demo':
        camera = {
          type: 'Demo',
          name: 'Demo Video',
          id: 'demoVideo',
          path: 'demoVideo',
          camStream: this.camStream,
        };
        break;
      case 'YouTube': {
        const url = this.youtubeUrl.trim();
        if (!url) return;
        camera = {
          type: 'YouTube',
          name: 'YouTube',
          id: 'youtube',
          path: url,
          camStream: this.camStream,
        };
        break;
      }
      case 'IP': {
        const path = this.ipPath.trim();
        if (!path) return;
        camera = {
          type: 'IP',
          name: 'IP Camera',
          id: 'ip',
          path,
          username: this.ipUsername || undefined,
          password: this.ipPassword || undefined,
          camStream: this.camStream,
        };
        break;
      }
      case 'Image': {
        const url = this.imageUrl.trim();
        if (!url) return;
        camera = {
          type: 'Image',
          name: 'Image',
          id: 'image',
          path: url,
          camStream: this.camStream,
        };
        break;
      }
      default:
        return;
    }

    // Post to backend
    try {
      await fetch(`${this.basepath}/cameras/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(camera),
      });
    } catch (err) {
      console.error('Failed to select camera:', err);
    }

    const camSetup: CamSetup = {
      camera,
      width: width ?? (Number(this.camSetup?.width) || 640),
      height: height ?? (Number(this.camSetup?.height) || 480),
    };

    this.camSetup = camSetup;

    this.dispatchEvent(
      new CustomEvent('camera-setup-changed', {
        detail: camSetup,
        bubbles: true,
        composed: true,
      }),
    );
  }

  private get tabIndex(): number {
    const map: Record<SourceTab, number> = { Local: 0, Demo: 1, YouTube: 2, IP: 3, Image: 4 };
    return map[this.activeTab] ?? 0;
  }

  private interfaceBadge(iface: string) {
    const labels: Record<string, string> = {
      usb: 'USB',
      csi: 'MIPI CSI',
      gmsl: 'GMSL',
      other: 'Other',
    };
    return html`<span class="interface-badge ${iface}">${labels[iface] ?? iface}</span>`;
  }

  private renderLocalTab() {
    return html`
      <div class="field-row">
        ${this.loading
          ? html`<p class="source-label">Detecting cameras…</p>`
          : this.localCameras.length === 0
            ? html`<p class="source-label">No local cameras detected</p>`
            : html`
                <div class="resolution-row">
                  <md-outlined-select
                    label="Camera"
                    .value=${this.selectedLocalId}
                    @change=${this.onLocalCameraChange}
                  >
                    ${repeat(
                      this.localCameras,
                      c => c.id,
                      c => html`
                        <md-select-option value="${c.id}">
                          <div slot="headline">${c.name} (${c.id}) ${this.interfaceBadge(c.interface ?? 'usb')}</div>
                        </md-select-option>
                      `,
                    )}
                  </md-outlined-select>
                </div>

                ${this.resolutionsForSelected.length > 0
                  ? html`
                      <md-outlined-select
                        label="Resolution"
                        .value=${this.selectedResolution}
                        @change=${this.onResolutionChange}
                      >
                        ${repeat(
                          this.resolutionsForSelected,
                          r => `${r.width}x${r.height}`,
                          r => html`
                            <md-select-option value="${r.width}x${r.height}">
                              <div slot="headline">${r.width} × ${r.height}</div>
                            </md-select-option>
                          `,
                        )}
                      </md-outlined-select>
                    `
                  : nothing}
              `}
      </div>
    `;
  }

  private renderDemoTab() {
    return html`
      <div class="demo-card">
        <span class="demo-icon">🎬</span>
        <p>Use the built-in demo video (luggage belt footage) for testing and evaluation.</p>
        <p class="hint">Click "Apply" to start the demo video stream.</p>
      </div>
    `;
  }

  private renderYouTubeTab() {
    return html`
      <div class="field-row">
        <p class="source-label">Enter a YouTube video or live stream URL:</p>
        <md-outlined-text-field
          label="YouTube URL"
          .value=${this.youtubeUrl}
          @input=${(ev: Event) => {
            this.youtubeUrl = (ev.target as MdOutlinedTextField).value;
          }}
          type="url"
          placeholder="https://www.youtube.com/watch?v=..."
        ></md-outlined-text-field>
        <p class="hint">Resolution will be auto-detected from the stream.</p>
      </div>
    `;
  }

  private renderIPTab() {
    return html`
      <div class="field-row">
        <p class="source-label">Enter the address for your camera stream. This can be an RTSP stream, HTTP stream, or other URL.</p>
        <md-outlined-text-field
          label="Camera URL"
          .value=${this.ipPath}
          @input=${(ev: Event) => {
            this.ipPath = (ev.target as MdOutlinedTextField).value;
          }}
          type="url"
          placeholder="rtsp://192.168.1.100:554/stream"
        ></md-outlined-text-field>
        <md-outlined-text-field
          label="Username (optional)"
          .value=${this.ipUsername}
          @input=${(ev: Event) => {
            this.ipUsername = (ev.target as MdOutlinedTextField).value;
          }}
          type="text"
        ></md-outlined-text-field>
        <md-outlined-text-field
          label="Password (optional)"
          .value=${this.ipPassword}
          @input=${(ev: Event) => {
            this.ipPassword = (ev.target as MdOutlinedTextField).value;
          }}
          type="password"
        ></md-outlined-text-field>
      </div>
    `;
  }

  private renderImageTab() {
    return html`
      <div class="field-row">
        <p class="source-label">Enter a URL to an image file (jpg, png, bmp, webp):</p>
        <md-outlined-text-field
          label="Image URL"
          .value=${this.imageUrl}
          @input=${(ev: Event) => {
            this.imageUrl = (ev.target as MdOutlinedTextField).value;
          }}
          type="url"
          placeholder="https://example.com/photo.jpg"
        ></md-outlined-text-field>
        <p class="hint">The image will be used as a static frame for detection. Inference runs once and re-runs automatically when you change settings (model, confidence, zones, etc.).</p>
      </div>
    `;
  }

  private get currentSourceLabel(): string {
    const cam = this.camSetup?.camera;
    if (!cam) return 'No camera configured';
    switch (cam.type) {
      case 'USB':
        return `Camera: ${cam.name ?? cam.id ?? 'Unknown'}`;
      case 'Demo':
        return 'Demo Video';
      case 'YouTube':
        return `YouTube: ${cam.path?.substring(0, 40) ?? ''}`;
      case 'IP':
        if (cam.path === 'demoVideo') return 'Demo Video';
        return `IP: ${cam.path?.substring(0, 40) ?? ''}`;
      case 'Image':
        return `Image: ${cam.path?.substring(0, 40) ?? ''}`;
      default:
        return cam.path ?? 'Unknown';
    }
  }

  render() {
    return html`
      <md-dialog
        id="camera-dialog"
        class="dialog"
        @cancel=${this.onDialogCancel}
        @close=${this.onDialogClose}
      >
        <div slot="headline">Camera Source Setup</div>
        <form slot="content" method="dialog">
          <md-tabs .activeTabIndex=${this.tabIndex} @change=${this.onTabChange}>
            <md-primary-tab>Local Camera</md-primary-tab>
            <md-primary-tab>Demo Video</md-primary-tab>
            <md-primary-tab>YouTube</md-primary-tab>
            <md-primary-tab>IP / RTSP</md-primary-tab>
            <md-primary-tab>Image</md-primary-tab>
          </md-tabs>

          <div class="tab-content">
            ${this.activeTab === 'Local' ? this.renderLocalTab() : nothing}
            ${this.activeTab === 'Demo' ? this.renderDemoTab() : nothing}
            ${this.activeTab === 'YouTube' ? this.renderYouTubeTab() : nothing}
            ${this.activeTab === 'IP' ? this.renderIPTab() : nothing}
            ${this.activeTab === 'Image' ? this.renderImageTab() : nothing}
          </div>
        </form>
        <div slot="actions">
          <md-text-button @click=${() => this.dialog?.close('cancel')}>Cancel</md-text-button>
          <md-text-button @click=${this.onApplyClick}>Apply</md-text-button>
        </div>
      </md-dialog>
    `;
  }
}
