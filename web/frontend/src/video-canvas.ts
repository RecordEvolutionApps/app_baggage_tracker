import { LitElement, html, css } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';

import '@material/web/elevation/elevation.js';
import './canvas-toolbox.js';
import './polygon-list.js';
import { mainStyles, CamSetup } from './utils.js';
@customElement('video-canvas')
export class VideoCanvas extends LitElement {
  canvasElement!: HTMLCanvasElement;
  polygonManager: PolygonManager;

  animationFrameId: number = -1;

  @property({ type: Object })
  declare video?: HTMLVideoElement;

  @property({ type: Number })
  declare width: number;

  @property({ type: Number })
  declare height: number;

  @property({ type: String})
  declare camStream: string

  @property({ type: Object })
  declare camSetup?: CamSetup;

  @state()
  declare loading: boolean;

  initialized = false;

  constructor() {
    super();
    this.polygonManager = new PolygonManager();
    this.width = 0;
    this.height = 0;
    this.camStream = 'frontCam';
    this.loading = true;

    this.getCursorPosition = this.getCursorPosition.bind(this);
  }

  firstUpdated() {
    this.canvasElement = this.shadowRoot?.getElementById(
      'canvas',
    ) as HTMLCanvasElement;

    // Set the camStream on the polygon manager now that it's available
    this.polygonManager.setCamStream(this.camStream);
  }

  drawPolygons(context: CanvasRenderingContext2D) {
    // Draw Polygons
    const { polygons } = this.polygonManager;
    for (const polygon of polygons) {
      const polygonPoints = polygon.getPoints();

      if (polygonPoints.length === 0) continue;

      // Set line width
      context.lineWidth = 2;
      context.strokeStyle = polygon.lineColor;
      context.fillStyle = polygon.fillColor;

      // Start drawing
      context.beginPath();

      // Move to the first point
      const firstPoint = polygonPoints[0];
      context.arc(firstPoint.x, firstPoint.y, 1, 0, 2 * Math.PI);
      context.moveTo(firstPoint.x, firstPoint.y);

      // Connect each point with a line
      for (var i = 1; i < polygonPoints.length; i++) {
        context.arc(polygonPoints[i].x, polygonPoints[i].y, 1, 0, 2 * Math.PI);
        context.lineTo(polygonPoints[i].x, polygonPoints[i].y);
      }

      context.stroke();

      if (polygon.committed) {
        context.closePath();
        context.fill();

        // const centroid = polygon.computeCenterPoint();
        // context.textAlign = 'center';
        // context.font = '32px serif';
        // context.fillStyle = 'black';
        // context.fillText(polygon.label, centroid.x, centroid.y);
      }
    }
  }

  step() {
    if (this.video?.paused || this.video?.ended) {
      return;
    }

    if (!this.video?.videoWidth || !this.video?.videoHeight) {
      this.animationFrameId = window.requestAnimationFrame(this.step.bind(this));
      return;
    }

    // Auto-resize canvas to match the actual video source resolution.
    // Compare against canvasElement dimensions (not this.width/height) because
    // the parent component may have already updated this.width/height via
    // property bindings before the canvas buffer was resized — causing a
    // false "no change" when comparing this.width vs video.videoWidth.
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (vw !== this.canvasElement.width || vh !== this.canvasElement.height) {
      this.canvasElement.width = vw;
      this.canvasElement.height = vh;
    }
    // Keep reactive properties in sync (used for polygon coordinate mapping)
    if (this.width !== vw) this.width = vw;
    if (this.height !== vh) this.height = vh;

    // First real frame — hide loading indicator
    if (this.loading) {
      this.loading = false;
    }

    // Draw Image
    const context = this.canvasElement?.getContext('2d', { alpha: false })!;
    context.drawImage(this.video!, 0, 0, vw, vh);

    this.drawPolygons(context);

    this.animationFrameId = window.requestAnimationFrame(this.step.bind(this));
  }

  getCursorPosition(event: any) {
    if (!this.canvasElement) return;

    const rect = this.canvasElement.getBoundingClientRect();
    const scaler = this.width / rect.width;

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const selectedPolygon = this.polygonManager.getSelected();
    selectedPolygon?.add(x * scaler , y * scaler);

    this.polygonManager.update();
  }

  update(changedProps: any) {
    super.update(changedProps);

    if (changedProps.has('camStream') && this.camStream) {
      this.polygonManager.setCamStream(this.camStream);
    }

    // If camSetup loaded and no source configured, stop showing spinner
    if (changedProps.has('camSetup') && this.camSetup && !this.camSetup.camera?.path) {
      this.loading = false;
    }

    if (!this.initialized && this.video && this.width && this.height) {
      this.canvasElement.addEventListener('mousedown', this.getCursorPosition);
      this.canvasElement.width = this.width;
      this.canvasElement.height = this.height;

      this.video.addEventListener('play', () => {
        this.animationFrameId = window.requestAnimationFrame(
          this.step.bind(this),
        );
      });

      // If video is already playing (we missed the play event), start immediately
      if (!this.video.paused && !this.video.ended) {
        this.animationFrameId = window.requestAnimationFrame(
          this.step.bind(this),
        );
      }

      this.initialized = true;
    }
  }

  static styles = [
    mainStyles,
    css`
      :host {
        width: 100%;
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      #canvas {
        max-width: 100%;
        max-height: 100%;
        background: #fff;
      }
      .container {
        height: 100%;
        flex: 1;
        display: flex;
        flex-direction: row;
        align-items: stretch;
        justify-content: space-between;
      }

      .sidebar {
        display: flex;
        flex-direction: column;
        gap: 16px;
        height: 100%;
        max-height: 100%;
        min-height: 0;
        max-width: 300px;
        min-width: 300px;
        padding: 10px;
        box-sizing: border-box;
        background: #e9eaf2;
        overflow-y: auto;
        overflow-x: hidden;
        flex-shrink: 0;
      }

      .surface {
        display: flex;
        position: relative;
        --md-elevation-level: 1;
        flex: 1;
        min-width: 0;
        min-height: 0;
        border-radius: 4px;
        align-items: center;
        justify-content: center;
        overflow: hidden;
      }

      .loading-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        z-index: 2;
        gap: 16px;
      }

      .loading-overlay[hidden] {
        display: none;
      }

      .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #d0d2db;
        border-top-color: #002e6a;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }

      .loading-text {
        font-family: sans-serif;
        font-size: 0.9rem;
        color: #5e5f61;
      }

      @media only screen and (max-width: 600px) {
        .container {
          flex-direction: column-reverse;
        }
        .sidebar {
          flex-direction: row;
          flex-wrap: wrap;
          justify-content: space-between;
          width: 100%;
          gap: 12px;
        }

      }


    `,
  ];

  render() {
    return html`<div class="container">
      <div class="sidebar">
        <canvas-toolbox
          .canvas=${this.canvasElement}
          .polygonManager=${this.polygonManager}
          .camSetup=${this.camSetup}
          .camStream=${this.camStream}
        ></canvas-toolbox>
        <polygon-list
          .canvas=${this.canvasElement}
          .polygonManager=${this.polygonManager}
        ></polygon-list>
      </div>
      <div class="surface">
        <div class="loading-overlay" ?hidden=${!this.loading}>
          <div class="spinner"></div>
          <span class="loading-text">Connecting to video stream…</span>
        </div>
        ${!this.loading && this.camSetup && !this.camSetup.camera?.path ? html`
          <div class="loading-overlay">
            <span class="loading-text">No video source configured</span>
          </div>
        ` : ''}
        <canvas id="canvas"></canvas>
      </div>
    </div>`;
  }
}
