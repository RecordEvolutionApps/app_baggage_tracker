import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';
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
  video?: HTMLVideoElement;

  @property({ type: Number })
  width: number = 0;

  @property({ type: Number })
  height: number = 0;

  @property({ type: String})
  camStream: string = 'frontCam'

  @property({ type: Object })
  camSetup?: CamSetup;

  initialized = false;

  constructor() {
    super();
    this.polygonManager = new PolygonManager();

    this.getCursorPosition = this.getCursorPosition.bind(this);
  }

  firstUpdated() {
    this.canvasElement = this.shadowRoot?.getElementById(
      'canvas',
    ) as HTMLCanvasElement;

    // // TODO: remove for production
    // setInterval(() => {
    //   const context = this.canvasElement?.getContext('2d', { alpha: false })!;

    //   context.fillStyle = 'white';
    //   context.fillRect(0, 0, this.width, this.height);

    //   this.drawPolygons(context);
    // }, 1000 / 30);
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

    // Draw Image
    const context = this.canvasElement?.getContext('2d', { alpha: false })!;
    context.drawImage(this.video!, 0, 0, this.width, this.height);

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

    if (!this.initialized && this.video && this.width && this.height) {
      this.canvasElement.addEventListener('mousedown', this.getCursorPosition);
      this.canvasElement.width = this.width;
      this.canvasElement.height = this.height;

      this.video.addEventListener('play', () => {
        this.animationFrameId = window.requestAnimationFrame(
          this.step.bind(this),
        );
      });

      this.initialized = true;
    }
  }

  static styles = [
    mainStyles,
    css`
      :host {
        width: 100%;
      }
      #canvas {
        width: 100%;
        background: #fff;
      }
      .container {
        height: 100%;
        display: flex;
        flex-direction: row;
        align-items: start;
        justify-content: space-between;
      }

      .sidebar {
        max-width: 280px;
        margin-right: 16px;
        height: 100%;
        padding: 10px;
        box-sizing: border-box;
        background: #e9eaf2;
        border-radius: 4px;
      }

      .surface {
        display: flex;
        position: relative;
        --md-elevation-level: 1;
        width: 100%;
        border-radius: 4px;
      }

      @media only screen and (max-width: 600px) {
        .container {
          flex-direction: column-reverse;
        }
        .sidebar {
          display: flex;
          max-width: fit-content;
          justify-content: space-between;
          width: 100%;
        }
        .surface {
          margin-bottom: 16px;
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
        <md-elevation></md-elevation>
        <canvas id="canvas"></canvas>
      </div>
    </div>`;
  }
}
