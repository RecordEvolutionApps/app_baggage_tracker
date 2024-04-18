import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';

import './canvas-toolbox.js';
import './polygon-list.js';
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

    setInterval(() => {
      const context = this.canvasElement?.getContext('2d', { alpha: false })!;

      context.fillStyle = 'white';
      context.fillRect(0, 0, this.width, this.height);

      this.drawPolygons(context);
    }, 1000 / 60);
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
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const selectedPolygon = this.polygonManager.getSelected();
    selectedPolygon?.add(x, y);
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

  static styles = css`
    .container {
      height: 100%;
      display: flex;
    }

    canvas {
      border: 1px solid black;
    }
  `;

  render() {
    return html`<div class="container">
      <canvas-toolbox
        .canvas=${this.canvasElement}
        .polygonManager=${this.polygonManager}
      ></canvas-toolbox>
      <canvas id="canvas"></canvas>
      <polygon-list
        .canvas=${this.canvasElement}
        .polygonManager=${this.polygonManager}
      ></polygon-list>
    </div>`;
  }
}
