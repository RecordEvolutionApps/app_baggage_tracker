import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js'

@customElement('video-canvas')
export class VideoCanvas extends LitElement {
  canvasElement!: HTMLCanvasElement;
  polygonManager: PolygonManager
  polygon: Polygon;

  animationFrameId: number = -1;

  @property({ type: Object })
  video?: HTMLVideoElement;

  @property({ type: Number })
  width: number = 0

  @property({ type: Number })
  height: number = 0

  initialized = false

  constructor() {
    super()
    this.polygonManager = new PolygonManager()
    this.polygon = this.polygonManager.create()

    this.getCursorPosition = this.getCursorPosition.bind(this)
  }

  firstUpdated() {
    this.canvasElement = this.shadowRoot?.getElementById('canvas') as HTMLCanvasElement
  }

  step() {
    if (this.video?.paused || this.video?.ended) {
      return;
    }

    // Draw Image
    const context = this.canvasElement?.getContext('2d', { alpha: false })!
    context.drawImage(this.video!, 0, 0, this.width, this.height);

    // Draw Polygons
    const { polygons } = this.polygonManager
    for (const polygon of polygons) {
      const polygonPoints = polygon.getPoints()

      if (polygonPoints.length < 2) continue

      // Set line width
      context.lineWidth = 10;

      // Start drawing
      context.beginPath();

      // Move to the first point
      context.moveTo(polygonPoints[0].x, polygonPoints[0].y);

      // Connect each point with a line
      for (var i = 1; i < polygonPoints.length; i++) {
        context.lineTo(polygonPoints[i].x, polygonPoints[i].y);
      }

      // Draw the line
      context.stroke();
    }

    this.animationFrameId = window.requestAnimationFrame(this.step.bind(this))
  }

  getCursorPosition(event: any) {
    if (!this.canvasElement) return

    const rect = this.canvasElement.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    this.polygon.add(x, y)
  }

  update(changedProps: any) {
    super.update(changedProps)

    if (!this.initialized && this.video && this.width && this.height) {
      this.canvasElement.addEventListener("mousedown", this.getCursorPosition)
      this.canvasElement.width = this.width
      this.canvasElement.height = this.height

      this.video.addEventListener("play", () => {
        this.animationFrameId = window.requestAnimationFrame(this.step.bind(this))
      })

      this.initialized = true
    }
  }

  render() {
    return html`
      <canvas id="canvas"></canvas>
    `;
  }
}
