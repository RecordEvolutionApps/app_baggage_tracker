import { LitElement, html, css } from 'lit';
import { property, customElement } from 'lit/decorators.js';
import './polygon.js'

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
      const polygonPoints = this.polygon.getPoints()
      for (const { x, y } of polygonPoints) {
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(x, y);
        context.stroke();
      }
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
