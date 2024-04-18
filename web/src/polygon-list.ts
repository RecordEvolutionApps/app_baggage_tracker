import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';
import { repeat } from 'lit/directives/repeat.js';
import { classMap } from 'lit/directives/class-map.js';

@customElement('polygon-list')
export class PolygonList extends LitElement {
  @property({ type: Object })
  canvas?: HTMLCanvasElement;

  @property({ type: Object })
  polygonManager?: PolygonManager;

  @state()
  polygons: Polygon[] = [];

  @state()
  selectedPolygon: Polygon | null = null;

  static styles = css`
    :host {
      width: 128px;
      border: 1px solid black;
      margin-left: 24px;
    }

    ul {
      list-style-type: none;
      margin-block-start: 0;
      margin-block-end: 0;
      margin-inline-start: 0px;
      margin-inline-end: 0px;
      padding-inline-start: 0;
    }

    .selected {
      background-color: lightgray;
    }
  `;

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

    this.polygons = this.polygonManager?.getAll();

    console.log(this.polygons);
  }

  selectPolygon(id: number) {
    return (ev: Event) => {
      ev.preventDefault();
      this.polygonManager?.select(id);
    };
  }

  deletePolygon(id: number) {
    return (ev: Event) => {
      ev.preventDefault();
      this.polygonManager?.remove(id);
    };
  }

  render() {
    return html`<div>
      <ul>
        ${repeat(
          this.polygons,
          c => c.id,
          c => {
            const selected = this.polygonManager?.selectedPolygon?.id === c.id;
            const classes = { selected: selected };

            return html`<li class=${classMap(classes)}>
              <a href="#" @click=${this.selectPolygon(c.id)}>${c.label}</a>
              <a href="#" @click=${this.deletePolygon(c.id)}>Delete</a>
            </li>`;
          },
        )}
      </ul>
    </div>`;
  }
}
