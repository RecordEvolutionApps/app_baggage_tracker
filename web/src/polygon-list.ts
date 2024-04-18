import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { PolygonManager, Polygon } from './polygon.js';
import { repeat } from 'lit/directives/repeat.js';
import { classMap } from 'lit/directives/class-map.js';

import '@material/web/list/list-item.js';
import '@material/web/list/list.js';
import '@material/web/icon/icon.js';
import { mainStyles } from './utils.js';

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

  static styles = [
    mainStyles,
    css`
      :host {
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

      md-list {
        width: 232px;
      }

      md-list-item:hover {
        cursor: pointer;
        background-color: #e1e1e1;
      }

      md-icon {
        --md-icon-size: 12px;
      }

      md-icon:hover {
        cursor: pointer;
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

    this.polygons = this.polygonManager?.getAll();
    this.requestUpdate();
  }

  selectPolygon(id: number) {
    return (ev: Event) => {
      ev.preventDefault();
      ev.stopPropagation();

      this.polygonManager?.select(id);
    };
  }

  deletePolygon(id: number) {
    return (ev: Event) => {
      ev.preventDefault();
      ev.stopPropagation();
      this.polygonManager?.remove(id);
    };
  }

  render() {
    return html` <md-list>
      ${repeat(
        this.polygons,
        c => c.id,
        c => {
          const selected = this.polygonManager?.selectedPolygon?.id === c.id;
          const classes = { selected: selected };

          return html`<md-list-item
            @click=${this.selectPolygon(c.id)}
            class=${classMap(classes)}
          >
            <div slot="headline">
              <div
                style="display: inline-block; margin-right: 4px; width: 10px; height: 10px; background-color: ${c.lineColor}; border: 1px solid black;"
              ></div>
              ${c.label}
            </div>
            <div slot="supporting-text">Points: ${c.points.length}</div>
            <md-icon slot="end" @click=${this.deletePolygon(c.id)}
              >delete</md-icon
            >
          </md-list-item>`;
        },
      )}
    </md-list>`;
  }
}
