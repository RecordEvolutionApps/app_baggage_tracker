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
        margin: 16px 0 0 24px;
        color: #47484c;
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
        background-color: #e8e8ee;
      }

      md-list {
        padding: 0;
        min-width: 170px;
        border-radius: 4px;
        --md-list-container-color: #f7faf9;
        --md-list-item-label-text-color: #005353;
        --md-list-item-supporting-text-color: #005353;
        --md-list-item-trailing-supporting-text-color: #005353;
      }

      md-list-item {
        border-radius: 4px;
      }

      md-list-item:hover {
        cursor: pointer;
        background-color: #e1e1e1;
      }

      md-icon {
        position: absolute;
        right: 2px;
      }

      md-icon:hover {
        cursor: pointer;
      }
      .header {
        display: flex;
        flex-direction: row;
        align-items: center;
      }
      .colorIcon {
        display: inline-block;
        margin-right: 4px;
        width: 14px;
        height: 14px;
        border-radius: 50%;
      }

      md-icon {
        color: #a1a1a3;
      }

      b {
        font-size: 15px;
        max-width: 90px;
        text-overflow: ellipsis;
        overflow: hidden;
        white-space: nowrap;
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
            <div slot="headline" class="header">
              <div
                class="colorIcon"
                style="background-color: ${c.lineColor};"
              ></div>
              <b title="${c.label}">${c.label}</b>
            </div>
            <!-- <div slot="supporting-text">Points: ${c.points.length}</div> -->
            <md-icon slot="end" @click=${this.deletePolygon(c.id)}
              >delete</md-icon
            >
          </md-list-item>`;
        },
      )}
    </md-list>`;
  }
}
