import { PolygonState, getRandomColor, hexToTransparent } from './utils';

const basepath = window.location.protocol + '//' + window.location.host;
let polygonID = 0;

export class Polygon extends EventTarget {
  public id: number;
  public committed: boolean = false;
  public lineColor: string;
  public fillColor: string;
  public label: string = '';
  public points: { x: number; y: number }[] = [];

  constructor() {
    super();
    this.id = polygonID++;
    this.lineColor = getRandomColor();
    this.fillColor = hexToTransparent(this.lineColor, 0.3);

    this.label = String(this.id);
  }

  add(x: number, y: number) {
    if (this.committed) {
      throw new Error('Cannot add pixels to committed polygon');
    }

    this.points.push({ x, y });
  }

  undo() {
    if (this.committed) {
      this.committed = false;
    }

    this.points.pop();
  }

  getPoints() {
    return this.points;
  }

  get isCommitable() {
    return this.points.length >= 3;
  }

  computeCenterPoint() {
    let sumX = 0;
    let sumY = 0;
    let signedArea = 0;

    for (let i = 0; i < this.points.length; i++) {
      let vertex1 = this.points[i];
      let vertex2 = this.points[(i + 1) % this.points.length];

      let x0 = vertex1.x;
      let y0 = vertex1.y;
      let x1 = vertex2.x;
      let y1 = vertex2.y;

      let crossProduct = x0 * y1 - x1 * y0;
      signedArea += crossProduct;
      sumX += (x0 + x1) * crossProduct;
      sumY += (y0 + y1) * crossProduct;
    }

    signedArea *= 0.5;

    let centerX = sumX / (6 * signedArea);
    let centerY = sumY / (6 * signedArea);

    return { x: centerX, y: centerY };
  }

  toJSON() {
    return {
      id: this.id,
      label: this.label,
      committed: this.committed,
      lineColor: this.lineColor,
      fillColor: this.fillColor,
      points: this.points,
    };
  }

  commit() {
    if (this.isCommitable) {
      const { x, y } = this.points[0];
      this.add(x, y);

      this.committed = true;
    }
  }

  static revive({
    points,
    committed,
    lineColor,
    fillColor,
    label,
  }: {
    label: string;
    lineColor: string;
    fillColor: string;
    committed: boolean;
    points: { x: number; y: number }[];
  }) {

    const polygon = new Polygon();
    polygon.points = points;
    polygon.label = label;
    polygon.committed = committed;
    polygon.lineColor = lineColor;
    polygon.fillColor = fillColor;

    return polygon;
  }
}

export class PolygonManager extends EventTarget {
  polygons: Polygon[] = [];
  selectedPolygon: Polygon | null = null;

  constructor() {
    super()
    this.importFromRemote()
  }

  async importFromRemote() {
    try {
      const result = await fetch(`${basepath}/mask`, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
        },
      }).then(res => res.json());

      this.import(result)
    } catch (error) {
      console.error('Failed to load polygons from remote', error)
    }
  }

  updateRemote(data: PolygonState) {
    return fetch(`${basepath}/mask/save`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
      },
      body: JSON.stringify(data),
    });
  }

  create(label?: string) {
    const polygon = new Polygon();
    this.polygons.push(polygon);

    if (label) {
      polygon.label = label;
    }

    this.selectedPolygon = polygon;

    this.update();

    return polygon;
  }

  remove(id: number) {
    let selectedPolygonIndex;
    // Select previous polygon in list if we remove the one currently selected
    if (this.selectedPolygon?.id === id) {
      selectedPolygonIndex = this.polygons.findIndex(p => p.id === id);

      // Make sure to unset the select Polygon as it no longer exists
      this.selectedPolygon = null;
    }

    this.polygons = this.polygons.filter(p => p.id !== id);

    if (selectedPolygonIndex !== undefined) {
      const previousPolygon = this.polygons[selectedPolygonIndex - 1];
      if (previousPolygon) {
        this.select(previousPolygon.id);
      } else if (this.polygons.length > 0) {
        this.select(this.polygons[0].id);
      }
    }

    this.update();
  }

  update() {
    this.dispatchEvent(
      new CustomEvent('update', {
        detail: {
          polygons: this.polygons,
          selectedPolygon: this.selectedPolygon,
        },
      }),
    );

    // only save committed polygons
    const { polygons, selectedPolygonId } = this.export()
    this.updateRemote({ polygons: polygons.filter((p) => p.committed), selectedPolygonId })
  }

  getSelected() {
    return this.selectedPolygon;
  }

  getAll() {
    return this.polygons;
  }

  export(): PolygonState {
    return {
      selectedPolygonId: this.selectedPolygon?.id,
      polygons: this.polygons.map(p => p.toJSON())
    };
  }


  select(id: number) {
    const polygon = this.polygons.find(p => p.id === id);
    this.selectedPolygon = polygon ?? null;

    this.update();
  }

  import(
    data: PolygonState
  ) {
    this.polygons = data.polygons.map(p => Polygon.revive(p));

    this.update()
  }
}
