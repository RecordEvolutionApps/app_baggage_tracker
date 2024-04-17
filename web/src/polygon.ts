let polygonID = 0

class Polygon {
    public id: number
    public label: string = ""
    public points: { x: number, y: number }[] = []

    constructor() {
        this.id = polygonID++
    }

    add(x: number, y: number) {
        this.points.push({ x, y })
    }

    undo() {
        this.points.pop()
    }

    getPoints() {
        return this.points
    }

    toJSON() {
        return {
            id: this.id,
            label: this.label,
            points: this.points
        };
    }

    setLabel(label: string) {
        this.label = label
    }

    setID(id: number) {
        this.id = id
    }

    setPoints(points: { x: number, y: number }[]) {
        this.points = points
    }

    static revive({ points, label }: {
        label: string,
        points: { x: number, y: number }[]
    }) {
        const polygon = new Polygon()

        polygon.setPoints(points)
        polygon.setLabel(label)

        return polygon
    }
}

class PolygonManager {

    polygons: Polygon[] = []

    create() {
        const polygon = new Polygon()
        this.polygons.push(polygon)

        return polygon
    }

    remove(id: number) {
        this.polygons.filter((p) => p.id !== id)
    }

    export() {
        return this.polygons.map((p) => p.toJSON())
    }

    import(data: {
        label: string,
        points: { x: number, y: number }[]
    }[]) {
        this.polygons = data.map((p) => Polygon.revive(p))
    }
}

