import { IronFlock } from 'ironflock'
import { mkdirSync, readFileSync, writeFileSync, renameSync } from 'node:fs'
import { join } from 'node:path'

const ENV = Bun.env.ENV || ''
const STUB_DIR = '/data/stub'

// ── File-backed StubIronFlock for DEV mode ─────────────────────────────────

class StubIronFlock {
    private _subs = new Map<string, ((row: any) => void)[]>()

    private _filePath(table: string): string {
        return join(STUB_DIR, `${table}.json`)
    }

    private _readTable(table: string): any[] {
        try {
            return JSON.parse(readFileSync(this._filePath(table), 'utf-8'))
        } catch {
            return []
        }
    }

    private _writeTable(table: string, rows: any[]): void {
        mkdirSync(STUB_DIR, { recursive: true })
        const tmp = this._filePath(table) + '.tmp'
        writeFileSync(tmp, JSON.stringify(rows, null, 2))
        renameSync(tmp, this._filePath(table))
    }

    async appendToTable(table: string, args?: unknown[], kwargs?: Record<string, unknown>) {
        const data = ((args && args.length > 0 ? args[0] : {}) ?? {}) as Record<string, any>
        const rows = this._readTable(table)
        const maxId = rows.reduce((m: number, r: any) => Math.max(m, r._rowId ?? 0), 0)
        const row = { ...data, _rowId: maxId + 1, _publisher: 'ts', latest_flag: true }

        // Maintain latest_flag for streams table (keyed on stream_name)
        if (table === 'streams' && data.stream_name) {
            for (const r of rows) {
                if (r.stream_name === data.stream_name) r.latest_flag = false
            }
        }

        rows.push(row)
        this._writeTable(table, rows)

        // Fire subscription callbacks (skip own if exclude_me)
        const excludeMe = (kwargs as any)?.exclude_me === true
        const cbs = this._subs.get(table)
        if (cbs && !excludeMe) {
            const { _rowId: _, _publisher: __, ...clean } = row
            for (const cb of cbs) cb(clean)
        }
    }

    async getHistory(table: string, params?: Record<string, any>): Promise<any[]> {
        let rows = this._readTable(table)

        // Apply filterAnd
        const filters: any[] = params?.filterAnd ?? []
        for (const f of filters) {
            rows = rows.filter((r: any) => {
                const val = r[f.column]
                if (f.operator === '=') return val === f.value
                if (f.operator === '!=') return val !== f.value
                return true
            })
        }

        // Newest first
        rows.sort((a: any, b: any) => (b._rowId ?? 0) - (a._rowId ?? 0))

        // Strip internal fields
        rows = rows.map(({ _rowId, _publisher, ...rest }: any) => rest)

        if (params?.limit) rows = rows.slice(0, params.limit)
        return rows
    }

    async subscribeToTable(table: string, callback: (row: any) => void) {
        if (!this._subs.has(table)) this._subs.set(table, [])
        this._subs.get(table)!.push(callback)

        // Also poll the file for cross-process changes (rows published by Python)
        let lastId = this._readTable(table).reduce((m: number, r: any) => Math.max(m, r._rowId ?? 0), 0)
        setInterval(() => {
            const rows = this._readTable(table)
            const newRows = rows.filter((r: any) => (r._rowId ?? 0) > lastId && r._publisher !== 'ts')
            if (newRows.length > 0) {
                lastId = rows.reduce((m: number, r: any) => Math.max(m, r._rowId ?? 0), 0)
                for (const r of newRows) {
                    const { _rowId, _publisher, ...clean } = r
                    callback(clean)
                }
            }
        }, 1500)
    }

    start() {}
    stop() {}
}

// ── IronFlock config for the frontend SDK (IronFlockOptions from env) ──────

export function getIronFlockConfig() {
    return {
        serialNumber: Bun.env.DEVICE_SERIAL_NUMBER,
        deviceName: Bun.env.DEVICE_NAME,
        deviceKey: Bun.env.DEVICE_KEY,
        appName: Bun.env.APP_NAME,
        swarmKey: parseInt(Bun.env.SWARM_KEY || '0', 10),
        appKey: parseInt(Bun.env.APP_KEY || '0', 10),
        env: Bun.env.ENV,
        reswarmUrl: Bun.env.RESWARM_URL,
    }
}

let ironflock: IronFlock | StubIronFlock

if (ENV === 'DEV') {
    ironflock = new StubIronFlock()
} else {
    ironflock = new IronFlock()
    await ironflock.start()
}

export { ironflock }
