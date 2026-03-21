const ENV = (window as any).__ENV__ || ''

// ── Determine DEV mode ─────────────────────────────────────────────────────
// In production the ironflock runtime injects connection details.
// In DEV mode we use a stub that proxies through our Elysia REST backend.
const isDev = ENV === 'DEV' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'

// ── REST-backed stub for DEV mode ──────────────────────────────────────────

type FilterItem = { column: string; operator: string; value: any }
type HistoryParams = { limit?: number; filterAnd?: FilterItem[] }

class StubIronFlock {
    private _basepath = window.location.protocol + '//' + window.location.host
    private _subs = new Map<string, { callback: (row: any) => void; lastSnapshot: string }[]>()
    private _pollTimers = new Map<string, ReturnType<typeof setInterval>>()

    async publishToTable(table: string, data: Record<string, any>, _options?: Record<string, any>) {
        if (table !== 'streams') return

        const camStream = data.stream_name
        if (!camStream) return

        // Check if stream already exists → PUT, otherwise POST
        const existing = await this._fetchJson(`${this._basepath}/streams/${encodeURIComponent(camStream)}`)

        if (data.deleted) {
            // Delete via REST
            await fetch(`${this._basepath}/streams/${encodeURIComponent(camStream)}`, { method: 'DELETE' })
            return
        }

        // Parse out the config from the stream_config JSON string
        let config: Record<string, any> = {}
        if (data.stream_config) {
            config = typeof data.stream_config === 'string' ? JSON.parse(data.stream_config) : data.stream_config
        }

        if (existing) {
            await fetch(`${this._basepath}/streams/${encodeURIComponent(camStream)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            })
        } else {
            await fetch(`${this._basepath}/streams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: config.name ?? camStream, camStream, ...config }),
            })
        }
    }

    async getHistory(table: string, params?: HistoryParams): Promise<any[]> {
        if (table !== 'streams') return []

        const filters = params?.filterAnd ?? []
        const streamNameFilter = filters.find(f => f.column === 'stream_name' && f.operator === '=')

        if (streamNameFilter) {
            // Single stream read
            const row = await this._fetchJson(
                `${this._basepath}/streams/${encodeURIComponent(streamNameFilter.value)}`
            )
            if (!row) return []
            return [this._configToRow(row)]
        }

        // List all streams
        const configs = await this._fetchJson(`${this._basepath}/streams`)
        if (!Array.isArray(configs)) return []
        return configs.map((c: any) => this._configToRow(c))
    }

    async subscribeToTable(table: string, callback: (row: any) => void) {
        if (table !== 'streams') return

        // Initial snapshot
        const initial = await this.getHistory('streams')
        const entry = { callback, lastSnapshot: JSON.stringify(initial) }

        if (!this._subs.has(table)) this._subs.set(table, [])
        this._subs.get(table)!.push(entry)

        // Start polling if not already
        if (!this._pollTimers.has(table)) {
            const timer = setInterval(async () => {
                const current = await this.getHistory('streams')
                const currentStr = JSON.stringify(current)
                for (const sub of this._subs.get(table) ?? []) {
                    if (currentStr !== sub.lastSnapshot) {
                        const prev = JSON.parse(sub.lastSnapshot) as any[]
                        sub.lastSnapshot = currentStr
                        // Fire callback for each changed/new row
                        for (const row of current) {
                            const old = prev.find((p: any) => p.stream_name === row.stream_name)
                            if (!old || JSON.stringify(old) !== JSON.stringify(row)) {
                                sub.callback(row)
                            }
                        }
                        // Fire for deleted rows
                        for (const old of prev) {
                            if (!current.find((c: any) => c.stream_name === old.stream_name)) {
                                sub.callback({ ...old, deleted: true })
                            }
                        }
                    }
                }
            }, 2000)
            this._pollTimers.set(table, timer)
        }
    }

    run() {}
    start() {}

    // ── Helpers ────────────────────────────────────────────────────────────

    /** Convert a StreamConfig from REST to a table row shape. */
    private _configToRow(config: any): Record<string, any> {
        return {
            stream_name: config.camStream ?? '',
            cam_path: config.path ?? '',
            stream_config: JSON.stringify(config),
            status: config.stopped ? 'stopped' : 'configured',
            deleted: false,
            latest_flag: true,
        }
    }

    private async _fetchJson(url: string): Promise<any | null> {
        try {
            const res = await fetch(url)
            if (!res.ok) return null
            return await res.json()
        } catch {
            return null
        }
    }
}

// ── Export singleton ────────────────────────────────────────────────────────

let ironflock: any
let deviceKey = ''

const ironflockReady: Promise<void> = (async () => {
    if (isDev) {
        ironflock = new StubIronFlock()
    } else {
        const { IronFlock } = await import('ironflock')
        ironflock = await IronFlock.fromServer('/api/ironflock-config')
        await ironflock.start()
    }
    try {
        const res = await fetch('/api/ironflock-config')
        if (res.ok) {
            const cfg = await res.json()
            deviceKey = cfg.deviceKey ?? ''
        }
    } catch { /* deviceKey stays empty */ }
})()

export { ironflock, ironflockReady, deviceKey }
