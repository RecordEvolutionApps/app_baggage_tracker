const ENV = (window as any).__ENV__ || ''

// ── Determine DEV mode ─────────────────────────────────────────────────────
// In production the ironflock runtime injects connection details.
// In DEV mode we use a stub that proxies through our Elysia REST backend.
const isDev = ENV === 'LOCAL' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'

// ── REST-backed stub for DEV mode ──────────────────────────────────────────

type FilterItem = { column: string; operator: string; value: any }
type HistoryParams = { limit?: number; filterAnd?: FilterItem[] }

class StubIronFlock {
    private _basepath = window.location.protocol + '//' + window.location.host
    private _subs = new Map<string, { callback: (row: any) => void; lastSnapshot: string }[]>()
    private _pollTimers = new Map<string, ReturnType<typeof setInterval>>()

    async appendToTable(table: string, data: Record<string, any> | Record<string, any>[], _options?: Record<string, any>) {
        // camera_hubs rows are written by the backend, not the browser; ignore them silently.
        if (table !== 'streams') return

        // The IronFlock SDK accepts an array of rows; handle both forms
        const rows = Array.isArray(data) ? data : [data]

        for (const row of rows) {
            const camStream = row.stream_name
            if (!camStream) continue

            // Check if stream already exists → PUT, otherwise POST
            const existing = await this._fetchJson(`${this._basepath}/streams/${encodeURIComponent(camStream)}`)

            if (row.deleted) {
                // Delete via REST
                await fetch(`${this._basepath}/streams/${encodeURIComponent(camStream)}`, { method: 'DELETE' })
                continue
            }

            // Parse out the config from the stream_config JSON string
            let config: Record<string, any> = {}
            if (row.stream_config) {
                config = typeof row.stream_config === 'string' ? JSON.parse(row.stream_config) : row.stream_config
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
    }

    async getHistory(table: string, params?: HistoryParams): Promise<any[]> {
        if (table === 'camera_hubs') {
            // In DEV mode the camera hub is the local server itself.
            // Build a synthetic row from the ironflock-config endpoint.
            const cfg = await this._fetchJson(`${this._basepath}/api/ironflock-config`)
            if (!cfg) return []
            return [{
                tsp: new Date().toISOString(),
                devname: cfg.deviceName ?? '',
                webpage: cfg.deviceKey ? `https://${cfg.deviceKey}-visionai-1100.app.ironflock.com` : '',
                devicelink: '',
                deleted: false,
                latest_flag: true,
            }]
        }

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
        if (table !== 'streams' && table !== 'camera_hubs') return

        // Initial snapshot — fire immediately for each existing row
        const initial = await this.getHistory(table)
        const entry = { callback, lastSnapshot: JSON.stringify(initial) }
        for (const row of initial) callback(row)

        if (!this._subs.has(table)) this._subs.set(table, [])
        this._subs.get(table)!.push(entry)

        // Start polling for this table if not already running
        if (!this._pollTimers.has(table)) {
            const timer = setInterval(async () => {
                const current = await this.getHistory(table)
                const currentStr = JSON.stringify(current)
                for (const sub of this._subs.get(table) ?? []) {
                    if (currentStr !== sub.lastSnapshot) {
                        const prev = JSON.parse(sub.lastSnapshot) as any[]
                        sub.lastSnapshot = currentStr
                        // Primary key differs by table
                        const pk = table === 'streams' ? 'stream_name' : 'devname'
                        // Fire callback for each changed/new row
                        for (const row of current) {
                            const old = prev.find((p: any) => p[pk] === row[pk])
                            if (!old || JSON.stringify(old) !== JSON.stringify(row)) {
                                sub.callback(row)
                            }
                        }
                        // Fire for deleted rows
                        for (const old of prev) {
                            if (!current.find((c: any) => c[pk] === old[pk])) {
                                sub.callback({ ...old, deleted: true })
                            }
                        }
                    }
                }
            }, 2000)
            this._pollTimers.set(table, timer)
        }
    }

    async callDeviceFunction(_deviceKey: string, topic: string, args?: unknown[], _kwargs?: Record<string, unknown>, options?: Record<string, any>): Promise<any> {
        const routeMap: Record<string, () => Promise<any>> = {
            getModels: () => this._fetchJson(`${this._basepath}/cameras/models`),
            getModelTags: () => this._fetchJson(`${this._basepath}/cameras/models/tags`),
            getCachedModels: () => this._fetchJson(`${this._basepath}/cameras/models/cache`),
            getModelClasses: () => {
                const modelId = args?.[0] as string
                return this._fetchJson(`${this._basepath}/cameras/models/${encodeURIComponent(modelId)}/classes`)
            },
            getBackendStatus: () => {
                const camStream = args?.[0] as string
                return this._fetchJson(`${this._basepath}/streams/${encodeURIComponent(camStream)}/backend`)
            },
            getModelStatus: () => {
                const modelId = args?.[0] as string
                return this._fetchJson(`${this._basepath}/cameras/models/${encodeURIComponent(modelId)}/status`)
            },
            deleteCachedModel: async () => {
                const modelId = args?.[0] as string
                const res = await fetch(`${this._basepath}/cameras/models/${encodeURIComponent(modelId)}/cache`, { method: 'DELETE' })
                return res.ok ? res.json() : null
            },
            clearAllCache: async () => {
                const res = await fetch(`${this._basepath}/cameras/models/cache`, { method: 'DELETE' })
                return res.ok ? res.json() : null
            },
            validateModel: async () => {
                const model = args?.[0] as string
                const res = await fetch(`${this._basepath}/cameras/models/validate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model }),
                })
                return res.ok ? res.json() : null
            },
            listCameras: () => this._fetchJson(`${this._basepath}/cameras`),
            prepareModel: () => this._sseToProgress(`${this._basepath}/cameras/models/prepare`, args?.[0] as string, options),
            buildTrt: () => this._sseToProgress(`${this._basepath}/cameras/models/build-trt`, args?.[0] as string, options),
            startStream: async () => {
                const camStream = args?.[0] as string
                const res = await fetch(`${this._basepath}/streams/${encodeURIComponent(camStream)}/start`, { method: 'POST' })
                return res.ok ? res.json() : null
            },
            stopStream: async () => {
                const camStream = args?.[0] as string
                const res = await fetch(`${this._basepath}/streams/${encodeURIComponent(camStream)}/stop`, { method: 'POST' })
                return res.ok ? res.json() : null
            },
            deleteStream: async () => {
                const camStream = args?.[0] as string
                const res = await fetch(`${this._basepath}/streams/${encodeURIComponent(camStream)}`, { method: 'DELETE' })
                return res.ok ? res.json() : null
            },
        }
        const handler = routeMap[topic]
        if (!handler) throw new Error(`StubIronFlock: unknown device function "${topic}"`)
        return handler()
    }

    /** Simulate progressive results by reading an SSE stream and calling on_progress. */
    private async _sseToProgress(url: string, model: string, options?: Record<string, any>): Promise<any> {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model }),
        })
        if (!res.ok || !res.body) return { status: 'error', progress: 0, message: `HTTP ${res.status}` }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let lastEvent: any = null

        while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.slice(6))
                        lastEvent = event
                        if (options?.on_progress) await options.on_progress(event)
                    } catch { /* skip */ }
                }
            }
        }
        return lastEvent
    }

    run() {}
    start() {}

    // ── Helpers ────────────────────────────────────────────────────────────

    /** Convert a StreamConfig from REST to a table row shape. */
    private _configToRow(config: any): Record<string, any> {
        return {
            stream_name: config.camStream ?? '',
            cam_path: config.source?.path ?? config.path ?? '',
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
