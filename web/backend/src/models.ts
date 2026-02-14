import type { Context } from "elysia";
import { VIDEO_API } from './shared.js'

// ── Fallback & cache ───────────────────────────────────────────────────────

const FALLBACK_MODELS = [
    { id: 'none', label: 'No Inference' },
    { id: 'rtmdet_tiny_8xb32-300e_coco', label: 'RTMDet Tiny' },
    { id: 'rtmdet_s_8xb32-300e_coco', label: 'RTMDet Small' },
    { id: 'rtmdet_m_8xb32-300e_coco', label: 'RTMDet Medium' },
]

let cachedModels: { 
    id: string; 
    label: string; 
    arch?: string; 
    dataset?: string;
    architecture?: string;
    task?: string;
    paper?: string;
    summary?: string;
    openVocab?: boolean;
    fileSize?: number;
}[] | null = null

async function fetchAvailableModels(): Promise<{ 
    id: string; 
    label: string; 
    arch?: string; 
    dataset?: string;
    architecture?: string;
    task?: string;
    paper?: string;
    summary?: string;
    openVocab?: boolean;
    fileSize?: number;
}[]> {
    if (cachedModels) return cachedModels
    try {
        const res = await fetch(`${VIDEO_API}/models`, { signal: AbortSignal.timeout(30000) })
        if (res.ok) {
            const models = await res.json()
            // Ensure openVocab flag is set even if the video API doesn't provide it
            const OV_KEYWORDS = ['grounding-dino', 'grounding_dino', 'glip', 'detic', 'yolo-world', 'yolo_world']
            for (const m of models) {
                if (m.openVocab === undefined) {
                    const id = (m.id || '').toLowerCase()
                    const arch = (m.architecture || '').toLowerCase()
                    m.openVocab = OV_KEYWORDS.some(kw => id.includes(kw) || arch.includes(kw))
                    if (m.openVocab && !m.label.includes('Open Vocab')) {
                        m.label = `${m.label} (Open Vocab)`
                    }
                }
            }
            cachedModels = [{ id: 'none', label: 'No Inference' }, ...models]
            console.log(`Fetched ${models.length} models from video API`)
            return cachedModels
        }
    } catch (err) {
        console.error('Failed to fetch models from video API, using fallback:', err)
    }
    return FALLBACK_MODELS
}

// ── Route handlers ─────────────────────────────────────────────────────────

export async function getModels(): Promise<any> {
    return fetchAvailableModels()
}

export async function getModelStatus(ctx: Context): Promise<any> {
    const url = new URL(ctx.request.url)
    const modelId = url.pathname.split('/cameras/models/')[1]?.split('/')[0]
    if (!modelId) {
        ctx.set.status = 400
        return { error: 'model id is required' }
    }
    try {
        const res = await fetch(`${VIDEO_API}/models/${encodeURIComponent(modelId)}/status`, { signal: AbortSignal.timeout(5000) })
        if (res.ok) return await res.json()
        ctx.set.status = res.status
        return { error: `Video API returned ${res.status}` }
    } catch (err) {
        console.error('Failed to fetch model status:', err)
        ctx.set.status = 502
        return { error: 'Could not reach video API' }
    }
}

export async function prepareModel(ctx: Context): Promise<Response> {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        return new Response(JSON.stringify({ error: 'Invalid JSON' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json' },
        })
    }
    const { model } = body
    if (!model) {
        return new Response(JSON.stringify({ error: 'model is required' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json' },
        })
    }
    try {
        const res = await fetch(`${VIDEO_API}/models/prepare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model }),
        })
        return new Response(res.body, {
            status: res.status,
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            },
        })
    } catch (err) {
        console.error('Failed to prepare model:', err)
        return new Response(JSON.stringify({ error: 'Could not reach video API' }), {
            status: 502,
            headers: { 'Content-Type': 'application/json' },
        })
    }
}

export async function getModelClasses(ctx: Context): Promise<any> {
    const url = new URL(ctx.request.url)
    const modelId = (ctx as any).params?.modelId ?? url.pathname.split('/cameras/models/')[1]?.split('/')[0]
    if (!modelId) {
        ctx.set.status = 400
        return { error: 'model id is required' }
    }
    try {
        const res = await fetch(`${VIDEO_API}/models/${encodeURIComponent(modelId)}/classes`, { signal: AbortSignal.timeout(30000) })
        if (res.ok) return await res.json()
        ctx.set.status = res.status
        return { error: `Video API returned ${res.status}` }
    } catch (err) {
        console.error('Failed to fetch model classes:', err)
        ctx.set.status = 502
        return { error: 'Could not reach video API' }
    }
}
