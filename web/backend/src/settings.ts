import type { Context } from "elysia";
import { streamSetup, streamSetupFile, writeStreamSettings } from './shared.js'
import { killVideoStream, startVideoStream } from './streams.js'

// ── Stream settings updates ────────────────────────────────────────────────

export async function updateStreamModel(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, model } = body
    if (!camStream || !model) {
        ctx.set.status = 400
        return { error: 'camStream and model are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.model = model
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)

    // Restart the stream so it picks up the new model
    if (cam.path) {
        console.log(`Restarting stream ${camStream} for model change to ${model}`)
        await killVideoStream(cam.path, camStream)
        startVideoStream(cam, camStream)
    }

    return { status: 'ok', camStream, model }
}

export async function updateStreamSahi(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, useSahi } = body
    if (!camStream || typeof useSahi !== 'boolean') {
        ctx.set.status = 400
        return { error: 'camStream (string) and useSahi (boolean) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.useSahi = useSahi
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, useSahi }
}

export async function updateStreamSmoothing(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, useSmoothing } = body
    if (!camStream || typeof useSmoothing !== 'boolean') {
        ctx.set.status = 400
        return { error: 'camStream (string) and useSmoothing (boolean) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.useSmoothing = useSmoothing
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, useSmoothing }
}

export async function updateStreamConfidence(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, confidence } = body
    if (!camStream || typeof confidence !== 'number') {
        ctx.set.status = 400
        return { error: 'camStream (string) and confidence (number) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.confidence = confidence
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, confidence }
}

export async function updateStreamFrameBuffer(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, frameBuffer } = body
    if (!camStream || typeof frameBuffer !== 'number' || frameBuffer < 0) {
        ctx.set.status = 400
        return { error: 'camStream (string) and frameBuffer (non-negative number) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.frameBuffer = frameBuffer
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, frameBuffer }
}

export async function updateStreamIou(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, iou } = body
    if (!camStream || typeof iou !== 'number' || iou < 0 || iou > 1) {
        ctx.set.status = 400
        return { error: 'camStream (string) and iou (number 0-1) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.iou = iou
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, iou }
}

export async function updateStreamOverlapRatio(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, overlapRatio } = body
    if (!camStream || typeof overlapRatio !== 'number' || overlapRatio < 0.01 || overlapRatio > 0.5) {
        ctx.set.status = 400
        return { error: 'camStream (string) and overlapRatio (number 0.01-0.5) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.overlapRatio = overlapRatio
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, overlapRatio }
}

export async function updateStreamClassList(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, classList } = body
    if (!camStream || !Array.isArray(classList)) {
        ctx.set.status = 400
        return { error: 'camStream (string) and classList (number[]) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.classList = classList
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, classList }
}

export async function updateStreamClassNames(ctx: Context) {
    let body: any
    try {
        body = typeof ctx.body === 'string' ? JSON.parse(ctx.body) : ctx.body
    } catch {
        ctx.set.status = 400
        return { error: 'Invalid JSON' }
    }
    const { camStream, classNames } = body
    if (!camStream || !Array.isArray(classNames)) {
        ctx.set.status = 400
        return { error: 'camStream (string) and classNames (string[]) are required' }
    }
    const cam = streamSetup[camStream]
    if (!cam) {
        ctx.set.status = 404
        return { error: `Stream "${camStream}" not found` }
    }
    cam.classNames = classNames
    await Bun.write(streamSetupFile, JSON.stringify(streamSetup))
    await writeStreamSettings(camStream, cam)
    return { status: 'ok', camStream, classNames }
}
