
import type { Context } from "elysia";
import { streams } from "./cameras";
import { FileSink } from "bun";
import { dirname } from "path";
import { mkdir } from "node:fs/promises";

export let currentMaskData: {
    selectedPolygonId: number | undefined;
    polygons: {
        id: number;
        label: string;
        type: 'ZONE' | 'LINE';
        lineColor: string;
        fillColor: string;
        committed: boolean;
        points: { x: number; y: number }[];
    }[]
} = { polygons: [] };

const maskPath = Bun.env.MASK_PATH || "/data/mask.json";

export const updateStreamsWithMask = () => {
    for (const stream of streams.values()) {
        if (stream.stdin) {
            const stdin = stream.stdin as FileSink

            stdin.write(JSON.stringify(currentMaskData) + "\n")
            stdin.flush()
        }
    }
}

export const saveMask = async (ctx: Context) => {
    try {
        // parse to check validity
        const state = JSON.parse(ctx.body as any)

        currentMaskData = state
        await mkdir(dirname(maskPath), { recursive: true });
        await Bun.write(maskPath, ctx.body)

        updateStreamsWithMask()
    } catch (error) {
        console.error("Failed to save mask", error)
    }
}

export const readMask = async () => {
    const maskFile = Bun.file(maskPath)

    try {
        const exists = await maskFile.exists()
        if (exists) {
            currentMaskData = await maskFile.json()
        }
    } catch (error) {
        console.error("Failed to read mask", error)
    }
}

export const getMask = () => {
    return currentMaskData
}

readMask()