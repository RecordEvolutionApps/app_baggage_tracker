
import type { Context } from "elysia";
import { join } from "path";
import { mkdir, readdir } from "node:fs/promises";

type MaskData = {
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
}

/** Per-stream mask data keyed by camStream name */
const maskStore: Record<string, MaskData> = {};

const masksDir = Bun.env.MASKS_DIR || "/data/masks";

function maskPathFor(camStream: string): string {
    return join(masksDir, `${camStream}.json`);
}

export const saveMask = async (ctx: Context) => {
    try {
        const params = new URLSearchParams(ctx.request.url.split('?')[1]);
        const camStream = params.get('camStream');
        if (!camStream) {
            ctx.set.status = 400;
            return { error: 'camStream query parameter is required' };
        }

        // parse to check validity
        const state = JSON.parse(ctx.body as any) as MaskData;

        maskStore[camStream] = state;
        await mkdir(masksDir, { recursive: true });
        await Bun.write(maskPathFor(camStream), ctx.body as string);
    } catch (error) {
        console.error("Failed to save mask", error);
    }
}

/** Load all per-stream mask files from disk on startup */
const readAllMasks = async () => {
    try {
        // Migrate legacy single mask file to per-stream format
        const legacyMaskPath = Bun.env.MASK_PATH || "/data/mask.json";
        const legacyFile = Bun.file(legacyMaskPath);
        if (await legacyFile.exists()) {
            try {
                const legacyData = await legacyFile.json();
                if (legacyData?.polygons) {
                    console.log('Migrating legacy mask.json to per-stream format (frontCam)');
                    await mkdir(masksDir, { recursive: true });
                    await Bun.write(maskPathFor('frontCam'), JSON.stringify(legacyData));
                    // Remove legacy file after migration
                    const { unlink } = await import("node:fs/promises");
                    await unlink(legacyMaskPath);
                }
            } catch (err) {
                console.error('Failed to migrate legacy mask file', err);
            }
        }

        await mkdir(masksDir, { recursive: true });
        const files = await readdir(masksDir);
        for (const file of files) {
            if (!file.endsWith('.json')) continue;
            const camStream = file.replace(/\.json$/, '');
            const maskFile = Bun.file(join(masksDir, file));
            try {
                maskStore[camStream] = await maskFile.json();
            } catch (err) {
                console.error(`Failed to read mask for ${camStream}`, err);
            }
        }
    } catch (error) {
        console.error("Failed to read masks directory", error);
    }
}

export const getMask = (ctx: Context) => {
    const params = new URLSearchParams(ctx.request.url.split('?')[1]);
    const camStream = params.get('camStream');
    if (!camStream) {
        ctx.set.status = 400;
        return { error: 'camStream query parameter is required' };
    }
    return maskStore[camStream] ?? { polygons: [] };
}

/** Return all masks keyed by camStream (used when starting streams) */
export const getMaskForStream = (camStream: string): MaskData => {
    return maskStore[camStream] ?? { polygons: [] };
}

readAllMasks();