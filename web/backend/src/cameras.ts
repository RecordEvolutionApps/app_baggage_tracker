/**
 * Barrel re-export — all camera/stream/model/settings handlers.
 *
 * The implementation is split across:
 *   shared.ts   – types, config, shared mutable state
 *   models.ts   – model discovery & preparation proxy
 *   streams.ts  – stream lifecycle, CRUD, camera discovery
 *   settings.ts – per-stream settings updates
 */
export { getModels, getModelStatus, prepareModel, getModelClasses } from './models.js'
export { getStreamSetup, listStreams, createStream, deleteStream, selectCamera, getUSBCameras } from './streams.js'
export { updateStreamModel, updateStreamSahi, updateStreamFrameBuffer, updateStreamClassList, updateStreamClassNames } from './settings.js'
