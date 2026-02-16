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
export { getStreamSetup, listStreams, createStream, deleteStream, selectCamera, getUSBCameras, getStreamBackendStatus } from './streams.js'
export { updateStreamModel, updateStreamSahi, updateStreamSmoothing, updateStreamConfidence, updateStreamFrameBuffer, updateStreamNmsIou, updateStreamSahiIou, updateStreamOverlapRatio, updateStreamClassList, updateStreamClassNames } from './settings.js'
