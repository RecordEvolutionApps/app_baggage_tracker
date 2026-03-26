/**
 * Barrel re-export — stream/camera handlers.
 *
 * The implementation is split across:
 *   shared.ts   – types, config, persistence helpers
 *   streams.ts  – stream lifecycle, CRUD, camera discovery
 */
export { handleListStreams, handleGetStream, handleCreateStream, handleUpdateStream, handleDeleteStream, handleStopStream, handleStartStream, getUSBCameras, getDeviceCameras, getStreamBackendStatus } from './streams.js'
