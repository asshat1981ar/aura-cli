
// Auto-generated assets metadata
import { Asset, initAssets } from '@botpress/runtime/runtime';

// Static asset metadata (populated at build time)
export const assetsMetadata: Record<string, Asset> = {};

// Local hashes from cache
export const localHashes: Record<string, string> = {};

// Initialize the assets runtime with metadata and local hashes
// The global object should be passed by the agent initialization code
export function initializeAssets(globalObj: any = globalThis) {
  initAssets(globalObj, assetsMetadata, localHashes);
}

// Auto-initialize if running in a supported environment
if (typeof globalThis !== 'undefined') {
  initializeAssets(globalThis);
} else if (typeof global !== 'undefined') {
  initializeAssets(global);
}
