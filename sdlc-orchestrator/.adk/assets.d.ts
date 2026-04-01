// Auto-generated asset types
import { Asset } from '@botpress/runtime';

export type AssetPaths = never;

export interface AssetPathMap {

}

// Runtime asset access
declare global {
  const assets: {
    get<T extends AssetPaths>(path: T): Asset;
    list(): Asset[];
    getSyncStatus(): { 
      synced: boolean; 
      neverSynced: string[]; 
      stale: string[];
      upToDate: string[];
    };
  };
}
