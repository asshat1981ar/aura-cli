// Global types for ADK assets
declare global {
  const assets: {
    get(path: string): Promise<{
      url: string;
      path: string;
      name: string;
      size: number;
      mime: string;
      hash: string;
      createdAt: string;
      updatedAt: string;
      fileId: string;
    }>;
  };
}

export {};