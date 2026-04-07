/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly DEV: boolean
  readonly PROD: boolean
  readonly MODE: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
  readonly hot?: {
    accept: () => void
    dispose: (callback: () => void) => void
  }
}
