/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig(({ mode }) => ({
  plugins: [
    react({
      // Optimize React runtime
      jsxRuntime: 'automatic',
      // Fast refresh for development
      fastRefresh: true,
    }),
    // Bundle analyzer for production
    mode === 'analyze' && visualizer({
      open: true,
      gzipSize: true,
      brotliSize: true,
    }),
  ].filter(Boolean),
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
    // Optimize dev server
    hmr: {
      overlay: false,
    },
  },
  
  build: {
    // Optimize chunk size
    chunkSizeWarningLimit: 500,
    
    // Code splitting strategy
    rollupOptions: {
      output: {
        // Manual chunk splitting for optimal caching
        manualChunks: {
          // React and core libraries
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          
          // State management
          'vendor-state': ['zustand'],
          
          // Charts (heavy)
          'vendor-charts': ['recharts'],
          
          // UI utilities
          'vendor-ui': ['lucide-react', 'clsx', 'tailwind-merge'],
          
          // Editor and heavy components
          'vendor-editor': ['@monaco-editor/react'],
        },
        
        // Optimize chunk file naming for caching
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name || ''
          if (info.endsWith('.css')) {
            return 'assets/styles/[name]-[hash][extname]'
          }
          if (info.endsWith('.png') || info.endsWith('.jpg') || info.endsWith('.svg')) {
            return 'assets/images/[name]-[hash][extname]'
          }
          return 'assets/[name]-[hash][extname]'
        },
      },
    },
    
    // Minification options
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info', 'console.debug'],
      },
      mangle: {
        safari10: true,
      },
    },
    
    // CSS optimization
    cssMinify: true,
    
    // Source maps for production debugging
    sourcemap: mode !== 'production',
    
    // Tree shaking
    treeshake: true,
  },
  
  // Optimize dependencies pre-bundling
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'zustand',
      'lucide-react',
      'recharts',
    ],
    exclude: ['@monaco-editor/react'],
  },
  
  // ESBuild options
  esbuild: {
    // Drop console in production
    drop: mode === 'production' ? ['console', 'debugger'] : [],
    // JSX transform
    jsx: 'automatic',
  },
  
  // Preview server options
  preview: {
    port: 4173,
  },
  
  // Test configuration
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
      ],
    },
  },
}))
