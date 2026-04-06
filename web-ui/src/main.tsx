import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { observeWebVitals, preloadCriticalResources } from './lib/performance.ts'

// Register service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/sw.js')
      .then((registration) => {
        console.log('SW registered:', registration.scope)
      })
      .catch((error) => {
        console.log('SW registration failed:', error)
      })
  })
}

// Preload critical resources
preloadCriticalResources([
  { href: '/assets/index.css', as: 'style' },
  { href: '/assets/index.js', as: 'script' },
])

// Monitor Web Vitals in development
// @ts-ignore - import.meta.env types
if (import.meta.env?.DEV) {
  observeWebVitals((vitals) => {
    console.log('[Web Vitals]', vitals)
  })
}

// Send Web Vitals to analytics in production
// @ts-ignore - import.meta.env types
if (import.meta.env?.PROD) {
  observeWebVitals((vitals) => {
    // Send to your analytics endpoint
    fetch('/api/analytics/vitals', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(vitals),
      keepalive: true,
    }).catch(() => {}) // Silent fail
  })
}

// Report Long Tasks
if ('PerformanceObserver' in window) {
  try {
    const longTaskObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration > 50) {
          console.warn(`[Long Task] ${entry.duration.toFixed(2)}ms`, entry)
        }
      }
    })
    longTaskObserver.observe({ entryTypes: ['longtask'] })
  } catch (e) {}
}

// React 18 Concurrent Features
const root = ReactDOM.createRoot(document.getElementById('root')!)

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

// Hot Module Replacement
// @ts-ignore - import.meta.hot types
if (import.meta.hot) {
  // @ts-ignore
  import.meta.hot.accept()
}
