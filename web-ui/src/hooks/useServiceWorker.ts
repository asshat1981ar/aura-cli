import { useEffect, useState, useCallback } from 'react'

interface ServiceWorkerState {
  isRegistered: boolean
  isUpdateAvailable: boolean
  offlineReady: boolean
}

export function useServiceWorker(): ServiceWorkerState & { updateServiceWorker: () => void } {
  const [state, setState] = useState<ServiceWorkerState>({
    isRegistered: false,
    isUpdateAvailable: false,
    offlineReady: false,
  })

  useEffect(() => {
    if ('serviceWorker' in navigator) {
      registerServiceWorker()
    }
  }, [])

  const registerServiceWorker = async () => {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/',
      })

      setState((prev) => ({ ...prev, isRegistered: true }))

      // Check for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              setState((prev) => ({ ...prev, isUpdateAvailable: true }))
            }
          })
        }
      })

      // Listen for messages from service worker
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data === 'offline-ready') {
          setState((prev) => ({ ...prev, offlineReady: true }))
        }
      })
    } catch (error) {
      console.error('Service Worker registration failed:', error)
    }
  }

  const updateServiceWorker = useCallback(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then((registration) => {
        registration.waiting?.postMessage('skipWaiting')
        window.location.reload()
      })
    }
  }, [])

  return { ...state, updateServiceWorker }
}

// Hook for network status
export function useNetworkStatus(): { isOnline: boolean; wasOffline: boolean } {
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [wasOffline, setWasOffline] = useState(false)

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true)
    }

    const handleOffline = () => {
      setIsOnline(false)
      setWasOffline(true)
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  return { isOnline, wasOffline }
}

// Hook for background sync
export function useBackgroundSync() {
  const sync = useCallback(async (tag: string) => {
    if ('serviceWorker' in navigator && 'SyncManager' in window) {
      try {
        const registration = await navigator.serviceWorker.ready
        // @ts-ignore - sync is not in the standard type yet
        await registration.sync.register(tag)
        return true
      } catch (error) {
        console.error('Background sync registration failed:', error)
        return false
      }
    }
    return false
  }, [])

  return { sync }
}

// Hook for caching strategies
export function useCacheStrategy() {
  const clearCache = useCallback(async () => {
    if ('caches' in window) {
      const cacheNames = await caches.keys()
      await Promise.all(cacheNames.map((name) => caches.delete(name)))
    }
  }, [])

  const preloadCache = useCallback(async (urls: string[]) => {
    if ('caches' in window) {
      const cache = await caches.open('aura-preload-cache')
      await Promise.all(
        urls.map(async (url) => {
          try {
            const response = await fetch(url)
            await cache.put(url, response)
          } catch (error) {
            console.error(`Failed to preload ${url}:`, error)
          }
        })
      )
    }
  }, [])

  return { clearCache, preloadCache }
}
