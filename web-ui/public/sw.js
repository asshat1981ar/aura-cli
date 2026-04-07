const CACHE_NAME = 'aura-cache-v1'
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/assets/index.css',
  '/assets/index.js',
]

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS)
    })
  )
  self.skipWaiting()
})

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    })
  )
  self.clients.claim()
})

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return
  }

  // API requests - network first with cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request))
    return
  }

  // Static assets - cache first
  if (isStaticAsset(url.pathname)) {
    event.respondWith(cacheFirst(request))
    return
  }

  // Everything else - stale while revalidate
  event.respondWith(staleWhileRevalidate(request))
})

// Cache strategies
async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME)
  const cached = await cache.match(request)
  
  if (cached) {
    return cached
  }
  
  const response = await fetch(request)
  cache.put(request, response.clone())
  return response
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE_NAME)
  
  try {
    const networkResponse = await fetch(request)
    cache.put(request, networkResponse.clone())
    return networkResponse
  } catch (error) {
    const cached = await cache.match(request)
    if (cached) {
      return cached
    }
    throw error
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME)
  const cached = await cache.match(request)
  
  // Fetch and update cache in background
  const fetchPromise = fetch(request)
    .then((networkResponse) => {
      cache.put(request, networkResponse.clone())
      return networkResponse
    })
    .catch(() => cached)
  
  // Return cached if available, otherwise wait for network
  return cached || fetchPromise
}

// Background sync for offline mutations
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-mutations') {
    event.waitUntil(syncMutations())
  }
})

async function syncMutations() {
  // Get pending mutations from IndexedDB and retry
  // This is a placeholder - implement with actual IndexedDB logic
  console.log('Syncing pending mutations...')
}

// Push notifications (optional)
self.addEventListener('push', (event) => {
  const data = event.data.json()
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: '/aura-icon.svg',
      badge: '/aura-badge.png',
      data: data.data,
    })
  )
})

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  event.waitUntil(
    clients.openWindow(event.notification.data?.url || '/')
  )
})

// Helper functions
function isStaticAsset(pathname) {
  return (
    pathname.startsWith('/assets/') ||
    pathname === '/' ||
    pathname === '/index.html'
  )
}

// Message handling from main thread
self.addEventListener('message', (event) => {
  if (event.data === 'skipWaiting') {
    self.skipWaiting()
  }
})
