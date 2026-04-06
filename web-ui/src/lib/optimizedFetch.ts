// Request deduplication and caching
interface CacheEntry<T> {
  data: T
  timestamp: number
  ttl: number
}

class RequestCache {
  private cache = new Map<string, CacheEntry<any>>()
  private pendingRequests = new Map<string, Promise<any>>()

  get<T>(key: string): T | null {
    const entry = this.cache.get(key)
    if (!entry) return null
    
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key)
      return null
    }
    
    return entry.data
  }

  set<T>(key: string, data: T, ttl: number = 30000): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    })
  }

  getPendingRequest<T>(key: string): Promise<T> | null {
    return this.pendingRequests.get(key) || null
  }

  setPendingRequest<T>(key: string, promise: Promise<T>): void {
    this.pendingRequests.set(key, promise)
    promise.finally(() => {
      this.pendingRequests.delete(key)
    })
  }

  clear(key?: string): void {
    if (key) {
      this.cache.delete(key)
      this.pendingRequests.delete(key)
    } else {
      this.cache.clear()
      this.pendingRequests.clear()
    }
  }
}

export const requestCache = new RequestCache()

// Optimized fetch with deduplication and caching
interface OptimizedFetchOptions {
  cacheKey?: string
  cacheTtl?: number
  deduplicate?: boolean
}

export async function optimizedFetch<T>(
  url: string,
  options?: RequestInit & OptimizedFetchOptions
): Promise<T> {
  const { cacheKey, cacheTtl = 30000, deduplicate = true, ...fetchOptions } = options || {}
  
  const key = cacheKey || url

  // Check cache first
  const cached = requestCache.get<T>(key)
  if (cached) {
    return cached
  }

  // Check for pending request (deduplication)
  if (deduplicate) {
    const pending = requestCache.getPendingRequest<T>(key)
    if (pending) {
      return pending
    }
  }

  // Make the request
  const promise = fetch(url, fetchOptions)
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      requestCache.set(key, data, cacheTtl)
      return data
    })

  if (deduplicate) {
    requestCache.setPendingRequest(key, promise)
  }

  return promise
}

// Request batching for multiple API calls
interface BatchRequest {
  url: string
  options?: RequestInit
  resolve: (value: any) => void
  reject: (error: any) => void
}

class RequestBatcher {
  private batch: BatchRequest[] = []
  private timeout: NodeJS.Timeout | null = null
  private batchWindow = 10 // ms

  add(url: string, options?: RequestInit): Promise<any> {
    return new Promise((resolve, reject) => {
      this.batch.push({ url, options, resolve, reject })
      
      if (!this.timeout) {
        this.timeout = setTimeout(() => this.flush(), this.batchWindow)
      }
    })
  }

  private async flush() {
    const currentBatch = this.batch
    this.batch = []
    this.timeout = null

    if (currentBatch.length === 1) {
      // Single request, just execute it
      const req = currentBatch[0]
      try {
        const response = await fetch(req.url, req.options)
        const data = await response.json()
        req.resolve(data)
      } catch (error) {
        req.reject(error)
      }
    } else if (currentBatch.length > 1) {
      // Multiple requests, use batch endpoint if available
      try {
        const response = await fetch('/api/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            requests: currentBatch.map(r => ({ url: r.url, options: r.options }))
          })
        })
        
        if (response.ok) {
          const results = await response.json()
          currentBatch.forEach((req, index) => {
            const result = results[index]
            if (result.error) {
              req.reject(new Error(result.error))
            } else {
              req.resolve(result.data)
            }
          })
        } else {
          // Fallback to individual requests
          await Promise.all(
            currentBatch.map(async (req) => {
              try {
                const response = await fetch(req.url, req.options)
                const data = await response.json()
                req.resolve(data)
              } catch (error) {
                req.reject(error)
              }
            })
          )
        }
      } catch (error) {
        currentBatch.forEach(req => req.reject(error))
      }
    }
  }
}

export const requestBatcher = new RequestBatcher()

// Stale-while-revalidate pattern
export async function swrFetch<T>(
  url: string,
  options?: RequestInit & { cacheTtl?: number }
): Promise<{ data: T; isStale: boolean }> {
  const cacheKey = url
  const cached = requestCache.get<T>(cacheKey)
  
  if (cached) {
    // Return cached data immediately
    // Then revalidate in background
    fetch(url, options)
      .then(r => r.json())
      .then(data => {
        requestCache.set(cacheKey, data, options?.cacheTtl)
      })
      .catch(() => {}) // Ignore revalidation errors
    
    return { data: cached, isStale: true }
  }
  
  // No cache, must fetch
  const data = await optimizedFetch<T>(url, options)
  return { data, isStale: false }
}

// Prefetching utilities
export function prefetchUrls(urls: string[], delay: number = 100): () => void {
  const controllers: AbortController[] = []
  
  const timeout = setTimeout(() => {
    urls.forEach((url, index) => {
      setTimeout(() => {
        const controller = new AbortController()
        controllers.push(controller)
        
        fetch(url, { signal: controller.signal })
          .then(r => r.json())
          .then(data => requestCache.set(url, data))
          .catch(() => {}) // Ignore prefetch errors
      }, index * 50) // Stagger requests
    })
  }, delay)
  
  return () => {
    clearTimeout(timeout)
    controllers.forEach(c => c.abort())
  }
}
