// Performance monitoring and optimization utilities

// Web Vitals monitoring
export interface WebVitals {
  LCP?: number // Largest Contentful Paint
  FID?: number // First Input Delay
  CLS?: number // Cumulative Layout Shift
  FCP?: number // First Contentful Paint
  TTFB?: number // Time to First Byte
}

export function observeWebVitals(callback: (vitals: WebVitals) => void): () => void {
  const vitals: WebVitals = {}
  const observers: PerformanceObserver[] = []

  // Largest Contentful Paint
  if ('PerformanceObserver' in window) {
    try {
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const lastEntry = entries[entries.length - 1]
        vitals.LCP = lastEntry.startTime
        callback({ ...vitals })
      })
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] })
      observers.push(lcpObserver)
    } catch (e) {}

    // First Input Delay
    try {
      const fidObserver = new PerformanceObserver((list) => {
        const entry = list.getEntries()[0] as PerformanceEventTiming
        vitals.FID = entry.processingStart - entry.startTime
        callback({ ...vitals })
      })
      fidObserver.observe({ entryTypes: ['first-input'] })
      observers.push(fidObserver)
    } catch (e) {}

    // Cumulative Layout Shift
    try {
      let clsValue = 0
      const clsObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!(entry as any).hadRecentInput) {
            clsValue += (entry as any).value
          }
        }
        vitals.CLS = clsValue
        callback({ ...vitals })
      })
      clsObserver.observe({ entryTypes: ['layout-shift'] })
      observers.push(clsObserver)
    } catch (e) {}

    // First Contentful Paint
    try {
      const fcpObserver = new PerformanceObserver((list) => {
        const entry = list.getEntries()[0]
        vitals.FCP = entry.startTime
        callback({ ...vitals })
        fcpObserver.disconnect()
      })
      fcpObserver.observe({ entryTypes: ['paint'] })
      observers.push(fcpObserver)
    } catch (e) {}
  }

  // Time to First Byte
  const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
  if (navigation) {
    vitals.TTFB = navigation.responseStart - navigation.startTime
    callback({ ...vitals })
  }

  // Cleanup function
  return () => {
    observers.forEach((observer) => observer.disconnect())
  }
}

// Measure component render time
export function measureRenderTime(componentName: string): () => void {
  const startTime = performance.now()
  
  return () => {
    const endTime = performance.now()
    const duration = endTime - startTime
    
    if (duration > 16) { // Log slow renders (>16ms)
      console.warn(`[Performance] ${componentName} rendered in ${duration.toFixed(2)}ms`)
    }
    
    // Send to analytics in production
    if (window.gtag) {
      window.gtag('event', 'timing_complete', {
        name: 'component_render',
        value: Math.round(duration),
        event_category: componentName,
      })
    }
  }
}

// Resource loading optimizer
export function preloadCriticalResources(resources: Array<{ href: string; as: string; type?: string }>) {
  resources.forEach((resource) => {
    const link = document.createElement('link')
    link.rel = 'preload'
    link.href = resource.href
    link.as = resource.as
    if (resource.type) {
      link.type = resource.type
    }
    document.head.appendChild(link)
  })
}

// Lazy load non-critical resources
export function lazyLoadResource(href: string, type: 'script' | 'style' | 'image'): Promise<void> {
  return new Promise((resolve, reject) => {
    let element: HTMLScriptElement | HTMLLinkElement | HTMLImageElement
    
    switch (type) {
      case 'script':
        element = document.createElement('script')
        element.src = href
        element.async = true
        break
      case 'style':
        element = document.createElement('link')
        element.rel = 'stylesheet'
        element.href = href
        break
      case 'image':
        element = new Image()
        element.src = href
        break
    }
    
    element.onload = () => resolve()
    element.onerror = () => reject(new Error(`Failed to load ${href}`))
    
    if (type === 'script' || type === 'style') {
      document.head.appendChild(element)
    }
  })
}

// Intersection Observer for lazy loading
export function createLazyLoader(
  callback: (entry: IntersectionObserverEntry) => void,
  options?: IntersectionObserverInit
): IntersectionObserver {
  return new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        callback(entry)
      }
    })
  }, {
    rootMargin: '50px',
    threshold: 0.01,
    ...options,
  })
}

// Memory usage monitoring
export function monitorMemoryUsage(callback: (usage: { usedJSHeapSize: number; totalJSHeapSize: number }) => void): () => void {
  if (!('memory' in performance)) {
    return () => {}
  }
  
  const interval = setInterval(() => {
    const memory = (performance as any).memory
    if (memory) {
      callback({
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
      })
    }
  }, 5000)
  
  return () => clearInterval(interval)
}

// Long task monitoring
export function observeLongTasks(callback: (duration: number) => void): () => void {
  if (!('PerformanceObserver' in window)) {
    return () => {}
  }
  
  try {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration > 50) { // Long task threshold
          callback(entry.duration)
        }
      }
    })
    
    observer.observe({ entryTypes: ['longtask'] })
    return () => observer.disconnect()
  } catch (e) {
    return () => {}
  }
}

// Animation frame scheduler for non-critical work
export function scheduleIdleWork(work: () => void, timeout?: number): number {
  if ('requestIdleCallback' in window) {
    return requestIdleCallback(work, { timeout })
  }
  return requestAnimationFrame(work)
}

export function cancelIdleWork(id: number) {
  if ('cancelIdleCallback' in window) {
    cancelIdleCallback(id)
  } else {
    cancelAnimationFrame(id)
  }
}

// Debounce function for performance
export function debounce<T extends (...args: any[]) => void>(
  func: T,
  wait: number,
  immediate: boolean = false
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  
  return function (this: any, ...args: Parameters<T>) {
    const later = () => {
      timeout = null
      if (!immediate) func.apply(this, args)
    }
    
    const callNow = immediate && !timeout
    
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(later, wait)
    
    if (callNow) func.apply(this, args)
  }
}

// Throttle function for performance
export function throttle<T extends (...args: any[]) => void>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false
  
  return function (this: any, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

// RAF throttle for scroll/resize handlers
export function rafThrottle<T extends (...args: any[]) => void>(
  callback: T
): (...args: Parameters<T>) => void {
  let rafId: number | null = null
  let lastArgs: Parameters<T>
  
  return function (this: any, ...args: Parameters<T>) {
    lastArgs = args
    
    if (rafId === null) {
      rafId = requestAnimationFrame(() => {
        rafId = null
        callback.apply(this, lastArgs)
      })
    }
  }
}

// Add gtag to window for TypeScript
declare global {
  interface Window {
    gtag?: (...args: any[]) => void
  }
}
