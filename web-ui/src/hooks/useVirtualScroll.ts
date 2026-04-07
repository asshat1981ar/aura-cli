import { useState, useEffect, useRef, useCallback } from 'react'

interface UseVirtualScrollOptions {
  itemHeight: number
  overscan?: number
  containerHeight: number
}

interface VirtualScrollState {
  startIndex: number
  endIndex: number
  virtualItems: Array<{ index: number; style: React.CSSProperties }>
  totalHeight: number
  scrollTop: number
}

export function useVirtualScroll<T>(
  items: T[],
  options: UseVirtualScrollOptions
): VirtualScrollState & { containerRef: React.RefObject<HTMLDivElement> } {
  const { itemHeight, overscan = 5, containerHeight } = options
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleScroll = () => {
      setScrollTop(container.scrollTop)
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  const virtualItems = useCallback(() => {
    const totalHeight = items.length * itemHeight
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
    const visibleCount = Math.ceil(containerHeight / itemHeight) + overscan * 2
    const endIndex = Math.min(items.length, startIndex + visibleCount)

    const virtualItems = []
    for (let i = startIndex; i < endIndex; i++) {
      virtualItems.push({
        index: i,
        style: {
          position: 'absolute' as const,
          top: i * itemHeight,
          height: itemHeight,
          left: 0,
          right: 0,
        },
      })
    }

    return {
      startIndex,
      endIndex,
      virtualItems,
      totalHeight,
      scrollTop,
    }
  }, [items.length, itemHeight, scrollTop, containerHeight, overscan])

  return {
    ...virtualItems(),
    containerRef,
  }
}

// Hook for infinite scrolling
interface UseInfiniteScrollOptions {
  threshold?: number
  onLoadMore: () => Promise<void>
  hasMore: boolean
  isLoading: boolean
}

export function useInfiniteScroll({
  threshold = 100,
  onLoadMore,
  hasMore,
  isLoading,
}: UseInfiniteScrollOptions) {
  const observerRef = useRef<IntersectionObserver | null>(null)
  const loadMoreRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isLoading || !hasMore) return

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          onLoadMore()
        }
      },
      { rootMargin: `${threshold}px` }
    )

    if (loadMoreRef.current) {
      observerRef.current.observe(loadMoreRef.current)
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect()
      }
    }
  }, [isLoading, hasMore, onLoadMore, threshold])

  return { loadMoreRef }
}

// Hook for debounced scrolling
export function useDebouncedScroll(callback: () => void, delay: number = 100) {
  const timeoutRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    const handleScroll = () => {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(callback, delay)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', handleScroll)
      clearTimeout(timeoutRef.current)
    }
  }, [callback, delay])
}
