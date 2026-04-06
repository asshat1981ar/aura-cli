import { useRef, useEffect, useState, useCallback } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface OptimizedListProps<T> {
  items: T[]
  renderItem: (item: T, index: number) => React.ReactNode
  itemHeight: number
  containerHeight: number
  className?: string
  overscan?: number
  onEndReached?: () => void
  hasMore?: boolean
  isLoading?: boolean
  emptyMessage?: string
}

export function OptimizedList<T>({
  items,
  renderItem,
  itemHeight,
  containerHeight,
  className,
  overscan = 5,
  onEndReached,
  hasMore = false,
  isLoading = false,
  emptyMessage = 'No items',
}: OptimizedListProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const endSentinelRef = useRef<HTMLDivElement>(null)

  // Calculate visible range
  const totalHeight = items.length * itemHeight
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
  const visibleCount = Math.ceil(containerHeight / itemHeight) + overscan * 2
  const endIndex = Math.min(items.length, startIndex + visibleCount)

  // Get visible items
  const visibleItems = items.slice(startIndex, endIndex)

  // Handle scroll
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleScroll = () => {
      setScrollTop(container.scrollTop)
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  // Intersection observer for infinite scroll
  useEffect(() => {
    if (!onEndReached || !hasMore || !endSentinelRef.current) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !isLoading) {
          onEndReached()
        }
      },
      { rootMargin: '100px' }
    )

    observer.observe(endSentinelRef.current)
    return () => observer.disconnect()
  }, [onEndReached, hasMore, isLoading])

  if (items.length === 0 && !isLoading) {
    return (
      <div className={cn("flex items-center justify-center p-8 text-muted-foreground", className)}>
        {emptyMessage}
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={cn("overflow-auto", className)}
      style={{ height: containerHeight }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        {visibleItems.map((item, index) => {
          const actualIndex = startIndex + index
          return (
            <div
              key={actualIndex}
              style={{
                position: 'absolute',
                top: actualIndex * itemHeight,
                height: itemHeight,
                left: 0,
                right: 0,
              }}
            >
              {renderItem(item, actualIndex)}
            </div>
          )
        })}
        
        {/* End sentinel for infinite scroll */}
        {hasMore && (
          <div
            ref={endSentinelRef}
            style={{
              position: 'absolute',
              top: totalHeight,
              height: 1,
              left: 0,
              right: 0,
            }}
          />
        )}
      </div>
      
      {/* Loading indicator */}
      {isLoading && (
        <div className="flex items-center justify-center p-4">
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
        </div>
      )}
    </div>
  )
}

// Windowed list for very large datasets
interface WindowedListProps<T> {
  items: T[]
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode
  itemHeight: number
  className?: string
  overscan?: number
}

export function WindowedList<T>({
  items,
  renderItem,
  itemHeight,
  className,
  overscan = 3,
}: WindowedListProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 20 })

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateVisibleRange = () => {
      const { scrollTop, clientHeight } = container
      const start = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
      const end = Math.min(
        items.length,
        Math.ceil((scrollTop + clientHeight) / itemHeight) + overscan
      )
      setVisibleRange({ start, end })
    }

    updateVisibleRange()
    container.addEventListener('scroll', updateVisibleRange, { passive: true })
    window.addEventListener('resize', updateVisibleRange)

    return () => {
      container.removeEventListener('scroll', updateVisibleRange)
      window.removeEventListener('resize', updateVisibleRange)
    }
  }, [items.length, itemHeight, overscan])

  const totalHeight = items.length * itemHeight
  const visibleItems = items.slice(visibleRange.start, visibleRange.end)

  return (
    <div
      ref={containerRef}
      className={cn("overflow-auto", className)}
      style={{ height: '100%' }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        {visibleItems.map((item, index) => {
          const actualIndex = visibleRange.start + index
          const style: React.CSSProperties = {
            position: 'absolute',
            top: actualIndex * itemHeight,
            height: itemHeight,
            left: 0,
            right: 0,
          }
          return (
            <div key={actualIndex} style={style}>
              {renderItem(item, actualIndex, style)}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Memoized list item wrapper
interface MemoizedItemProps<T> {
  item: T
  index: number
  renderItem: (item: T, index: number) => React.ReactNode
  isSelected?: boolean
  onSelect?: (item: T, index: number) => void
}

import { memo } from 'react'

export const MemoizedListItem = memo(function MemoizedListItem<T>({
  item,
  index,
  renderItem,
  isSelected,
  onSelect,
}: MemoizedItemProps<T>) {
  const handleClick = useCallback(() => {
    onSelect?.(item, index)
  }, [item, index, onSelect])

  return (
    <div
      onClick={handleClick}
      className={cn(
        "transition-colors",
        isSelected && "bg-primary/10",
        onSelect && "cursor-pointer hover:bg-accent"
      )}
    >
      {renderItem(item, index)}
    </div>
  )
}) as <T>(props: MemoizedItemProps<T>) => JSX.Element
