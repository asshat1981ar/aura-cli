import * as React from 'react'
import { cn } from '@/lib/utils'

interface ResizablePanelGroupProps {
  direction: 'horizontal' | 'vertical'
  children: React.ReactNode
  className?: string
}

export function ResizablePanelGroup({ 
  direction, 
  children, 
  className 
}: ResizablePanelGroupProps) {
  return (
    <div 
      className={cn(
        'flex w-full h-full',
        direction === 'horizontal' ? 'flex-row' : 'flex-col',
        className
      )}
    >
      {children}
    </div>
  )
}

interface ResizablePanelProps {
  defaultSize?: number
  minSize?: number
  maxSize?: number
  children: React.ReactNode
  className?: string
}

export function ResizablePanel({ 
  defaultSize = 50, 
  children,
  className 
}: ResizablePanelProps) {
  return (
    <div 
      className={cn('relative', className)}
      style={{ flex: `1 1 ${defaultSize}%` }}
    >
      {children}
    </div>
  )
}

export function ResizableHandle() {
  return (
    <div className="w-1 bg-border cursor-col-resize hover:bg-primary/20 transition-colors flex-shrink-0" />
  )
}
