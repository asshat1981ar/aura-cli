import { useState, useEffect } from 'react'
import { Keyboard, X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Shortcut {
  keys: string[]
  description: string
}

const SHORTCUTS: { category: string; items: Shortcut[] }[] = [
  {
    category: 'Navigation',
    items: [
      { keys: ['Ctrl', 'K'], description: 'Open command palette' },
      { keys: ['G', 'D'], description: 'Go to Dashboard' },
      { keys: ['G', 'C'], description: 'Go to AI Chat' },
      { keys: ['G', 'E'], description: 'Go to Editor' },
      { keys: ['G', 'G'], description: 'Go to Goals' },
      { keys: ['G', 'A'], description: 'Go to Agents' },
      { keys: ['G', 'W'], description: 'Go to Workflows' },
      { keys: ['G', 'T'], description: 'Go to Terminal' },
      { keys: ['G', 'L'], description: 'Go to Logs' },
      { keys: ['G', ','], description: 'Go to Settings' },
    ]
  },
  {
    category: 'General',
    items: [
      { keys: ['Ctrl', '/'], description: 'Show keyboard shortcuts' },
      { keys: ['Esc'], description: 'Close modal / Cancel' },
      { keys: ['Ctrl', 'R'], description: 'Reload page' },
    ]
  },
  {
    category: 'Terminal',
    items: [
      { keys: ['↑'], description: 'Previous command' },
      { keys: ['↓'], description: 'Next command' },
      { keys: ['Tab'], description: 'Autocomplete' },
      { keys: ['Ctrl', 'C'], description: 'Copy output' },
    ]
  }
]

export function KeyboardShortcuts() {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault()
        setIsOpen(prev => !prev)
      }
      if (e.key === 'Escape') {
        setIsOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={() => setIsOpen(false)}
      />
      
      {/* Modal */}
      <div className="relative w-full max-w-2xl mx-4 bg-card rounded-xl shadow-2xl border overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <Keyboard className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold">Keyboard Shortcuts</h2>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="p-2 hover:bg-accent rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-auto">
          <div className="grid gap-8">
            {SHORTCUTS.map((section) => (
              <div key={section.category}>
                <h3 className="text-sm font-medium text-muted-foreground uppercase mb-4">
                  {section.category}
                </h3>
                <div className="space-y-3">
                  {section.items.map((shortcut, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between py-2"
                    >
                      <span className="text-sm">{shortcut.description}</span>
                      <div className="flex items-center gap-1">
                        {shortcut.keys.map((key, keyIdx) => (
                          <span key={keyIdx} className="flex items-center">
                            <kbd className="px-2 py-1 bg-muted rounded text-xs font-mono min-w-[24px] text-center">
                              {key}
                            </kbd>
                            {keyIdx < shortcut.keys.length - 1 && (
                              <span className="mx-1 text-muted-foreground">+</span>
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Footer */}
        <div className="px-6 py-3 border-t text-center text-sm text-muted-foreground">
          Press <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs">Esc</kbd> to close
        </div>
      </div>
    </div>
  )
}
