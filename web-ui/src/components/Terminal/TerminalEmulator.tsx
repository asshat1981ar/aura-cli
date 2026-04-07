import { useEffect, useRef, useState, useCallback } from 'react'
import { Terminal, Play, Trash2, X, Copy, Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useTerminalStore } from '@/stores/terminalStore'

interface TerminalEmulatorProps {
  sessionId: string
  onClose?: () => void
}

const COMMON_COMMANDS = [
  'ls -la',
  'pwd',
  'git status',
  'git log --oneline -10',
  'python3 --version',
  'npm --version',
  'find . -name "*.py" | head -20',
  'cat README.md | head -30'
]

export function TerminalEmulator({ sessionId, onClose }: TerminalEmulatorProps) {
  const { 
    sessions, 
    isExecuting, 
    executeCommand, 
    clearSession,
    currentCwd
  } = useTerminalStore()
  
  const [input, setInput] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [copied, setCopied] = useState(false)
  
  const terminalRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  
  const session = sessions.find(s => s.id === sessionId)
  const commands = session?.commands || []
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [commands])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [sessionId])

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isExecuting) return
    
    const cmd = input.trim()
    setInput('')
    setShowSuggestions(false)
    setHistoryIndex(-1)
    
    await executeCommand(cmd, sessionId)
  }, [input, isExecuting, executeCommand, sessionId])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      const commandHistory = commands.map(c => c.command).reverse()
      if (historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1
        setHistoryIndex(newIndex)
        setInput(commandHistory[newIndex])
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1
        setHistoryIndex(newIndex)
        const commandHistory = commands.map(c => c.command).reverse()
        setInput(commandHistory[newIndex])
      } else if (historyIndex === 0) {
        setHistoryIndex(-1)
        setInput('')
      }
    } else if (e.key === 'Tab') {
      e.preventDefault()
      const matching = COMMON_COMMANDS.find(cmd => 
        cmd.startsWith(input.toLowerCase()) && cmd !== input
      )
      if (matching) {
        setInput(matching)
      }
    }
  }, [commands, historyIndex, input])

  const handleCopyOutput = async () => {
    const output = commands.map(c => `$ ${c.command}\n${c.output}`).join('\n\n')
    await navigator.clipboard.writeText(output)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getPrompt = () => {
    const cwd = session?.cwd || currentCwd
    const shortCwd = cwd.replace('/home/westonaaron675/aura-cli', '~')
    return `${shortCwd} $`
  }

  return (
    <div className="flex flex-col h-full bg-[#1e1e1e] text-[#d4d4d4] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#2d2d2d] border-b border-[#3e3e3e]">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#858585]" />
          <span className="text-sm font-medium">Terminal</span>
          <span className="text-xs text-[#858585] ml-2">{sessionId.slice(0, 8)}</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleCopyOutput}
            className="p-1.5 rounded hover:bg-[#3e3e3e] transition-colors"
            title="Copy all output"
          >
            {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
          </button>
          <button
            onClick={() => clearSession(sessionId)}
            className="p-1.5 rounded hover:bg-[#3e3e3e] transition-colors"
            title="Clear terminal"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="p-1.5 rounded hover:bg-[#3e3e3e] transition-colors"
              title="Close terminal"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Terminal Output */}
      <div 
        ref={terminalRef}
        className="flex-1 overflow-auto p-4 font-mono text-sm space-y-1"
        onClick={() => inputRef.current?.focus()}
      >
        {commands.length === 0 && (
          <div className="text-[#858585] italic">
            Welcome to AURA Terminal. Type commands below or select from suggestions.
          </div>
        )}
        
        {commands.map((cmd) => (
          <div key={cmd.id} className="space-y-1">
            <div className="flex items-start gap-2">
              <span className="text-[#858585]">{getPrompt()}</span>
              <span className="text-[#d4d4d4]">{cmd.command}</span>
            </div>
            <div className={cn(
              "pl-4 whitespace-pre-wrap",
              cmd.exit_code !== 0 ? "text-[#f48771]" : "text-[#d4d4d4]"
            )}>
              {cmd.output || <span className="italic text-[#858585]">No output</span>}
            </div>
          </div>
        ))}

        {/* Current Input */}
        <form onSubmit={handleSubmit} className="flex items-center gap-2 pt-2">
          <span className="text-[#858585]">{getPrompt()}</span>
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => {
                setInput(e.target.value)
                setShowSuggestions(e.target.value.length > 0)
              }}
              onKeyDown={handleKeyDown}
              className="w-full bg-transparent outline-none text-[#d4d4d4] font-mono"
              placeholder="Type command..."
              disabled={isExecuting}
              autoComplete="off"
              spellCheck={false}
            />
            
            {/* Autocomplete Suggestions */}
            {showSuggestions && input.length > 0 && (
              <div className="absolute bottom-full left-0 mb-1 bg-[#2d2d2d] border border-[#3e3e3e] rounded-lg shadow-lg min-w-[200px] z-10">
                {COMMON_COMMANDS
                  .filter(cmd => cmd.startsWith(input.toLowerCase()) && cmd !== input)
                  .slice(0, 5)
                  .map((cmd, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => {
                        setInput(cmd)
                        setShowSuggestions(false)
                        inputRef.current?.focus()
                      }}
                      className="w-full px-3 py-1.5 text-left text-sm hover:bg-[#3e3e3e] transition-colors first:rounded-t-lg last:rounded-b-lg"
                    >
                      {cmd}
                    </button>
                  ))}
              </div>
            )}
          </div>
          <button
            type="submit"
            disabled={isExecuting || !input.trim()}
            className="p-1.5 rounded hover:bg-[#3e3e3e] transition-colors disabled:opacity-50"
          >
            <Play className={cn("w-4 h-4", isExecuting && "animate-pulse")} />
          </button>
        </form>

        {/* Quick Commands */}
        {commands.length === 0 && (
          <div className="pt-4 space-y-2">
            <p className="text-xs text-[#858585] uppercase">Quick Commands</p>
            <div className="flex flex-wrap gap-2">
              {['ls -la', 'git status', 'pwd', 'python3 --version'].map((cmd) => (
                <button
                  key={cmd}
                  onClick={() => {
                    setInput(cmd)
                    inputRef.current?.focus()
                  }}
                  className="px-2 py-1 text-xs bg-[#2d2d2d] hover:bg-[#3e3e3e] rounded transition-colors"
                >
                  {cmd}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-1.5 bg-[#2d2d2d] border-t border-[#3e3e3e] text-xs">
        <div className="flex items-center gap-4">
          <span className="text-[#858585]">
            {isExecuting ? 'Running...' : 'Ready'}
          </span>
          <span className="text-[#858585]">
            {commands.length} commands
          </span>
        </div>
        <div className="flex items-center gap-2 text-[#858585]">
          <span>UTF-8</span>
          <span>bash</span>
        </div>
      </div>
    </div>
  )
}
