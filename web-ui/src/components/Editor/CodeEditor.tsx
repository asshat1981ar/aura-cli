import { useCallback, useEffect, useState } from 'react'
import Editor from '@monaco-editor/react'
import { 
  X, 
  Save, 
  Circle,
  FileCode
} from 'lucide-react'
import { useEditorStore } from '../../stores/editorStore'

interface CodeEditorProps {
  className?: string
}

export function CodeEditor({ className }: CodeEditorProps) {
  const {
    openFiles,
    activeFilePath,
    setActiveFile,
    closeFile,
    updateFileContent,
    saveFile,
  } = useEditorStore()
  
  const [isSaving, setIsSaving] = useState(false)
  
  const activeFile = openFiles.find((f) => f.path === activeFilePath)
  
  const handleEditorChange = useCallback((value: string | undefined) => {
    if (activeFilePath && value !== undefined) {
      updateFileContent(activeFilePath, value)
    }
  }, [activeFilePath, updateFileContent])
  
  const handleSave = async () => {
    if (!activeFilePath) return
    setIsSaving(true)
    await saveFile(activeFilePath)
    setIsSaving(false)
  }
  
  // Keyboard shortcut for save
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault()
        handleSave()
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [activeFilePath])
  
  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Tab Bar */}
      <div className="flex items-center border-b bg-muted/30 overflow-x-auto">
        {openFiles.map((file) => (
          <div
            key={file.path}
            onClick={() => setActiveFile(file.path)}
            className={`flex items-center gap-2 px-4 py-2 text-sm cursor-pointer border-r min-w-fit transition-colors ${
              file.path === activeFilePath
                ? 'bg-card border-t-2 border-t-primary'
                : 'hover:bg-accent text-muted-foreground'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            <span className="truncate max-w-[150px]">{file.path.split('/').pop()}</span>
            
            {file.isDirty && (
              <Circle className="w-2 h-2 fill-current" />
            )}
            
            <button
              onClick={(e) => {
                e.stopPropagation()
                closeFile(file.path)
              }}
              className="p-0.5 hover:bg-accent rounded opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
        
        {openFiles.length === 0 && (
          <div className="px-4 py-2 text-sm text-muted-foreground">
            No files open
          </div>
        )}
      </div>
      
      {/* Editor Area */}
      <div className="flex-1 relative">
        {activeFile ? (
          <>
            {/* Toolbar */}
            <div className="absolute top-2 right-2 z-10 flex items-center gap-2">
              {activeFile.isDirty && (
                <span className="text-xs text-amber-600 bg-amber-100 dark:bg-amber-900/30 px-2 py-1 rounded">
                  Unsaved
                </span>
              )}
              <button
                onClick={handleSave}
                disabled={isSaving || !activeFile.isDirty}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-primary text-primary-foreground text-sm rounded hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {isSaving ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-3.5 h-3.5" />
                    Save
                  </>
                )}
              </button>
            </div>
            
            {/* Monaco Editor */}
            <Editor
              height="100%"
              language={activeFile.language}
              value={activeFile.content}
              onChange={handleEditorChange}
              theme="vs-dark"
              options={{
                minimap: { enabled: true },
                fontSize: 14,
                wordWrap: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                smoothScrolling: true,
                cursorBlinking: 'smooth',
                formatOnPaste: true,
                formatOnType: true,
                tabSize: 2,
                insertSpaces: true,
              }}
              loading={
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  Loading editor...
                </div>
              }
            />
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <div className="text-center">
              <FileCode className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p className="text-lg font-medium">No file open</p>
              <p className="text-sm">Select a file from the explorer to start editing</p>
              <p className="text-xs mt-4 text-muted-foreground">
                Tip: Use Ctrl+S to save files
              </p>
            </div>
          </div>
        )}
      </div>
      
      {/* Status Bar */}
      {activeFile && (
        <div className="h-6 bg-primary text-primary-foreground text-xs flex items-center px-3 justify-between">
          <div className="flex items-center gap-4">
            <span>{activeFile.path}</span>
            <span className="opacity-70">{activeFile.language}</span>
          </div>
          <div className="flex items-center gap-4">
            {activeFile.isLoading ? (
              <span>Loading...</span>
            ) : (
              <>
                <span>{activeFile.content.split('\n').length} lines</span>
                <span>{activeFile.content.length} chars</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
