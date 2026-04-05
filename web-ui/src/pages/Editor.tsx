import { FileTree } from '../components/Editor/FileTree'
import { CodeEditor } from '../components/Editor/CodeEditor'
import { useEditorStore } from '../stores/editorStore'
import { Code2 } from 'lucide-react'

export function Editor() {
  const { activeFilePath, openFile } = useEditorStore()
  
  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      <div className="flex items-center justify-between p-6 border-b">
        <div>
          <h2 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Code2 className="w-8 h-8" />
            Code Editor
          </h2>
          <p className="text-muted-foreground">
            Browse and edit files with AI assistance
          </p>
        </div>
      </div>
      
      <div className="flex-1 flex overflow-hidden">
        {/* File Tree Sidebar */}
        <div className="w-64 border-r">
          <FileTree 
            onFileSelect={openFile}
            selectedPath={activeFilePath}
          />
        </div>
        
        {/* Editor */}
        <div className="flex-1">
          <CodeEditor />
        </div>
      </div>
    </div>
  )
}
