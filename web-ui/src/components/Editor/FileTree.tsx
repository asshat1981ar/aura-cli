import { useEffect } from 'react'
import { 
  Folder, 
  FolderOpen, 
  FileCode, 
  FileText, 
  ChevronRight, 
  ChevronDown,
  RefreshCw
} from 'lucide-react'
import { useEditorStore } from '../../stores/editorStore'

interface FileNode {
  name: string
  path: string
  type: 'file' | 'directory'
  children?: FileNode[]
  isExpanded?: boolean
}

interface FileTreeProps {
  onFileSelect?: (path: string) => void
  selectedPath?: string | null
}

export function FileTree({ onFileSelect, selectedPath }: FileTreeProps) {
  const { fileTree, isLoadingTree, loadFileTree, toggleDirectory, openFile } = useEditorStore()
  
  useEffect(() => {
    loadFileTree()
  }, [loadFileTree])
  
  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b flex items-center justify-between">
        <h3 className="font-semibold text-sm">Explorer</h3>
        <button
          onClick={() => loadFileTree()}
          disabled={isLoadingTree}
          className="p-1.5 hover:bg-accent rounded transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 ${isLoadingTree ? 'animate-spin' : ''}`} />
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2">
        {fileTree.map((node) => (
          <TreeNode
            key={node.path}
            node={node}
            depth={0}
            onToggle={toggleDirectory}
            onSelect={(path) => {
              openFile(path)
              onFileSelect?.(path)
            }}
            selectedPath={selectedPath}
          />
        ))}
        
        {fileTree.length === 0 && !isLoadingTree && (
          <div className="text-center text-muted-foreground text-sm py-8">
            No files found
          </div>
        )}
      </div>
    </div>
  )
}

interface TreeNodeProps {
  node: FileNode
  depth: number
  onToggle: (path: string) => void
  onSelect: (path: string) => void
  selectedPath?: string | null
}

function TreeNode({ node, depth, onToggle, onSelect, selectedPath }: TreeNodeProps) {
  const isSelected = node.path === selectedPath
  const isDirectory = node.type === 'directory'
  
  const getFileIcon = (name: string) => {
    const ext = name.split('.').pop()?.toLowerCase()
    switch (ext) {
      case 'py':
        return <FileCode className="w-4 h-4 text-blue-500" />
      case 'ts':
      case 'tsx':
      case 'js':
      case 'jsx':
        return <FileCode className="w-4 h-4 text-yellow-500" />
      case 'json':
        return <FileCode className="w-4 h-4 text-green-500" />
      case 'md':
        return <FileText className="w-4 h-4 text-gray-500" />
      default:
        return <FileText className="w-4 h-4 text-gray-400" />
    }
  }
  
  return (
    <div>
      <div
        className={`flex items-center gap-1 px-2 py-1 cursor-pointer text-sm rounded transition-colors ${
          isSelected ? 'bg-primary/10 text-primary' : 'hover:bg-accent'
        }`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={() => {
          if (isDirectory) {
            onToggle(node.path)
          } else {
            onSelect(node.path)
          }
        }}
      >
        {isDirectory ? (
          <>
            {node.isExpanded ? (
              <ChevronDown className="w-3 h-3 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-3 h-3 text-muted-foreground" />
            )}
            {node.isExpanded ? (
              <FolderOpen className="w-4 h-4 text-yellow-500" />
            ) : (
              <Folder className="w-4 h-4 text-yellow-500" />
            )}
          </>
        ) : (
          <>
            <span className="w-3" />
            {getFileIcon(node.name)}
          </>
        )}
        
        <span className="truncate ml-1">{node.name}</span>
      </div>
      
      {isDirectory && node.isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              onToggle={onToggle}
              onSelect={onSelect}
              selectedPath={selectedPath}
            />
          ))}
        </div>
      )}
    </div>
  )
}
