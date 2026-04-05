import { create } from 'zustand'

export interface FileNode {
  name: string
  path: string
  type: 'file' | 'directory'
  children?: FileNode[]
  isExpanded?: boolean
}

export interface OpenFile {
  path: string
  content: string
  language: string
  isDirty: boolean
  isLoading?: boolean
}

interface EditorState {
  // File tree
  rootPath: string
  fileTree: FileNode[]
  isLoadingTree: boolean
  
  // Open files
  openFiles: OpenFile[]
  activeFilePath: string | null
  
  // Actions
  loadFileTree: (path?: string) => Promise<void>
  toggleDirectory: (path: string) => void
  openFile: (path: string) => Promise<void>
  closeFile: (path: string) => void
  updateFileContent: (path: string, content: string) => void
  saveFile: (path: string) => Promise<void>
  setActiveFile: (path: string | null) => void
  getFileLanguage: (path: string) => string
}

const getLanguageFromPath = (path: string): string => {
  const ext = path.split('.').pop()?.toLowerCase()
  const languageMap: Record<string, string> = {
    'py': 'python',
    'ts': 'typescript',
    'tsx': 'typescript',
    'js': 'javascript',
    'jsx': 'javascript',
    'json': 'json',
    'md': 'markdown',
    'yml': 'yaml',
    'yaml': 'yaml',
    'html': 'html',
    'css': 'css',
    'scss': 'scss',
    'sh': 'shell',
    'bash': 'shell',
    'dockerfile': 'dockerfile',
    'rs': 'rust',
    'go': 'go',
    'java': 'java',
    'cpp': 'cpp',
    'c': 'c',
    'h': 'c',
  }
  return languageMap[ext || ''] || 'plaintext'
}

export const useEditorStore = create<EditorState>((set, get) => ({
  rootPath: '/home/westonaaron675/aura-cli',
  fileTree: [],
  isLoadingTree: false,
  openFiles: [],
  activeFilePath: null,

  loadFileTree: async (path) => {
    set({ isLoadingTree: true })
    try {
      const response = await fetch(`/api/files/tree?path=${encodeURIComponent(path || '')}`)
      if (!response.ok) throw new Error('Failed to load file tree')
      const tree = await response.json()
      set({ fileTree: tree, isLoadingTree: false })
    } catch (error) {
      console.error('Failed to load file tree:', error)
      set({ isLoadingTree: false })
    }
  },

  toggleDirectory: (path) => {
    set((state) => ({
      fileTree: toggleNode(state.fileTree, path),
    }))
  },

  openFile: async (path) => {
    const { openFiles } = get()
    
    // Check if already open
    const existingFile = openFiles.find((f) => f.path === path)
    if (existingFile) {
      set({ activeFilePath: path })
      return
    }

    // Add placeholder while loading
    const newFile: OpenFile = {
      path,
      content: '',
      language: getLanguageFromPath(path),
      isDirty: false,
      isLoading: true,
    }
    
    set((state) => ({
      openFiles: [...state.openFiles, newFile],
      activeFilePath: path,
    }))

    try {
      const response = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`)
      if (!response.ok) throw new Error('Failed to load file')
      const { content } = await response.json()
      
      set((state) => ({
        openFiles: state.openFiles.map((f) =>
          f.path === path ? { ...f, content, isLoading: false } : f
        ),
      }))
    } catch (error) {
      console.error('Failed to load file:', error)
      set((state) => ({
        openFiles: state.openFiles.filter((f) => f.path !== path),
      }))
    }
  },

  closeFile: (path) => {
    set((state) => {
      const newOpenFiles = state.openFiles.filter((f) => f.path !== path)
      const newActiveFile =
        state.activeFilePath === path
          ? newOpenFiles[newOpenFiles.length - 1]?.path || null
          : state.activeFilePath
      return {
        openFiles: newOpenFiles,
        activeFilePath: newActiveFile,
      }
    })
  },

  updateFileContent: (path, content) => {
    set((state) => ({
      openFiles: state.openFiles.map((f) =>
        f.path === path ? { ...f, content, isDirty: true } : f
      ),
    }))
  },

  saveFile: async (path) => {
    const { openFiles } = get()
    const file = openFiles.find((f) => f.path === path)
    if (!file) return

    try {
      const response = await fetch('/api/files/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, content: file.content }),
      })

      if (!response.ok) throw new Error('Failed to save file')

      set((state) => ({
        openFiles: state.openFiles.map((f) =>
          f.path === path ? { ...f, isDirty: false } : f
        ),
      }))
    } catch (error) {
      console.error('Failed to save file:', error)
    }
  },

  setActiveFile: (path) => set({ activeFilePath: path }),

  getFileLanguage: (path) => getLanguageFromPath(path),
}))

// Helper to toggle directory expansion in tree
function toggleNode(nodes: FileNode[], path: string): FileNode[] {
  return nodes.map((node) => {
    if (node.path === path) {
      return { ...node, isExpanded: !node.isExpanded }
    }
    if (node.children) {
      return { ...node, children: toggleNode(node.children, path) }
    }
    return node
  })
}
