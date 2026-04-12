/**
 * CodeEditor.tsx — Sprint 7
 *
 * Monaco-powered code editor with:
 *  • File-tree sidebar (initially pre-populated with key project files)
 *  • Read-only mode by default (toggle with the toolbar button)
 *  • File content fetched from GET /api/files?path=<encoded-path>
 *    or GET /api/file/<encoded-path> (tries both)
 *
 * Dependencies: @monaco-editor/react (already in package.json)
 */

import { useState, useCallback, Suspense, lazy } from 'react'
import { Code2, Lock, Unlock, ChevronRight, ChevronDown, FileText, FolderOpen, Folder } from 'lucide-react'

// Lazy-load Monaco to keep the initial bundle small
const MonacoEditor = lazy(() => import('@monaco-editor/react'))

// ── Types ─────────────────────────────────────────────────────────────────────

interface FileNode {
  name: string
  path: string
  type: 'file' | 'dir'
  children?: FileNode[]
  language?: string
}

// ── Static file tree ──────────────────────────────────────────────────────────

const FILE_TREE: FileNode[] = [
  {
    name: 'aura_cli', path: 'aura_cli', type: 'dir',
    children: [
      { name: 'server.py',    path: 'aura_cli/server.py',    type: 'file', language: 'python' },
      { name: '__init__.py',  path: 'aura_cli/__init__.py',  type: 'file', language: 'python' },
    ],
  },
  {
    name: 'core', path: 'core', type: 'dir',
    children: [
      { name: 'logging_utils.py',   path: 'core/logging_utils.py',   type: 'file', language: 'python' },
      { name: 'mcp_contracts.py',   path: 'core/mcp_contracts.py',   type: 'file', language: 'python' },
      { name: 'mcp_registry.py',    path: 'core/mcp_registry.py',    type: 'file', language: 'python' },
    ],
  },
  {
    name: 'web-ui/src', path: 'web-ui/src', type: 'dir',
    children: [
      { name: 'App.tsx',   path: 'web-ui/src/App.tsx',   type: 'file', language: 'typescript' },
      { name: 'main.tsx',  path: 'web-ui/src/main.tsx',  type: 'file', language: 'typescript' },
    ],
  },
  { name: 'README.md',      path: 'README.md',      type: 'file', language: 'markdown' },
  { name: 'pyproject.toml', path: 'pyproject.toml', type: 'file', language: 'ini' },
]

function inferLanguage(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase()
  const MAP: Record<string, string> = {
    py: 'python', ts: 'typescript', tsx: 'typescript',
    js: 'javascript', jsx: 'javascript', json: 'json',
    md: 'markdown', yml: 'yaml', yaml: 'yaml',
    toml: 'ini', sh: 'shell', css: 'css', html: 'html',
  }
  return MAP[ext ?? ''] ?? 'plaintext'
}

// ── API ───────────────────────────────────────────────────────────────────────

async function fetchFileContent(filePath: string): Promise<string> {
  const encoded = encodeURIComponent(filePath)
  for (const url of [
    `/api/files?path=${encoded}`,
    `/api/file/${encoded}`,
  ]) {
    try {
      const res = await fetch(url)
      if (res.ok) return res.text()
    } catch { /* next */ }
  }
  throw new Error(`Could not load file: ${filePath}`)
}

// ── FileTreeNode ──────────────────────────────────────────────────────────────

interface FileTreeNodeProps {
  node: FileNode
  depth: number
  selectedPath: string | null
  onSelect: (node: FileNode) => void
}

function FileTreeNode({ node, depth, selectedPath, onSelect }: FileTreeNodeProps) {
  const [expanded, setExpanded] = useState(depth === 0)
  const isSelected = selectedPath === node.path

  if (node.type === 'dir') {
    return (
      <li>
        <button
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          aria-label={`${expanded ? 'Collapse' : 'Expand'} folder ${node.name}`}
          className={`w-full flex items-center gap-1.5 px-2 py-1 text-left text-sm rounded hover:bg-accent transition-colors`}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
        >
          {expanded
            ? <ChevronDown className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" aria-hidden="true" />
            : <ChevronRight className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" aria-hidden="true" />}
          {expanded
            ? <FolderOpen className="w-4 h-4 flex-shrink-0 text-yellow-500" aria-hidden="true" />
            : <Folder className="w-4 h-4 flex-shrink-0 text-yellow-500" aria-hidden="true" />}
          <span className="truncate">{node.name}</span>
        </button>
        {expanded && node.children && (
          <ul role="group" aria-label={`Contents of ${node.name}`}>
            {node.children.map((child) => (
              <FileTreeNode
                key={child.path}
                node={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelect={onSelect}
              />
            ))}
          </ul>
        )}
      </li>
    )
  }

  return (
    <li>
      <button
        onClick={() => onSelect(node)}
        aria-selected={isSelected}
        aria-label={`Open file ${node.name}`}
        className={`w-full flex items-center gap-1.5 px-2 py-1 text-left text-sm rounded transition-colors ${
          isSelected
            ? 'bg-primary/10 text-primary'
            : 'hover:bg-accent text-foreground'
        }`}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        <FileText className="w-4 h-4 flex-shrink-0 text-muted-foreground" aria-hidden="true" />
        <span className="truncate">{node.name}</span>
      </button>
    </li>
  )
}

// ── CodeEditorView ────────────────────────────────────────────────────────────

export function CodeEditorView() {
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null)
  const [content, setContent] = useState<string>('')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loadingFile, setLoadingFile] = useState(false)
  const [readOnly, setReadOnly] = useState(true)

  const handleSelect = useCallback(async (node: FileNode) => {
    if (node.type === 'dir') return
    setSelectedFile(node)
    setLoadingFile(true)
    setLoadError(null)
    setContent('')
    try {
      const text = await fetchFileContent(node.path)
      setContent(text)
    } catch (err) {
      setLoadError((err as Error).message)
      setContent(`// ${(err as Error).message}\n// File content could not be loaded from the server.\n// Ensure the AURA server is running and exposes GET /api/files?path=...`)
    } finally {
      setLoadingFile(false)
    }
  }, [])

  const language = selectedFile?.language ?? inferLanguage(selectedFile?.path ?? '')

  return (
    <main
      aria-labelledby="editor-heading"
      className="flex flex-col h-[calc(100vh-4rem)]"
    >
      {/* ── Toolbar ──────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-4 py-3 border-b bg-card shrink-0">
        <div className="flex items-center gap-2">
          <Code2 className="w-5 h-5 text-primary" aria-hidden="true" />
          <h1 id="editor-heading" className="text-lg font-semibold">Code Editor</h1>
          {selectedFile && (
            <>
              <span className="text-muted-foreground mx-1" aria-hidden="true">/</span>
              <span className="text-sm text-muted-foreground font-mono">{selectedFile.path}</span>
            </>
          )}
        </div>
        <button
          onClick={() => setReadOnly((r) => !r)}
          aria-label={readOnly ? 'Switch to editable mode' : 'Switch to read-only mode'}
          aria-pressed={!readOnly}
          className="flex items-center gap-2 px-3 py-1.5 text-sm border rounded-lg bg-card hover:bg-accent transition-colors"
        >
          {readOnly
            ? <Lock className="w-3.5 h-3.5" aria-hidden="true" />
            : <Unlock className="w-3.5 h-3.5" aria-hidden="true" />}
          {readOnly ? 'Read-only' : 'Editable'}
        </button>
      </header>

      {/* ── Body ─────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* File Tree */}
        <nav
          aria-label="File tree"
          className="w-56 border-r bg-card overflow-y-auto shrink-0 py-2"
        >
          <ul role="tree" aria-label="Project files">
            {FILE_TREE.map((node) => (
              <FileTreeNode
                key={node.path}
                node={node}
                depth={0}
                selectedPath={selectedFile?.path ?? null}
                onSelect={handleSelect}
              />
            ))}
          </ul>
        </nav>

        {/* Editor Pane */}
        <div className="flex-1 overflow-hidden relative">
          {loadError && (
            <div
              role="alert"
              className="absolute top-2 left-2 right-2 z-10 p-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400 text-xs"
            >
              ⚠️ {loadError}
            </div>
          )}

          {loadingFile && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/60 z-10">
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" aria-hidden="true" />
                Loading…
              </div>
            </div>
          )}

          {!selectedFile ? (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-2">
              <FileText className="w-12 h-12 opacity-20" aria-hidden="true" />
              <p className="text-sm">Select a file from the tree to view it</p>
            </div>
          ) : (
            <Suspense
              fallback={
                <div className="flex items-center justify-center h-full text-muted-foreground text-sm gap-2">
                  <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" aria-hidden="true" />
                  Loading editor…
                </div>
              }
            >
              <MonacoEditor
                height="100%"
                language={language}
                value={content}
                options={{
                  readOnly,
                  minimap: { enabled: false },
                  scrollBeyondLastLine: false,
                  fontSize: 13,
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                  lineNumbers: 'on',
                  wordWrap: 'on',
                  automaticLayout: true,
                  tabSize: 2,
                  renderLineHighlight: 'line',
                  scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
                }}
                aria-label={`Code editor for ${selectedFile.name}`}
              />
            </Suspense>
          )}
        </div>
      </div>
    </main>
  )
}

export default CodeEditorView
