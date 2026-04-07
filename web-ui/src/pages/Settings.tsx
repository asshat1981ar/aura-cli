import { useEffect, useState } from 'react'
import { 
  Save, 
  Check, 
  Moon, 
  Sun, 
  Monitor,
  Bell,
  Globe,
  Server,
  Puzzle,
  Shield,
  RefreshCw,
  TestTube,
  Trash2
} from 'lucide-react'
import { useThemeStore } from '../stores/themeStore'
import { useSettingsStore } from '../stores/settingsStore'
import { cn } from '@/lib/utils'

type SettingsTab = 'general' | 'notifications' | 'api' | 'mcp'

export function Settings() {
  const { isDark, toggleTheme } = useThemeStore()
  const { 
    settings, 
    mcpConfig, 
    saving, 
    isLoading,
    fetchSettings, 
    updateSettings,
    fetchMCPConfig,
    updateMCPConfig,
    testMCPConnection
  } = useSettingsStore()
  
  const [activeTab, setActiveTab] = useState<SettingsTab>('general')
  const [saved, setSaved] = useState(false)
  const [localSettings, setLocalSettings] = useState(settings)
  const [, setEditingMCP] = useState<string | null>(null)
  const [testingMCP, setTestingMCP] = useState<string | null>(null)

  useEffect(() => {
    fetchSettings()
    fetchMCPConfig()
  }, [fetchSettings, fetchMCPConfig])

  useEffect(() => {
    setLocalSettings(settings)
  }, [settings])

  const handleSave = async () => {
    if (!localSettings) return
    const success = await updateSettings(localSettings)
    if (success) {
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    }
  }

  const handleThemeChange = (theme: 'light' | 'dark' | 'system') => {
    setLocalSettings(prev => prev ? { ...prev, theme } : prev)
    if (theme === 'dark') {
      if (!isDark) toggleTheme()
    } else if (theme === 'light') {
      if (isDark) toggleTheme()
    }
  }

  const handleTestMCP = async (serverId: string) => {
    setTestingMCP(serverId)
    await testMCPConnection(serverId)
    setTestingMCP(null)
  }

  const handleDeleteMCP = (serverId: string) => {
    if (!mcpConfig) return
    const newConfig = { ...mcpConfig }
    delete newConfig.mcpServers[serverId]
    updateMCPConfig(newConfig)
  }

  const tabs = [
    { id: 'general' as SettingsTab, label: 'General', icon: Monitor },
    { id: 'notifications' as SettingsTab, label: 'Notifications', icon: Bell },
    { id: 'api' as SettingsTab, label: 'API & WebSocket', icon: Globe },
    { id: 'mcp' as SettingsTab, label: 'MCP Servers', icon: Puzzle },
  ]

  if (isLoading && !settings) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex">
      {/* Sidebar Tabs */}
      <div className="w-64 border-r bg-card p-4">
        <h2 className="text-xl font-bold mb-6">Settings</h2>
        <nav className="space-y-1">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                  activeTab === tab.id
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            )
          })}
        </nav>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-8">
        <div className="max-w-3xl">
          {/* General Settings */}
          {activeTab === 'general' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-1">General Settings</h3>
                <p className="text-muted-foreground">Configure appearance and behavior</p>
              </div>

              {/* Theme */}
              <section className="bg-card border rounded-xl p-6">
                <h4 className="font-medium mb-4 flex items-center gap-2">
                  <Sun className="w-4 h-4" />
                  Appearance
                </h4>
                <div className="grid grid-cols-3 gap-4">
                  {(['light', 'dark', 'system'] as const).map((theme) => (
                    <button
                      key={theme}
                      onClick={() => handleThemeChange(theme)}
                      className={cn(
                        "p-4 rounded-lg border text-center transition-all",
                        localSettings?.theme === theme
                          ? "border-primary bg-primary/5"
                          : "hover:bg-accent"
                      )}
                    >
                      {theme === 'light' && <Sun className="w-6 h-6 mx-auto mb-2" />}
                      {theme === 'dark' && <Moon className="w-6 h-6 mx-auto mb-2" />}
                      {theme === 'system' && <Monitor className="w-6 h-6 mx-auto mb-2" />}
                      <span className="text-sm font-medium capitalize">{theme}</span>
                    </button>
                  ))}
                </div>
              </section>

              {/* Features */}
              <section className="bg-card border rounded-xl p-6 space-y-4">
                <h4 className="font-medium">Features</h4>
                
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Auto Refresh</p>
                    <p className="text-sm text-muted-foreground">
                      Automatically refresh data every {localSettings?.refresh_interval || 5}s
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings?.features.auto_refresh}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      features: { ...prev.features, auto_refresh: e.target.checked }
                    } : prev)}
                    className="w-5 h-5"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Confirm Destructive Actions</p>
                    <p className="text-sm text-muted-foreground">
                      Show confirmation dialogs for delete operations
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings?.features.confirm_destructive}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      features: { ...prev.features, confirm_destructive: e.target.checked }
                    } : prev)}
                    className="w-5 h-5"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Show Telemetry</p>
                    <p className="text-sm text-muted-foreground">
                      Display telemetry data in dashboard
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings?.features.show_telemetry}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      features: { ...prev.features, show_telemetry: e.target.checked }
                    } : prev)}
                    className="w-5 h-5"
                  />
                </div>
              </section>
            </div>
          )}

          {/* Notifications */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-1">Notifications</h3>
                <p className="text-muted-foreground">Configure when to receive notifications</p>
              </div>

              <section className="bg-card border rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Goal Completion</p>
                    <p className="text-sm text-muted-foreground">
                      Notify when goals are completed
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings?.notifications.goal_completion}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      notifications: { ...prev.notifications, goal_completion: e.target.checked }
                    } : prev)}
                    className="w-5 h-5"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Agent Status Changes</p>
                    <p className="text-sm text-muted-foreground">
                      Notify when agents go online/offline
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings?.notifications.agent_status}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      notifications: { ...prev.notifications, agent_status: e.target.checked }
                    } : prev)}
                    className="w-5 h-5"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Errors</p>
                    <p className="text-sm text-muted-foreground">
                      Notify on system errors
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings?.notifications.errors}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      notifications: { ...prev.notifications, errors: e.target.checked }
                    } : prev)}
                    className="w-5 h-5"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Webhook Events</p>
                    <p className="text-sm text-muted-foreground">
                      Notify on incoming webhook events
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings?.notifications.webhook_events}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      notifications: { ...prev.notifications, webhook_events: e.target.checked }
                    } : prev)}
                    className="w-5 h-5"
                  />
                </div>
              </section>
            </div>
          )}

          {/* API Settings */}
          {activeTab === 'api' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-1">API & WebSocket</h3>
                <p className="text-muted-foreground">Configure connection settings</p>
              </div>

              <section className="bg-card border rounded-xl p-6 space-y-4">
                <div>
                  <label className="block font-medium mb-1">API Endpoint</label>
                  <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-muted-foreground" />
                    <input
                      type="text"
                      value={localSettings?.api.endpoint}
                      onChange={(e) => setLocalSettings(prev => prev ? {
                        ...prev,
                        api: { ...prev.api, endpoint: e.target.value }
                      } : prev)}
                      className="flex-1 px-4 py-2 border rounded-lg bg-background"
                    />
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    The base URL for AURA API requests
                  </p>
                </div>

                <div>
                  <label className="block font-medium mb-1">WebSocket Endpoint</label>
                  <div className="flex items-center gap-2">
                    <Server className="w-4 h-4 text-muted-foreground" />
                    <input
                      type="text"
                      value={localSettings?.api.websocket}
                      onChange={(e) => setLocalSettings(prev => prev ? {
                        ...prev,
                        api: { ...prev.api, websocket: e.target.value }
                      } : prev)}
                      className="flex-1 px-4 py-2 border rounded-lg bg-background"
                    />
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    The WebSocket URL for real-time updates
                  </p>
                </div>

                <div>
                  <label className="block font-medium mb-1">Request Timeout (seconds)</label>
                  <input
                    type="number"
                    value={localSettings?.api.timeout}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      api: { ...prev.api, timeout: parseInt(e.target.value) }
                    } : prev)}
                    min={5}
                    max={300}
                    className="w-32 px-4 py-2 border rounded-lg bg-background"
                  />
                </div>

                <div>
                  <label className="block font-medium mb-1">Refresh Interval (seconds)</label>
                  <input
                    type="number"
                    value={localSettings?.refresh_interval}
                    onChange={(e) => setLocalSettings(prev => prev ? {
                      ...prev,
                      refresh_interval: parseInt(e.target.value)
                    } : prev)}
                    min={1}
                    max={60}
                    className="w-32 px-4 py-2 border rounded-lg bg-background"
                  />
                </div>
              </section>
            </div>
          )}

          {/* MCP Servers */}
          {activeTab === 'mcp' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold mb-1">MCP Servers</h3>
                  <p className="text-muted-foreground">
                    Manage Model Context Protocol server configurations
                  </p>
                </div>
                <button
                  onClick={() => setEditingMCP('new')}
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
                >
                  <Puzzle className="w-4 h-4" />
                  Add Server
                </button>
              </div>

              <div className="space-y-4">
                {Object.entries(mcpConfig?.mcpServers || {}).map(([id, config]) => (
                  <div key={id} className="bg-card border rounded-xl p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <Puzzle className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-medium">{id}</h4>
                          <p className="text-sm text-muted-foreground">
                            {config.type.toUpperCase()} • {config.url || config.command}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleTestMCP(id)}
                          disabled={testingMCP === id}
                          className="p-2 text-muted-foreground hover:text-primary transition-colors disabled:opacity-50"
                          title="Test Connection"
                        >
                          <TestTube className={cn("w-4 h-4", testingMCP === id && "animate-spin")} />
                        </button>
                        <button
                          onClick={() => setEditingMCP(id)}
                          className="p-2 text-muted-foreground hover:text-primary transition-colors"
                          title="Edit"
                        >
                          <Shield className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDeleteMCP(id)}
                          className="p-2 text-muted-foreground hover:text-destructive transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}

                {Object.keys(mcpConfig?.mcpServers || {}).length === 0 && (
                  <div className="text-center py-12 text-muted-foreground border rounded-xl">
                    <Puzzle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No MCP servers configured</p>
                    <p className="text-sm">Add a server to enable tool integration</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Save Button */}
          {activeTab !== 'mcp' && (
            <div className="flex justify-end pt-6 border-t mt-6">
              <button
                onClick={handleSave}
                disabled={saving}
                className={cn(
                  "flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-colors",
                  saved
                    ? 'bg-green-600 text-white'
                    : 'bg-primary text-primary-foreground hover:bg-primary/90'
                )}
              >
                {saving ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : saved ? (
                  <>
                    <Check className="w-4 h-4" />
                    Saved!
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    Save Changes
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
