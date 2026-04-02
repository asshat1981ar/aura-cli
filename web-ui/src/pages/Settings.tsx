import { useState } from 'react'
import { Save, Check } from 'lucide-react'
import { useThemeStore } from '../stores/themeStore'

export function Settings() {
  const { isDark, toggleTheme } = useThemeStore()
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
        <p className="text-muted-foreground">
          Configure your AURA dashboard preferences.
        </p>
      </div>

      <div className="space-y-6">
        <section className="bg-card border rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Appearance</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Dark Mode</p>
                <p className="text-sm text-muted-foreground">
                  Toggle between light and dark themes
                </p>
              </div>
              <button
                onClick={toggleTheme}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  isDark ? 'bg-primary' : 'bg-muted'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    isDark ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </section>

        <section className="bg-card border rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Notifications</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Goal Completion</p>
                <p className="text-sm text-muted-foreground">
                  Notify when goals are completed
                </p>
              </div>
              <input type="checkbox" defaultChecked className="w-5 h-5" />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Agent Status Changes</p>
                <p className="text-sm text-muted-foreground">
                  Notify when agents go online/offline
                </p>
              </div>
              <input type="checkbox" defaultChecked className="w-5 h-5" />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Errors</p>
                <p className="text-sm text-muted-foreground">
                  Notify on system errors
                </p>
              </div>
              <input type="checkbox" defaultChecked className="w-5 h-5" />
            </div>
          </div>
        </section>

        <section className="bg-card border rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">System</h3>
          <div className="space-y-4">
            <div>
              <label className="block font-medium mb-1">API Endpoint</label>
              <input
                type="text"
                defaultValue="http://localhost:8000"
                className="w-full px-4 py-2 border rounded-lg bg-background"
              />
              <p className="text-sm text-muted-foreground mt-1">
                The base URL for AURA API requests
              </p>
            </div>
            <div>
              <label className="block font-medium mb-1">
                WebSocket Endpoint
              </label>
              <input
                type="text"
                defaultValue="ws://localhost:8000/ws"
                className="w-full px-4 py-2 border rounded-lg bg-background"
              />
              <p className="text-sm text-muted-foreground mt-1">
                The WebSocket URL for real-time updates
              </p>
            </div>
            <div>
              <label className="block font-medium mb-1">
                Refresh Interval (seconds)
              </label>
              <input
                type="number"
                defaultValue={5}
                min={1}
                max={60}
                className="w-full px-4 py-2 border rounded-lg bg-background"
              />
            </div>
          </div>
        </section>

        <div className="flex justify-end">
          <button
            onClick={handleSave}
            className={`flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-colors ${
              saved
                ? 'bg-green-600 text-white'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
            }`}
          >
            {saved ? (
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
      </div>
    </div>
  )
}
