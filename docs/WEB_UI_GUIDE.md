# AURA Web UI User Guide

## Getting Started

### Accessing the Dashboard
Open your browser and navigate to:
```
http://localhost:8000
```

Default login credentials (for development):
- Username: `admin`
- Password: (configured in your environment)

## Dashboard Overview

### Main Navigation
The sidebar provides access to all major features:

| Page | Description | Use Case |
|------|-------------|----------|
| **Dashboard** | System overview with key metrics | Quick status check |
| **AI Chat** | Multi-agent chat interface | Get help from AI agents |
| **Editor** | Code editor with file tree | Browse and edit code |
| **Goals** | Goal queue management | Track work items |
| **Agents** | Agent monitoring and control | Manage AI agents |
| **SADD** | Design decomposition | Break down complex tasks |
| **Workflows** | n8n workflow visualizer | View automation workflows |
| **MCP Tools** | MCP server tools | Execute AI tools |
| **Terminal** | Integrated shell | Run commands directly |
| **Coverage** | Test coverage dashboard | View code quality |
| **Logs** | System logs & telemetry | Debug and monitor |
| **Settings** | Configuration panel | Customize the UI |

## Feature Guides

### AI Chat

The AI Chat interface allows you to interact with AURA's agents:

1. **Select an Agent**: Choose from available agents (planner, coder, debugger, etc.)
2. **Send Messages**: Type your request and press Enter
3. **View History**: Scroll through conversation history
4. **Multiple Sessions**: Create separate chat sessions for different topics

**Tips:**
- Use `@agent_name` to direct questions to specific agents
- Code blocks are syntax highlighted automatically
- Export conversations using the download button

### n8n Workflow Visualizer

View and interact with n8n workflows:

1. **Browse Workflows**: List of all workflows in `n8n-workflows/`
2. **Visual Canvas**: Interactive node-based diagram
3. **Execute Workflows**: Trigger workflows directly from the UI
4. **View Node Details**: Click nodes to see configuration

**Controls:**
- Mouse wheel: Zoom in/out
- Drag: Pan the canvas
- Click node: View details
- Double-click: Center on node

### MCP Tool Panel

Execute tools from configured MCP servers:

1. **Select Server**: Choose from 12 configured MCP servers
2. **Browse Tools**: View available tools with descriptions
3. **Execute Tools**: Fill parameters and run tools
4. **View Results**: See execution output in real-time

**Available MCP Servers:**
- `filesystem` - File operations
- `brave-search` - Web search
- `github` - GitHub integration
- `memory` - Persistent memory
- `playwright` - Browser automation
- `n8n-mcp` - n8n workflow control
- And more...

### Agent Observatory

Monitor and manage AI agents:

1. **Agent Grid**: Visual overview of all agents
2. **Real-time Metrics**: Executions, success rate, latency
3. **Lifecycle Controls**: Pause, resume, restart agents
4. **Detailed View**: Click agent to see logs and history

**Status Colors:**
- 🟢 Green - Idle/Healthy
- 🔵 Blue - Busy/Active
- 🟡 Yellow - Paused
- 🔴 Red - Error

### Terminal Console

Execute shell commands directly in the browser:

1. **Create Sessions**: Multiple terminal tabs
2. **Command History**: Up/Down arrows to navigate history
3. **Autocomplete**: Tab for command suggestions
4. **Quick Commands**: Click common commands

**Security Features:**
- Dangerous commands are blocked (sudo, rm -rf /, etc.)
- 30-second timeout on commands
- Session isolation

### Coverage Dashboard

Track test coverage and code quality:

1. **Overall Coverage**: Project-wide statistics
2. **Heatmap Visualization**: Coverage by module (Treemap)
3. **Coverage Gaps**: Functions needing tests
4. **Test Status**: Pass/fail statistics

**Severity Levels:**
- 🔴 Critical - High impact, no tests
- 🟠 High - Important function uncovered
- 🟡 Medium - Should have tests
- 🔵 Low - Nice to have coverage

### System Logs & Telemetry

Monitor system health and performance:

#### Logs Tab
- Real-time log streaming via WebSocket
- Filter by level (DEBUG, INFO, WARN, ERROR)
- Search within logs
- Export to JSON

#### Metrics Tab
- System health status
- Performance charts
- Latency over time
- Agent distribution

## Settings

### General Settings
- **Theme**: Light/Dark/System
- **Auto Refresh**: Automatic data updates
- **Confirm Actions**: Safety dialogs

### Notifications
Enable/disable notifications for:
- Goal completion
- Agent status changes
- System errors
- Webhook events

### API Configuration
- API endpoint URL
- WebSocket endpoint
- Request timeout
- Refresh interval

### MCP Server Configuration
View and manage MCP server connections:
- Test connection status
- View server configuration
- Delete servers

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + K` | Command palette (coming soon) |
| `Ctrl + /` | Toggle sidebar |
| `Ctrl + 1-9` | Switch to page by number |
| `Escape` | Close modals/drawers |

## Tips & Best Practices

### Performance
- The UI uses code splitting - initial load is fast
- Service worker caches assets for offline use
- Large lists use virtual scrolling
- WebSocket provides real-time updates

### Browser Support
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

### Troubleshooting

**Page not loading:**
- Check API server is running on port 8000
- Check browser console for errors
- Clear browser cache

**WebSocket disconnected:**
- Check network connection
- Refresh the page
- Verify WebSocket endpoint in settings

**Slow performance:**
- Disable auto-refresh in settings
- Close unused terminal sessions
- Clear old logs

## Web UI Architecture

### Tech Stack
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State**: Zustand
- **Charts**: Recharts
- **Icons**: Lucide React

### Performance Features
- Code splitting by route
- Lazy loading of heavy components
- Service worker caching
- Request deduplication
- Virtual scrolling for lists
- Optimized bundle splitting

## Getting Help

- Check the [API Documentation](API.md)
- Review logs in the Logs page
- Check agent status in Agent Observatory
- Use AI Chat for assistance
