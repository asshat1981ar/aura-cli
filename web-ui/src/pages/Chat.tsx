import { ChatInterface } from '../components/Chat/ChatInterface'
import { MessageSquare } from 'lucide-react'

export function Chat() {
  return (
    <div className="h-[calc(100vh-4rem)]">
      <div className="h-full flex flex-col">
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-3xl font-bold tracking-tight flex items-center gap-3">
              <MessageSquare className="w-8 h-8" />
              AI Chat
            </h2>
            <p className="text-muted-foreground">
              Chat with any AURA agent for assistance
            </p>
          </div>
        </div>
        
        <div className="flex-1 relative">
          <ChatInterface isOpen={true} onClose={() => {}} />
        </div>
      </div>
    </div>
  )
}
