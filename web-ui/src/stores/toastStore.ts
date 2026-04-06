import { create } from 'zustand'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface Toast {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
}

interface ToastState {
  toasts: Toast[]
  
  // Actions
  addToast: (toast: Omit<Toast, 'id'>) => void
  removeToast: (id: string) => void
  clearAll: () => void
}

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],

  addToast: (toast) => {
    const id = crypto.randomUUID()
    const duration = toast.duration || 5000
    
    set((state) => ({
      toasts: [...state.toasts, { ...toast, id, duration }]
    }))
    
    // Auto-remove after duration
    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id)
      }))
    }, duration)
  },

  removeToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id)
    }))
  },

  clearAll: () => {
    set({ toasts: [] })
  }
}))

// Helper functions for common toast patterns
export const toast = {
  success: (title: string, message?: string) => {
    useToastStore.getState().addToast({
      type: 'success',
      title,
      message,
      duration: 5000
    })
  },
  
  error: (title: string, message?: string) => {
    useToastStore.getState().addToast({
      type: 'error',
      title,
      message,
      duration: 8000
    })
  },
  
  warning: (title: string, message?: string) => {
    useToastStore.getState().addToast({
      type: 'warning',
      title,
      message,
      duration: 6000
    })
  },
  
  info: (title: string, message?: string) => {
    useToastStore.getState().addToast({
      type: 'info',
      title,
      message,
      duration: 5000
    })
  }
}
