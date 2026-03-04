'use client';

import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  X,
} from 'lucide-react';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: number;
}

const TOAST_LIFETIME_MS = 5000;
const MAX_TOASTS = 5;

const iconMap = {
  success: CheckCircle2,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
};

const colorMap = {
  success: {
    border: 'border-profit/40',
    bg: 'bg-profit/5',
    icon: 'text-profit',
    bar: 'bg-profit/50',
  },
  error: {
    border: 'border-loss/40',
    bg: 'bg-loss/5',
    icon: 'text-loss',
    bar: 'bg-loss/50',
  },
  warning: {
    border: 'border-yellow-500/40',
    bg: 'bg-yellow-500/5',
    icon: 'text-yellow-500',
    bar: 'bg-yellow-500/50',
  },
  info: {
    border: 'border-primary/40',
    bg: 'bg-primary/5',
    icon: 'text-primary',
    bar: 'bg-primary/50',
  },
};

function formatRelativeTime(timestamp: number): string {
  const diff = Math.floor((Date.now() - timestamp) / 1000);
  if (diff < 5) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  return `${Math.floor(diff / 60)}m ago`;
}

export function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Auto-dismiss after TOAST_LIFETIME_MS
  useEffect(() => {
    const interval = setInterval(() => {
      setToasts((prev) => prev.filter((t) => Date.now() - t.timestamp < TOAST_LIFETIME_MS));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Listen for custom trading-toast events from WebSocket dispatcher
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<Toast>).detail;
      if (!detail) return;
      setToasts((prev) => [...prev.slice(-(MAX_TOASTS - 1)), detail]);
    };
    window.addEventListener('trading-toast', handler);
    return () => window.removeEventListener('trading-toast', handler);
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <div
      className="fixed top-20 right-4 z-[110] flex flex-col gap-2 pointer-events-none"
      style={{ maxWidth: 380 }}
      role="region"
      aria-label="Toast notifications"
      aria-live="polite"
    >
      <AnimatePresence mode="popLayout">
        {toasts.map((toast) => {
          const Icon = iconMap[toast.type];
          const colors = colorMap[toast.type];

          return (
            <motion.div
              key={toast.id}
              layout
              initial={{ opacity: 0, x: 60, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 60, scale: 0.85, transition: { duration: 0.2 } }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              className={cn(
                'pointer-events-auto rounded-xl border shadow-2xl backdrop-blur-xl',
                colors.border,
                colors.bg
              )}
            >
              <div className="flex items-start gap-3 p-3.5">
                <div className={cn('mt-0.5 shrink-0', colors.icon)}>
                  <Icon className="h-4 w-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-semibold text-foreground truncate">
                      {toast.title}
                    </p>
                    <span className="text-[9px] text-muted-foreground whitespace-nowrap font-mono">
                      {formatRelativeTime(toast.timestamp)}
                    </span>
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-0.5 leading-relaxed">
                    {toast.message}
                  </p>
                </div>
                <button
                  onClick={() => dismiss(toast.id)}
                  className="shrink-0 rounded-lg p-1 hover:bg-muted/50 transition-colors"
                  aria-label="Dismiss notification"
                >
                  <X className="h-3 w-3 text-muted-foreground" />
                </button>
              </div>
              {/* Auto-dismiss progress bar */}
              <motion.div
                className={cn('h-[2px] rounded-full mx-3 mb-2', colors.bar)}
                initial={{ width: '100%' }}
                animate={{ width: '0%' }}
                transition={{ duration: TOAST_LIFETIME_MS / 1000, ease: 'linear' }}
              />
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

/**
 * Dispatch a toast event programmatically.
 * Can be called from anywhere in the app.
 */
export function dispatchToast(
  type: Toast['type'],
  title: string,
  message: string
) {
  const toast: Toast = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type,
    title,
    message,
    timestamp: Date.now(),
  };
  window.dispatchEvent(new CustomEvent('trading-toast', { detail: toast }));
}
