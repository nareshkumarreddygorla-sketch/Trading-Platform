'use client';

import { useStore } from '@/store/useStore';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

/**
 * Displays WebSocket connection status as a colored dot + label.
 * Reads the wsStatus field from the Zustand store (set by useWebSocket hook).
 *
 * States:
 *   - connected: green dot + "Connected"
 *   - connecting / reconnecting: yellow pulsing dot + "Reconnecting..."
 *   - disconnected: red dot + "Disconnected"
 */
export function ConnectionStatus() {
  const wsStatus = useStore((s) => s.wsStatus);

  const config = {
    connected: {
      dotColor: 'bg-profit',
      glowColor: 'shadow-[0_0_6px_hsl(152,69%,53%,0.6)]',
      label: 'Connected',
      labelColor: 'text-profit',
      pulse: false,
    },
    connecting: {
      dotColor: 'bg-yellow-500',
      glowColor: 'shadow-[0_0_6px_hsl(38,92%,50%,0.5)]',
      label: 'Connecting...',
      labelColor: 'text-yellow-500',
      pulse: true,
    },
    reconnecting: {
      dotColor: 'bg-yellow-500',
      glowColor: 'shadow-[0_0_6px_hsl(38,92%,50%,0.5)]',
      label: 'Reconnecting...',
      labelColor: 'text-yellow-500',
      pulse: true,
    },
    disconnected: {
      dotColor: 'bg-loss',
      glowColor: 'shadow-[0_0_6px_hsl(0,84%,60%,0.5)]',
      label: 'Disconnected',
      labelColor: 'text-loss',
      pulse: false,
    },
  };

  const state = config[wsStatus] || config.disconnected;

  return (
    <div className="flex items-center gap-1.5" title={`WebSocket: ${state.label}`} role="status" aria-live="polite" aria-label={`WebSocket connection: ${state.label}`}>
      <span className="relative flex h-2 w-2">
        {state.pulse && (
          <motion.span
            className={cn(
              'absolute inline-flex h-full w-full rounded-full opacity-75',
              state.dotColor
            )}
            animate={{ scale: [1, 1.8, 1], opacity: [0.75, 0, 0.75] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
          />
        )}
        <span
          className={cn(
            'relative inline-flex h-2 w-2 rounded-full',
            state.dotColor,
            state.glowColor
          )}
        />
      </span>
      <span className={cn('text-[11px] font-medium hidden xl:inline', state.labelColor)}>
        {state.label}
      </span>
    </div>
  );
}
