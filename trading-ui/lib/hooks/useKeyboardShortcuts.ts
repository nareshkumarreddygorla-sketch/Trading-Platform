'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

/**
 * Global keyboard shortcuts for power users.
 *
 * Shortcuts (uses Cmd on Mac, Ctrl on Windows/Linux):
 *   - Cmd+K: Toggle command palette (dispatches custom event for future use)
 *   - Cmd+D: Navigate to Dashboard
 *   - Cmd+P: Navigate to Positions
 *   - Cmd+T: Navigate to Trades
 *   - Escape: Close modals/overlays (dispatches custom event)
 */
export function useKeyboardShortcuts() {
  const router = useRouter();

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const isMod = e.metaKey || e.ctrlKey;

      // Ignore when typing in inputs/textareas
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        // Only allow Escape to propagate from inputs
        if (e.key !== 'Escape') return;
      }

      if (isMod && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent('toggle-command-palette'));
        return;
      }

      if (isMod && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        router.push('/dashboard');
        return;
      }

      if (isMod && e.key.toLowerCase() === 'p') {
        e.preventDefault();
        router.push('/positions');
        return;
      }

      if (isMod && e.key.toLowerCase() === 't') {
        e.preventDefault();
        router.push('/trades');
        return;
      }

      if (e.key === 'Escape') {
        window.dispatchEvent(new CustomEvent('close-modal'));
        return;
      }
    }

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [router]);
}
