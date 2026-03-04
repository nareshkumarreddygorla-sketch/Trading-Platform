'use client';

import { cn } from '@/lib/utils';

/**
 * Base shimmer bar used internally by composite skeleton components.
 */
function ShimmerBar({ className }: { className?: string }) {
  return (
    <div
      className={cn('shimmer rounded-lg', className)}
    />
  );
}

/**
 * Skeleton placeholder for a glass-card KPI or stat card.
 * Mimics the layout: icon top-right, label, large value, sub-value.
 */
export function CardSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('glass-card p-5 space-y-3', className)}>
      <div className="flex items-start justify-between">
        <div className="flex flex-col gap-2 flex-1">
          <ShimmerBar className="h-3 w-20" />
          <ShimmerBar className="h-8 w-32" />
          <ShimmerBar className="h-3 w-24" />
        </div>
        <ShimmerBar className="h-10 w-10 rounded-xl shrink-0" />
      </div>
    </div>
  );
}

/**
 * Skeleton placeholder for tabular data.
 * Renders a header row + N body rows with shimmer columns.
 */
export function TableSkeleton({
  rows = 5,
  cols = 4,
  className,
}: {
  rows?: number;
  cols?: number;
  className?: string;
}) {
  return (
    <div className={cn('glass-card p-6 space-y-4', className)}>
      {/* Header */}
      <div className="flex items-center gap-4">
        {Array.from({ length: cols }).map((_, i) => (
          <ShimmerBar
            key={`hdr-${i}`}
            className={cn('h-3', i === 0 ? 'w-32' : 'w-20', 'flex-shrink-0')}
          />
        ))}
      </div>
      <div className="h-px bg-border/30" />
      {/* Rows */}
      {Array.from({ length: rows }).map((_, r) => (
        <div key={`row-${r}`} className="flex items-center gap-4">
          {Array.from({ length: cols }).map((_, c) => (
            <ShimmerBar
              key={`cell-${r}-${c}`}
              className={cn(
                'h-4',
                c === 0 ? 'w-28' : c === cols - 1 ? 'w-16' : 'w-20',
                'flex-shrink-0'
              )}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

/**
 * Skeleton placeholder for a chart area.
 * Shows a large shimmer rectangle mimicking a chart viewport.
 */
export function ChartSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('glass-card p-6', className)}>
      {/* Chart header */}
      <div className="flex items-center justify-between mb-5">
        <div className="space-y-1.5">
          <ShimmerBar className="h-4 w-28" />
          <ShimmerBar className="h-3 w-40" />
        </div>
        <div className="flex gap-1.5">
          {[1, 2, 3, 4].map((i) => (
            <ShimmerBar key={i} className="h-7 w-10 rounded-lg" />
          ))}
        </div>
      </div>
      {/* Chart body */}
      <div className="relative h-72 rounded-xl overflow-hidden">
        <ShimmerBar className="absolute inset-0 rounded-xl" />
        {/* Fake axis lines */}
        <div className="absolute inset-0 flex flex-col justify-between py-4 px-8 pointer-events-none">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="h-px w-full"
              style={{ background: 'hsl(217 33% 15% / 0.4)' }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * A simple loading spinner with the platform's primary color.
 */
export function Spinner({ className, size = 'md' }: { className?: string; size?: 'sm' | 'md' | 'lg' }) {
  const sizeClasses = {
    sm: 'h-4 w-4 border-2',
    md: 'h-6 w-6 border-2',
    lg: 'h-8 w-8 border-[3px]',
  };

  return (
    <div
      className={cn(
        'animate-spin rounded-full border-primary/30 border-t-primary',
        sizeClasses[size],
        className
      )}
    />
  );
}
