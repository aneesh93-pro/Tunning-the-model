import { cn } from '../../lib/utils';

/**
 * ReactBits-inspired subtle technical DotGrid surface backdrop.
 */
export function DotGrid({ className }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={cn(
        'pointer-events-none absolute inset-0 opacity-40 dark:opacity-25',
        className
      )}
      style={{
        backgroundImage:
          'radial-gradient(circle at 1px 1px, rgba(148, 163, 184, 0.28) 1px, transparent 0)',
        backgroundSize: '20px 20px',
      }}
    />
  );
}
