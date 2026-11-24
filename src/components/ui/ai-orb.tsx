'use client';

import { type ClassValue, clsx } from 'clsx';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { type ComponentProps, useSyncExternalStore } from 'react';
import { twMerge } from 'tailwind-merge';
import { type OrbSize, ThinkingOrb } from 'thinking-orbs';
import {
  AI_ORB_LABELS,
  AI_ORB_STATES,
  type AgentActivity,
  type AiOrbState,
  type AiOrbTone,
  ORB_VISUAL_TABLE,
  orbStateFor,
} from '@/utils/aiOrbState';

export { AI_ORB_LABELS, AI_ORB_STATES, orbStateFor };
export type { AgentActivity, AiOrbState, AiOrbTone };

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const SIZE_MAP: Record<'sm' | 'md' | 'lg', OrbSize> = {
  sm: 20,
  md: 32,
  lg: 64,
};

const TONE_TOKENS: Record<
  AiOrbTone,
  { vars: string[]; darkInk: string; lightInk: string }
> = {
  muted: {
    vars: [],
    darkInk: '#CBD5E1',
    lightInk: '#475569',
  },
  primary: {
    vars: ['--color-ring', '--ring'],
    darkInk: '#60A5FA',
    lightInk: '#1D4ED8',
  },
  attention: {
    vars: ['--color-warning', '--warning'],
    darkInk: '#FBBF24',
    lightInk: '#B45309',
  },
  critical: {
    vars: ['--color-destructive', '--destructive'],
    darkInk: '#F87171',
    lightInk: '#DC2626',
  },
};

const TONE_TEXT_CLASSES: Record<AiOrbTone, string> = {
  muted: 'text-slate-600 dark:text-slate-300',
  primary: 'text-blue-600 dark:text-blue-400',
  attention: 'text-amber-600 dark:text-amber-400',
  critical: 'text-red-600 dark:text-red-400',
};

function subscribeTheme(onChange: () => void): () => void {
  if (typeof document === 'undefined') return () => {};
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class', 'data-theme', 'style'],
  });
  const media =
    typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-color-scheme: dark)')
      : null;
  media?.addEventListener('change', onChange);
  return () => {
    observer.disconnect();
    media?.removeEventListener('change', onChange);
  };
}

function getThemeSnapshot(): string {
  if (typeof document === 'undefined') return 'dark';
  const root = document.documentElement;
  if (root.classList.contains('dark') || root.getAttribute('data-theme') === 'dark') return 'dark';
  if (root.classList.contains('light') || root.getAttribute('data-theme') === 'light') return 'light';
  return typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
}

function resolveToneColor(tone: AiOrbTone, themeSnapshot: string): string {
  const spec = TONE_TOKENS[tone];
  const fallback = themeSnapshot === 'dark' ? spec.darkInk : spec.lightInk;
  if (
    spec.vars.length === 0 ||
    typeof document === 'undefined' ||
    typeof getComputedStyle === 'undefined'
  ) {
    return fallback;
  }
  const styles = getComputedStyle(document.documentElement);
  for (const token of spec.vars) {
    const raw = styles.getPropertyValue(token).trim();
    if (/^(#([0-9a-f]{3}|[0-9a-f]{6})|rgba?\([^)]+\))$/i.test(raw)) {
      return raw;
    }
  }
  return fallback;
}

export interface AiOrbProps extends Omit<ComponentProps<'div'>, 'children'> {
  /** Semantic activity state of the model or workflow. */
  state?: AiOrbState;
  /** Render scale mapped to ThinkingOrb's tuned presets (20 / 32 / 64 px). */
  size?: 'sm' | 'md' | 'lg';
  /** Optional tone override instead of using the state's default palette. */
  tone?: AiOrbTone;
  /** Accessible status string (also rendered inline when showLabel is true). */
  label?: string;
  /** Render the crisp status text alongside the animated orb. */
  showLabel?: boolean;
  /** Hide from assistive technology when paired with adjacent status copy. */
  decorative?: boolean;
  /** Enable subtle pointer gravitational pull when hovered. */
  gravity?: boolean;
  /** Optional animation speed multiplier. */
  speedMultiplier?: number;
}

export function AiOrb({
  state = 'idle',
  size = 'md',
  tone: toneOverride,
  label,
  showLabel = false,
  decorative = false,
  gravity = false,
  speedMultiplier = 1,
  className,
  ...rest
}: AiOrbProps) {
  const themeSnapshot = useSyncExternalStore(subscribeTheme, getThemeSnapshot, () => 'dark');
  const prefersReducedMotion = useReducedMotion();

  const visual = ORB_VISUAL_TABLE[state];
  const resolvedTone = toneOverride ?? visual.tone;
  const resolvedColor = resolveToneColor(resolvedTone, themeSnapshot);
  const resolvedLabel = label ?? AI_ORB_LABELS[state];
  const pixelSize = SIZE_MAP[size];
  const dotSizeScale = size === 'sm' ? 1.28 : size === 'md' ? 1.12 : 1;
  const dotCountScale = size === 'sm' ? 1.1 : 1;

  const a11yProps = decorative
    ? ({ 'aria-hidden': true } as const)
    : ({
        role: 'status',
        'aria-live': 'polite',
        'aria-label': resolvedLabel,
      } as const);

  return (
    <div
      data-ai-orb-state={state}
      data-ai-orb-tone={resolvedTone}
      className={cn('inline-flex items-center gap-2 shrink-0 select-none', className)}
      {...a11yProps}
      {...rest}
    >
      <div
        className="relative inline-flex items-center justify-center shrink-0 rounded-full"
        style={{ width: pixelSize, height: pixelSize }}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={`${visual.orb}-${resolvedTone}`}
            initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0.65, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={prefersReducedMotion ? { opacity: 0 } : { opacity: 0.5, scale: 0.94 }}
            transition={{ duration: prefersReducedMotion ? 0 : 0.2, ease: 'easeOut' }}
            className="inline-flex items-center justify-center"
          >
            <ThinkingOrb
              state={visual.orb}
              size={pixelSize}
              theme={themeSnapshot === 'dark' ? 'dark' : 'light'}
              color={resolvedColor}
              speed={visual.speed * speedMultiplier}
              dots={dotCountScale}
              dotSize={dotSizeScale}
              paused={visual.paused || Boolean(prefersReducedMotion)}
              gravity={gravity}
              aria-hidden="true"
              role="presentation"
            />
          </motion.div>
        </AnimatePresence>
      </div>

      {showLabel && (
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={`${state}-${resolvedLabel}`}
            initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 2 }}
            animate={{ opacity: 1, y: 0 }}
            exit={prefersReducedMotion ? { opacity: 0 } : { opacity: 0, y: -2 }}
            transition={{ duration: prefersReducedMotion ? 0 : 0.16 }}
            className={cn(
              'font-mono text-xs font-medium tracking-tight whitespace-nowrap',
              TONE_TEXT_CLASSES[resolvedTone]
            )}
          >
            {resolvedLabel}
          </motion.span>
        </AnimatePresence>
      )}
    </div>
  );
}
