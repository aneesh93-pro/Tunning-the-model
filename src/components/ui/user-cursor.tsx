'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AnimatePresence,
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
} from 'motion/react';
import { cn } from '@/lib/utils';
import {
  computeDirectionTiltDeg,
  computeDynamicLabelOffset,
  computeLabelTiltDeg,
  isCoarsePointerDevice,
  resolveContrastInk,
  resolvePressScale,
  type CursorPoint,
  type SpringConfig,
} from '@/utils/userCursor';

export interface UserCursorClassNames {
  root?: string;
  cursor?: string;
  arrow?: string;
  label?: string;
  labelText?: string;
}

export interface UserCursorProps {
  children?: React.ReactNode;
  className?: string;
  /** Collaborator display name shown inside the name tag pill. */
  name?: string;
  /** Optional custom React node rendered inside the label pill instead of `name`. */
  label?: React.ReactNode;
  /** Primary color applied to the cursor pointer and label pill. */
  color?: string;
  /** Resting visual rotation of the cursor arrow in degrees. */
  rotation?: number;
  /** Smoothly rotates the arrow and trailing label along pointer movement velocity. */
  directionAwareTilt?: boolean;
  /** Maximum angular deflection on the label pill while moving (degrees). */
  labelTiltStrength?: number;
  /** Visibility trigger mode. */
  trigger?: 'always' | 'hover' | 'press';
  /** Toggles visibility of the collaborator name tag pill. */
  showLabel?: boolean;
  /** Hides the native OS cursor over the active surface. */
  hideNativeCursor?: boolean;
  /** Tracks viewport-wide instead of scoping to a container element. */
  fullScreen?: boolean;
  /** Spring parameters for the primary arrow follower. */
  spring?: SpringConfig;
  /** Spring parameters for the trailing name tag pill. */
  labelSpring?: SpringConfig;
  /** Pixel offset from the pointer coordinates to the arrow tip. */
  offset?: Partial<CursorPoint>;
  /** Pixel offset from the pointer tip to the label pill. */
  labelOffset?: Partial<CursorPoint>;
  /** Scale applied to the cursor while the pointer button is pressed. */
  pressScale?: number;
  /** Granular class overrides for cursor layers. */
  classNames?: UserCursorClassNames;
  /** Skips rendering on coarse/touch pointer devices. */
  hideOnTouch?: boolean;
  /** Z-index applied to the floating cursor layer. */
  zIndex?: number;
}

const DEFAULT_ARROW_SPRING: Required<SpringConfig> = {
  stiffness: 380,
  damping: 32,
  mass: 0.6,
};

const DEFAULT_LABEL_SPRING: Required<SpringConfig> = {
  stiffness: 220,
  damping: 26,
  mass: 0.7,
};

export function UserCursor({
  children,
  className,
  name = 'Sophie',
  label,
  color = '#F39C2A',
  rotation = -14,
  directionAwareTilt = false,
  labelTiltStrength = 8,
  trigger = 'hover',
  showLabel = true,
  hideNativeCursor = true,
  fullScreen = false,
  spring,
  labelSpring,
  offset,
  labelOffset,
  pressScale = 0.92,
  classNames,
  hideOnTouch = true,
  zIndex = 50,
}: UserCursorProps) {
  const prefersReducedMotion = useReducedMotion();
  const rootRef = useRef<HTMLDivElement>(null);
  const lastPointRef = useRef<CursorPoint | null>(null);
  const tiltTimerRef = useRef<number | null>(null);

  const [isCoarse, setIsCoarse] = useState(false);
  const [isInside, setIsInside] = useState(trigger === 'always');
  const [isPressed, setIsPressed] = useState(false);

  const arrowSpringCfg = useMemo(
    () => ({ ...DEFAULT_ARROW_SPRING, ...spring }),
    [spring]
  );
  const pillSpringCfg = useMemo(
    () => ({ ...DEFAULT_LABEL_SPRING, ...labelSpring }),
    [labelSpring]
  );

  const resolvedOffset = useMemo(
    () => ({ x: offset?.x ?? 0, y: offset?.y ?? 0 }),
    [offset?.x, offset?.y]
  );
  const resolvedLabelOffset = useMemo(
    () => computeDynamicLabelOffset(labelOffset, rotation),
    [labelOffset, rotation]
  );
  const pillTextInk = useMemo(() => resolveContrastInk(color), [color]);

  // Raw pointer motion values (arrow)
  const rawX = useMotionValue(-200);
  const rawY = useMotionValue(-200);
  const rawArrowAngle = useMotionValue(rotation);
  const rawLabelAngle = useMotionValue(0);

  const arrowX = useSpring(rawX, prefersReducedMotion ? { stiffness: 2000, damping: 120, mass: 0.1 } : arrowSpringCfg);
  const arrowY = useSpring(rawY, prefersReducedMotion ? { stiffness: 2000, damping: 120, mass: 0.1 } : arrowSpringCfg);
  const arrowRotate = useSpring(rawArrowAngle, { stiffness: 280, damping: 24, mass: 0.5 });

  // Slightly lagging follower for the collaborator name tag pill
  const labelX = useSpring(rawX, prefersReducedMotion ? { stiffness: 1600, damping: 100, mass: 0.1 } : pillSpringCfg);
  const labelY = useSpring(rawY, prefersReducedMotion ? { stiffness: 1600, damping: 100, mass: 0.1 } : pillSpringCfg);
  const labelRotate = useSpring(rawLabelAngle, { stiffness: 240, damping: 20, mass: 0.5 });

  useEffect(() => {
    setIsCoarse(isCoarsePointerDevice());
  }, []);

  useEffect(() => {
    rawArrowAngle.set(rotation);
  }, [rotation, rawArrowAngle]);

  useEffect(() => {
    if (hideOnTouch && isCoarse) return undefined;

    const updateCoordinates = (clientX: number, clientY: number) => {
      let nextX = clientX + resolvedOffset.x;
      let nextY = clientY + resolvedOffset.y;

      if (!fullScreen && rootRef.current) {
        const rect = rootRef.current.getBoundingClientRect();
        nextX = clientX - rect.left + resolvedOffset.x;
        nextY = clientY - rect.top + resolvedOffset.y;
      }

      const prev = lastPointRef.current;
      const dx = prev ? nextX - prev.x : 0;
      const dy = prev ? nextY - prev.y : 0;
      lastPointRef.current = { x: nextX, y: nextY };

      rawX.set(nextX);
      rawY.set(nextY);

      if (!prefersReducedMotion) {
        rawArrowAngle.set(
          computeDirectionTiltDeg(dx, dy, rotation, directionAwareTilt)
        );
        rawLabelAngle.set(
          directionAwareTilt ? computeLabelTiltDeg(dx, labelTiltStrength) : 0
        );

        if (tiltTimerRef.current !== null) {
          window.clearTimeout(tiltTimerRef.current);
        }
        tiltTimerRef.current = window.setTimeout(() => {
          rawArrowAngle.set(rotation);
          rawLabelAngle.set(0);
        }, 120);
      }
    };

    if (fullScreen) {
      if (trigger === 'always' && lastPointRef.current === null && typeof window !== 'undefined') {
        const initialX = Math.round(window.innerWidth * 0.5);
        const initialY = Math.round(window.innerHeight * 0.45);
        lastPointRef.current = { x: initialX, y: initialY };
        rawX.set(initialX);
        rawY.set(initialY);
      }

      const onMove = (e: PointerEvent) => {
        if (hideOnTouch && e.pointerType === 'touch') return;
        setIsInside(true);
        updateCoordinates(e.clientX, e.clientY);
      };
      const onDown = (e: PointerEvent) => {
        if (hideOnTouch && e.pointerType === 'touch') return;
        setIsInside(true);
        setIsPressed(true);
        updateCoordinates(e.clientX, e.clientY);
      };
      const onUp = () => setIsPressed(false);
      const onLeaveWindow = () => {
        setIsPressed(false);
        if (trigger !== 'always') setIsInside(false);
      };

      window.addEventListener('pointermove', onMove, { passive: true });
      window.addEventListener('pointerdown', onDown, { passive: true });
      window.addEventListener('pointerup', onUp, { passive: true });
      window.addEventListener('pointercancel', onUp, { passive: true });
      document.addEventListener('mouseleave', onLeaveWindow);

      return () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerdown', onDown);
        window.removeEventListener('pointerup', onUp);
        window.removeEventListener('pointercancel', onUp);
        document.removeEventListener('mouseleave', onLeaveWindow);
        if (tiltTimerRef.current !== null) window.clearTimeout(tiltTimerRef.current);
      };
    }

    const node = rootRef.current;
    if (!node) return undefined;

    const onEnter = (e: PointerEvent) => {
      if (hideOnTouch && e.pointerType === 'touch') return;
      setIsInside(true);
      updateCoordinates(e.clientX, e.clientY);
    };
    const onMove = (e: PointerEvent) => {
      if (hideOnTouch && e.pointerType === 'touch') return;
      setIsInside(true);
      updateCoordinates(e.clientX, e.clientY);
    };
    const onDown = (e: PointerEvent) => {
      if (hideOnTouch && e.pointerType === 'touch') return;
      setIsPressed(true);
      updateCoordinates(e.clientX, e.clientY);
    };
    const onUp = () => setIsPressed(false);
    const onLeave = () => {
      setIsPressed(false);
      if (trigger !== 'always') setIsInside(false);
    };

    node.addEventListener('pointerenter', onEnter);
    node.addEventListener('pointermove', onMove);
    node.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    node.addEventListener('pointerleave', onLeave);

    return () => {
      node.removeEventListener('pointerenter', onEnter);
      node.removeEventListener('pointermove', onMove);
      node.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointerup', onUp);
      node.removeEventListener('pointerleave', onLeave);
      if (tiltTimerRef.current !== null) window.clearTimeout(tiltTimerRef.current);
    };
  }, [
    directionAwareTilt,
    fullScreen,
    hideOnTouch,
    isCoarse,
    labelTiltStrength,
    prefersReducedMotion,
    rawArrowAngle,
    rawLabelAngle,
    rawX,
    rawY,
    resolvedOffset.x,
    resolvedOffset.y,
    rotation,
    trigger,
  ]);

  const isVisible =
    !(hideOnTouch && isCoarse) &&
    (trigger === 'always'
      ? true
      : trigger === 'press'
        ? isPressed
        : isInside);

  const shouldHideNative = hideNativeCursor && isVisible;
  const activeScale = resolvePressScale(isPressed, pressScale);

  useEffect(() => {
    if (!fullScreen || typeof document === 'undefined') return undefined;
    if (shouldHideNative) {
      document.documentElement.classList.add('user-cursor-hide-native');
    } else {
      document.documentElement.classList.remove('user-cursor-hide-native');
    }
    return () => {
      document.documentElement.classList.remove('user-cursor-hide-native');
    };
  }, [fullScreen, shouldHideNative]);

  const cursorLayer = (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          key="user-cursor-layer"
          data-testid="user-cursor-overlay"
          data-pressed={isPressed ? 'true' : 'false'}
          aria-hidden="true"
          initial={{ opacity: 0, scale: 0.72 }}
          animate={{ opacity: 1, scale: activeScale }}
          exit={{ opacity: 0, scale: 0.75 }}
          transition={{ duration: 0.14, ease: [0.22, 1, 0.36, 1] }}
          style={{ zIndex }}
          className={cn(
            fullScreen ? 'fixed inset-0' : 'absolute inset-0',
            'pointer-events-none select-none overflow-hidden',
            classNames?.cursor
          )}
        >
          {/* Collaborator Cursor Arrow */}
          <motion.div
            data-testid="user-cursor-arrow"
            style={{
              x: arrowX,
              y: arrowY,
              rotate: arrowRotate,
              transformOrigin: '2px 2px',
            }}
            className={cn('absolute top-0 left-0 will-change-transform', classNames?.arrow)}
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="drop-shadow-[0_2px_8px_rgba(255,255,255,0.22)] dark:drop-shadow-[0_2px_10px_rgba(255,255,255,0.18)]"
            >
              <path
                d="M3.2 2.6L20.4 9.7C21.6 10.2 21.5 11.9 20.2 12.2L12.7 13.9L9.8 20.9C9.3 22.1 7.6 22 7.2 20.8L2.2 4.1C1.9 3.1 2.4 2.3 3.2 2.6Z"
                fill={color}
                stroke="#FFFFFF"
                strokeWidth="1.65"
                strokeLinejoin="round"
              />
            </svg>
          </motion.div>

          {/* Trailing Collaborator Name Tag */}
          {showLabel && (
            <motion.div
              data-testid="user-cursor-label"
              style={{
                x: labelX,
                y: labelY,
                rotate: labelRotate,
                marginLeft: `${resolvedLabelOffset.x}px`,
                marginTop: `${resolvedLabelOffset.y}px`,
                backgroundColor: color,
                color: pillTextInk,
              }}
              className={cn(
                'absolute top-0 left-0 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold tracking-tight leading-none whitespace-nowrap shadow-[0_6px_18px_rgba(2,6,23,0.45)] ring-1 ring-white/75 dark:ring-white/45 will-change-transform',
                classNames?.label
              )}
            >
              <span
                className="w-1.5 h-1.5 rounded-full bg-current opacity-85 animate-pulse"
                aria-hidden="true"
              />
              <span className={cn('font-display', classNames?.labelText)}>
                {label ?? name}
              </span>
            </motion.div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );

  if (fullScreen && !children) {
    return cursorLayer;
  }

  return (
    <div
      ref={rootRef}
      className={cn(
        'relative',
        shouldHideNative && !fullScreen ? 'user-cursor-hide-native' : '',
        className,
        classNames?.root
      )}
    >
      {children}
      {cursorLayer}
    </div>
  );
}

export default UserCursor;
