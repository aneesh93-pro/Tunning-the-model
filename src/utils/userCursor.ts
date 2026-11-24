export interface CursorPoint {
  x: number;
  y: number;
}

export interface SpringConfig {
  stiffness?: number;
  damping?: number;
  mass?: number;
}

export function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/**
 * Computes arrow rotation (degrees) from pointer movement delta,
 * blending smoothly around the resting rotation when motion is small.
 */
export function computeDirectionTiltDeg(
  dx: number,
  dy: number,
  restingRotation = -14,
  directionAwareTilt = false
): number {
  if (!directionAwareTilt) return restingRotation;
  const speed = Math.hypot(dx, dy);
  if (speed < 1.2) return restingRotation;
  // Horizontal + vertical velocity tilt kept bounded so the pointer tip stays oriented forward
  const vxTilt = clampNumber(dx * 0.9, -22, 22);
  const vyTilt = clampNumber(dy * 0.35, -10, 10);
  return clampNumber(restingRotation + vxTilt + vyTilt * 0.3, -48, 32);
}

/**
 * Computes subtle trailing rotation on the name tag pill during rapid movement.
 */
export function computeLabelTiltDeg(
  dx: number,
  labelTiltStrength = 8
): number {
  if (labelTiltStrength <= 0 || Math.abs(dx) < 0.75) return 0;
  return clampNumber(-dx * 0.55, -labelTiltStrength, labelTiltStrength);
}

/**
 * Computes normalized press scale factor when mouse button is down.
 */
export function resolvePressScale(isPressed: boolean, pressScale = 0.92): number {
  if (!isPressed) return 1;
  return clampNumber(pressScale, 0.65, 1.15);
}

/**
 * Computes pixel offset for the name tag pill relative to the cursor arrow tip.
 */
export function computeDynamicLabelOffset(
  customOffset?: Partial<CursorPoint>,
  rotation = -14
): CursorPoint {
  if (
    customOffset &&
    typeof customOffset.x === 'number' &&
    typeof customOffset.y === 'number'
  ) {
    return { x: customOffset.x, y: customOffset.y };
  }
  const tiltShift = rotation > 0 ? 18 : 14;
  return {
    x: customOffset?.x ?? tiltShift,
    y: customOffset?.y ?? 20,
  };
}

/**
 * Picks accessible foreground text ink (#0F172A or #FFFFFF) for a hex color pill.
 */
export function resolveContrastInk(hexColor: string): '#0F172A' | '#FFFFFF' {
  const cleaned = hexColor.trim().replace(/^#/, '');
  const full =
    cleaned.length === 3
      ? cleaned
          .split('')
          .map((c) => c + c)
          .join('')
      : cleaned;
  if (!/^[0-9a-fA-F]{6}$/.test(full)) {
    return '#FFFFFF';
  }
  const r = parseInt(full.slice(0, 2), 16) / 255;
  const g = parseInt(full.slice(2, 4), 16) / 255;
  const b = parseInt(full.slice(4, 6), 16) / 255;
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 0.62 ? '#0F172A' : '#FFFFFF';
}

export function isCoarsePointerDevice(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false;
  }
  return window.matchMedia('(pointer: coarse)').matches;
}
