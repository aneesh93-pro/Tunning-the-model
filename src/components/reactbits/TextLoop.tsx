import React, { useEffect, useId, useMemo, useRef } from 'react';
import gsap from 'gsap';

export type TextLoopShape = 'wave' | 'circle' | 'infinity' | 'arch' | 'line';
export type TextLoopDirection = 'forward' | 'reverse';

export interface TextLoopProps {
  text?: string;
  shape?: TextLoopShape;
  path?: string;
  viewBox?: string;
  speed?: number;
  direction?: TextLoopDirection;
  separator?: string;
  curviness?: number;
  tilt?: number;
  cross?: boolean;
  crossColor?: string;
  crossAngle?: number;
  draggable?: boolean;
  scrollBoost?: number;
  fontSize?: number;
  fontWeight?: number | string;
  letterSpacing?: number;
  uppercase?: boolean;
  outline?: boolean;
  outlineWidth?: number;
  color?: string;
  ribbon?: boolean;
  ribbonColor?: string;
  ribbonWidth?: number;
  edgeFade?: number;
  pauseOnHover?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

function contrastTextColor(hex: string): string {
  const clean = hex.replace('#', '');
  const full = clean.length === 3 ? clean.replace(/./g, (c) => c + c) : clean;
  const num = parseInt(full.slice(0, 6), 16);
  if (Number.isNaN(num)) return '#ffffff';
  const r = (num >> 16) & 255;
  const g = (num >> 8) & 255;
  const b = num & 255;
  const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return lum > 0.62 ? '#0f172a' : '#ffffff';
}

function buildShapePath(shape: TextLoopShape, curviness: number): string {
  const amp = Math.max(12, Math.min(210, curviness));
  switch (shape) {
    case 'line':
      return 'M -120 260 L 1320 260';
    case 'arch':
      return `M -60 ${280 + amp * 0.55} Q 600 ${280 - amp * 1.9} 1260 ${280 + amp * 0.55}`;
    case 'circle': {
      const r = Math.max(70, Math.min(210, curviness * 1.9));
      const cx = 600;
      const cy = 260;
      return `M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy} A ${r} ${r} 0 1 1 ${cx - r} ${cy}`;
    }
    case 'infinity': {
      const rx = Math.max(140, Math.min(440, curviness * 3.6));
      const ry = Math.max(55, Math.min(185, curviness * 1.45));
      const cx = 600;
      const cy = 260;
      return `M ${cx} ${cy} C ${cx + rx * 0.55} ${cy - ry}, ${cx + rx} ${cy - ry}, ${cx + rx} ${cy} C ${cx + rx} ${cy + ry}, ${cx + rx * 0.55} ${cy + ry}, ${cx} ${cy} C ${cx - rx * 0.55} ${cy - ry}, ${cx - rx} ${cy - ry}, ${cx - rx} ${cy} C ${cx - rx} ${cy + ry}, ${cx - rx * 0.55} ${cy + ry}, ${cx} ${cy}`;
    }
    case 'wave':
    default: {
      // Smooth continuous multi-cycle wave spanning beyond viewBox edges for seamless wrapping
      return `M -240 260 C -60 ${260 - amp}, 120 ${260 + amp}, 300 260 C 480 ${260 - amp}, 660 ${260 + amp}, 840 260 C 1020 ${260 - amp}, 1200 ${260 + amp}, 1380 260`;
    }
  }
}

export function TextLoop({
  text = 'React ✦ Bits',
  shape = 'wave',
  path,
  viewBox = '0 0 1200 520',
  speed = 90,
  direction = 'forward',
  separator = '✦',
  curviness = 90,
  tilt = 0,
  cross = false,
  crossColor,
  crossAngle = 16,
  draggable = false,
  scrollBoost = 0,
  fontSize = 46,
  fontWeight = 800,
  letterSpacing = 2,
  uppercase = true,
  outline = false,
  outlineWidth = 1.5,
  color,
  ribbon = false,
  ribbonColor = '#5227FF',
  ribbonWidth = 86,
  edgeFade = 0,
  pauseOnHover = true,
  className = '',
  style,
}: TextLoopProps) {
  const uid = useId().replace(/:/g, '');
  const pathIdMain = `textloop-main-${uid}`;
  const pathIdCross = `textloop-cross-${uid}`;
  const maskId = `textloop-fade-${uid}`;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const pathMainRef = useRef<SVGPathElement | null>(null);
  const pathCrossRef = useRef<SVGPathElement | null>(null);
  const tpMainRef = useRef<SVGTextPathElement | null>(null);
  const tpCrossRef = useRef<SVGTextPathElement | null>(null);

  const hoverRef = useRef(false);
  const dragStateRef = useRef<{ active: boolean; lastX: number; velocity: number }>({
    active: false,
    lastX: 0,
    velocity: 0,
  });
  const scrollBoostRef = useRef(0);

  const activePath = useMemo(
    () => (path && path.trim().length > 0 ? path : buildShapePath(shape, curviness)),
    [path, shape, curviness]
  );

  const unitString = useMemo(() => {
    const cleanText = uppercase ? text.toUpperCase() : text;
    const sep = separator ? ` ${separator} ` : ' ';
    return `${cleanText}${sep}`;
  }, [text, separator, uppercase]);

  const repeatedString = useMemo(() => {
    // Repeat enough times to saturate the entire path circumference with seamless overlap
    return Array.from({ length: 14 }, () => unitString).join('');
  }, [unitString]);

  const resolvedFill = color ?? (ribbon ? contrastTextColor(ribbonColor) : '#F8FAFC');
  const resolvedCrossColor = crossColor ?? (ribbon ? '#E0E7FF' : 'rgba(148,163,184,0.45)');

  useEffect(() => {
    const mainPath = pathMainRef.current;
    const tpMain = tpMainRef.current;
    if (!mainPath || !tpMain) return undefined;

    const prefersReduced =
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    const totalLen = Math.max(100, mainPath.getTotalLength());
    const crossLen = Math.max(100, pathCrossRef.current?.getTotalLength() ?? totalLen);
    const dirSign = direction === 'reverse' ? -1 : 1;

    let offsetMain = 0;
    let offsetCross = 0;
    let lastTime = performance.now();

    const onScroll = () => {
      if (!scrollBoost) return;
      scrollBoostRef.current = Math.max(-6, Math.min(6, scrollBoostRef.current + scrollBoost * 0.4));
    };
    if (scrollBoost !== 0) {
      window.addEventListener('scroll', onScroll, { passive: true });
    }

    const applyOffsets = () => {
      const m = ((offsetMain % totalLen) + totalLen) % totalLen;
      tpMain.setAttribute('startOffset', `${-m}`);
      if (tpCrossRef.current) {
        const c = ((offsetCross % crossLen) + crossLen) % crossLen;
        tpCrossRef.current.setAttribute('startOffset', `${-c}`);
      }
    };

    if (prefersReduced) {
      offsetMain = totalLen * 0.08;
      applyOffsets();
      return () => {
        if (scrollBoost !== 0) window.removeEventListener('scroll', onScroll);
      };
    }

    const tick = () => {
      const now = performance.now();
      const dt = Math.min(0.05, Math.max(0, (now - lastTime) / 1000));
      lastTime = now;

      const paused = pauseOnHover && hoverRef.current && !dragStateRef.current.active;
      const baseVel = paused ? 0 : speed * dirSign;
      scrollBoostRef.current *= Math.pow(0.08, dt);
      dragStateRef.current.velocity *= Math.pow(0.05, dt);

      const delta = (baseVel + scrollBoostRef.current * 45 + dragStateRef.current.velocity) * dt;
      offsetMain += delta;
      offsetCross -= delta;
      applyOffsets();
    };

    gsap.ticker.add(tick);
    return () => {
      gsap.ticker.remove(tick);
      if (scrollBoost !== 0) window.removeEventListener('scroll', onScroll);
    };
  }, [activePath, speed, direction, pauseOnHover, scrollBoost, cross, repeatedString]);

  const fadeStop = Math.max(0, Math.min(0.45, edgeFade / 200));

  return (
    <div
      ref={containerRef}
      onMouseEnter={() => {
        hoverRef.current = true;
      }}
      onMouseLeave={() => {
        hoverRef.current = false;
        dragStateRef.current.active = false;
      }}
      onPointerDown={(e) => {
        if (!draggable) return;
        dragStateRef.current.active = true;
        dragStateRef.current.lastX = e.clientX;
        dragStateRef.current.velocity = 0;
      }}
      onPointerMove={(e) => {
        if (!draggable || !dragStateRef.current.active) return;
        const dx = e.clientX - dragStateRef.current.lastX;
        dragStateRef.current.lastX = e.clientX;
        dragStateRef.current.velocity = -dx * 16;
      }}
      onPointerUp={() => {
        dragStateRef.current.active = false;
      }}
      className={`relative w-full overflow-hidden select-none ${
        draggable ? 'cursor-grab active:cursor-grabbing' : ''
      } ${className}`.trim()}
      style={style}
    >
      <svg
        viewBox={viewBox}
        preserveAspectRatio="xMidYMid slice"
        className="w-full h-full block overflow-visible"
        aria-label={text}
        role="img"
      >
        <defs>
          <path id={pathIdMain} ref={pathMainRef} d={activePath} />
          {cross && <path id={pathIdCross} ref={pathCrossRef} d={activePath} />}
          {edgeFade > 0 && (
            <linearGradient id={`${maskId}-grad`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="black" />
              <stop offset={`${Math.round(fadeStop * 100)}%`} stopColor="white" />
              <stop offset={`${Math.round((1 - fadeStop) * 100)}%`} stopColor="white" />
              <stop offset="100%" stopColor="black" />
            </linearGradient>
          )}
          {edgeFade > 0 && (
            <mask id={maskId}>
              <rect x="-200" y="-200" width="1600" height="920" fill={`url(#${maskId}-grad)`} />
            </mask>
          )}
        </defs>

        <g mask={edgeFade > 0 ? `url(#${maskId})` : undefined}>
          {cross && (
            <g transform={`rotate(${tilt - crossAngle / 2} 600 260)`} opacity={0.85}>
              {ribbon && (
                <use
                  href={`#${pathIdCross}`}
                  fill="none"
                  stroke={resolvedCrossColor}
                  strokeWidth={ribbonWidth}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
              <text
                fontSize={fontSize}
                fontWeight={fontWeight}
                letterSpacing={letterSpacing}
                dominantBaseline="central"
                fill={outline ? 'none' : ribbon ? contrastTextColor(resolvedCrossColor) : resolvedCrossColor}
                stroke={outline ? (ribbon ? contrastTextColor(resolvedCrossColor) : resolvedCrossColor) : 'none'}
                strokeWidth={outline ? outlineWidth : 0}
              >
                <textPath ref={tpCrossRef} href={`#${pathIdCross}`} startOffset="0">
                  {repeatedString}
                </textPath>
              </text>
            </g>
          )}

          <g transform={`rotate(${cross ? tilt + crossAngle / 2 : tilt} 600 260)`}>
            {ribbon && (
              <use
                href={`#${pathIdMain}`}
                fill="none"
                stroke={ribbonColor}
                strokeWidth={ribbonWidth}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}
            <text
              fontSize={fontSize}
              fontWeight={fontWeight}
              letterSpacing={letterSpacing}
              dominantBaseline="central"
              fill={outline ? 'none' : resolvedFill}
              stroke={outline ? resolvedFill : 'none'}
              strokeWidth={outline ? outlineWidth : 0}
            >
              <textPath ref={tpMainRef} href={`#${pathIdMain}`} startOffset="0">
                {repeatedString}
              </textPath>
            </text>
          </g>
        </g>
      </svg>
    </div>
  );
}

export default TextLoop;
