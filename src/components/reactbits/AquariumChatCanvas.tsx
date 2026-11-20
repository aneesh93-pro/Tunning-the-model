import React, { useEffect, useRef } from 'react';

interface FishAgent {
  species: 'blue_yellow' | 'minnow' | 'gold' | 'clown';
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseSpeed: number;
  dir: 1 | -1;
  phase: number;
  schoolCenterY: number;
}

interface BubbleParticle {
  x: number;
  y: number;
  vy: number;
  wobblePhase: number;
  size: 1 | 2;
  alpha: number;
}

interface WaterRipple {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  alpha: number;
}

interface KelpBlade {
  x: number;
  height: number;
  phase: number;
  speed: number;
}

interface AquariumChatCanvasProps {
  className?: string;
  dimmed?: boolean;
  darkTheme?: boolean;
}

const BAND_COLORS_LIGHT = [
  '#a5e9f4',
  '#7adbec',
  '#51cae0',
  '#2eb6cf',
  '#159db9',
  '#0f839d',
];

const BAND_COLORS_DARK = [
  '#15485d',
  '#123f53',
  '#0f3649',
  '#0c2e40',
  '#0a2737',
  '#07202e',
];

// Classic 4x4 Bayer matrix normalized to [0..15]
const BAYER_4X4 = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
];

function drawPixelSprite(
  ctx: CanvasRenderingContext2D,
  grid: string[],
  palette: Record<string, string>,
  cx: number,
  cy: number,
  flipLeft: boolean
) {
  const h = grid.length;
  const w = grid[0].length;
  const ox = Math.round(cx - w / 2);
  const oy = Math.round(cy - h / 2);

  for (let row = 0; row < h; row++) {
    const line = grid[row];
    for (let col = 0; col < w; col++) {
      const ch = line[flipLeft ? w - 1 - col : col];
      if (ch === '.') continue;
      const color = palette[ch];
      if (!color) continue;
      ctx.fillStyle = color;
      ctx.fillRect(ox + col, oy + row, 1, 1);
    }
  }
}

// Crisp pixel sprites matching the aquarium species
const BLUE_YELLOW_SPRITE = [
  '....YYYY......',
  '...YYYYYY.....',
  '..BBBBBBBB..YY',
  '.BBBBBBEBBBYYY',
  'BBBBBBBBBBBYYY',
  '.BBBBBBBBBB.YY',
  '..BBBBBBBB....',
  '...BBBBBB.....',
];

const CLOWN_SPRITE = [
  '...OOO..OO....',
  '..OOWOOWOOO..K',
  '.OOOWOOWOOEOKK',
  'OOOOWOOWOOOOKK',
  '.OOOWOOWOOOO.K',
  '..OOWOOWOOO...',
  '...OOO..OO....',
];

const GOLD_SPRITE = [
  '...GGGGG....',
  '..GGGGGGG.GG',
  '.GGGGGGGEGGG',
  'GGGGGGGGGGGG',
  '.GGGGGGGGGGG',
  '..GGGGGGG.GG',
  '...GGGGG....',
];

const MINNOW_SPRITE = [
  '..SSSS..',
  '.SSSSSE.',
  'SSSSSS.S',
  '.SSSSS..',
];

export const AquariumChatCanvas: React.FC<AquariumChatCanvasProps> = ({
  className = '',
  dimmed = false,
  darkTheme = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointerRef = useRef<{ x: number; y: number; active: boolean; lastRippleAt: number }>({
    x: -100,
    y: -100,
    active: false,
    lastRippleAt: 0,
  });
  const ripplesRef = useRef<WaterRipple[]>([]);
  const bubblesRef = useRef<BubbleParticle[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    const pixelScale = 4;
    let gridW = 190;
    let gridH = 110;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(240, Math.floor(rect.width));
      const h = Math.max(180, Math.floor(rect.height));
      gridW = Math.max(96, Math.floor(w / pixelScale));
      gridH = Math.max(64, Math.floor(h / pixelScale));
      canvas.width = gridW;
      canvas.height = gridH;
    };
    resize();

    const resizeObserver = new ResizeObserver(() => resize());
    resizeObserver.observe(canvas);

    // Initialize schools matching the screenshot composition
    const fish: FishAgent[] = [];
    // Top right/center blue & yellow tropical school (7 fish)
    for (let i = 0; i < 7; i++) {
      fish.push({
        species: 'blue_yellow',
        x: gridW * (0.44 + (i % 4) * 0.11),
        y: gridH * (0.12 + (i % 3) * 0.08),
        vx: 9 + (i % 3) * 2.2,
        vy: 0,
        baseSpeed: 8 + (i % 3) * 2,
        dir: i % 2 === 0 ? 1 : -1,
        phase: i * 0.85,
        schoolCenterY: gridH * 0.19,
      });
    }
    // Silver minnows darting near center-top (10 tiny minnows)
    for (let i = 0; i < 10; i++) {
      fish.push({
        species: 'minnow',
        x: gridW * (0.4 + (i % 5) * 0.05),
        y: gridH * (0.25 + (i % 4) * 0.045),
        vx: 13 + (i % 4) * 1.8,
        vy: 0,
        baseSpeed: 12 + (i % 3) * 2.5,
        dir: 1,
        phase: i * 0.5,
        schoolCenterY: gridH * 0.29,
      });
    }
    // Golden school mid-left (5 fish)
    for (let i = 0; i < 5; i++) {
      fish.push({
        species: 'gold',
        x: gridW * (0.22 + i * 0.065),
        y: gridH * (0.46 + (i % 3) * 0.06),
        vx: -7.5 - (i % 2) * 1.5,
        vy: 0,
        baseSpeed: 7 + (i % 3) * 1.4,
        dir: -1,
        phase: i * 1.1,
        schoolCenterY: gridH * 0.52,
      });
    }
    // Clownfish lower right school (6 fish)
    for (let i = 0; i < 6; i++) {
      fish.push({
        species: 'clown',
        x: gridW * (0.54 + (i % 4) * 0.07),
        y: gridH * (0.63 + (i % 3) * 0.075),
        vx: 6.5 + (i % 3) * 1.6,
        vy: 0,
        baseSpeed: 6.5 + (i % 3) * 1.4,
        dir: i === 2 ? -1 : 1,
        phase: i * 1.3,
        schoolCenterY: gridH * 0.68,
      });
    }

    // Initialize ambient bubbles
    bubblesRef.current = Array.from({ length: 18 }, (_, i) => ({
      x: ((i * 29) % 92) * 0.01 * gridW + 6,
      y: gridH * (0.25 + ((i * 17) % 65) * 0.01),
      vy: 6 + (i % 4) * 2.1,
      wobblePhase: i * 0.9,
      size: i % 3 === 0 ? 2 : 1,
      alpha: 0.45 + (i % 4) * 0.12,
    }));

    // Kelp stalks across bottom seabed
    const kelp: KelpBlade[] = Array.from({ length: 16 }, (_, i) => ({
      x: Math.round(((i + 0.4) / 16) * gridW),
      height: 11 + ((i * 7) % 9),
      phase: i * 0.72,
      speed: 1.5 + (i % 3) * 0.35,
    }));

    let rafId = 0;
    let lastTime = performance.now();

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      const t = now * 0.001;

      const bands = darkTheme ? BAND_COLORS_DARK : BAND_COLORS_LIGHT;
      const bandHeight = gridH / bands.length;

      // 1. Dithered ocean depth bands
      for (let y = 0; y < gridH - 8; y++) {
        const exactBand = Math.min(bands.length - 1, Math.floor(y / bandHeight));
        const nextBand = Math.min(bands.length - 1, exactBand + 1);
        const localFrac = (y - exactBand * bandHeight) / bandHeight;
        const primary = bands[exactBand];
        const secondary = bands[nextBand];

        if (localFrac < 0.72 || exactBand === nextBand) {
          ctx.fillStyle = primary;
          ctx.fillRect(0, y, gridW, 1);
        } else {
          const threshold = ((localFrac - 0.72) / 0.28) * 16;
          for (let x = 0; x < gridW; x++) {
            const useNext = BAYER_4X4[y & 3][x & 3] < threshold;
            ctx.fillStyle = useNext ? secondary : primary;
            ctx.fillRect(x, y, 1, 1);
          }
        }
      }

      // 2. Soft diagonal sunlight shafts & organic caustic blotches
      ctx.save();
      ctx.fillStyle = darkTheme ? 'rgba(125, 211, 252, 0.06)' : 'rgba(255, 255, 255, 0.11)';
      for (let s = 0; s < 3; s++) {
        const baseOffset = gridW * (0.04 + s * 0.28) + Math.sin(t * 0.35 + s) * 6;
        for (let y = 0; y < gridH - 12; y += 2) {
          const xStart = Math.round(baseOffset + y * 0.42);
          const rayW = Math.round(14 + Math.sin(y * 0.08 + t + s) * 4);
          ctx.fillRect(xStart, y, rayW, 2);
        }
      }
      ctx.restore();

      // 3. Gently swaying green pixel kelp
      for (const k of kelp) {
        const kxBase = Math.min(gridW - 2, Math.max(1, k.x));
        for (let step = 0; step < k.height; step++) {
          const sway = Math.round(Math.sin(t * k.speed + k.phase + step * 0.38) * (step * 0.12));
          const ky = gridH - 7 - step;
          ctx.fillStyle = step % 3 === 0 ? '#3fa35b' : step % 2 === 0 ? '#2b8248' : '#1e6336';
          ctx.fillRect(kxBase + sway, ky, 1, 1);
          if (step < k.height - 2 && step % 4 === 1) {
            ctx.fillStyle = '#4bb96b';
            ctx.fillRect(kxBase + sway + 1, ky, 1, 1);
          }
        }
      }

      // 4. Sandy seabed wave
      for (let x = 0; x < gridW; x++) {
        const ridge = Math.round(
          7 + Math.sin(x * 0.11) * 1.4 + Math.cos(x * 0.047) * 1.1
        );
        const topY = gridH - ridge;
        ctx.fillStyle = darkTheme ? '#827155' : '#ead4ab';
        ctx.fillRect(x, topY, 1, 2);
        ctx.fillStyle = darkTheme ? '#6d5d44' : '#d8be8f';
        ctx.fillRect(x, topY + 2, 1, 3);
        ctx.fillStyle = darkTheme ? '#564935' : '#c6a775';
        ctx.fillRect(x, topY + 5, 1, ridge);
        if ((x * 13) % 19 === 0) {
          ctx.fillStyle = darkTheme ? '#9a8666' : '#f5e6c8';
          ctx.fillRect(x, topY + 3, 1, 1);
        }
      }

      // 5. Bubbles rising with pixel '+' sparkle motif
      const bubbles = bubblesRef.current;
      for (let i = bubbles.length - 1; i >= 0; i--) {
        const b = bubbles[i];
        b.y -= b.vy * dt;
        const bx = Math.round(b.x + Math.sin(t * 2.2 + b.wobblePhase) * 1.2);
        const by = Math.round(b.y);
        if (by < 4) {
          if (bubbles.length > 20) {
            bubbles.splice(i, 1);
          } else {
            b.y = gridH - 12 - Math.random() * 8;
            b.x = 6 + Math.random() * (gridW - 12);
          }
          continue;
        }
        ctx.fillStyle = `rgba(255, 255, 255, ${b.alpha.toFixed(2)})`;
        if (b.size === 2) {
          ctx.fillRect(bx, by - 1, 1, 1);
          ctx.fillRect(bx, by + 1, 1, 1);
          ctx.fillRect(bx - 1, by, 1, 1);
          ctx.fillRect(bx + 1, by, 1, 1);
        } else {
          ctx.fillRect(bx, by, 1, 1);
        }
      }

      // 6. Interactive water ripples
      const ripples = ripplesRef.current;
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rp = ripples[i];
        rp.radius += 26 * dt;
        rp.alpha -= 0.72 * dt;
        if (rp.alpha <= 0.03 || rp.radius >= rp.maxRadius) {
          ripples.splice(i, 1);
          continue;
        }
        ctx.fillStyle = `rgba(255, 255, 255, ${rp.alpha.toFixed(2)})`;
        const steps = 20;
        for (let s = 0; s < steps; s++) {
          const ang = (s / steps) * Math.PI * 2;
          const rx = Math.round(rp.x + Math.cos(ang) * rp.radius);
          const ry = Math.round(rp.y + Math.sin(ang) * rp.radius * 0.65);
          ctx.fillRect(rx, ry, 1, 1);
        }
      }

      // 7. Update and render fish schools
      const pointer = pointerRef.current;
      for (const f of fish) {
        // Gentle school cruising + sine vertical bob
        const targetVx = f.dir * f.baseSpeed;
        f.vx += (targetVx - f.vx) * Math.min(1, dt * 1.8);
        f.vy += ((f.schoolCenterY + Math.sin(t * 1.4 + f.phase) * 4 - f.y) * 0.6 - f.vy) * Math.min(1, dt * 2.2);

        // Pointer water wake interaction
        if (pointer.active) {
          const dx = f.x - pointer.x;
          const dy = f.y - pointer.y;
          const dist = Math.hypot(dx, dy);
          if (dist > 0.1 && dist < 24) {
            const force = ((24 - dist) / 24) * 32;
            f.vx += (dx / dist) * force * dt;
            f.vy += (dy / dist) * force * dt;
            if (Math.abs(f.vx) > 2) {
              f.dir = f.vx >= 0 ? 1 : -1;
            }
          }
        }

        f.x += f.vx * dt;
        f.y += f.vy * dt;

        // Soft tank bounds
        if (f.x < 10) {
          f.x = 10;
          f.dir = 1;
          f.vx = Math.abs(f.vx);
        } else if (f.x > gridW - 10) {
          f.x = gridW - 10;
          f.dir = -1;
          f.vx = -Math.abs(f.vx);
        }
        if (f.y < 7) f.y = 7;
        if (f.y > gridH - 14) f.y = gridH - 14;

        const flipLeft = f.dir === -1;
        if (f.species === 'blue_yellow') {
          drawPixelSprite(
            ctx,
            BLUE_YELLOW_SPRITE,
            {
              B: '#1554df',
              Y: '#facc15',
              E: '#0f172a',
            },
            f.x,
            f.y,
            flipLeft
          );
        } else if (f.species === 'clown') {
          drawPixelSprite(
            ctx,
            CLOWN_SPRITE,
            {
              O: '#f46224',
              W: '#ffffff',
              K: '#1e293b',
              E: '#0f172a',
            },
            f.x,
            f.y,
            flipLeft
          );
        } else if (f.species === 'gold') {
          drawPixelSprite(
            ctx,
            GOLD_SPRITE,
            {
              G: '#eab308',
              E: '#1e293b',
            },
            f.x,
            f.y,
            flipLeft
          );
        } else {
          drawPixelSprite(
            ctx,
            MINNOW_SPRITE,
            {
              S: '#94a3b8',
              E: '#0f172a',
            },
            f.x,
            f.y,
            flipLeft
          );
        }
      }

      if (dimmed) {
        ctx.fillStyle = darkTheme ? 'rgba(9, 11, 16, 0.45)' : 'rgba(248, 250, 252, 0.35)';
        ctx.fillRect(0, 0, gridW, gridH);
      }

      rafId = requestAnimationFrame(frame);
    };

    rafId = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
    };
  }, [dimmed, darkTheme]);

  const toGridCoords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { gx: 0, gy: 0 };
    const rect = canvas.getBoundingClientRect();
    const gx = ((e.clientX - rect.left) / Math.max(1, rect.width)) * canvas.width;
    const gy = ((e.clientY - rect.top) / Math.max(1, rect.height)) * canvas.height;
    return { gx, gy };
  };

  return (
    <canvas
      ref={canvasRef}
      aria-label="Interactive pixel aquarium water backdrop"
      onPointerMove={(e) => {
        const { gx, gy } = toGridCoords(e);
        const now = performance.now();
        pointerRef.current.x = gx;
        pointerRef.current.y = gy;
        pointerRef.current.active = true;
        if (now - pointerRef.current.lastRippleAt > 140) {
          pointerRef.current.lastRippleAt = now;
          ripplesRef.current.push({
            x: gx,
            y: gy,
            radius: 2,
            maxRadius: 11,
            alpha: 0.42,
          });
        }
      }}
      onPointerLeave={() => {
        pointerRef.current.active = false;
      }}
      onPointerDown={(e) => {
        const { gx, gy } = toGridCoords(e);
        pointerRef.current.x = gx;
        pointerRef.current.y = gy;
        pointerRef.current.active = true;
        ripplesRef.current.push({
          x: gx,
          y: gy,
          radius: 1.5,
          maxRadius: 18,
          alpha: 0.72,
        });
        for (let i = 0; i < 4; i++) {
          bubblesRef.current.push({
            x: gx + (i - 1.5) * 2.5,
            y: gy + (i % 2) * 1.5,
            vy: 9 + i * 1.5,
            wobblePhase: i * 1.1,
            size: i % 2 === 0 ? 2 : 1,
            alpha: 0.75,
          });
        }
      }}
      className={`w-full h-full block cursor-crosshair ${className}`}
      style={{ imageRendering: 'pixelated' }}
    />
  );
};
