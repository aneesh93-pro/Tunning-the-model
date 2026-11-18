import {
  Component,
  Suspense,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
  type Ref,
} from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { ContactShadows, Environment, Lightformer, useFBX } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import * as THREE from 'three';
import {
  buildMiniMindEmblemGroup,
  disposeObject3DResources,
  isValidGlbOrGltfBuffer,
} from '../../utils/minimindEmblem';
import { applyThreeShaderSafetyFixes } from '../../utils/threeShaderFixes';

applyThreeShaderSafetyFixes();

const THREE_CLOCK_WARNING_INSTALLED = Symbol.for('minimind.threeClockWarnFiltered');
if (typeof window !== 'undefined' && !Reflect.get(globalThis, THREE_CLOCK_WARNING_INSTALLED)) {
  Reflect.set(globalThis, THREE_CLOCK_WARNING_INSTALLED, true);
  const originalWarn = console.warn.bind(console);
  console.warn = (...args: unknown[]) => {
    const first = typeof args[0] === 'string' ? args[0] : '';
    const second = typeof args[1] === 'string' ? args[1] : '';
    if (first.includes('THREE.Clock: This module has been deprecated')) {
      return;
    }
    if (
      first.includes('THREE.WebGLProgram: Program Info Log') &&
      second.includes('warning X4008: floating point division by zero') &&
      !second.includes('ERROR')
    ) {
      return;
    }
    originalWarn(...args);
  };
}

export type ModelViewerEnvironment =
  | 'studio'
  | 'city'
  | 'sunset'
  | 'dawn'
  | 'warehouse'
  | 'apartment'
  | 'lobby'
  | 'night'
  | 'park'
  | 'forest'
  | 'none';

export type ModelViewerIntro = 'rise' | 'spin' | 'fade' | 'none';

export interface ModelViewerHandle {
  capture: (type?: string) => string | null;
}

export interface ModelViewerProps {
  url: string;
  width?: number | string;
  height?: number | string;
  environment?: ModelViewerEnvironment;
  environmentIntensity?: number;
  exposure?: number;
  intro?: ModelViewerIntro;
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  autoRotateDelay?: number;
  dragRotate?: boolean;
  turntable?: boolean;
  zoom?: boolean;
  minZoom?: number;
  maxZoom?: number;
  hoverTilt?: number;
  float?: number;
  shadow?: boolean;
  shadowOpacity?: number;
  fit?: number;
  offsetX?: number;
  offsetY?: number;
  startAngle?: number;
  tilt?: number;
  resetOnDoubleClick?: boolean;
  onModelLoaded?: () => void;
  ref?: Ref<ModelViewerHandle>;
  className?: string;
  style?: CSSProperties;
}

interface BoundsInfo {
  scale: number;
  center: THREE.Vector3;
  radius: number;
  floor: number;
}

interface MotionState {
  yaw: number;
  pitch: number;
  yawVelocity: number;
  pitchVelocity: number;
  zoom: number;
  zoomGoal: number;
  tiltX: number;
  tiltY: number;
  tiltGoalX: number;
  tiltGoalY: number;
  spinSpeed: number;
  idle: number;
  dragging: boolean;
  intro: number;
  clock: number;
  reset: { yaw: number; pitch: number } | null;
}

interface SettingsSnapshot {
  autoRotate: boolean;
  autoRotateSpeed: number;
  autoRotateDelay: number;
  dragRotate: boolean;
  turntable: boolean;
  zoom: boolean;
  minZoom: number;
  maxZoom: number;
  hoverTilt: number;
  float: number;
  shadow: boolean;
  shadowOpacity: number;
  intro: ModelViewerIntro;
  startAngle: number;
  tilt: number;
  offsetX: number;
  offsetY: number;
  fit: number;
  resetOnDoubleClick: boolean;
  reducedMotion: boolean;
}

const FOV = 30;
const INTRO_SECONDS = 1.3;
const DRAG_SPEED = 0.0065;
const PITCH_LIMIT = (65 * Math.PI) / 180;

const toRadians = (degrees: number) => (degrees * Math.PI) / 180;

const measured = new WeakMap<THREE.Object3D, BoundsInfo>();
const announced = new WeakSet<THREE.Object3D>();

const measure = (object: THREE.Object3D): BoundsInfo => {
  const existing = measured.get(object);
  if (existing) return existing;
  const box = new THREE.Box3().setFromObject(object);
  const extent = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const scale = 2 / Math.max(extent.x, extent.y, extent.z, 0.0001);
  const bounds: BoundsInfo = {
    scale,
    center,
    radius: (extent.length() / 2) * scale,
    floor: (extent.y / 2) * scale,
  };
  measured.set(object, bounds);
  return bounds;
};

const ENV_PALETTES: Record<
  Exclude<ModelViewerEnvironment, 'none'>,
  { top: string; key: string; fill: string; rim: string }
> = {
  studio: { top: '#ffffff', key: '#dbeafe', fill: '#38bdf8', rim: '#818cf8' },
  city: { top: '#e0f2fe', key: '#93c5fd', fill: '#0ea5e9', rim: '#6366f1' },
  sunset: { top: '#fed7aa', key: '#fb923c', fill: '#f472b6', rim: '#8b5cf6' },
  dawn: { top: '#fce7f3', key: '#93c5fd', fill: '#f9a8d4', rim: '#38bdf8' },
  warehouse: { top: '#f1f5f9', key: '#cbd5e1', fill: '#64748b', rim: '#3b82f6' },
  apartment: { top: '#fef3c7', key: '#fde68a', fill: '#93c5fd', rim: '#cbd5e1' },
  lobby: { top: '#fef9c3', key: '#e2e8f0', fill: '#60a5fa', rim: '#a78bfa' },
  night: { top: '#1e3a8a', key: '#3b82f6', fill: '#06b6d4', rim: '#8b5cf6' },
  park: { top: '#ecfccb', key: '#bae6fd', fill: '#34d399', rim: '#60a5fa' },
  forest: { top: '#dcfce7', key: '#a7f3d0', fill: '#38bdf8', rim: '#818cf8' },
};

const FallbackEmblemModel = ({
  children,
}: {
  children: (obj: THREE.Object3D) => ReactNode;
}) => {
  const object = useMemo(() => buildMiniMindEmblemGroup(), []);
  useEffect(() => () => disposeObject3DResources(object), [object]);
  return <>{children(object)}</>;
};

class ModelSceneErrorBoundary extends Component<
  { fallback: ReactNode; children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch() {
    // Fallback renders procedural MiniMind 3D Emblem cleanly
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

const SafeGltfModel = ({
  url,
  children,
}: {
  url: string;
  children: (obj: THREE.Object3D) => ReactNode;
}) => {
  const [object, setObject] = useState<THREE.Object3D | null>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    let loadedObj: THREE.Object3D | null = null;

    const useFallback = () => {
      if (!active) return;
      const fallback = buildMiniMindEmblemGroup();
      loadedObj = fallback;
      setObject(fallback);
    };

    const loadValidatedGlb = async () => {
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) {
          useFallback();
          return;
        }
        const contentType = (res.headers.get('content-type') || '').toLowerCase();
        if (contentType.includes('text/html')) {
          useFallback();
          return;
        }
        const buffer = await res.arrayBuffer();
        if (!active) return;
        const validation = isValidGlbOrGltfBuffer(buffer);
        if (!validation.valid) {
          useFallback();
          return;
        }

        const loader = new GLTFLoader();
        const basePath = url.slice(0, url.lastIndexOf('/') + 1);
        loader.parse(
          buffer,
          basePath,
          (gltf) => {
            if (!active) return;
            const cloned = gltf.scene.clone(true);
            loadedObj = cloned;
            setObject(cloned);
          },
          () => {
            useFallback();
          }
        );
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        useFallback();
      }
    };

    void loadValidatedGlb();

    return () => {
      active = false;
      controller.abort();
      if (loadedObj) {
        disposeObject3DResources(loadedObj);
      }
    };
  }, [url]);

  if (!object) return null;
  return <>{children(object)}</>;
};

const FbxModel = ({
  url,
  children,
}: {
  url: string;
  children: (obj: THREE.Object3D) => ReactNode;
}) => {
  const fbx = useFBX(url);
  const object = useMemo(() => fbx.clone(true), [fbx]);
  return <>{children(object)}</>;
};

const ObjModel = ({
  url,
  children,
}: {
  url: string;
  children: (obj: THREE.Object3D) => ReactNode;
}) => {
  const obj = useLoader(OBJLoader, url);
  const object = useMemo(() => obj.clone(true), [obj]);
  return <>{children(object)}</>;
};

const Model = ({
  url,
  children,
}: {
  url: string;
  children: (obj: THREE.Object3D) => ReactNode;
}) => {
  const extension = url.split('?')[0].split('.').pop()?.toLowerCase() ?? '';
  if (extension === 'fbx') return <FbxModel url={url}>{children}</FbxModel>;
  if (extension === 'obj') return <ObjModel url={url}>{children}</ObjModel>;
  return <SafeGltfModel url={url}>{children}</SafeGltfModel>;
};

const Exposure = ({ value }: { value: number }) => {
  const gl = useThree((state) => state.gl);
  const invalidate = useThree((state) => state.invalidate);
  useEffect(() => {
    gl.toneMapping = THREE.NeutralToneMapping;
    gl.toneMappingExposure = value;
    invalidate();
  }, [gl, value, invalidate]);
  return null;
};

const Lighting = ({
  environment,
  intensity,
}: {
  environment: ModelViewerEnvironment;
  intensity: number;
}) => {
  const invalidate = useThree((state) => state.invalidate);
  useEffect(() => {
    invalidate();
  }, [environment, intensity, invalidate]);

  if (environment === 'none') return null;
  const palette = ENV_PALETTES[environment] ?? ENV_PALETTES.studio;

  return (
    <Environment resolution={256} frames={1} environmentIntensity={intensity}>
      <Lightformer
        form="rect"
        intensity={2.4}
        position={[0, 6, 2]}
        rotation={[Math.PI / 2.2, 0, 0]}
        scale={[10, 6, 1]}
        color={palette.top}
      />
      <Lightformer
        form="rect"
        intensity={1.9}
        position={[-5, 2, 4]}
        rotation={[0, Math.PI / 4, 0]}
        scale={[6, 5, 1]}
        color={palette.key}
      />
      <Lightformer
        form="rect"
        intensity={1.7}
        position={[5, 0.5, 3]}
        rotation={[0, -Math.PI / 3, 0]}
        scale={[6, 5, 1]}
        color={palette.fill}
      />
      <Lightformer
        form="ring"
        intensity={1.4}
        position={[0, -4, -3]}
        rotation={[-Math.PI / 2.4, 0, 0]}
        scale={[7, 7, 1]}
        color={palette.rim}
      />
    </Environment>
  );
};

const Stage = ({
  object,
  settingsRef,
  visibleRef,
  onReady,
}: {
  object: THREE.Object3D;
  settingsRef: React.RefObject<SettingsSnapshot>;
  visibleRef: React.RefObject<boolean>;
  onReady: () => void;
}) => {
  const { camera, gl, size, invalidate } = useThree();
  const placeRef = useRef<THREE.Group>(null);
  const floatRef = useRef<THREE.Group>(null);
  const tiltRef = useRef<THREE.Group>(null);
  const spinRef = useRef<THREE.Group>(null);
  const fitRef = useRef<THREE.Group>(null);
  const bounds = useMemo(() => measure(object), [object]);
  const motionRef = useRef<MotionState | null>(null);

  if (!motionRef.current) {
    const s = settingsRef.current;
    motionRef.current = {
      yaw: toRadians(s.startAngle),
      pitch: toRadians(s.tilt),
      yawVelocity: 0,
      pitchVelocity: 0,
      zoom: 1,
      zoomGoal: 1,
      tiltX: 0,
      tiltY: 0,
      tiltGoalX: 0,
      tiltGoalY: 0,
      spinSpeed: 0,
      idle: 0,
      dragging: false,
      intro: s.intro === 'none' ? 1 : 0,
      clock: 0,
      reset: null,
    };
  }
  const motion = motionRef.current;

  useEffect(() => {
    if (announced.has(object)) return;
    announced.add(object);
    onReady?.();
  }, [object, onReady]);

  useEffect(() => {
    const element = gl.domElement;
    const pointers = new Map<number, { x: number; y: number; t: number }>();
    let pinchStart = 0;
    let zoomStart = 1;

    const wake = () => invalidate();

    const onDown = (event: PointerEvent) => {
      const s = settingsRef.current;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY, t: event.timeStamp });
      if (pointers.size === 2 && s.zoom) {
        const [a, b] = [...pointers.values()];
        pinchStart = Math.hypot(a.x - b.x, a.y - b.y);
        zoomStart = motion.zoomGoal;
        motion.dragging = false;
      } else if (
        pointers.size === 1 &&
        s.dragRotate &&
        (event.pointerType !== 'mouse' || event.button === 0)
      ) {
        motion.dragging = true;
        motion.yawVelocity = 0;
        motion.pitchVelocity = 0;
        element.setPointerCapture?.(event.pointerId);
        element.style.cursor = 'grabbing';
      }
      motion.idle = 0;
      wake();
    };

    const onMove = (event: PointerEvent) => {
      const s = settingsRef.current;
      const rect = element.getBoundingClientRect();
      if (event.pointerType === 'mouse' || event.pointerType === 'pen') {
        const nx = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
        const ny = ((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1;
        motion.tiltGoalX = ny * s.hoverTilt * 0.6;
        motion.tiltGoalY = nx * s.hoverTilt;
      }
      const last = pointers.get(event.pointerId);
      if (last) {
        const dx = event.clientX - last.x;
        const dy = event.clientY - last.y;
        const elapsed = Math.max((event.timeStamp - last.t) / 1000, 1 / 240);
        last.x = event.clientX;
        last.y = event.clientY;
        last.t = event.timeStamp;
        if (pointers.size === 2 && s.zoom && pinchStart > 0) {
          const [a, b] = [...pointers.values()];
          const distance = Math.hypot(a.x - b.x, a.y - b.y);
          motion.zoomGoal = THREE.MathUtils.clamp(
            (zoomStart * distance) / pinchStart,
            s.minZoom,
            s.maxZoom
          );
        } else if (motion.dragging) {
          const m = motion;
          m.yaw += dx * DRAG_SPEED;
          m.yawVelocity = m.yawVelocity * 0.4 + ((dx * DRAG_SPEED) / elapsed) * 0.6;
          if (!s.turntable) {
            m.pitch = THREE.MathUtils.clamp(
              m.pitch + dy * DRAG_SPEED,
              -PITCH_LIMIT,
              PITCH_LIMIT
            );
            m.pitchVelocity = m.pitchVelocity * 0.4 + ((dy * DRAG_SPEED) / elapsed) * 0.6;
          }
          m.idle = 0;
        }
      }
      wake();
    };

    const onUp = (event: PointerEvent) => {
      const last = pointers.get(event.pointerId);
      if (last && event.timeStamp - last.t > 80) {
        motion.yawVelocity = 0;
        motion.pitchVelocity = 0;
      }
      pointers.delete(event.pointerId);
      if (pointers.size < 2) pinchStart = 0;
      if (pointers.size === 0) {
        motion.dragging = false;
        element.style.cursor = '';
      }
      wake();
    };

    const onLeave = () => {
      motion.tiltGoalX = 0;
      motion.tiltGoalY = 0;
      wake();
    };

    const onWheel = (event: WheelEvent) => {
      const s = settingsRef.current;
      if (!s.zoom || !(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      const m = motion;
      m.zoomGoal = THREE.MathUtils.clamp(
        m.zoomGoal * Math.exp(-event.deltaY * 0.01),
        s.minZoom,
        s.maxZoom
      );
      m.idle = 0;
      wake();
    };

    const onDoubleClick = () => {
      const s = settingsRef.current;
      if (!s.resetOnDoubleClick) return;
      const m = motion;
      const turns = Math.round((m.yaw - toRadians(s.startAngle)) / (Math.PI * 2));
      m.reset = { yaw: toRadians(s.startAngle) + turns * Math.PI * 2, pitch: toRadians(s.tilt) };
      m.zoomGoal = 1;
      m.yawVelocity = 0;
      m.pitchVelocity = 0;
      m.idle = 0;
      wake();
    };

    element.addEventListener('pointerdown', onDown);
    element.addEventListener('pointermove', onMove);
    element.addEventListener('pointerup', onUp);
    element.addEventListener('pointercancel', onUp);
    element.addEventListener('pointerleave', onLeave);
    element.addEventListener('wheel', onWheel, { passive: false });
    element.addEventListener('dblclick', onDoubleClick);
    return () => {
      element.removeEventListener('pointerdown', onDown);
      element.removeEventListener('pointermove', onMove);
      element.removeEventListener('pointerup', onUp);
      element.removeEventListener('pointercancel', onUp);
      element.removeEventListener('pointerleave', onLeave);
      element.removeEventListener('wheel', onWheel);
      element.removeEventListener('dblclick', onDoubleClick);
    };
  }, [gl, invalidate, settingsRef, motion]);

  useFrame((_, delta) => {
    const place = placeRef.current;
    const floatGroup = floatRef.current;
    const tiltGroup = tiltRef.current;
    const spinGroup = spinRef.current;
    if (!place || !floatGroup || !tiltGroup || !spinGroup) return;

    const s = settingsRef.current;
    const m = motion;
    const dt = Math.min(delta, 0.05);
    m.clock += dt;
    m.idle += dt;
    const reduce = s.reducedMotion;

    if (m.intro < 1) m.intro = reduce ? 1 : Math.min(1, m.intro + dt / INTRO_SECONDS);
    const eased = 1 - Math.pow(1 - m.intro, 3);

    if (m.reset) {
      m.yaw += (m.reset.yaw - m.yaw) * (1 - Math.exp(-dt * 7));
      m.pitch += (m.reset.pitch - m.pitch) * (1 - Math.exp(-dt * 7));
      if (Math.abs(m.reset.yaw - m.yaw) < 0.001 && Math.abs(m.reset.pitch - m.pitch) < 0.001) {
        m.reset = null;
      }
    } else if (!m.dragging) {
      const decay = Math.exp(-dt * 4.5);
      m.yaw += m.yawVelocity * dt;
      m.pitch = THREE.MathUtils.clamp(m.pitch + m.pitchVelocity * dt, -PITCH_LIMIT, PITCH_LIMIT);
      m.yawVelocity *= decay;
      m.pitchVelocity *= decay;
      if (s.turntable) m.pitch += (toRadians(s.tilt) - m.pitch) * (1 - Math.exp(-dt * 4));
    }

    const spinning = s.autoRotate && !reduce && !m.dragging && m.idle > s.autoRotateDelay;
    m.spinSpeed +=
      ((spinning ? s.autoRotateSpeed : 0) - m.spinSpeed) * (1 - Math.exp(-dt * 2));
    m.yaw += m.spinSpeed * dt;

    m.zoom += (m.zoomGoal - m.zoom) * (1 - Math.exp(-dt * 8));
    m.tiltX += (m.tiltGoalX - m.tiltX) * (1 - Math.exp(-dt * 6));
    m.tiltY += (m.tiltGoalY - m.tiltY) * (1 - Math.exp(-dt * 6));

    const aspect = size.width / Math.max(size.height, 1);
    const halfView = Math.tan(toRadians(FOV) / 2);
    const fitDistance =
      bounds.radius / (Math.max(s.fit, 0.05) * halfView * Math.min(1, aspect));
    const distance = fitDistance / m.zoom;
    camera.position.set(0, 0, distance);
    camera.near = distance / 50;
    camera.far = distance * 50;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();

    const halfHeight = fitDistance * halfView;
    place.position.set(s.offsetX * halfHeight * aspect, s.offsetY * halfHeight, 0);

    const rise = s.intro === 'rise' ? (1 - eased) * -0.45 : 0;
    const grow = s.intro === 'rise' ? 0.9 + 0.1 * eased : 1;
    const twist =
      s.intro === 'spin'
        ? (1 - eased) * -Math.PI * 1.2
        : s.intro === 'rise'
          ? (1 - eased) * -0.5
          : 0;
    const bob = reduce ? 0 : Math.sin(m.clock * 1.3) * s.float * 0.06;

    floatGroup.position.y = rise + bob;
    floatGroup.scale.setScalar(grow);
    tiltGroup.rotation.set(m.tiltX, m.tiltY, 0);
    spinGroup.rotation.set(m.pitch, m.yaw + twist, 0, 'YXZ');

    const settling =
      m.intro < 1 ||
      m.dragging ||
      !!m.reset ||
      Math.abs(m.yawVelocity) + Math.abs(m.pitchVelocity) > 0.0005 ||
      Math.abs(m.zoomGoal - m.zoom) > 0.0005 ||
      Math.abs(m.tiltGoalX - m.tiltX) + Math.abs(m.tiltGoalY - m.tiltY) > 0.0005 ||
      m.spinSpeed > 0.0005 ||
      (s.autoRotate && !reduce) ||
      (s.float > 0 && !reduce);

    if (settling && visibleRef.current) invalidate();
  });

  return (
    <group ref={placeRef}>
      <group ref={floatRef}>
        <group ref={tiltRef}>
          <group ref={spinRef}>
            <group
              ref={fitRef}
              scale={bounds.scale}
              position={[
                -bounds.center.x * bounds.scale,
                -bounds.center.y * bounds.scale,
                -bounds.center.z * bounds.scale,
              ]}
            >
              <primitive object={object} />
            </group>
          </group>
        </group>
      </group>
      {settingsRef.current.shadow && (
        <ContactShadows
          position={[0, -bounds.floor - 0.08, 0]}
          opacity={settingsRef.current.shadowOpacity}
          scale={bounds.radius * 4}
          blur={2.6}
          far={bounds.floor * 2 + 0.5}
          resolution={512}
          color="#000000"
        />
      )}
    </group>
  );
};

export const ModelViewer = ({
  url,
  width = 400,
  height = 400,
  environment = 'studio',
  environmentIntensity = 1,
  exposure = 1,
  autoRotate = true,
  autoRotateSpeed = 0.4,
  autoRotateDelay = 2,
  dragRotate = true,
  turntable = false,
  zoom = true,
  minZoom = 0.7,
  maxZoom = 2,
  hoverTilt = 0.15,
  float = 0.3,
  shadow = true,
  shadowOpacity = 0.45,
  intro = 'rise',
  startAngle = -30,
  tilt = 10,
  offsetX = 0,
  offsetY = 0,
  fit = 0.8,
  resetOnDoubleClick = true,
  className = '',
  style,
  onModelLoaded,
  ref,
}: ModelViewerProps) => {
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const invalidateRef = useRef<(() => void) | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const visibleRef = useRef(true);
  const [ready, setReady] = useState(false);

  const settings: SettingsSnapshot = {
    autoRotate,
    autoRotateSpeed,
    autoRotateDelay: Math.max(0, autoRotateDelay),
    dragRotate,
    turntable,
    zoom,
    minZoom: Math.min(minZoom, maxZoom),
    maxZoom: Math.max(minZoom, maxZoom),
    hoverTilt: Math.max(0, hoverTilt),
    float: Math.max(0, float),
    shadow,
    shadowOpacity,
    intro,
    startAngle,
    tilt,
    offsetX,
    offsetY,
    fit,
    resetOnDoubleClick,
    reducedMotion:
      typeof window !== 'undefined' &&
      (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false),
  };

  const settingsRef = useRef<SettingsSnapshot>(settings);
  settingsRef.current = settings;

  useImperativeHandle(
    ref,
    () => ({
      capture: (type = 'image/png') => glRef.current?.domElement.toDataURL(type) ?? null,
    }),
    []
  );

  useEffect(() => {
    setReady(false);
  }, [url]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    const observer = new IntersectionObserver((entries) => {
      visibleRef.current = entries.some((entry) => entry.isIntersecting);
      if (visibleRef.current) invalidateRef.current?.();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleReady = useCallback(() => {
    setReady(true);
    onModelLoaded?.();
  }, [onModelLoaded]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ position: 'relative', width, height, ...style }}
    >
      <Canvas
        frameloop="demand"
        dpr={[1, 2]}
        camera={{ fov: FOV, position: [0, 0, 6], near: 0.01, far: 100 }}
        gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
        onCreated={(state) => {
          glRef.current = state.gl;
          invalidateRef.current = state.invalidate;
        }}
        style={{
          touchAction: 'pan-y',
          cursor: dragRotate ? 'grab' : 'default',
          opacity: ready || intro === 'none' ? 1 : 0,
          transition: intro === 'none' ? 'none' : 'opacity 0.6s ease',
        }}
      >
        <Exposure value={exposure} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[3, 5, 4]} intensity={0.85} />
        <Lighting environment={environment} intensity={environmentIntensity} />
        <Suspense fallback={null}>
          {url && (
            <ModelSceneErrorBoundary
              fallback={
                <FallbackEmblemModel>
                  {(object) => (
                    <Stage
                      object={object}
                      settingsRef={settingsRef}
                      visibleRef={visibleRef}
                      onReady={handleReady}
                    />
                  )}
                </FallbackEmblemModel>
              }
            >
              <Model url={url}>
                {(object) => (
                  <Stage
                    object={object}
                    settingsRef={settingsRef}
                    visibleRef={visibleRef}
                    onReady={handleReady}
                  />
                )}
              </Model>
            </ModelSceneErrorBoundary>
          )}
        </Suspense>
      </Canvas>
    </div>
  );
};

export default ModelViewer;
