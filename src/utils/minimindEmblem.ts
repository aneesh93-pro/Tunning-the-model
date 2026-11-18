import * as THREE from 'three';

export interface GlbValidationResult {
  valid: boolean;
  kind: 'glb' | 'gltf' | 'invalid';
  reason?: string;
}

const GLB_MAGIC = 0x46546c67; // 'glTF' in little-endian

export function isValidGlbOrGltfBuffer(buffer: ArrayBuffer): GlbValidationResult {
  if (!buffer || buffer.byteLength < 12) {
    return { valid: false, kind: 'invalid', reason: 'Buffer too small to be GLB/glTF' };
  }

  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic === GLB_MAGIC) {
    const version = view.getUint32(4, true);
    const totalLength = view.getUint32(8, true);
    if (version < 1 || version > 2 || totalLength < 12 || totalLength > buffer.byteLength) {
      return { valid: false, kind: 'invalid', reason: 'Invalid GLB binary header length or version' };
    }
    return { valid: true, kind: 'glb' };
  }

  // Inspect leading bytes for JSON glTF vs HTML fallback (e.g. <!doctype html>)
  const sampleLen = Math.min(buffer.byteLength, 2048);
  const snippet = new TextDecoder('utf-8', { fatal: false })
    .decode(new Uint8Array(buffer, 0, sampleLen))
    .trimStart();

  if (snippet.startsWith('<')) {
    return { valid: false, kind: 'invalid', reason: 'Received HTML document instead of GLTF/GLB' };
  }

  if (snippet.startsWith('{') && snippet.includes('"asset"')) {
    return { valid: true, kind: 'gltf' };
  }

  return { valid: false, kind: 'invalid', reason: 'Unrecognized asset format' };
}

export function buildMiniMindEmblemGroup(): THREE.Group {
  const root = new THREE.Group();
  root.name = 'MiniMindAI_Emblem';

  // 1. Frozen Base Slab (W0) — dark anodized hexagonal plate with chamfered bevel
  const baseGroup = new THREE.Group();
  baseGroup.name = 'FrozenBaseSlab_W0';

  const hexShape = new THREE.Shape();
  const radius = 1.08;
  for (let i = 0; i < 6; i++) {
    const angle = (i / 6) * Math.PI * 2 + Math.PI / 6;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    if (i === 0) hexShape.moveTo(x, y);
    else hexShape.lineTo(x, y);
  }
  hexShape.closePath();

  const slabGeo = new THREE.ExtrudeGeometry(hexShape, {
    depth: 0.14,
    bevelEnabled: true,
    bevelSegments: 3,
    steps: 1,
    bevelSize: 0.04,
    bevelThickness: 0.035,
  });
  slabGeo.rotateX(Math.PI / 2);
  slabGeo.center();

  const slabMat = new THREE.MeshStandardMaterial({
    color: '#1e293b',
    metalness: 0.65,
    roughness: 0.58,
  });
  const slabMesh = new THREE.Mesh(slabGeo, slabMat);
  slabMesh.position.y = -0.62;
  baseGroup.add(slabMesh);

  const trimGeo = new THREE.TorusGeometry(0.84, 0.024, 12, 48);
  trimGeo.rotateX(Math.PI / 2);
  const trimMat = new THREE.MeshStandardMaterial({
    color: '#64748b',
    metalness: 0.9,
    roughness: 0.24,
  });
  const trimMesh = new THREE.Mesh(trimGeo, trimMat);
  trimMesh.position.y = -0.5;
  baseGroup.add(trimMesh);
  root.add(baseGroup);

  // 2. Dual LoRA Matrix Rings (A * B)
  const ringAGeo = new THREE.TorusGeometry(0.76, 0.048, 20, 64);
  const ringAMat = new THREE.MeshStandardMaterial({
    color: '#2563eb',
    emissive: '#1d4ed8',
    emissiveIntensity: 0.35,
    metalness: 0.88,
    roughness: 0.22,
  });
  const ringA = new THREE.Mesh(ringAGeo, ringAMat);
  ringA.name = 'LoRA_Matrix_A';
  ringA.rotation.set(Math.PI / 4, Math.PI / 6, 0);
  ringA.position.y = 0.06;
  root.add(ringA);

  const ringBGeo = new THREE.TorusGeometry(0.64, 0.04, 20, 64);
  const ringBMat = new THREE.MeshStandardMaterial({
    color: '#06b6d4',
    emissive: '#0891b2',
    emissiveIntensity: 0.4,
    metalness: 0.85,
    roughness: 0.2,
  });
  const ringB = new THREE.Mesh(ringBGeo, ringBMat);
  ringB.name = 'LoRA_Matrix_B';
  ringB.rotation.set(-Math.PI / 4, -Math.PI / 5, Math.PI / 7);
  ringB.position.y = 0.06;
  root.add(ringB);

  // 3. NF4 Neural Core & 4 Workflow Nodes
  const coreGroup = new THREE.Group();
  coreGroup.name = 'NF4_NeuralCore';
  coreGroup.position.y = 0.06;

  const coreGeo = new THREE.OctahedronGeometry(0.32, 0);
  const coreMat = new THREE.MeshStandardMaterial({
    color: '#8b5cf6',
    emissive: '#6d28d9',
    emissiveIntensity: 0.55,
    metalness: 0.35,
    roughness: 0.18,
    flatShading: true,
  });
  const coreMesh = new THREE.Mesh(coreGeo, coreMat);
  coreGroup.add(coreMesh);

  const nodeGeo = new THREE.SphereGeometry(0.085, 20, 20);
  const nodeMat = new THREE.MeshStandardMaterial({
    color: '#38bdf8',
    emissive: '#0284c7',
    emissiveIntensity: 0.45,
    metalness: 0.75,
    roughness: 0.16,
  });
  const strutGeo = new THREE.CylinderGeometry(0.014, 0.014, 0.48, 8);
  const strutMat = new THREE.MeshStandardMaterial({
    color: '#94a3b8',
    metalness: 0.88,
    roughness: 0.28,
  });

  const nodeAngles = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];
  for (const ang of nodeAngles) {
    const nx = Math.cos(ang) * 0.48;
    const nz = Math.sin(ang) * 0.48;
    const node = new THREE.Mesh(nodeGeo, nodeMat);
    node.position.set(nx, 0, nz);
    coreGroup.add(node);

    const strut = new THREE.Mesh(strutGeo, strutMat);
    strut.position.set(nx * 0.5, 0, nz * 0.5);
    strut.rotation.z = Math.PI / 2;
    strut.rotation.y = -ang;
    coreGroup.add(strut);
  }
  root.add(coreGroup);

  // 4. Grounded Verification Halo
  const haloGeo = new THREE.TorusGeometry(0.42, 0.025, 16, 48);
  haloGeo.rotateX(Math.PI / 2);
  const haloMat = new THREE.MeshStandardMaterial({
    color: '#e2e8f0',
    emissive: '#cbd5e1',
    emissiveIntensity: 0.2,
    metalness: 0.94,
    roughness: 0.15,
  });
  const halo = new THREE.Mesh(haloGeo, haloMat);
  halo.name = 'VerificationHalo';
  halo.position.y = 0.72;
  root.add(halo);

  return root;
}

export function disposeObject3DResources(root: THREE.Object3D): void {
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.geometry) {
      mesh.geometry.dispose();
    }
    if (mesh.material) {
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((m) => m.dispose());
      } else {
        mesh.material.dispose();
      }
    }
  });
}
