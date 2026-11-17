import * as THREE from 'three';

export function patchLightsFragmentBegin(source: string): string {
  return source.replace(
    '1.0 / EssMs - 1.0',
    '1.0 / max( EssMs, 1e-5 ) - 1.0',
  );
}

export function patchLightsPhysicalParsFragment(source: string): string {
  let out = source;
  out = out.replace(
    '( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 )',
    '( f - vec3( f90 ) * x5 ) / max( 1.0 - x5, 1e-4 )',
  );
  out = out.replace(
    'FssEss * Favg / ( 1.0 - Ems * Favg )',
    'FssEss * Favg / max( vec3( 1.0 ) - Ems * Favg, vec3( 1e-5 ) )',
  );
  out = out.replace(
    '1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) )',
    '1.0 / max( 4.0 * ( dotNL + dotNV - dotNL * dotNV ), 1e-5 )',
  );
  return out;
}

export function patchDefaultNormalVertex(source: string): string {
  let out = source;
  out = out.replace(
    'transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );',
    'transformedNormal /= max( vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) ), vec3( 1e-6 ) );',
  );
  out = out.replace(
    'transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );',
    'transformedNormal /= max( vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) ), vec3( 1e-6 ) );',
  );
  return out;
}

const APPLIED_KEY = Symbol.for('minimind.threeShaderSafetyFixesApplied');

export function applyThreeShaderSafetyFixes(): void {
  if (Reflect.get(globalThis, APPLIED_KEY)) return;
  Reflect.set(globalThis, APPLIED_KEY, true);

  if (THREE.ShaderChunk.lights_fragment_begin) {
    THREE.ShaderChunk.lights_fragment_begin = patchLightsFragmentBegin(
      THREE.ShaderChunk.lights_fragment_begin,
    );
  }
  if (THREE.ShaderChunk.lights_physical_pars_fragment) {
    THREE.ShaderChunk.lights_physical_pars_fragment = patchLightsPhysicalParsFragment(
      THREE.ShaderChunk.lights_physical_pars_fragment,
    );
  }
  if (THREE.ShaderChunk.defaultnormal_vertex) {
    THREE.ShaderChunk.defaultnormal_vertex = patchDefaultNormalVertex(
      THREE.ShaderChunk.defaultnormal_vertex,
    );
  }
}
