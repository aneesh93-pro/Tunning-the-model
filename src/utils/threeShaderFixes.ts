import * as THREE from 'three';

export function patchLightsFragmentBegin(source: string): string {
  let out = source.replace(
    '1.0 / EssMs - 1.0',
    '1.0 / max( EssMs, 1e-5 ) - 1.0',
  );
  out = out.replace(
    'vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w',
    'vSpotLightCoord[ i ].xyz / ( vSpotLightCoord[ i ].w < 0.0 ? min( vSpotLightCoord[ i ].w, -1e-6 ) : max( vSpotLightCoord[ i ].w, 1e-6 ) )',
  );
  return out;
}

export function patchLightsParsBegin(source: string): string {
  return source.replace(
    'lightDistance / cutoffDistance',
    'lightDistance / max( cutoffDistance, 1e-5 )',
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
  out = out.replace(
    'return RECIPROCAL_PI * a2 / pow2( denom );',
    'return RECIPROCAL_PI * a2 / max( pow2( denom ), 1e-6 );',
  );
  out = out.replace(
    'float w2 = a2 / v2;',
    'float w2 = a2 / max( v2, 1e-6 );',
  );
  out = out.replace(
    'float invAlpha = 1.0 / alpha;',
    'float invAlpha = 1.0 / max( alpha, 1e-5 );',
  );
  out = out.replaceAll(
    'float A = 1.0 / ( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness );',
    'float A = 1.0 / max( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness, 1e-5 );',
  );
  return out;
}

export function patchShadowmapParsFragment(source: string): string {
  let out = source;
  out = out.replaceAll(
    'shadowCoord.xyz /= shadowCoord.w;',
    'shadowCoord.xyz /= ( shadowCoord.w < 0.0 ? min( shadowCoord.w, -1e-6 ) : max( shadowCoord.w, 1e-6 ) );',
  );
  out = out.replace(
    'vec2 texelSize = vec2( 1.0 ) / shadowMapSize;',
    'vec2 texelSize = vec2( 1.0 ) / max( shadowMapSize, vec2( 1.0 ) );',
  );
  out = out.replace(
    'float texelSize = shadowRadius / shadowMapSize.x;',
    'float texelSize = shadowRadius / max( shadowMapSize.x, 1.0 );',
  );
  return out;
}

export function patchIridescenceFragment(source: string): string {
  return source.replace(
    'vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );',
    'vec3 Rs = pow2( T121 ) * R23 / max( vec3( 1.0 ) - R123, vec3( 1e-5 ) );',
  );
}

export function patchTransmissionParsFragment(source: string): string {
  return source.replace(
    'vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;',
    'vec3 attenuationCoefficient = -log( max( attenuationColor, vec3( 1e-5 ) ) ) / max( attenuationDistance, 1e-5 );',
  );
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

function isBenignThreeWarning(first: string, second: string): boolean {
  if (first.includes('THREE.Clock: This module has been deprecated')) {
    return true;
  }
  if (
    first.includes('THREE.WebGLProgram: Program Info Log') &&
    second.includes('warning X4008: floating point division by zero') &&
    !second.includes('ERROR')
  ) {
    return true;
  }
  return false;
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
  if (THREE.ShaderChunk.lights_pars_begin) {
    THREE.ShaderChunk.lights_pars_begin = patchLightsParsBegin(
      THREE.ShaderChunk.lights_pars_begin,
    );
  }
  if (THREE.ShaderChunk.lights_physical_pars_fragment) {
    THREE.ShaderChunk.lights_physical_pars_fragment = patchLightsPhysicalParsFragment(
      THREE.ShaderChunk.lights_physical_pars_fragment,
    );
  }
  if (THREE.ShaderChunk.shadowmap_pars_fragment) {
    THREE.ShaderChunk.shadowmap_pars_fragment = patchShadowmapParsFragment(
      THREE.ShaderChunk.shadowmap_pars_fragment,
    );
  }
  if (THREE.ShaderChunk.iridescence_fragment) {
    THREE.ShaderChunk.iridescence_fragment = patchIridescenceFragment(
      THREE.ShaderChunk.iridescence_fragment,
    );
  }
  if (THREE.ShaderChunk.transmission_pars_fragment) {
    THREE.ShaderChunk.transmission_pars_fragment = patchTransmissionParsFragment(
      THREE.ShaderChunk.transmission_pars_fragment,
    );
  }
  if (THREE.ShaderChunk.defaultnormal_vertex) {
    THREE.ShaderChunk.defaultnormal_vertex = patchDefaultNormalVertex(
      THREE.ShaderChunk.defaultnormal_vertex,
    );
  }

  if (typeof window !== 'undefined' && typeof THREE.setConsoleFunction === 'function') {
    const nativeWarn = console.warn.bind(console);
    const nativeError = console.error.bind(console);
    const nativeLog = console.log.bind(console);
    THREE.setConsoleFunction((type, message, ...params) => {
      if (type === 'warn') {
        const second = typeof params[0] === 'string' ? params[0] : '';
        if (isBenignThreeWarning(message, second)) return;
        nativeWarn(message, ...params);
      } else if (type === 'error') {
        nativeError(message, ...params);
      } else {
        nativeLog(message, ...params);
      }
    });
  }
}
