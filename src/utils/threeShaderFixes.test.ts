import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  applyThreeShaderSafetyFixes,
  patchDefaultNormalVertex,
  patchLightsFragmentBegin,
  patchLightsPhysicalParsFragment,
} from './threeShaderFixes.ts';

test('patchLightsFragmentBegin guards EssMs denominator', () => {
  const patched = patchLightsFragmentBegin(THREE.ShaderChunk.lights_fragment_begin);
  assert.ok(patched.includes('1.0 / max( EssMs, 1e-5 ) - 1.0'));
  assert.ok(!patched.includes('1.0 / EssMs - 1.0'));
});

test('patchLightsPhysicalParsFragment guards Fms, Schlick_to_F0, and Charlie denominators', () => {
  const patched = patchLightsPhysicalParsFragment(THREE.ShaderChunk.lights_physical_pars_fragment);
  assert.ok(patched.includes('FssEss * Favg / max( vec3( 1.0 ) - Ems * Favg, vec3( 1e-5 ) )'));
  assert.ok(patched.includes('( f - vec3( f90 ) * x5 ) / max( 1.0 - x5, 1e-4 )'));
  assert.ok(patched.includes('1.0 / max( 4.0 * ( dotNL + dotNV - dotNL * dotNV ), 1e-5 )'));
});

test('patchDefaultNormalVertex guards batching and instancing dot denominators', () => {
  const patched = patchDefaultNormalVertex(THREE.ShaderChunk.defaultnormal_vertex);
  assert.ok(patched.includes('transformedNormal /= max( vec3( dot( bm[ 0 ], bm[ 0 ] )'));
  assert.ok(patched.includes('transformedNormal /= max( vec3( dot( im[ 0 ], im[ 0 ] )'));
});

test('applyThreeShaderSafetyFixes patches THREE.ShaderChunk idempotently', () => {
  applyThreeShaderSafetyFixes();
  applyThreeShaderSafetyFixes();
  assert.ok(THREE.ShaderChunk.lights_fragment_begin.includes('max( EssMs, 1e-5 )'));
  assert.ok(
    THREE.ShaderChunk.lights_physical_pars_fragment.includes('max( vec3( 1.0 ) - Ems * Favg, vec3( 1e-5 ) )'),
  );
});
