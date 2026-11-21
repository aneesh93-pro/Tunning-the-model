import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  applyThreeShaderSafetyFixes,
  patchDefaultNormalVertex,
  patchIridescenceFragment,
  patchLightsFragmentBegin,
  patchLightsParsBegin,
  patchLightsPhysicalParsFragment,
  patchShadowmapParsFragment,
  patchTransmissionParsFragment,
} from './threeShaderFixes.ts';

test('patchLightsFragmentBegin guards EssMs and spotLightCoord denominators', () => {
  const patched = patchLightsFragmentBegin(THREE.ShaderChunk.lights_fragment_begin);
  assert.ok(patched.includes('1.0 / max( EssMs, 1e-5 ) - 1.0'));
  assert.ok(!patched.includes('1.0 / EssMs - 1.0'));
  assert.ok(!patched.includes('vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w'));
});

test('patchLightsPhysicalParsFragment guards Fms, Schlick_to_F0, Charlie, and GGX denominators', () => {
  const patched = patchLightsPhysicalParsFragment(THREE.ShaderChunk.lights_physical_pars_fragment);
  assert.ok(patched.includes('FssEss * Favg / max( vec3( 1.0 ) - Ems * Favg, vec3( 1e-5 ) )'));
  assert.ok(patched.includes('( f - vec3( f90 ) * x5 ) / max( 1.0 - x5, 1e-4 )'));
  assert.ok(patched.includes('1.0 / max( 4.0 * ( dotNL + dotNV - dotNL * dotNV ), 1e-5 )'));
  assert.ok(patched.includes('return RECIPROCAL_PI * a2 / max( pow2( denom ), 1e-6 );'));
  assert.ok(patched.includes('float invAlpha = 1.0 / max( alpha, 1e-5 );'));
});

test('patchShadowmapParsFragment guards shadowCoord.w and texelSize denominators', () => {
  const patched = patchShadowmapParsFragment(THREE.ShaderChunk.shadowmap_pars_fragment);
  assert.ok(!patched.includes('shadowCoord.xyz /= shadowCoord.w;'));
  assert.ok(patched.includes('vec2 texelSize = vec2( 1.0 ) / max( shadowMapSize, vec2( 1.0 ) );'));
  assert.ok(patched.includes('float texelSize = shadowRadius / max( shadowMapSize.x, 1.0 );'));
});

test('patchLightsParsBegin, patchIridescenceFragment, and patchTransmissionParsFragment guard divisions', () => {
  assert.ok(
    patchLightsParsBegin(THREE.ShaderChunk.lights_pars_begin).includes(
      'lightDistance / max( cutoffDistance, 1e-5 )',
    ),
  );
  assert.ok(
    patchIridescenceFragment(THREE.ShaderChunk.iridescence_fragment).includes(
      'max( vec3( 1.0 ) - R123, vec3( 1e-5 ) )',
    ),
  );
  assert.ok(
    patchTransmissionParsFragment(THREE.ShaderChunk.transmission_pars_fragment).includes(
      'max( attenuationDistance, 1e-5 )',
    ),
  );
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
  assert.ok(!THREE.ShaderChunk.shadowmap_pars_fragment.includes('shadowCoord.xyz /= shadowCoord.w;'));
});
