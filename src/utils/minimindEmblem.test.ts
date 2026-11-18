import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildMiniMindEmblemGroup,
  disposeObject3DResources,
  isValidGlbOrGltfBuffer,
} from './minimindEmblem.ts';

test('isValidGlbOrGltfBuffer accepts valid binary GLB v2 header', () => {
  const buf = new ArrayBuffer(24);
  const view = new DataView(buf);
  view.setUint32(0, 0x46546c67, true); // 'glTF'
  view.setUint32(4, 2, true); // version 2
  view.setUint32(8, 24, true); // total length

  const res = isValidGlbOrGltfBuffer(buf);
  assert.equal(res.valid, true);
  assert.equal(res.kind, 'glb');
});

test('isValidGlbOrGltfBuffer rejects HTML SPA fallback payload (<!doctype html>)', () => {
  const html = '<!doctype html><html><body>MiniMind AI</body></html>';
  const bytes = new TextEncoder().encode(html);
  const res = isValidGlbOrGltfBuffer(bytes.buffer);
  assert.equal(res.valid, false);
  assert.equal(res.kind, 'invalid');
  assert.match(res.reason ?? '', /HTML/i);
});

test('isValidGlbOrGltfBuffer accepts valid glTF JSON asset', () => {
  const gltfJson = JSON.stringify({ asset: { version: '2.0' }, scenes: [] });
  const bytes = new TextEncoder().encode(gltfJson);
  const res = isValidGlbOrGltfBuffer(bytes.buffer);
  assert.equal(res.valid, true);
  assert.equal(res.kind, 'gltf');
});

test('isValidGlbOrGltfBuffer rejects truncated or corrupt buffers', () => {
  const empty = new ArrayBuffer(4);
  assert.equal(isValidGlbOrGltfBuffer(empty).valid, false);

  const badLen = new ArrayBuffer(16);
  const view = new DataView(badLen);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, 9999, true); // exceeds byteLength
  assert.equal(isValidGlbOrGltfBuffer(badLen).valid, false);
});

test('buildMiniMindEmblemGroup constructs composite 3D emblem and disposes resources', () => {
  const group = buildMiniMindEmblemGroup();
  assert.equal(group.name, 'MiniMindAI_Emblem');
  const childNames = group.children.map((c) => c.name);
  assert.ok(childNames.includes('FrozenBaseSlab_W0'));
  assert.ok(childNames.includes('LoRA_Matrix_A'));
  assert.ok(childNames.includes('LoRA_Matrix_B'));
  assert.ok(childNames.includes('NF4_NeuralCore'));
  assert.ok(childNames.includes('VerificationHalo'));
  disposeObject3DResources(group);
});
