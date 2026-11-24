import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeDirectionTiltDeg,
  computeDynamicLabelOffset,
  computeLabelTiltDeg,
  resolveContrastInk,
  resolvePressScale,
} from './userCursor';

test('userCursor math helpers', async (t) => {
  await t.test('returns resting rotation when directionAwareTilt is disabled or delta is tiny', () => {
    assert.equal(computeDirectionTiltDeg(24, -6, -14, false), -14);
    assert.equal(computeDirectionTiltDeg(0.4, 0.2, -14, true), -14);
  });

  await t.test('applies bounded directional tilt when enabled and pointer moves', () => {
    const rightTilt = computeDirectionTiltDeg(18, 0, -14, true);
    const leftTilt = computeDirectionTiltDeg(-18, 0, -14, true);
    assert.ok(rightTilt > -14 && rightTilt <= 32);
    assert.ok(leftTilt < -14 && leftTilt >= -48);
  });

  await t.test('clamps label tilt strength cleanly during rapid motion', () => {
    assert.equal(computeLabelTiltDeg(0.2, 8), 0);
    assert.equal(computeLabelTiltDeg(50, 8), -8);
    assert.equal(computeLabelTiltDeg(-50, 8), 8);
    assert.equal(computeLabelTiltDeg(10, 0), 0);
  });

  await t.test('resolves pressScale with safe bounds', () => {
    assert.equal(resolvePressScale(false, 0.92), 1);
    assert.equal(resolvePressScale(true, 0.92), 0.92);
    assert.equal(resolvePressScale(true, 0.2), 0.65);
  });

  await t.test('computes dynamic label offset and honors custom overrides', () => {
    assert.deepEqual(computeDynamicLabelOffset(undefined, -14), { x: 14, y: 20 });
    assert.deepEqual(computeDynamicLabelOffset({ x: 22, y: 26 }, -14), { x: 22, y: 26 });
  });

  await t.test('selects accessible contrast ink for pill backgrounds', () => {
    assert.equal(resolveContrastInk('#000000'), '#FFFFFF');
    assert.equal(resolveContrastInk('#09090B'), '#FFFFFF');
    assert.equal(resolveContrastInk('#2563EB'), '#FFFFFF');
    assert.equal(resolveContrastInk('#3B82F6'), '#FFFFFF');
    assert.equal(resolveContrastInk('#F39C2A'), '#0F172A');
    assert.equal(resolveContrastInk('#FDE047'), '#0F172A');
  });
});
