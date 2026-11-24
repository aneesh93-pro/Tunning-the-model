import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  AI_ORB_LABELS,
  AI_ORB_STATES,
  ORB_VISUAL_TABLE,
  mapExperimentStatusToOrbState,
  mapJobRunnerToOrbState,
  mapPlaygroundToOrbState,
  orbStateFor,
} from './aiOrbState';

describe('aiOrbState helpers', () => {
  it('defines visual configuration and labels for every AI_ORB_STATES entry', () => {
    assert.equal(AI_ORB_STATES.length, 10);
    for (const state of AI_ORB_STATES) {
      assert.ok(ORB_VISUAL_TABLE[state], `Missing visual table entry for ${state}`);
      assert.ok(AI_ORB_LABELS[state], `Missing label for ${state}`);
      assert.ok(ORB_VISUAL_TABLE[state].speed > 0, `Speed must be positive for ${state}`);
    }
  });

  it('resolves priority order in orbStateFor cleanly', () => {
    assert.equal(orbStateFor({}), 'idle');
    assert.equal(orbStateFor({ done: true }), 'done');
    assert.equal(orbStateFor({ listening: true, done: true }), 'listening');
    assert.equal(orbStateFor({ streaming: true, listening: true }), 'streaming');
    assert.equal(orbStateFor({ toolCalling: true, streaming: true }), 'tool-call');
    assert.equal(orbStateFor({ waitingApproval: true, toolCalling: true }), 'waiting-approval');
    assert.equal(orbStateFor({ error: true, waitingApproval: true, streaming: true }), 'error');
  });

  it('maps experiment statuses to semantic orb states', () => {
    assert.equal(mapExperimentStatusToOrbState(undefined), 'idle');
    assert.equal(mapExperimentStatusToOrbState('preparing'), 'thinking');
    assert.equal(mapExperimentStatusToOrbState('running'), 'reasoning');
    assert.equal(mapExperimentStatusToOrbState('paused'), 'waiting-approval');
    assert.equal(mapExperimentStatusToOrbState('completed'), 'done');
    assert.equal(mapExperimentStatusToOrbState('failed'), 'error');
    assert.equal(mapExperimentStatusToOrbState('cancelled'), 'error');
  });

  it('maps playground and runner states accurately', () => {
    assert.equal(mapPlaygroundToOrbState({}), 'idle');
    assert.equal(mapPlaygroundToOrbState({ hasDraftInput: true }), 'listening');
    assert.equal(mapPlaygroundToOrbState({ streaming: true, hasDraftInput: true }), 'streaming');
    assert.equal(mapPlaygroundToOrbState({ testingAllModels: true, streaming: true }), 'tool-call');
    assert.equal(mapPlaygroundToOrbState({ hasMessages: true }), 'done');
    assert.equal(mapPlaygroundToOrbState({ error: 'Network timeout', hasMessages: true }), 'error');

    assert.equal(mapJobRunnerToOrbState({}), 'idle');
    assert.equal(mapJobRunnerToOrbState({ running: true, hasResult: true }), 'tool-call');
    assert.equal(mapJobRunnerToOrbState({ hasResult: true }), 'done');
    assert.equal(mapJobRunnerToOrbState({ error: 'Bad split' }), 'error');
  });
});
