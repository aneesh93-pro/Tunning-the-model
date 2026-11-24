import test from 'node:test';
import assert from 'node:assert/strict';
import { pickPreferredDatasetId } from './datasetSelection';

test('pickPreferredDatasetId returns empty string for empty dataset list', () => {
  assert.equal(pickPreferredDatasetId([]), '');
  assert.equal(pickPreferredDatasetId([], 'ds_1'), '');
});

test('pickPreferredDatasetId prefers explicit preferredId when present in list', () => {
  const datasets = [
    { id: 'ds_raw', status: 'validated' as const },
    { id: 'ds_ready', status: 'training_ready' as const },
  ];
  assert.equal(pickPreferredDatasetId(datasets, 'ds_raw'), 'ds_raw');
});

test('pickPreferredDatasetId picks first training_ready dataset when preferredId is missing or absent', () => {
  const datasets = [
    { id: 'ds_raw_1', status: 'validated' as const },
    { id: 'ds_uploaded', status: 'uploaded' as const },
    { id: 'ds_ready_1', status: 'training_ready' as const },
    { id: 'ds_ready_2', status: 'training_ready' as const },
  ];
  assert.equal(pickPreferredDatasetId(datasets), 'ds_ready_1');
  assert.equal(pickPreferredDatasetId(datasets, 'nonexistent'), 'ds_ready_1');
});

test('pickPreferredDatasetId falls back to first dataset when none are training_ready', () => {
  const datasets = [
    { id: 'ds_uploaded', status: 'uploaded' as const },
    { id: 'ds_validated', status: 'validated' as const },
  ];
  assert.equal(pickPreferredDatasetId(datasets), 'ds_uploaded');
});
