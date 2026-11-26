import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildExperimentDefaultsFromDataset,
  buildPlaygroundPresetsFromDataset,
  pickPreferredDatasetId,
} from './datasetSelection';

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

test('buildExperimentDefaultsFromDataset reflects imported dataset name, format, and training method', () => {
  const ds = {
    name: 'Support Tickets Q4 (18)',
    format: 'instruction' as const,
    rowCount: 18,
    columnCount: 4,
  };
  const loraDefaults = buildExperimentDefaultsFromDataset(ds, 'lora');
  assert.equal(loraDefaults.experimentName, 'Support Tickets Q4 LoRA v1');
  assert.match(loraDefaults.description, /Support Tickets Q4 \(18\)/);
  assert.match(loraDefaults.description, /18 rows, 4 attributes/);

  const qloraDefaults = buildExperimentDefaultsFromDataset(ds, 'qlora');
  assert.equal(qloraDefaults.experimentName, 'Support Tickets Q4 QLoRA v1');
});

test('buildPlaygroundPresetsFromDataset extracts discrete choices and prompts from imported dataset rows', () => {
  const rows = [
    { ticket: 'App crashes when exporting CSV', priority: 'urgent_bug' },
    { ticket: 'How do I change billing email?', priority: 'billing_question' },
    { ticket: 'Add dark mode toggle to exports', priority: 'feature_request' },
  ];
  const presets = buildPlaygroundPresetsFromDataset(
    { name: 'Support Tickets', columnMapping: null },
    rows
  );
  assert.equal(presets.suggestedChoices, 'urgent_bug, billing_question, feature_request');
  assert.equal(presets.quickPrompts.length, 3);
  assert.equal(presets.quickPrompts[0].prompt, 'App crashes when exporting CSV');
  assert.equal(presets.quickPrompts[0].choices, 'urgent_bug, billing_question, feature_request');
  assert.match(presets.quickPrompts[2].label, /Anti-Hallucination Probe/);
});
