import test from 'node:test';
import assert from 'node:assert/strict';
import type { DatasetRecord, RegisteredModelRecord, TrainingConfig } from '../shared/types';
import { buildAdapterBundleDocs, buildDatasetBundleDocs } from './folderBundles';

const baseConfig: TrainingConfig = {
  experimentName: 'exp-1',
  description: 'test run',
  datasetId: 'ds-1',
  modelId: 'qwen2.5-0.5b-instruct',
  method: 'qlora',
  loraR: 8,
  loraAlpha: 16,
  loraDropout: 0.05,
  targetModules: ['q_proj', 'v_proj'],
  bias: 'none',
  quantizationBits: 4,
  quantType: 'nf4',
  doubleQuant: true,
  computeDtype: 'bfloat16',
  gradientCheckpointing: true,
  learningRate: 0.0002,
  epochs: 3,
  batchSize: 4,
  gradientAccumulation: 2,
  maxSeqLength: 512,
  weightDecay: 0.01,
  warmupRatio: 0.05,
  optimizer: 'adamw',
  scheduler: 'cosine',
  earlyStoppingPatience: 2,
  seed: 42,
};

test('buildDatasetBundleDocs returns 6 capped files when training_ready with splits', () => {
  const dataset: DatasetRecord = {
    id: 'ds-bank',
    name: 'Banking Intent Classification',
    fileName: 'banking_intent.jsonl',
    fileType: 'jsonl',
    format: 'instruction',
    status: 'training_ready',
    rowCount: 100,
    columnCount: 3,
    sizeBytes: 12800,
    missingCount: 0,
    duplicateCount: 2,
    validationReport: null,
    splits: {
      trainCount: 68,
      valCount: 15,
      testCount: 15,
      trainRatio: 0.7,
      valRatio: 0.15,
      testRatio: 0.15,
      leakageFree: true,
    },
    columnMapping: null,
    createdAt: '2026-11-22T00:00:00Z',
    updatedAt: '2026-11-22T00:00:00Z',
  };

  const docs = buildDatasetBundleDocs(dataset, 'csv');
  assert.equal(docs.length, 6);
  assert.equal(docs[0].id, 'cleaned');
  assert.equal(docs[0].rowCount, 98);
  assert.equal(docs[0].fileName, 'banking_intent_classification_cleaned.csv');
  assert.equal(docs[1].id, 'train');
  assert.equal(docs[1].exportUrl, '/api/datasets/ds-bank/export?split=train&format=csv');
  assert.equal(docs[5].id, 'zip');
  assert.equal(docs[5].exportUrl, '/api/datasets/ds-bank/export?split=all&format=zip');
});

test('buildDatasetBundleDocs returns compact raw export bundle before preprocessing', () => {
  const dataset: DatasetRecord = {
    id: 'ds-raw',
    name: 'Raw Notes',
    fileName: 'notes.txt',
    fileType: 'txt',
    format: 'raw_text',
    status: 'uploaded',
    rowCount: 12,
    columnCount: 1,
    sizeBytes: 1024,
    missingCount: 1,
    duplicateCount: 0,
    validationReport: null,
    splits: null,
    columnMapping: null,
    createdAt: '2026-11-22T00:00:00Z',
    updatedAt: '2026-11-22T00:00:00Z',
  };

  const docs = buildDatasetBundleDocs(dataset, 'jsonl');
  assert.equal(docs.length, 3);
  assert.equal(docs[0].id, 'raw');
  assert.equal(docs[2].id, 'zip');
  assert.match(docs[2].exportUrl, /format=jsonl/);
});

test('buildAdapterBundleDocs produces 5 artifact cards reflecting metrics and PEFT params', () => {
  const model: RegisteredModelRecord = {
    id: 'mod-1',
    name: 'banking-qwen-lora',
    version: 'v1.0.0',
    baseModelId: 'qwen2.5-0.5b-instruct',
    baseModelName: 'Qwen2.5-0.5B-Instruct',
    method: 'qlora',
    experimentId: 'exp-1',
    datasetId: 'ds-1',
    datasetName: 'Banking Intent',
    adapterPath: '/tmp/mod-1',
    adapterSizeBytes: 2_400_000,
    trainableParams: 540_000,
    status: 'deployed',
    latestAccuracy: 0.9167,
    latestMacroF1: 0.9012,
    latestPerplexity: 2.34,
    trainingConfig: baseConfig,
    createdAt: '2026-11-22T00:00:00Z',
  };

  const docs = buildAdapterBundleDocs(model);
  assert.equal(docs.length, 5);
  assert.equal(docs[0].fileName, 'adapter_config.json');
  assert.match(docs[0].previewCode, /"r": 8/);
  assert.equal(docs[3].fileName, 'README.md');
  assert.match(docs[3].subtitle, /91\.67%/);
  assert.equal(docs[4].id, 'eval');
  assert.match(docs[4].previewCode, /"status": "deployed"/);
});
