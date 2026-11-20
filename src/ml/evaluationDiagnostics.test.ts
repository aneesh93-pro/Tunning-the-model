import test from 'node:test';
import assert from 'node:assert/strict';
import type { EvaluationRunRecord } from '../shared/types';
import { analyzeEvaluationRun } from './evaluationDiagnostics';

test('analyzeEvaluationRun explains frozen base errors, confusion pairs, and invalid outputs', () => {
  const sampleEval: EvaluationRunRecord = {
    id: 'eval_test_1',
    experimentId: null,
    registeredModelId: null,
    modelName: 'SmolLM2-360M-Instruct (Frozen Base)',
    baseModelId: 'smollm2-360m-instruct',
    method: 'base',
    datasetId: 'ds_1',
    datasetName: 'Banking Intent',
    taskType: 'classification',
    testSampleCount: 4,
    accuracy: 0.5,
    precision: 0.72,
    recall: 0.45,
    macroF1: 0.52,
    weightedF1: 0.51,
    exactMatch: null,
    tokenF1: null,
    perplexity: 9.2,
    avgLoss: 2.21,
    invalidOutputCount: 1,
    avgLatencyMs: 0.5,
    confusionMatrix: {
      labels: ['balance_inquiry', 'fees_and_rates'],
      matrix: [
        [2, 0],
        [2, 0],
      ],
    },
    samples: [
      {
        index: 0,
        prompt: 'What is the APR?',
        groundTruth: 'fees_and_rates',
        prediction: 'balance_inquiry',
        correct: false,
        invalidOutput: true,
        confidence: 0.52,
        loss: 3.1,
        latencyMs: 0.5,
      },
    ],
    createdAt: new Date().toISOString(),
  };

  const report = analyzeEvaluationRun(sampleEval);
  assert.equal(report.overallHealth, 'critical');
  assert.equal(report.topConfusions.length, 1);
  assert.equal(report.topConfusions[0].actual, 'fees_and_rates');
  assert.equal(report.topConfusions[0].predicted, 'balance_inquiry');
  assert.ok(report.diagnostics.some((d) => d.id === 'frozen-base-gap'));
  assert.ok(report.diagnostics.some((d) => d.id === 'invalid-outputs'));
  assert.ok(report.diagnostics.some((d) => d.id === 'high-perplexity'));
  assert.ok(report.diagnostics.some((d) => d.id === 'precision-recall-skew'));
});
