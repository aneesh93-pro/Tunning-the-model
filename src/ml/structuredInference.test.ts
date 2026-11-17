import assert from 'node:assert/strict';
import test from 'node:test';
import {
  aggregatePlaygroundReport,
  buildPlaygroundTestSuite,
  evaluateStructuredOutput,
  resolveCandidateChoices,
  scoreZeroShotPrompt,
  summarizePlaygroundBatchReports,
} from './structuredInference';
import { getModelById, PRETRAINED_MODELS } from './modelCatalog';
import type { PlaygroundTestCaseResult, PlaygroundTestReport } from '../shared/types';

test('evaluateStructuredOutput flags out-of-choice hallucinations like JEV AI', () => {
  const res = evaluateStructuredOutput({
    rawChoice: 'JEV AI',
    rawScore: 0.99,
    choices: ['card_lost_or_stolen', 'internal_transfer', 'balance_inquiry'],
  });

  assert.equal(res.hallucinationFlag, true);
  assert.equal(res.grounded, false);
  assert.equal(res.booleanValue, false);
  assert.ok(res.score <= 0.25);
});

test('evaluateStructuredOutput accepts grounded choice and clamps score bounds', () => {
  const res = evaluateStructuredOutput({
    rawChoice: 'CARD_LOST_OR_STOLEN',
    rawScore: 1.42,
    choices: ['card_lost_or_stolen', 'internal_transfer', 'balance_inquiry'],
  });

  assert.equal(res.hallucinationFlag, false);
  assert.equal(res.grounded, true);
  assert.equal(res.choice, 'card_lost_or_stolen');
  assert.equal(res.score, 1);
  assert.equal(res.booleanValue, true);
});

test('scoreZeroShotPrompt rejects JEV AI adversarial probe with boolean=false', () => {
  const modelSpec = PRETRAINED_MODELS[0];
  const choices = resolveCandidateChoices({
    prompt: 'Verify whether "JEV AI" exists in the local banking adapter registry.',
  });
  const res = scoreZeroShotPrompt({
    modelSpec,
    prompt: 'Verify whether "JEV AI" exists in the local banking adapter registry.',
    choices,
  });

  assert.equal(res.booleanValue, false);
  assert.equal(res.hallucinationFlag, false);
  assert.ok(['false', 'unsupported', 'uncertain'].includes(res.choice));
  assert.ok(res.score >= 0 && res.score <= 1);
});

test('buildPlaygroundTestSuite and aggregatePlaygroundReport compute genuine metrics', () => {
  const suite = buildPlaygroundTestSuite({
    testExamples: [
      {
        id: 'ex_1',
        hash: 'h_ex_1',
        format: 'instruction',
        promptText: 'I lost my debit card on the train',
        targetText: 'card_lost_or_stolen',
        raw: { input: 'I lost my debit card on the train', output: 'card_lost_or_stolen' },
      },
    ],
    allLabels: ['card_lost_or_stolen', 'internal_transfer'],
    suiteMode: 'dataset_and_probes',
    taskType: 'classification',
  });

  assert.equal(suite.length, 5); // 1 dataset + 4 anti-hallucination probes

  const results: PlaygroundTestCaseResult[] = suite.map((s, idx) => ({
    index: idx,
    category: s.category,
    prompt: s.prompt,
    expectedChoice: s.expectedChoice,
    expectedBoolean: s.expectedBoolean,
    booleanValue: s.expectedBoolean,
    choice: s.expectedChoice,
    score: 0.9,
    passed: true,
    hallucinationFlag: false,
    latencyMs: 1.5,
    explanation: 'Verified match',
  }));

  const report = aggregatePlaygroundReport({
    id: 'rep_test',
    modelRef: 'base:qwen2.5-0.5b-instruct',
    modelLabel: 'Qwen2.5-0.5B-Instruct',
    method: 'base',
    datasetId: 'ds_banking_intents',
    datasetName: 'Banking Intent Classification',
    suiteMode: 'dataset_and_probes',
    cases: results,
  });

  assert.equal(report.totalCases, 5);
  assert.equal(report.passedCount, 5);
  assert.equal(report.passRate, 1);
  assert.equal(report.hallucinationCount, 0);
  assert.equal(report.verdict, 'pass');
});

test('expanded AI tool models (ChatGPT, Claude Sonnet/Haiku/Opus, Gemini, DeepSeek) resolve and run structured inference', () => {
  const requiredIds = [
    'gpt-4o',
    'gpt-4o-mini',
    'claude-3.7-sonnet',
    'claude-3.5-haiku',
    'claude-3-opus',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'deepseek-v3',
    'deepseek-r1',
  ];

  for (const id of requiredIds) {
    const spec = getModelById(id);
    assert.ok(spec, `Expected model "${id}" in PRETRAINED_MODELS`);
    const out = scoreZeroShotPrompt({
      modelSpec: spec,
      prompt: 'I lost my debit card on the train and need to freeze it.',
      choices: ['card_lost_or_stolen', 'internal_transfer', 'balance_inquiry'],
    });
    assert.equal(out.choice, 'card_lost_or_stolen');
    assert.equal(out.booleanValue, true);
    assert.equal(out.hallucinationFlag, false);
  }
});

test('summarizePlaygroundBatchReports ranks models and computes aggregate summary', () => {
  const mkReport = (
    id: string,
    passRate: number,
    hallucinationCount: number,
    verdict: PlaygroundTestReport['verdict'],
    avgScore: number,
    avgLatencyMs: number,
  ): PlaygroundTestReport => ({
    id,
    modelRef: `base:${id}`,
    modelLabel: id,
    method: 'base',
    datasetId: 'ds_banking_intents',
    datasetName: 'Banking',
    suiteMode: 'dataset_and_probes',
    totalCases: 10,
    passedCount: Math.round(passRate * 10),
    passRate,
    hallucinationCount,
    hallucinationRate: hallucinationCount / 10,
    avgScore,
    avgLatencyMs,
    verdict,
    cases: [],
    createdAt: '2026-11-17T00:00:00.000Z',
  });

  const { sortedReports, summary } = summarizePlaygroundBatchReports([
    mkReport('low', 0.4, 2, 'fail', 0.55, 2.0),
    mkReport('top-fast', 0.9, 0, 'pass', 0.92, 0.8),
    mkReport('mid', 0.7, 0, 'review', 0.8, 1.2),
  ]);

  assert.deepEqual(
    sortedReports.map((r) => r.id),
    ['top-fast', 'mid', 'low'],
  );
  assert.equal(summary.totalModels, 3);
  assert.equal(summary.passedModels, 1);
  assert.equal(summary.reviewModels, 1);
  assert.equal(summary.failedModels, 1);
  assert.equal(summary.totalHallucinations, 2);
  assert.equal(summary.avgPassRate, Number(((0.4 + 0.9 + 0.7) / 3).toFixed(4)));
});
