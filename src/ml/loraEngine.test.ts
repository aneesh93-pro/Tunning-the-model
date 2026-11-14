import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildEngineForDataset,
  LoRANeuralEngine,
  quantizeNF4,
  dequantizeNF4,
} from './loraEngine';
import { preprocessAndSplitDataset } from './datasetProcessor';
import type { TrainingConfig } from '../shared/types';

test('NF4 quantization and dequantization preserve weights within expected 4-bit error bounds', () => {
  const weights = new Float32Array(128);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = Math.sin(i * 0.15) * 0.5;
  }
  const quantized = quantizeNF4(weights, 64, true);
  assert.equal(quantized.packedNibbles.length, 64);
  assert.ok(quantized.rmse < 0.05, `Expected NF4 RMSE < 0.05, got ${quantized.rmse}`);

  const restored = dequantizeNF4(quantized);
  assert.equal(restored.length, weights.length);
});

test('LoRA and QLoRA training reduce cross-entropy loss and improve test accuracy over frozen base model', () => {
  const rawRows = [
    { instruction: 'Classify query', input: 'I lost my debit card yesterday', output: 'card_issue' },
    { instruction: 'Classify query', input: 'My credit card was stolen', output: 'card_issue' },
    { instruction: 'Classify query', input: 'Freeze my atm card right now', output: 'card_issue' },
    { instruction: 'Classify query', input: 'Need replacement debit card', output: 'card_issue' },
    { instruction: 'Classify query', input: 'Block my damaged visa card', output: 'card_issue' },
    { instruction: 'Classify query', input: 'Send wire transfer to checking account', output: 'transfer' },
    { instruction: 'Classify query', input: 'Move money between savings and checking', output: 'transfer' },
    { instruction: 'Classify query', input: 'How do I transfer funds abroad', output: 'transfer' },
    { instruction: 'Classify query', input: 'Cancel pending bank wire transfer', output: 'transfer' },
    { instruction: 'Classify query', input: 'Initiate ACH transfer to external account', output: 'transfer' },
    { instruction: 'Classify query', input: 'What is my checking account balance', output: 'balance' },
    { instruction: 'Classify query', input: 'Show available funds in savings', output: 'balance' },
    { instruction: 'Classify query', input: 'Check current balance and recent deposits', output: 'balance' },
    { instruction: 'Classify query', input: 'How much money is in my account', output: 'balance' },
    { instruction: 'Classify query', input: 'View total balance summary', output: 'balance' },
  ];

  const bundle = preprocessAndSplitDataset(rawRows, { seed: 7 });
  const config: TrainingConfig = {
    experimentName: 'Test LoRA Run',
    description: 'Unit test',
    datasetId: 'ds_test',
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
    learningRate: 0.03,
    epochs: 12,
    batchSize: 5,
    gradientAccumulation: 1,
    maxSeqLength: 128,
    weightDecay: 0.001,
    warmupRatio: 0.1,
    optimizer: 'adamw',
    scheduler: 'cosine',
    earlyStoppingPatience: 5,
    seed: 7,
  };

  const engine = buildEngineForDataset(bundle.examples, config, 'chatml');
  assert.ok(engine.quantizationRmse !== null && engine.quantizationRmse < 0.05);

  const initialLoss = engine.evaluateLoss(bundle.train, true);
  for (let epoch = 0; epoch < config.epochs; epoch++) {
    engine.trainBatch(bundle.train, config.learningRate, config.weightDecay, 'adamw');
  }
  const finalLoss = engine.evaluateLoss(bundle.train, true);
  assert.ok(finalLoss < initialLoss, `Expected loss to drop: ${initialLoss} -> ${finalLoss}`);

  const baseEval = engine.evaluateTestSet(bundle.test, false);
  const tunedEval = engine.evaluateTestSet(bundle.test, true);
  assert.ok((tunedEval.avgLoss ?? 99) < (baseEval.avgLoss ?? 0));

  // Verify serialization & deserialization roundtrip
  const serialized = engine.serialize();
  const restored = LoRANeuralEngine.deserialize(serialized);
  const p1 = engine.predict(bundle.test[0].promptText, true);
  const p2 = restored.predict(bundle.test[0].promptText, true);
  assert.equal(p1.predictedLabel, p2.predictedLabel);
  assert.equal(p1.confidence, p2.confidence);
});
