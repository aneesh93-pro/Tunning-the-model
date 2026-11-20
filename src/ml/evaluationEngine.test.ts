import test from 'node:test';
import assert from 'node:assert/strict';
import { preprocessAndSplitDataset } from './datasetProcessor';
import { evaluateBaseModelOnBundle } from './evaluationEngine';
import { getModelById } from './modelCatalog';

const sampleRows = [
  { instruction: 'Classify intent', input: 'I lost my debit card on the train', output: 'card_lost_or_stolen' },
  { instruction: 'Classify intent', input: 'Freeze my stolen visa card immediately', output: 'card_lost_or_stolen' },
  { instruction: 'Classify intent', input: 'Missing credit card at the airport', output: 'card_lost_or_stolen' },
  { instruction: 'Classify intent', input: 'Send 300 from checking to savings vault', output: 'internal_transfer' },
  { instruction: 'Classify intent', input: 'Move money between my accounts now', output: 'internal_transfer' },
  { instruction: 'Classify intent', input: 'Recurring ACH wire transfer weekly', output: 'internal_transfer' },
  { instruction: 'Classify intent', input: 'How much money is in checking balance', output: 'balance_inquiry' },
  { instruction: 'Classify intent', input: 'Did my Friday paycheck deposit clear yet', output: 'balance_inquiry' },
  { instruction: 'Classify intent', input: 'Check available funds and pending balance', output: 'balance_inquiry' },
  { instruction: 'Classify intent', input: 'What is the monthly maintenance fee and APR', output: 'fees_and_rates' },
  { instruction: 'Classify intent', input: 'Overdraft charge and foreign transaction fee', output: 'fees_and_rates' },
  { instruction: 'Classify intent', input: 'Current savings APY interest rate', output: 'fees_and_rates' },
  { instruction: 'Classify intent', input: 'Lock my debit card before someone uses it', output: 'card_lost_or_stolen' },
  { instruction: 'Classify intent', input: 'Transfer $500 to my savings account today', output: 'internal_transfer' },
  { instruction: 'Classify intent', input: 'Show current ledger and available balance', output: 'balance_inquiry' },
  { instruction: 'Classify intent', input: 'Waive overdraft fee on checking account', output: 'fees_and_rates' },
];

test('evaluateBaseModelOnBundle returns deterministic differentiated metrics across models', () => {
  const bundle = preprocessAndSplitDataset(sampleRows, { seed: 42 });
  const smol = getModelById('smollm2-360m-instruct')!;
  const nemotron = getModelById('nvidia-llama-3.1-nemotron-70b')!;
  const glm = getModelById('zai-glm-4.5')!;
  const gpt4o = getModelById('gpt-4o')!;

  const rSmol1 = evaluateBaseModelOnBundle(smol, bundle);
  const rSmol2 = evaluateBaseModelOnBundle(smol, bundle);
  const rNemotron = evaluateBaseModelOnBundle(nemotron, bundle);
  const rGlm = evaluateBaseModelOnBundle(glm, bundle);
  const rGpt4o = evaluateBaseModelOnBundle(gpt4o, bundle);

  assert.equal(rSmol1.accuracy, rSmol2.accuracy);
  assert.equal(rSmol1.perplexity, rSmol2.perplexity);
  assert.ok(rNemotron.avgLoss !== rSmol1.avgLoss);
  assert.ok(rGlm.perplexity !== rGpt4o.perplexity);
  assert.ok((rGpt4o.accuracy ?? 0) >= (rSmol1.accuracy ?? 0));
  assert.ok(rNemotron.confusionMatrix !== null);
  assert.equal(rNemotron.samples.length, bundle.test.length);
});
