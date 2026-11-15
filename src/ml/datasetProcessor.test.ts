import test from 'node:test';
import assert from 'node:assert/strict';
import {
  detectDatasetFormat,
  parseDatasetContent,
  preprocessAndSplitDataset,
  rowsToCsv,
  rowsToJsonl,
  validateDatasetRows,
} from './datasetProcessor';

test('parseDatasetContent parses quoted CSV rows accurately', () => {
  const csv = [
    'instruction,input,output',
    '"Classify message","I lost my debit card, please help!",cash_or_card',
    '"Classify message","What is my APR?",rates_and_fees',
    '"Classify message","Transfer $200 to savings",transfers',
  ].join('\n');

  const rows = parseDatasetContent(csv, 'csv');
  assert.equal(rows.length, 3);
  assert.equal(rows[0].input, 'I lost my debit card, please help!');
  assert.equal(rows[0].output, 'cash_or_card');
  assert.equal(detectDatasetFormat(rows), 'instruction');
});

test('validateDatasetRows detects duplicates and missing values', () => {
  const rows = [
    { instruction: 'Classify', input: 'Lost card', output: 'card' },
    { instruction: 'Classify', input: 'Lost card', output: 'card' },
    { instruction: 'Classify', input: '', output: '' },
  ];
  const report = validateDatasetRows(rows);
  assert.equal(report.duplicateCount, 1);
  assert.equal(report.missingValuesTotal, 2);
  assert.equal(report.valid, true);
});

test('preprocessAndSplitDataset deduplicates and guarantees zero data leakage across splits', () => {
  const rows = Array.from({ length: 18 }, (_, i) => ({
    instruction: 'Classify intent',
    input: `Sample banking query number ${i} about ${i % 3 === 0 ? 'card' : i % 3 === 1 ? 'transfer' : 'loan'}`,
    output: i % 3 === 0 ? 'card' : i % 3 === 1 ? 'transfer' : 'loan',
  }));
  // Add an intentional duplicate
  rows.push({ ...rows[0] });

  const bundle = preprocessAndSplitDataset(rows, {
    trainRatio: 0.7,
    valRatio: 0.15,
    testRatio: 0.15,
    deduplicate: true,
    dropMissing: true,
    seed: 42,
  });

  assert.equal(bundle.examples.length, 18);
  assert.equal(bundle.splits.leakageFree, true);
  assert.equal(bundle.report.overlapCount, 0);
  assert.ok(bundle.train.length > 0);
  assert.ok(bundle.validation.length > 0);
  assert.ok(bundle.test.length > 0);
});

test('rowsToCsv and rowsToJsonl serialize processed rows cleanly with round-trip fidelity', () => {
  const rows = [
    { instruction: 'Classify intent', input: 'Card "missing", urgent!', output: 'card_lost_or_stolen' },
    { instruction: 'Classify intent', input: 'Wire $500 to checking', output: 'internal_transfer' },
  ];
  const csv = rowsToCsv(rows);
  const parsedFromCsv = parseDatasetContent(csv, 'csv');
  assert.equal(parsedFromCsv.length, 2);
  assert.equal(parsedFromCsv[0].input, 'Card "missing", urgent!');
  assert.equal(parsedFromCsv[1].output, 'internal_transfer');

  const jsonl = rowsToJsonl(rows);
  const parsedFromJsonl = parseDatasetContent(jsonl, 'jsonl');
  assert.equal(parsedFromJsonl.length, 2);
  assert.equal(parsedFromJsonl[0].output, 'card_lost_or_stolen');
});
