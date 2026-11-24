import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseDatasetContent,
  preprocessAndSplitDataset,
} from './datasetProcessor';

test('device CSV upload parses and produces non-empty train/val/test splits for >= 3 rows', () => {
  const csv = [
    'instruction,input,output',
    'Classify ticket,Card lost on subway,card_issue',
    'Classify ticket,Reset my login PIN,security',
    'Classify ticket,Check savings APY,fees',
    'Classify ticket,Transfer to checking,transfer',
  ].join('\n');

  const rows = parseDatasetContent(csv, 'csv');
  assert.equal(rows.length, 4);

  const bundle = preprocessAndSplitDataset(rows);
  assert.equal(bundle.format, 'instruction');
  assert.ok(bundle.train.length >= 1, 'train split should have at least 1 row');
  assert.ok(bundle.validation.length >= 1, 'validation split should have at least 1 row');
  assert.ok(bundle.test.length >= 1, 'test split should have at least 1 row');
  assert.equal(
    bundle.train.length + bundle.validation.length + bundle.test.length,
    bundle.examples.length
  );
  assert.equal(bundle.splits.leakageFree, true);
});

test('device JSON wrapped dataset ({ data: [...] }) or raw TXT prepares cleanly', () => {
  const wrappedJson = JSON.stringify({
    data: [
      { prompt: 'Summarize quarterly revenue up 12%', answer: 'Revenue rose 12% QoQ.' },
      { prompt: 'Explain net promoter score gain', answer: 'NPS improved by 6 points.' },
      { prompt: 'List top risk factor', answer: 'Supply chain lead time variance.' },
    ],
  });

  const jsonRows = parseDatasetContent(wrappedJson, 'json');
  assert.equal(jsonRows.length, 3);
  const jsonBundle = preprocessAndSplitDataset(jsonRows);
  assert.equal(jsonBundle.train.length, 1);
  assert.equal(jsonBundle.validation.length, 1);
  assert.equal(jsonBundle.test.length, 1);
  assert.equal(jsonBundle.splits.leakageFree, true);

  const txtContent = [
    'LoRA reduces trainable parameter counts by injecting low-rank matrices.',
    'QLoRA stores base weights in 4-bit NormalFloat precision.',
    'Held-out test splits guard against overfitting and evaluation leakage.',
  ].join('\n');
  const txtRows = parseDatasetContent(txtContent, 'txt');
  const txtBundle = preprocessAndSplitDataset(txtRows);
  assert.equal(txtBundle.format, 'raw_text');
  assert.ok(txtBundle.train.length >= 1);
  assert.ok(txtBundle.test.length >= 1);
});

test('tiny 1-row and 2-row device datasets degrade gracefully without empty splits', () => {
  const twoRows = parseDatasetContent(
    '{"instruction":"Tag","input":"Hello","output":"greeting"}\n{"instruction":"Tag","input":"Bye","output":"farewell"}',
    'jsonl'
  );
  const bundle2 = preprocessAndSplitDataset(twoRows);
  assert.equal(bundle2.train.length, 1);
  assert.equal(bundle2.validation.length, 1);
  assert.equal(bundle2.test.length, 1);

  const oneRow = parseDatasetContent('Single short text corpus line.', 'txt');
  const bundle1 = preprocessAndSplitDataset(oneRow);
  assert.equal(bundle1.train.length, 1);
  assert.equal(bundle1.test.length, 1);
});
