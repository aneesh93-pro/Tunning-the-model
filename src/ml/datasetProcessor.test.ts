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

test('parseDatasetContent handles flexible JSON wrappers, single objects, dictionaries, BOM, and TSV', () => {
  // Wrapper with "records" + UTF-8 BOM
  const recordsJson = `\uFEFF${JSON.stringify({
    meta: { source: 'export' },
    records: [
      { question: 'How do I freeze my card?', answer: 'card_lost_or_stolen' },
      { question: 'What is the checking APY?', answer: 'fees_and_rates' },
    ],
  })}`;
  const parsedRecords = parseDatasetContent(recordsJson, 'json');
  assert.equal(parsedRecords.length, 2);
  assert.equal(detectDatasetFormat(parsedRecords), 'instruction');

  // Deeply nested array + single object fallback
  const singleObjJson = JSON.stringify({ instruction: 'Summarize', input: 'Text here', output: 'Short' });
  const parsedSingle = parseDatasetContent(singleObjJson, 'json');
  assert.equal(parsedSingle.length, 1);
  assert.equal(parsedSingle[0].output, 'Short');

  // Dictionary of objects
  const dictJson = JSON.stringify({
    item1: { text: 'First document sample' },
    item2: { text: 'Second document sample' },
  });
  assert.equal(parseDatasetContent(dictJson, 'json').length, 2);

  // JSON lines stored in a .json file
  const ndjsonInJson = '{"text":"Line one"}\n{"text":"Line two"}';
  assert.equal(parseDatasetContent(ndjsonInJson, 'json').length, 2);

  // TSV parsing
  const tsv = 'instruction\toutput\nClassify\tcard_lost\nRate check\tfees';
  const parsedTsv = parseDatasetContent(tsv, 'tsv');
  assert.equal(parsedTsv.length, 2);
  assert.equal(parsedTsv[1].output, 'fees');
});

test('multi-attribute CSV, JSON columnar tables, and Windows TXT preserve all attributes with zero errors', () => {
  // Multi-attribute CSV database with arbitrary business columns and missing middle cell
  const businessCsv = [
    'id,customer,plan,region,mrr',
    'c1,Acme Corp,Enterprise,US-East,2400',
    'c2,,Growth,EU-West,890',
    'c3,Nimbus LLC,Starter,APAC,199',
    'c4,Horizon AI,Enterprise,US-West,3200',
    'c5,Pulse Health,Pro,US-East,1150',
    'c6,Orbit Labs,Growth,CA-Central,750',
  ].join('\r\n');

  const csvRows = parseDatasetContent(businessCsv, 'csv');
  assert.equal(csvRows.length, 6);
  const csvReport = validateDatasetRows(csvRows);
  assert.equal(csvReport.valid, true);
  assert.equal(csvReport.errors.length, 0);
  assert.equal(csvReport.columnCount, 5);
  assert.deepEqual(
    csvReport.columns.map((c) => c.name),
    ['id', 'customer', 'plan', 'region', 'mrr']
  );

  const csvBundle = preprocessAndSplitDataset(csvRows, {
    trainRatio: 0.7,
    valRatio: 0.15,
    testRatio: 0.15,
    seed: 7,
  });
  assert.equal(csvBundle.examples.length, 6);
  assert.equal(csvBundle.splits.leakageFree, true);
  assert.ok(csvBundle.examples[0].promptText.includes('plan:'));
  assert.ok(csvBundle.examples[1].promptText.includes('region: EU-West'));
  assert.ok(csvBundle.examples[0].targetText.startsWith('mrr:'));

  // Semicolon CSV + capitalized headers + explicit mapping
  const semiCsv = 'ID;Department;TicketText;Priority\n1;Billing;Refund invoice #44;High\n2;Auth;2FA code expired;Medium\n3;Billing;Double charge;High';
  const semiRows = parseDatasetContent(semiCsv, 'csv');
  assert.equal(semiRows.length, 3);
  const mappedReport = validateDatasetRows(semiRows, {
    instruction: 'Department',
    input: 'TicketText',
    output: 'Priority',
  });
  assert.equal(mappedReport.valid, true);
  assert.equal(mappedReport.format, 'instruction');

  // Columnar JSON wrapper + trailing comma + pandas orient='columns'
  const columnarJson = `{
    "columns": ["sku", "category", "stock", "price"],
    "data": [
      ["A-100", "hardware", 14, 299],
      ["A-200", "accessories", 88, 39],
      ["A-300", "hardware", 5, 499],
    ]
  }`;
  const colRows = parseDatasetContent(columnarJson, 'json');
  assert.equal(colRows.length, 3);
  assert.equal(colRows[0].sku, 'A-100');
  assert.equal(colRows[2].price, 499);
  assert.equal(validateDatasetRows(colRows).errors.length, 0);

  const pandasOrientCols = JSON.stringify({
    name: { '0': 'Alpha', '1': 'Beta' },
    score: { '0': 98, '1': 87 },
  });
  const pandasRows = parseDatasetContent(pandasOrientCols, 'json');
  assert.equal(pandasRows.length, 2);
  assert.equal(pandasRows[1].name, 'Beta');
  assert.equal(pandasRows[1].score, 87);

  // Windows CRLF double-newline TXT paragraphs + JSON inside TXT
  const winTxt = 'First training paragraph about domain adapters.\r\n\r\nSecond training paragraph about quantization.\r\n\r\nThird paragraph about evaluation.';
  const txtRows = parseDatasetContent(winTxt, 'txt');
  assert.equal(txtRows.length, 3);
  assert.equal(validateDatasetRows(txtRows).valid, true);

  const jsonInsideTxt = '[{"topic":"LoRA","notes":"Low rank adapters"},{"topic":"QLoRA","notes":"4-bit NF4"}]';
  const parsedJsonTxt = parseDatasetContent(jsonInsideTxt, 'txt');
  assert.equal(parsedJsonTxt.length, 2);
  assert.equal(parsedJsonTxt[1].topic, 'QLoRA');
});


