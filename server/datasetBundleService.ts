import fs from 'node:fs';
import path from 'node:path';
import { DATASETS_DIR, db, logAudit } from './db.ts';
import {
  parseDatasetContent,
  preprocessAndSplitDataset,
  type ProcessedDatasetBundle,
} from '../src/ml/datasetProcessor.ts';
import type { DatasetRecord } from '../src/shared/types.ts';

export interface DatasetBundleRow {
  id: string;
  name: string;
  file_name: string;
  file_type: string;
  status: string;
  raw_path: string;
  processed_path: string | null;
  column_mapping_json: string | null;
}

/**
 * Loads the preprocessed bundle for any dataset—and if a user-uploaded dataset
 * hasn't gone through manual "Preprocess & Split" yet, automatically prepares a
 * reproducible 70/15/15 cleaned & leakage-checked bundle on disk so Fine-Tuning
 * Studio, Evaluation Center, Model Comparison, and Model Playground can use it
 * immediately without errors.
 */
export function ensureDatasetBundlePrepared(datasetId: string): {
  datasetRow: DatasetBundleRow;
  bundle: ProcessedDatasetBundle;
  autoPrepared: boolean;
} {
  const row = db
    .prepare(
      'SELECT id, name, file_name, file_type, status, raw_path, processed_path, column_mapping_json FROM datasets WHERE id = ?'
    )
    .get(datasetId) as unknown as DatasetBundleRow | undefined;

  if (!row) {
    throw new Error(`Dataset "${datasetId}" not found.`);
  }

  if (row.processed_path && fs.existsSync(String(row.processed_path))) {
    const bundle = JSON.parse(
      fs.readFileSync(String(row.processed_path), 'utf-8')
    ) as ProcessedDatasetBundle;
    if (bundle.examples && bundle.examples.length > 0 && bundle.train && bundle.train.length > 0) {
      return { datasetRow: row, bundle, autoPrepared: false };
    }
  }

  if (!row.raw_path || !fs.existsSync(String(row.raw_path))) {
    throw new Error(`Raw dataset file for "${row.name}" is missing on disk.`);
  }

  const rawContent = fs.readFileSync(String(row.raw_path), 'utf-8');
  const rawRows = parseDatasetContent(rawContent, row.file_type as DatasetRecord['fileType']);
  if (rawRows.length === 0) {
    throw new Error(`Dataset "${row.name}" contains 0 usable rows.`);
  }

  const columnMapping = row.column_mapping_json
    ? (JSON.parse(String(row.column_mapping_json)) as {
        instruction?: string;
        input?: string;
        output?: string;
        messages?: string;
      })
    : null;

  const bundle = preprocessAndSplitDataset(rawRows, {
    trainRatio: 0.7,
    valRatio: 0.15,
    testRatio: 0.15,
    deduplicate: true,
    dropMissing: true,
    seed: 42,
    columnMapping,
  });

  if (bundle.examples.length === 0) {
    throw new Error(`Dataset "${row.name}" had 0 valid examples after cleaning empty values.`);
  }

  const procPath = path.join(DATASETS_DIR, `${row.id}.processed.json`);
  fs.writeFileSync(procPath, JSON.stringify(bundle, null, 2), 'utf-8');
  const now = new Date().toISOString();

  db.prepare(
    `UPDATE datasets SET format = ?, status = ?, row_count = ?, missing_count = ?, duplicate_count = ?,
     validation_report_json = ?, splits_json = ?, processed_path = ?, updated_at = ? WHERE id = ?`
  ).run(
    bundle.format,
    'training_ready',
    bundle.examples.length,
    bundle.report.missingValuesTotal,
    bundle.report.duplicateCount,
    JSON.stringify(bundle.report),
    JSON.stringify(bundle.splits),
    procPath,
    now,
    row.id
  );

  logAudit(
    'AUTO_PREPARE_DATASET',
    'dataset',
    row.id,
    `Auto-prepared device dataset "${row.name}" (${bundle.splits.trainCount}/${bundle.splits.valCount}/${bundle.splits.testCount})`
  );

  const updatedRow = db
    .prepare(
      'SELECT id, name, file_name, file_type, status, raw_path, processed_path, column_mapping_json FROM datasets WHERE id = ?'
    )
    .get(datasetId) as unknown as DatasetBundleRow;

  return { datasetRow: updatedRow, bundle, autoPrepared: true };
}
