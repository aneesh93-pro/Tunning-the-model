import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express, { type Request, type Response } from 'express';
import JSZip from 'jszip';
import multer from 'multer';
import { ARTIFACTS_DIR, DATASETS_DIR, db, logAudit } from './db.ts';
import { getHardwareTelemetry } from './hardware.ts';
import {
  listPlaygroundTestReports,
  runPlaygroundModelDiagnosticReport,
  streamModelInference,
} from './inferenceService.ts';
import {
  cancelTrainingJob,
  getExperimentById,
  getJobEmitter,
  getJobLogs,
  listExperiments,
  loadTrainedEngineFromAdapter,
  pauseTrainingJob,
  resumeTrainingJob,
  startTrainingJob,
} from './trainingManager.ts';
import {
  parseDatasetContent,
  preprocessAndSplitDataset,
  rowsToCsv,
  rowsToJsonl,
  validateDatasetRows,
  type ProcessedDatasetBundle,
  type RawRow,
} from '../src/ml/datasetProcessor.ts';
import { buildEngineForDataset } from '../src/ml/loraEngine.ts';
import { getModelById, PRETRAINED_MODELS } from '../src/ml/modelCatalog.ts';
import type {
  DatasetRecord,
  EvaluationRunRecord,
  RegisteredModelRecord,
  TrainingConfig,
  TrainingMetricPoint,
} from '../src/shared/types.ts';

const ALLOWED_EXTS = new Set(['csv', 'json', 'jsonl', 'txt']);
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB limit
});

// Simple per-IP in-memory rate limiter for API safety
const rateMap = new Map<string, { count: number; resetAt: number }>();
function rateLimit(req: Request, res: Response, next: () => void) {
  if (req.path.endsWith('/stream')) {
    next();
    return;
  }
  const key = req.ip || 'local';
  const now = Date.now();
  const entry = rateMap.get(key);
  if (!entry || now > entry.resetAt) {
    rateMap.set(key, { count: 1, resetAt: now + 60_000 });
    next();
    return;
  }
  entry.count++;
  if (entry.count > 240) {
    res.status(429).json({ error: 'Rate limit exceeded. Please wait a moment.' });
    return;
  }
  next();
}

function assertSafeId(id: string): string {
  if (!/^[a-zA-Z0-9_-]{2,80}$/.test(id)) {
    throw new Error('Invalid resource identifier.');
  }
  return id;
}

function mapDatasetRow(row: Record<string, unknown>): DatasetRecord {
  return {
    id: String(row.id),
    name: String(row.name),
    fileName: String(row.file_name),
    fileType: row.file_type as DatasetRecord['fileType'],
    format: row.format as DatasetRecord['format'],
    status: row.status as DatasetRecord['status'],
    rowCount: Number(row.row_count),
    columnCount: Number(row.column_count),
    sizeBytes: Number(row.size_bytes),
    missingCount: Number(row.missing_count),
    duplicateCount: Number(row.duplicate_count),
    validationReport: row.validation_report_json
      ? JSON.parse(String(row.validation_report_json))
      : null,
    splits: row.splits_json ? JSON.parse(String(row.splits_json)) : null,
    columnMapping: row.column_mapping_json
      ? JSON.parse(String(row.column_mapping_json))
      : null,
    createdAt: String(row.created_at),
    updatedAt: String(row.updated_at),
  };
}

function mapRegisteredModelRow(row: Record<string, unknown>): RegisteredModelRecord {
  return {
    id: String(row.id),
    name: String(row.name),
    version: String(row.version),
    baseModelId: String(row.base_model_id),
    baseModelName: String(row.base_model_name),
    method: row.method as RegisteredModelRecord['method'],
    experimentId: String(row.experiment_id),
    datasetId: String(row.dataset_id),
    datasetName: String(row.dataset_name),
    adapterPath: String(row.adapter_path),
    adapterSizeBytes: Number(row.adapter_size_bytes),
    trainableParams: Number(row.trainable_params),
    status: row.status as RegisteredModelRecord['status'],
    latestAccuracy:
      row.latest_accuracy !== null && row.latest_accuracy !== undefined
        ? Number(row.latest_accuracy)
        : null,
    latestMacroF1:
      row.latest_macro_f1 !== null && row.latest_macro_f1 !== undefined
        ? Number(row.latest_macro_f1)
        : null,
    latestPerplexity:
      row.latest_perplexity !== null && row.latest_perplexity !== undefined
        ? Number(row.latest_perplexity)
        : null,
    trainingConfig: JSON.parse(String(row.training_config_json)) as TrainingConfig,
    createdAt: String(row.created_at),
  };
}

function mapEvaluationRow(row: Record<string, unknown>): EvaluationRunRecord {
  return {
    id: String(row.id),
    experimentId: row.experiment_id ? String(row.experiment_id) : null,
    registeredModelId: row.registered_model_id ? String(row.registered_model_id) : null,
    modelName: String(row.model_name),
    baseModelId: String(row.base_model_id),
    method: row.method as EvaluationRunRecord['method'],
    datasetId: String(row.dataset_id),
    datasetName: String(row.dataset_name),
    taskType: row.task_type as EvaluationRunRecord['taskType'],
    testSampleCount: Number(row.test_sample_count),
    accuracy: row.accuracy !== null && row.accuracy !== undefined ? Number(row.accuracy) : null,
    precision:
      row.precision_score !== null && row.precision_score !== undefined
        ? Number(row.precision_score)
        : null,
    recall:
      row.recall_score !== null && row.recall_score !== undefined
        ? Number(row.recall_score)
        : null,
    macroF1: row.macro_f1 !== null && row.macro_f1 !== undefined ? Number(row.macro_f1) : null,
    weightedF1:
      row.weighted_f1 !== null && row.weighted_f1 !== undefined ? Number(row.weighted_f1) : null,
    exactMatch:
      row.exact_match !== null && row.exact_match !== undefined ? Number(row.exact_match) : null,
    tokenF1: row.token_f1 !== null && row.token_f1 !== undefined ? Number(row.token_f1) : null,
    perplexity: Number(row.perplexity),
    avgLoss: Number(row.avg_loss),
    invalidOutputCount: Number(row.invalid_output_count ?? 0),
    avgLatencyMs: Number(row.avg_latency_ms),
    confusionMatrix: row.confusion_matrix_json
      ? JSON.parse(String(row.confusion_matrix_json))
      : null,
    samples: JSON.parse(String(row.samples_json)),
    createdAt: String(row.created_at),
  };
}

function evaluateModelAgainstTestSet(modelRef: string, datasetId: string): EvaluationRunRecord {
  const dsRow = db
    .prepare('SELECT id, name, processed_path FROM datasets WHERE id = ?')
    .get(datasetId) as { id: string; name: string; processed_path: string | null } | undefined;
  if (!dsRow || !dsRow.processed_path || !fs.existsSync(dsRow.processed_path)) {
    throw new Error('Selected dataset must be preprocessed with a held-out test split before evaluation.');
  }

  const bundle = JSON.parse(
    fs.readFileSync(dsRow.processed_path, 'utf-8')
  ) as ProcessedDatasetBundle;
  if (!bundle.test || bundle.test.length === 0) {
    throw new Error('Dataset test split is empty.');
  }

  let experimentId: string | null = null;
  let registeredModelId: string | null = null;
  let modelName = '';
  let baseModelId = '';
  let method: EvaluationRunRecord['method'] = 'base';
  let result;
  let taskType: 'classification' | 'generation' = 'classification';

  if (modelRef.startsWith('base:')) {
    const rawId = modelRef.replace(/^base:/, '');
    const spec = getModelById(rawId);
    if (!spec) throw new Error(`Base model "${rawId}" not found.`);
    baseModelId = spec.id;
    modelName = `${spec.name} (Frozen Base)`;
    const dummyConfig: TrainingConfig = {
      experimentName: 'base-eval',
      description: '',
      datasetId,
      modelId: spec.id,
      method: 'lora',
      loraR: 8,
      loraAlpha: 16,
      loraDropout: 0,
      targetModules: spec.targetModules.slice(0, 2),
      bias: 'none',
      quantizationBits: 16,
      quantType: 'nf4',
      doubleQuant: false,
      computeDtype: 'float32',
      gradientCheckpointing: false,
      learningRate: 0.001,
      epochs: 1,
      batchSize: 4,
      gradientAccumulation: 1,
      maxSeqLength: 256,
      weightDecay: 0,
      warmupRatio: 0,
      optimizer: 'adamw',
      scheduler: 'cosine',
      earlyStoppingPatience: 3,
      seed: 42,
    };
    const allExamples = [...bundle.train, ...bundle.validation, ...bundle.test];
    const baseEngine = buildEngineForDataset(allExamples, dummyConfig, spec.chatTemplate);
    taskType = baseEngine.taskType;
    result = baseEngine.evaluateTestSet(bundle.test, false);
  } else {
    const regId = modelRef.replace(/^reg:/, '');
    const regRow = db
      .prepare('SELECT * FROM registered_models WHERE id = ?')
      .get(regId) as Record<string, unknown> | undefined;
    if (!regRow) throw new Error(`Registered model "${regId}" not found.`);
    registeredModelId = String(regRow.id);
    experimentId = String(regRow.experiment_id);
    modelName = String(regRow.name);
    baseModelId = String(regRow.base_model_id);
    method = regRow.method as EvaluationRunRecord['method'];

    const engine = loadTrainedEngineFromAdapter(String(regRow.adapter_path));
    taskType = engine.taskType;
    result = engine.evaluateTestSet(bundle.test, true);
  }

  const evalId = `eval_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
  const now = new Date().toISOString();

  db.prepare(
    `INSERT INTO evaluation_runs (
      id, workspace_id, experiment_id, registered_model_id, model_name, base_model_id, method,
      dataset_id, dataset_name, task_type, test_sample_count, accuracy, precision_score, recall_score,
      macro_f1, weighted_f1, exact_match, token_f1, perplexity, avg_loss, invalid_output_count,
      avg_latency_ms, confusion_matrix_json, samples_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    evalId,
    'ws_default',
    experimentId,
    registeredModelId,
    modelName,
    baseModelId,
    method,
    dsRow.id,
    dsRow.name,
    taskType,
    bundle.test.length,
    result.accuracy,
    result.precision,
    result.recall,
    result.macroF1,
    result.weightedF1,
    result.exactMatch,
    result.tokenF1,
    result.perplexity,
    result.avgLoss,
    result.invalidOutputCount,
    result.avgLatencyMs,
    result.confusionMatrix ? JSON.stringify(result.confusionMatrix) : null,
    JSON.stringify(result.samples),
    now
  );

  if (registeredModelId) {
    db.prepare(
      `UPDATE registered_models SET latest_accuracy = ?, latest_macro_f1 = ?, latest_perplexity = ? WHERE id = ?`
    ).run(result.accuracy ?? result.exactMatch, result.macroF1 ?? result.tokenF1, result.perplexity, registeredModelId);
  }

  logAudit('RUN_EVALUATION', 'evaluation', evalId, `Evaluated ${modelName} on ${dsRow.name} (${bundle.test.length} test rows)`);
  const row = db.prepare('SELECT * FROM evaluation_runs WHERE id = ?').get(evalId) as Record<string, unknown>;
  return mapEvaluationRow(row);
}

export function createApiRouter() {
  const router = express.Router();
  router.use(rateLimit);

  // --- OVERVIEW & HARDWARE ---
  router.get('/overview', (_req, res) => {
    const experiments = listExperiments();
    const completed = experiments.filter((e) => e.status === 'completed');
    const failed = experiments.filter((e) => e.status === 'failed');
    const datasetsCount = (db.prepare('SELECT COUNT(*) as c FROM datasets').get() as { c: number }).c;
    const registeredRows = db
      .prepare('SELECT * FROM registered_models ORDER BY created_at DESC')
      .all() as Record<string, unknown>[];
    const evalRows = db
      .prepare('SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 12')
      .all() as Record<string, unknown>[];

    const avgDurationMs =
      completed.length > 0
        ? Math.round(
            completed.reduce((acc, e) => acc + (e.durationMs ?? 0), 0) / completed.length
          )
        : null;

    const totalTrainableParams = experiments.reduce((acc, e) => acc + e.trainableParams, 0);
    const latestEval = evalRows[0] ? mapEvaluationRow(evalRows[0]) : null;

    res.json({
      stats: {
        totalExperiments: experiments.length,
        completedRuns: completed.length,
        failedRuns: failed.length,
        totalDatasets: datasetsCount,
        totalRegisteredModels: registeredRows.length,
        totalPretrainedModels: PRETRAINED_MODELS.length,
        avgDurationMs,
        totalTrainableParams,
        latestPerformance: latestEval
          ? {
              modelName: latestEval.modelName,
              accuracy: latestEval.accuracy,
              macroF1: latestEval.macroF1,
              perplexity: latestEval.perplexity,
              taskType: latestEval.taskType,
            }
          : null,
      },
      hardware: getHardwareTelemetry(),
      recentExperiments: experiments.slice(0, 6),
      recentModels: registeredRows.slice(0, 5).map(mapRegisteredModelRow),
      recentEvaluations: evalRows.map(mapEvaluationRow),
    });
  });

  // --- DATASETS ---
  router.get('/datasets', (_req, res) => {
    const rows = db
      .prepare('SELECT * FROM datasets ORDER BY created_at DESC')
      .all() as Record<string, unknown>[];
    res.json(rows.map(mapDatasetRow));
  });

  router.post('/datasets/upload', upload.single('file'), (req, res) => {
    try {
      let name = String(req.body?.name ?? '').trim();
      let fileName = '';
      let content = '';

      if (req.file) {
        fileName = path.basename(req.file.originalname);
        content = req.file.buffer.toString('utf-8');
      } else if (typeof req.body?.content === 'string') {
        fileName = path.basename(String(req.body.fileName || 'uploaded_dataset.jsonl'));
        content = req.body.content;
      } else {
        res.status(400).json({ error: 'No dataset file or content provided.' });
        return;
      }

      const ext = (fileName.split('.').pop() || '').toLowerCase();
      if (!ALLOWED_EXTS.has(ext)) {
        res.status(400).json({ error: `Unsupported file type ".${ext}". Allowed: CSV, JSON, JSONL, TXT.` });
        return;
      }
      const fileType = ext as DatasetRecord['fileType'];
      const rows = parseDatasetContent(content, fileType);
      if (rows.length === 0) {
        res.status(400).json({ error: 'Dataset parsed to 0 rows. Check formatting and headers.' });
        return;
      }

      const report = validateDatasetRows(rows);
      const id = `ds_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
      if (!name) name = fileName.replace(/\.[^.]+$/, '');

      const rawPath = path.join(DATASETS_DIR, `${id}.${fileType}`);
      fs.writeFileSync(rawPath, content, 'utf-8');
      const now = new Date().toISOString();

      db.prepare(
        `INSERT INTO datasets (
          id, workspace_id, name, file_name, file_type, format, status,
          row_count, column_count, size_bytes, missing_count, duplicate_count,
          validation_report_json, splits_json, column_mapping_json, raw_path, processed_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        id,
        'ws_default',
        name,
        fileName,
        fileType,
        report.format,
        report.valid ? 'validated' : 'uploaded',
        rows.length,
        report.columnCount,
        Buffer.byteLength(content, 'utf-8'),
        report.missingValuesTotal,
        report.duplicateCount,
        JSON.stringify(report),
        null,
        null,
        rawPath,
        null,
        now,
        now
      );

      logAudit('UPLOAD_DATASET', 'dataset', id, `Uploaded ${fileName} (${rows.length} rows)`);
      const inserted = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id) as Record<string, unknown>;
      res.status(201).json(mapDatasetRow(inserted));
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/datasets/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const row = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id) as Record<string, unknown> | undefined;
      if (!row) {
        res.status(404).json({ error: 'Dataset not found.' });
        return;
      }
      const dataset = mapDatasetRow(row);
      const rawContent = fs.readFileSync(String(row.raw_path), 'utf-8');
      const rawRows = parseDatasetContent(rawContent, dataset.fileType);
      let processedBundle: ProcessedDatasetBundle | null = null;
      if (row.processed_path && fs.existsSync(String(row.processed_path))) {
        processedBundle = JSON.parse(fs.readFileSync(String(row.processed_path), 'utf-8'));
      }

      res.json({
        dataset,
        previewRows: rawRows.slice(0, 25),
        processedPreview: processedBundle
          ? {
              cleanedSample: processedBundle.cleanedRows.slice(0, 12),
              trainSample: processedBundle.train.slice(0, 12).map((e) => e.raw),
              valSample: processedBundle.validation.slice(0, 12).map((e) => e.raw),
              testSample: processedBundle.test.slice(0, 12).map((e) => e.raw),
            }
          : null,
      });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.post('/datasets/:id/validate', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const row = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id) as Record<string, unknown> | undefined;
      if (!row) {
        res.status(404).json({ error: 'Dataset not found.' });
        return;
      }
      const rawContent = fs.readFileSync(String(row.raw_path), 'utf-8');
      const rawRows = parseDatasetContent(rawContent, row.file_type as DatasetRecord['fileType']);
      const columnMapping = req.body?.columnMapping ?? null;
      const report = validateDatasetRows(rawRows, columnMapping);
      const now = new Date().toISOString();

      db.prepare(
        `UPDATE datasets SET format = ?, status = ?, missing_count = ?, duplicate_count = ?, validation_report_json = ?, column_mapping_json = ?, updated_at = ? WHERE id = ?`
      ).run(
        report.format,
        report.valid ? 'validated' : 'uploaded',
        report.missingValuesTotal,
        report.duplicateCount,
        JSON.stringify(report),
        columnMapping ? JSON.stringify(columnMapping) : (row.column_mapping_json ? String(row.column_mapping_json) : null),
        now,
        id
      );

      const updated = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id) as Record<string, unknown>;
      res.json(mapDatasetRow(updated));
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.post('/datasets/:id/preprocess', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const row = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id) as Record<string, unknown> | undefined;
      if (!row) {
        res.status(404).json({ error: 'Dataset not found.' });
        return;
      }

      const trainRatio = Number(req.body?.trainRatio ?? 0.7);
      const valRatio = Number(req.body?.valRatio ?? 0.15);
      const testRatio = Number(req.body?.testRatio ?? 0.15);
      if (
        !Number.isFinite(trainRatio) ||
        !Number.isFinite(valRatio) ||
        !Number.isFinite(testRatio) ||
        trainRatio <= 0 ||
        valRatio <= 0 ||
        testRatio <= 0 ||
        Math.abs(trainRatio + valRatio + testRatio - 1) > 0.02
      ) {
        res.status(400).json({ error: 'Split ratios must be positive numbers summing to 1.00.' });
        return;
      }

      const rawContent = fs.readFileSync(String(row.raw_path), 'utf-8');
      const rawRows = parseDatasetContent(rawContent, row.file_type as DatasetRecord['fileType']);
      const bundle = preprocessAndSplitDataset(rawRows, {
        trainRatio,
        valRatio,
        testRatio,
        deduplicate: req.body?.deduplicate !== false,
        dropMissing: req.body?.dropMissing !== false,
        seed: Number.isFinite(Number(req.body?.seed)) ? Number(req.body?.seed) : 42,
        columnMapping: req.body?.columnMapping ?? (row.column_mapping_json ? JSON.parse(String(row.column_mapping_json)) : null),
      });

      const procPath = path.join(DATASETS_DIR, `${id}.processed.json`);
      fs.writeFileSync(procPath, JSON.stringify(bundle, null, 2), 'utf-8');
      const now = new Date().toISOString();

      db.prepare(
        `UPDATE datasets SET format = ?, status = ?, row_count = ?, missing_count = ?, duplicate_count = ?,
         validation_report_json = ?, splits_json = ?, column_mapping_json = ?, processed_path = ?, updated_at = ? WHERE id = ?`
      ).run(
        bundle.format,
        'training_ready',
        bundle.examples.length,
        bundle.report.missingValuesTotal,
        bundle.report.duplicateCount,
        JSON.stringify(bundle.report),
        JSON.stringify(bundle.splits),
        req.body?.columnMapping ? JSON.stringify(req.body.columnMapping) : (row.column_mapping_json ? String(row.column_mapping_json) : null),
        procPath,
        now,
        id
      );

      logAudit('PREPROCESS_DATASET', 'dataset', id, `Prepared splits (${bundle.splits.trainCount}/${bundle.splits.valCount}/${bundle.splits.testCount})`);
      const updated = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id) as Record<string, unknown>;
      res.json(mapDatasetRow(updated));
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/datasets/:id/export', async (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const row = db.prepare('SELECT * FROM datasets WHERE id = ?').get(id) as Record<string, unknown> | undefined;
      if (!row) {
        res.status(404).json({ error: 'Dataset not found.' });
        return;
      }

      const splitParam = String(req.query.split ?? 'cleaned').toLowerCase();
      const formatParam = String(req.query.format ?? 'jsonl').toLowerCase();
      const validSplits = new Set(['cleaned', 'train', 'val', 'test', 'all']);
      const validFormats = new Set(['jsonl', 'csv', 'json', 'zip']);
      const split = validSplits.has(splitParam) ? splitParam : 'cleaned';
      const format = validFormats.has(formatParam) ? formatParam : 'jsonl';

      const hasProcessed = Boolean(row.processed_path && fs.existsSync(String(row.processed_path)));
      let bundle: ProcessedDatasetBundle | null = null;
      if (hasProcessed) {
        bundle = JSON.parse(fs.readFileSync(String(row.processed_path), 'utf-8')) as ProcessedDatasetBundle;
      }

      if ((split === 'train' || split === 'val' || split === 'test' || format === 'zip') && !bundle) {
        res.status(400).json({
          error: 'Run "Preprocess & Split" first before exporting train/val/test splits or a processed bundle.',
        });
        return;
      }

      if (format === 'zip' && bundle) {
        const zip = new JSZip();
        const trainRows = bundle.train.map((e) => e.raw);
        const valRows = bundle.validation.map((e) => e.raw);
        const testRows = bundle.test.map((e) => e.raw);

        zip.file('cleaned_dataset.jsonl', rowsToJsonl(bundle.cleanedRows));
        zip.file('cleaned_dataset.csv', rowsToCsv(bundle.cleanedRows));
        zip.file('splits/train.jsonl', rowsToJsonl(trainRows));
        zip.file('splits/train.csv', rowsToCsv(trainRows));
        zip.file('splits/validation.jsonl', rowsToJsonl(valRows));
        zip.file('splits/validation.csv', rowsToCsv(valRows));
        zip.file('splits/test.jsonl', rowsToJsonl(testRows));
        zip.file('splits/test.csv', rowsToCsv(testRows));
        zip.file('reports/validation_report.json', JSON.stringify(bundle.report, null, 2));
        zip.file('reports/splits_summary.json', JSON.stringify(bundle.splits, null, 2));
        zip.file(
          'README.md',
          `# ${String(row.name)} — Processed Dataset Bundle\n\n- Format: \`${bundle.format}\`\n- Cleaned Rows: \`${bundle.cleanedRows.length}\`\n- Train Split: \`${trainRows.length}\`\n- Validation Split: \`${valRows.length}\`\n- Test Split: \`${testRows.length}\`\n- Leakage-Free: \`${bundle.splits.leakageFree}\`\n`
        );

        const buffer = await zip.generateAsync({ type: 'nodebuffer' });
        res.setHeader('Content-Type', 'application/zip');
        res.setHeader('Content-Disposition', `attachment; filename="${id}_processed_bundle.zip"`);
        res.send(buffer);
        return;
      }

      let targetRows: RawRow[];
      if (bundle) {
        if (split === 'train') targetRows = bundle.train.map((e) => e.raw);
        else if (split === 'val') targetRows = bundle.validation.map((e) => e.raw);
        else if (split === 'test') targetRows = bundle.test.map((e) => e.raw);
        else targetRows = bundle.cleanedRows;
      } else {
        const raw = fs.readFileSync(String(row.raw_path), 'utf-8');
        if (!req.query.split && !req.query.format) {
          res.setHeader('Content-Type', 'text/plain; charset=utf-8');
          res.setHeader('Content-Disposition', `attachment; filename="${String(row.file_name)}"`);
          res.send(raw);
          return;
        }
        targetRows = parseDatasetContent(raw, row.file_type as DatasetRecord['fileType']);
      }

      const fileSuffix = split === 'cleaned' ? 'cleaned' : split;
      if (format === 'csv') {
        res.setHeader('Content-Type', 'text/csv; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${id}_${fileSuffix}.csv"`);
        res.send(rowsToCsv(targetRows));
        return;
      }
      if (format === 'json') {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${id}_${fileSuffix}.json"`);
        res.send(JSON.stringify(targetRows, null, 2));
        return;
      }

      res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="${id}_${fileSuffix}.jsonl"`);
      res.send(rowsToJsonl(targetRows));
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.delete('/datasets/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      db.prepare('DELETE FROM datasets WHERE id = ?').run(id);
      logAudit('DELETE_DATASET', 'dataset', id, 'Removed dataset');
      res.json({ deleted: true });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // --- MODELS HUB & REGISTRY ---
  router.get('/models/available', (_req, res) => {
    res.json({
      models: PRETRAINED_MODELS,
      hardware: getHardwareTelemetry(),
    });
  });

  router.post('/models/validate-hf', async (req, res) => {
    const repoId = String(req.body?.repoId ?? '').trim();
    if (!/^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/.test(repoId)) {
      res.status(400).json({ error: 'Invalid Hugging Face repository ID. Expected format: org/model-name.' });
      return;
    }

    const catalogMatch = getModelById(repoId);
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 3500);
      const hfRes = await fetch(`https://huggingface.co/api/models/${repoId}`, {
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (hfRes.ok) {
        const meta = (await hfRes.json()) as Record<string, unknown>;
        res.json({
          valid: true,
          reachable: true,
          repoId,
          downloads: meta.downloads ?? null,
          likes: meta.likes ?? null,
          pipelineTag: meta.pipeline_tag ?? 'text-generation',
          architecture: catalogMatch?.architecture ?? 'transformer',
          recommendedModules: catalogMatch?.targetModules ?? ['q_proj', 'v_proj'],
        });
        return;
      }
    } catch {
      // Offline/sandboxed fallback for trusted catalog models
    }

    res.json({
      valid: Boolean(catalogMatch),
      reachable: false,
      repoId,
      architecture: catalogMatch?.architecture ?? 'unknown',
      recommendedModules: catalogMatch?.targetModules ?? ['q_proj', 'v_proj'],
      note: catalogMatch
        ? 'Verified against local MiniMind trusted SLM manifest (Hugging Face external network unreachable in sandbox).'
        : 'Repository could not be reached from this sandbox.',
    });
  });

  router.get('/models/registry', (_req, res) => {
    const rows = db
      .prepare('SELECT * FROM registered_models ORDER BY created_at DESC')
      .all() as Record<string, unknown>[];
    res.json(rows.map(mapRegisteredModelRow));
  });

  router.get('/models/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const row = db
        .prepare('SELECT * FROM registered_models WHERE id = ?')
        .get(id) as Record<string, unknown> | undefined;
      if (!row) {
        res.status(404).json({ error: 'Registered model not found.' });
        return;
      }
      const model = mapRegisteredModelRow(row);
      const exp = getExperimentById(model.experimentId);
      const evals = (
        db
          .prepare('SELECT * FROM evaluation_runs WHERE registered_model_id = ? ORDER BY created_at DESC')
          .all(id) as Record<string, unknown>[]
      ).map(mapEvaluationRow);

      res.json({ model, experiment: exp, evaluations: evals });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.post('/models/:id/deploy', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const existing = db.prepare('SELECT id FROM registered_models WHERE id = ?').get(id) as
        | { id: string }
        | undefined;
      if (!existing) {
        res.status(404).json({ error: 'Registered model not found.' });
        return;
      }
      const status = req.body?.status === 'ready' ? 'ready' : 'deployed';
      db.prepare('UPDATE registered_models SET status = ? WHERE id = ?').run(status, id);
      logAudit('DEPLOY_MODEL', 'registered_model', id, `Updated deployment status to ${status}`);
      const row = db.prepare('SELECT * FROM registered_models WHERE id = ?').get(id) as Record<string, unknown>;
      res.json(mapRegisteredModelRow(row));
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/models/:id/download', async (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const row = db
        .prepare('SELECT * FROM registered_models WHERE id = ?')
        .get(id) as Record<string, unknown> | undefined;
      if (!row) {
        res.status(404).json({ error: 'Registered model not found.' });
        return;
      }
      const model = mapRegisteredModelRow(row);
      const expDir = path.join(ARTIFACTS_DIR, model.experimentId);
      const zip = new JSZip();

      const filesToInclude = [
        { disk: path.join(expDir, 'adapter', 'adapter_model.json'), zipPath: 'adapter/adapter_model.json' },
        { disk: path.join(expDir, 'adapter', 'adapter_config.json'), zipPath: 'adapter/adapter_config.json' },
        { disk: path.join(expDir, 'tokenizer', 'tokenizer.json'), zipPath: 'tokenizer/tokenizer.json' },
        { disk: path.join(expDir, 'training_config.json'), zipPath: 'training_config.json' },
        { disk: path.join(expDir, 'metrics.json'), zipPath: 'metrics.json' },
        { disk: path.join(expDir, 'train_peft.py'), zipPath: 'train_peft.py' },
      ];

      for (const f of filesToInclude) {
        if (fs.existsSync(f.disk)) {
          zip.file(f.zipPath, fs.readFileSync(f.disk));
        }
      }

      const modelCard = `# ${model.name} (${model.version})\n\n- Base Model: \`${model.baseModelName}\`\n- Method: \`${model.method.toUpperCase()}\`\n- Dataset: \`${model.datasetName}\`\n- Trainable Parameters: \`${model.trainableParams.toLocaleString()}\`\n- Latest Accuracy / EM: \`${model.latestAccuracy ?? 'Not evaluated'}\`\n- Latest Perplexity: \`${model.latestPerplexity ?? 'Not evaluated'}\`\n`;
      zip.file('README.md', modelCard);

      const buffer = await zip.generateAsync({ type: 'nodebuffer' });
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', `attachment; filename="${model.name}-${model.version}.zip"`);
      res.send(buffer);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.delete('/models/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      db.prepare('DELETE FROM registered_models WHERE id = ?').run(id);
      logAudit('DELETE_MODEL', 'registered_model', id, 'Deleted registered model');
      res.json({ deleted: true });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // --- TRAINING & MONITORING ---
  router.post('/training/start', (req, res) => {
    try {
      const cfg = req.body as TrainingConfig;
      if (!cfg?.experimentName?.trim() || !cfg?.datasetId || !cfg?.modelId) {
        res.status(400).json({ error: 'experimentName, datasetId, and modelId are required.' });
        return;
      }
      assertSafeId(String(cfg.datasetId));
      if (
        (cfg.method !== 'full' && (!Number.isFinite(cfg.loraR) || cfg.loraR < 1)) ||
        !Number.isFinite(cfg.epochs) ||
        cfg.epochs < 1 ||
        !Number.isFinite(cfg.batchSize) ||
        cfg.batchSize < 1 ||
        !Number.isFinite(cfg.learningRate) ||
        cfg.learningRate <= 0
      ) {
        res.status(400).json({
          error: 'Training hyperparameters (loraR, epochs, batchSize, learningRate) must be positive finite numbers.',
        });
        return;
      }
      const exp = startTrainingJob(cfg);
      res.status(201).json(exp);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/training', (_req, res) => {
    res.json(listExperiments());
  });

  router.get('/training/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const exp = getExperimentById(id);
      if (!exp) {
        res.status(404).json({ error: 'Experiment not found.' });
        return;
      }
      const metrics = (
        db
          .prepare('SELECT * FROM training_metrics WHERE experiment_id = ? ORDER BY step ASC')
          .all(id) as Record<string, unknown>[]
      ).map((m): TrainingMetricPoint => ({
        step: Number(m.step),
        epoch: Number(m.epoch),
        trainLoss: Number(m.train_loss),
        valLoss: m.val_loss !== null && m.val_loss !== undefined ? Number(m.val_loss) : null,
        learningRate: Number(m.learning_rate),
        gradNorm: Number(m.grad_norm),
        memoryMb: Number(m.memory_mb),
        tokensPerSec: Number(m.tokens_per_sec),
        timestamp: String(m.timestamp),
      }));

      const checkpoints = db
        .prepare('SELECT * FROM checkpoints WHERE experiment_id = ? ORDER BY step DESC')
        .all(id) as Record<string, unknown>[];

      res.json({
        experiment: exp,
        metrics,
        checkpoints: checkpoints.map((c) => ({
          id: String(c.id),
          experimentId: String(c.experiment_id),
          step: Number(c.step),
          epoch: Number(c.epoch),
          trainLoss: Number(c.train_loss),
          valLoss: c.val_loss !== null && c.val_loss !== undefined ? Number(c.val_loss) : null,
          filePath: String(c.file_path),
          sizeBytes: Number(c.size_bytes),
          createdAt: String(c.created_at),
        })),
        logs: getJobLogs(id),
        hardware: getHardwareTelemetry(),
      });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/training/:id/stream', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders?.();

      const emitter = getJobEmitter(id);
      const sendEvent = (event: string, data: unknown) => {
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      };

      const exp = getExperimentById(id);
      if (exp) {
        sendEvent('init', { status: exp.status, experiment: exp });
      }

      if (!emitter) {
        res.end();
        return;
      }

      const onMetric = (metric: TrainingMetricPoint) => sendEvent('metric', metric);
      const onLog = (line: string) => sendEvent('log', { line });
      const onStatus = (payload: Record<string, unknown>) => {
        sendEvent('status', payload);
        if (
          payload.status === 'completed' ||
          payload.status === 'failed' ||
          payload.status === 'cancelled'
        ) {
          cleanup();
          res.end();
        }
      };

      const cleanup = () => {
        emitter.off('metric', onMetric);
        emitter.off('log', onLog);
        emitter.off('status', onStatus);
      };

      emitter.on('metric', onMetric);
      emitter.on('log', onLog);
      emitter.on('status', onStatus);
      req.on('close', cleanup);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.post('/training/:id/pause', (req, res) => {
    try {
      const ok = pauseTrainingJob(assertSafeId(req.params.id));
      res.json({ paused: ok });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.post('/training/:id/resume', (req, res) => {
    try {
      const ok = resumeTrainingJob(assertSafeId(req.params.id));
      res.json({ resumed: ok });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.post('/training/:id/cancel', (req, res) => {
    try {
      const ok = cancelTrainingJob(assertSafeId(req.params.id));
      res.json({ cancelled: ok });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/training/:id/logs', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const logs = getJobLogs(id);
      if (req.query.download === '1') {
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${id}.log"`);
        res.send(logs.join('\n'));
        return;
      }
      res.json({ logs });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // --- EVALUATION & COMPARISON ---
  router.post('/evaluation/run', (req, res) => {
    try {
      const modelRef = String(req.body?.modelRef ?? '');
      const datasetId = String(req.body?.datasetId ?? '');
      if (!modelRef || !datasetId) {
        res.status(400).json({ error: 'modelRef and datasetId are required.' });
        return;
      }
      const evaluation = evaluateModelAgainstTestSet(modelRef, datasetId);
      res.status(201).json(evaluation);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/evaluation', (_req, res) => {
    const rows = db
      .prepare('SELECT * FROM evaluation_runs ORDER BY created_at DESC')
      .all() as Record<string, unknown>[];
    res.json(rows.map(mapEvaluationRow));
  });

  router.get('/evaluation/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      const row = db
        .prepare('SELECT * FROM evaluation_runs WHERE id = ?')
        .get(id) as Record<string, unknown> | undefined;
      if (!row) {
        res.status(404).json({ error: 'Evaluation run not found.' });
        return;
      }
      res.json(mapEvaluationRow(row));
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.post('/comparison/run', (req, res) => {
    try {
      const leftModelRef = String(req.body?.leftModelRef ?? '');
      const rightModelRef = String(req.body?.rightModelRef ?? '');
      const datasetId = String(req.body?.datasetId ?? '');
      if (!leftModelRef || !rightModelRef || !datasetId) {
        res.status(400).json({ error: 'leftModelRef, rightModelRef, and datasetId are required.' });
        return;
      }

      const leftEval = evaluateModelAgainstTestSet(leftModelRef, datasetId);
      const rightEval = evaluateModelAgainstTestSet(rightModelRef, datasetId);

      const getMeta = (ref: string) => {
        if (ref.startsWith('base:')) {
          const spec = getModelById(ref.replace(/^base:/, ''));
          return {
            durationMs: 0,
            adapterSizeBytes: 0,
            trainableParams: 0,
            estimatedMemoryGb: spec?.estimatedFp16MemoryGb ?? 1.8,
          };
        }
        const regId = ref.replace(/^reg:/, '');
        const reg = db
          .prepare('SELECT * FROM registered_models WHERE id = ?')
          .get(regId) as Record<string, unknown> | undefined;
        const exp = reg ? getExperimentById(String(reg.experiment_id)) : null;
        const spec = reg ? getModelById(String(reg.base_model_id)) : null;
        return {
          durationMs: exp?.durationMs ?? 0,
          adapterSizeBytes: Number(reg?.adapter_size_bytes ?? 0),
          trainableParams: Number(reg?.trainable_params ?? 0),
          estimatedMemoryGb:
            reg?.method === 'qlora'
              ? spec?.estimatedQloraMemoryGb ?? 0.9
              : spec?.estimatedFp16MemoryGb ?? 2.4,
        };
      };

      const leftMeta = getMeta(leftModelRef);
      const rightMeta = getMeta(rightModelRef);

      const sampleDiffs = leftEval.samples.map((s, i) => ({
        index: s.index,
        prompt: s.prompt,
        groundTruth: s.groundTruth,
        leftPrediction: s.prediction,
        leftCorrect: s.correct,
        leftConfidence: s.confidence,
        rightPrediction: rightEval.samples[i]?.prediction ?? '',
        rightCorrect: rightEval.samples[i]?.correct ?? false,
        rightConfidence: rightEval.samples[i]?.confidence ?? 0,
      }));

      res.json({
        left: { evaluation: leftEval, meta: leftMeta },
        right: { evaluation: rightEval, meta: rightMeta },
        sampleDiffs,
      });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // --- INFERENCE & PLAYGROUND ---
  router.get('/inference/models', (_req, res) => {
    const regRows = db
      .prepare('SELECT * FROM registered_models ORDER BY created_at DESC')
      .all() as Record<string, unknown>[];
    const registered = regRows.map(mapRegisteredModelRow);
    const base = PRETRAINED_MODELS.map((m) => ({
      ref: `base:${m.id}`,
      label: m.aiTool && m.aiTool !== 'Open SLM' ? `[${m.aiTool}] ${m.name}` : `${m.name} (Open SLM Base)`,
      method: 'base' as const,
      baseModelId: m.id,
      parameterCount: m.parameterCount,
      provider: m.provider ?? m.source,
      aiTool: m.aiTool ?? 'Open SLM',
      modelTier: m.modelTier ?? 'Edge SLM',
      trainableLocally: m.trainableLocally ?? true,
    }));
    res.json({
      models: [
        ...registered.map((r) => ({
          ref: `reg:${r.id}`,
          label: `${r.name} (${r.method.toUpperCase()} · ${r.baseModelName})`,
          method: r.method,
          baseModelId: r.baseModelId,
          datasetName: r.datasetName,
          status: r.status,
          aiTool: 'Trained Adapter',
          provider: 'MiniMind Registry',
        })),
        ...base,
      ],
    });
  });

  router.post('/inference/chat', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    const modelRef = String(req.body?.modelRef ?? '');
    const prompt = String(req.body?.prompt ?? '').trim();
    if (!modelRef || !prompt) {
      res.write(`data: ${JSON.stringify({ type: 'error', error: 'modelRef and prompt are required.' })}\n\n`);
      res.end();
      return;
    }

    try {
      for await (const chunk of streamModelInference({
        modelRef,
        prompt,
        systemPrompt: req.body?.systemPrompt,
        choices: req.body?.choices,
        history: req.body?.history,
        temperature: req.body?.temperature,
        topP: req.body?.topP,
        maxNewTokens: req.body?.maxNewTokens,
      })) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
    } catch (err) {
      res.write(`data: ${JSON.stringify({ type: 'error', error: (err as Error).message })}\n\n`);
    }
    res.end();
  });

  router.post('/inference/test-report', (req, res) => {
    try {
      const modelRef = String(req.body?.modelRef ?? '').trim();
      if (!modelRef) {
        res.status(400).json({ error: 'modelRef is required.' });
        return;
      }
      const suiteMode = ['dataset_and_probes', 'dataset_only', 'anti_hallucination'].includes(
        req.body?.suiteMode
      )
        ? req.body.suiteMode
        : 'dataset_and_probes';
      const report = runPlaygroundModelDiagnosticReport({
        modelRef,
        datasetId: req.body?.datasetId ? String(req.body.datasetId) : undefined,
        suiteMode,
      });
      res.status(201).json(report);
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/inference/test-reports', (_req, res) => {
    res.json(listPlaygroundTestReports());
  });

  router.delete('/inference/test-reports/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      db.prepare('DELETE FROM playground_reports WHERE id = ?').run(id);
      res.json({ deleted: true });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.get('/inference/conversations', (_req, res) => {
    const rows = db
      .prepare('SELECT * FROM playground_conversations ORDER BY updated_at DESC LIMIT 20')
      .all() as Record<string, unknown>[];
    res.json(
      rows.map((r) => ({
        id: String(r.id),
        modelRef: String(r.model_ref),
        modelLabel: String(r.model_label),
        systemPrompt: String(r.system_prompt),
        messages: JSON.parse(String(r.messages_json)),
        updatedAt: String(r.updated_at),
      }))
    );
  });

  router.post('/inference/conversations', (req, res) => {
    try {
      const modelRef = String(req.body?.modelRef ?? '').trim();
      if (!modelRef) {
        res.status(400).json({ error: 'modelRef is required.' });
        return;
      }
      const id = req.body?.id
        ? assertSafeId(String(req.body.id))
        : `conv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`;
      const now = new Date().toISOString();
      const modelLabel = String(req.body?.modelLabel || modelRef);
      const systemPrompt = String(req.body?.systemPrompt || '');
      const messages = Array.isArray(req.body?.messages) ? req.body.messages : [];

      const existing = db.prepare('SELECT id FROM playground_conversations WHERE id = ?').get(id);
      if (existing) {
        db.prepare(
          `UPDATE playground_conversations SET model_ref = ?, model_label = ?, system_prompt = ?, messages_json = ?, updated_at = ? WHERE id = ?`
        ).run(modelRef, modelLabel, systemPrompt, JSON.stringify(messages), now, id);
      } else {
        db.prepare(
          `INSERT INTO playground_conversations (id, workspace_id, model_ref, model_label, system_prompt, messages_json, updated_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
        ).run(id, 'ws_default', modelRef, modelLabel, systemPrompt, JSON.stringify(messages), now, now);
      }
      res.json({ id, saved: true });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  router.delete('/inference/conversations/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      db.prepare('DELETE FROM playground_conversations WHERE id = ?').run(id);
      res.json({ deleted: true });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // --- SETTINGS, WORKSPACE & AUDIT ---
  router.get('/settings', (_req, res) => {
    const workspace = db.prepare('SELECT * FROM workspaces WHERE id = ?').get('ws_default');
    const users = db.prepare('SELECT * FROM users WHERE workspace_id = ?').all('ws_default');
    const apiKeys = db
      .prepare('SELECT id, name, prefix, role, last_used_at, created_at FROM api_keys WHERE workspace_id = ? ORDER BY created_at DESC')
      .all('ws_default');
    const auditLogs = db
      .prepare('SELECT * FROM audit_logs ORDER BY id DESC LIMIT 30')
      .all();

    res.json({
      workspace,
      users,
      apiKeys,
      auditLogs,
      hardware: getHardwareTelemetry(),
      geminiConfigured: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'),
    });
  });

  router.post('/settings/workspace', (req, res) => {
    const name = String(req.body?.name ?? '').trim();
    const description = String(req.body?.description ?? '').trim();
    if (!name) {
      res.status(400).json({ error: 'Workspace name is required.' });
      return;
    }
    db.prepare('UPDATE workspaces SET name = ?, description = ? WHERE id = ?').run(
      name,
      description,
      'ws_default'
    );
    logAudit('UPDATE_WORKSPACE', 'workspace', 'ws_default', `Renamed workspace to ${name}`);
    res.json({ updated: true });
  });

  router.post('/settings/api-keys', (req, res) => {
    const name = String(req.body?.name ?? '').trim();
    const role = ['admin', 'engineer', 'viewer'].includes(req.body?.role) ? req.body.role : 'engineer';
    if (!name) {
      res.status(400).json({ error: 'Token name is required.' });
      return;
    }
    const rawSecret = `mm_${crypto.randomBytes(20).toString('hex')}`;
    const prefix = rawSecret.slice(0, 10) + '...';
    const hash = crypto.createHash('sha256').update(rawSecret).digest('hex');
    const id = `key_${Date.now().toString(36)}`;
    const now = new Date().toISOString();

    db.prepare(
      `INSERT INTO api_keys (id, name, prefix, key_hash, role, workspace_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).run(id, name, prefix, hash, role, 'ws_default', now);
    logAudit('CREATE_API_KEY', 'api_key', id, `Generated ${role} token "${name}"`);

    res.status(201).json({ id, name, prefix, role, secretToken: rawSecret, createdAt: now });
  });

  router.delete('/settings/api-keys/:id', (req, res) => {
    try {
      const id = assertSafeId(req.params.id);
      db.prepare('DELETE FROM api_keys WHERE id = ?').run(id);
      logAudit('REVOKE_API_KEY', 'api_key', id, 'Revoked API token');
      res.json({ deleted: true });
    } catch (err) {
      res.status(400).json({ error: (err as Error).message });
    }
  });

  // Ensure artifacts dir stays referenced cleanly
  void ARTIFACTS_DIR;

  return router;
}
