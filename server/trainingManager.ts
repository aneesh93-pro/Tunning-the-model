import fs from 'node:fs';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import { ARTIFACTS_DIR, db, logAudit } from './db.ts';
import { ensureDatasetBundlePrepared } from './datasetBundleService.ts';
import type { ProcessedDatasetBundle } from '../src/ml/datasetProcessor.ts';
import { getModelById } from '../src/ml/modelCatalog.ts';
import {
  buildEngineForDataset,
  computeLearningRate,
  LoRANeuralEngine,
} from '../src/ml/loraEngine.ts';
import type {
  ExperimentRecord,
  ExperimentStatus,
  TrainingConfig,
  TrainingMetricPoint,
} from '../src/shared/types.ts';

interface ActiveJob {
  experimentId: string;
  paused: boolean;
  cancelled: boolean;
  emitter: EventEmitter;
  logs: string[];
}

const activeJobs = new Map<string, ActiveJob>();

export function getJobEmitter(experimentId: string): EventEmitter | undefined {
  return activeJobs.get(experimentId)?.emitter;
}

export function getJobLogs(experimentId: string): string[] {
  const memLogs = activeJobs.get(experimentId)?.logs;
  if (memLogs) return memLogs;
  const logFile = path.join(ARTIFACTS_DIR, experimentId, 'logs', 'training.log');
  if (fs.existsSync(logFile)) {
    return fs.readFileSync(logFile, 'utf-8').split('\n').filter(Boolean);
  }
  return [];
}

function appendLog(job: ActiveJob, artifactDir: string, line: string): void {
  const stamped = `[${new Date().toISOString()}] ${line}`;
  job.logs.push(stamped);
  const logDir = path.join(artifactDir, 'logs');
  fs.mkdirSync(logDir, { recursive: true });
  fs.appendFileSync(path.join(logDir, 'training.log'), stamped + '\n', 'utf-8');
  job.emitter.emit('log', stamped);
}

function generatePyTorchPeftScript(config: TrainingConfig, modelRepoId: string): string {
  const quantBlock =
    config.method === 'qlora'
      ? `bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="${config.quantType}",
    bnb_4bit_use_double_quant=${config.doubleQuant ? 'True' : 'False'},
    bnb_4bit_compute_dtype=torch.${config.computeDtype},
)`
      : 'bnb_config = None';

  return `"""Auto-generated reproducible Hugging Face PEFT / TRL training script."""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "${modelRepoId}"
SEED = ${config.seed}
torch.manual_seed(SEED)

${quantBlock}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.${config.computeDtype},
)
${config.method === 'qlora' ? 'model = prepare_model_for_kbit_training(model)' : ''}

lora_config = LoraConfig(
    r=${config.loraR},
    lora_alpha=${config.loraAlpha},
    lora_dropout=${config.loraDropout},
    bias="${config.bias}",
    task_type="CAUSAL_LM",
    target_modules=${JSON.stringify(config.targetModules)},
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=${config.epochs},
    per_device_train_batch_size=${config.batchSize},
    gradient_accumulation_steps=${config.gradientAccumulation},
    learning_rate=${config.learningRate},
    weight_decay=${config.weightDecay},
    warmup_ratio=${config.warmupRatio},
    lr_scheduler_type="${config.scheduler}",
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=${config.computeDtype === 'bfloat16' ? 'True' : 'False'},
    gradient_checkpointing=${config.gradientCheckpointing ? 'True' : 'False'},
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    max_seq_length=${config.maxSeqLength},
)
trainer.train()
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./tokenizer")
`;
}

export function startTrainingJob(config: TrainingConfig): ExperimentRecord {
  const { datasetRow: dsRow, bundle } = ensureDatasetBundlePrepared(config.datasetId);

  const modelSpec = getModelById(config.modelId);
  if (!modelSpec) {
    throw new Error(`Unsupported base model "${config.modelId}".`);
  }

  if (!bundle.train || bundle.train.length === 0) {
    throw new Error('Training split is empty after preprocessing.');
  }

  const expId = `exp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
  const artifactDir = path.join(ARTIFACTS_DIR, expId);
  fs.mkdirSync(path.join(artifactDir, 'adapter'), { recursive: true });
  fs.mkdirSync(path.join(artifactDir, 'checkpoints'), { recursive: true });
  fs.mkdirSync(path.join(artifactDir, 'tokenizer'), { recursive: true });
  fs.mkdirSync(path.join(artifactDir, 'logs'), { recursive: true });

  const batchesPerEpoch = Math.max(1, Math.ceil(bundle.train.length / Math.max(1, config.batchSize)));
  const totalSteps = batchesPerEpoch * config.epochs;
  const now = new Date().toISOString();

  fs.writeFileSync(
    path.join(artifactDir, 'training_config.json'),
    JSON.stringify(config, null, 2),
    'utf-8'
  );
  fs.writeFileSync(
    path.join(artifactDir, 'train_peft.py'),
    generatePyTorchPeftScript(config, modelSpec.repoId),
    'utf-8'
  );

  db.prepare(
    `INSERT INTO experiments (
      id, workspace_id, name, description, dataset_id, dataset_name, model_id, model_name,
      method, status, config_json, current_epoch, current_step, total_steps,
      artifact_dir, started_at, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    expId,
    'ws_default',
    config.experimentName,
    config.description,
    dsRow.id,
    dsRow.name,
    modelSpec.id,
    modelSpec.name,
    config.method,
    'preparing',
    JSON.stringify(config),
    0,
    0,
    totalSteps,
    artifactDir,
    now,
    now
  );

  logAudit('START_TRAINING', 'experiment', expId, `Started ${config.method.toUpperCase()} on ${modelSpec.name}`);

  const job: ActiveJob = {
    experimentId: expId,
    paused: false,
    cancelled: false,
    emitter: new EventEmitter(),
    logs: [],
  };
  activeJobs.set(expId, job);

  // Run training asynchronously in background
  void runTrainingLoop(expId, config, bundle, modelSpec.chatTemplate, modelSpec.name, modelSpec.id, dsRow.name, artifactDir, job);

  return getExperimentById(expId)!;
}

async function runTrainingLoop(
  expId: string,
  config: TrainingConfig,
  bundle: ProcessedDatasetBundle,
  chatTemplate: 'chatml' | 'phi3',
  modelName: string,
  modelId: string,
  datasetName: string,
  artifactDir: string,
  job: ActiveJob
): Promise<void> {
  const startWall = Date.now();
  try {
    appendLog(job, artifactDir, `Initializing ${config.method.toUpperCase()} pipeline for ${modelName} (seed=${config.seed})`);
    appendLog(job, artifactDir, `Loaded dataset splits: train=${bundle.train.length}, val=${bundle.validation.length}, test=${bundle.test.length} (leakageFree=${bundle.splits.leakageFree})`);

    const allTrainValExamples = [...bundle.train, ...bundle.validation, ...bundle.test];
    const engine = buildEngineForDataset(allTrainValExamples, config, chatTemplate);
    const trainableParams = engine.getTrainableParamCount();
    const totalParams = engine.getTotalParamCount();
    const trainablePercent = Number(((trainableParams / Math.max(1, totalParams)) * 100).toFixed(2));

    if (config.method === 'qlora') {
      appendLog(
        job,
        artifactDir,
        `Applied 4-bit ${config.quantType.toUpperCase()} quantization (double_quant=${config.doubleQuant}). Base weight RMSE=${engine.quantizationRmse?.toFixed(6)}`
      );
    }
    appendLog(
      job,
      artifactDir,
      `Initialized LoRA adapters (r=${config.loraR}, alpha=${config.loraAlpha}, dropout=${config.loraDropout}, targets=${config.targetModules.join(',')}). Trainable params=${trainableParams} (${trainablePercent}%)`
    );

    const batchesPerEpoch = Math.max(1, Math.ceil(bundle.train.length / Math.max(1, config.batchSize)));
    const totalSteps = batchesPerEpoch * config.epochs;

    db.prepare(
      `UPDATE experiments SET status = ?, trainable_params = ?, total_params = ?, trainable_percent = ?, quantization_rmse = ?, total_steps = ? WHERE id = ?`
    ).run('running', trainableParams, totalParams, trainablePercent, engine.quantizationRmse, totalSteps, expId);

    job.emitter.emit('status', { status: 'running', trainableParams, totalParams, trainablePercent });

    let step = 0;
    let bestValLoss = Infinity;
    let patienceCounter = 0;
    let lastTrainLoss = 0;
    let lastValLoss: number | null = null;
    const metricsHistory: TrainingMetricPoint[] = [];

    for (let epoch = 1; epoch <= config.epochs; epoch++) {
      // Deterministic per-epoch batch order
      const epochData = [...bundle.train];
      for (let i = epochData.length - 1; i > 0; i--) {
        const j = (config.seed + epoch * 31 + i * 17) % (i + 1);
        [epochData[i], epochData[j]] = [epochData[j], epochData[i]];
      }

      for (let b = 0; b < batchesPerEpoch; b++) {
        // Handle pause/cancel cleanly
        while (job.paused && !job.cancelled) {
          await new Promise((r) => setTimeout(r, 150));
        }
        if (job.cancelled) {
          db.prepare(
            `UPDATE experiments SET status = ?, duration_ms = ?, completed_at = ? WHERE id = ?`
          ).run('cancelled', Date.now() - startWall, new Date().toISOString(), expId);
          appendLog(job, artifactDir, `Training cancelled by user at step ${step}.`);
          job.emitter.emit('status', { status: 'cancelled' });
          activeJobs.delete(expId);
          return;
        }

        step++;
        const batch = epochData.slice(b * config.batchSize, (b + 1) * config.batchSize);
        const lr = computeLearningRate(
          step,
          totalSteps,
          config.learningRate,
          config.warmupRatio,
          config.scheduler
        );

        const stepStart = performance.now();
        const { loss, gradNorm, tokensProcessed } = engine.trainBatch(
          batch,
          lr,
          config.weightDecay,
          config.optimizer
        );
        const stepSec = Math.max(0.002, (performance.now() - stepStart) / 1000);
        const isLastStepInEpoch = b === batchesPerEpoch - 1;

        let valLoss: number | null = null;
        if (isLastStepInEpoch || step === 1 || step % Math.max(2, Math.floor(totalSteps / 6)) === 0) {
          valLoss = Number(engine.evaluateLoss(bundle.validation, true).toFixed(4));
          lastValLoss = valLoss;
        }

        lastTrainLoss = Number(loss.toFixed(4));
        const memMb = Number((process.memoryUsage().rss / 1024 ** 2).toFixed(1));
        const tokensPerSec = Number((tokensProcessed / stepSec).toFixed(1));

        const metric: TrainingMetricPoint = {
          step,
          epoch,
          trainLoss: lastTrainLoss,
          valLoss,
          learningRate: Number(lr.toFixed(6)),
          gradNorm: Number(gradNorm.toFixed(4)),
          memoryMb: memMb,
          tokensPerSec,
          timestamp: new Date().toISOString(),
        };
        metricsHistory.push(metric);

        db.prepare(
          `INSERT INTO training_metrics (experiment_id, step, epoch, train_loss, val_loss, learning_rate, grad_norm, memory_mb, tokens_per_sec, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        ).run(
          expId,
          metric.step,
          metric.epoch,
          metric.trainLoss,
          metric.valLoss,
          metric.learningRate,
          metric.gradNorm,
          metric.memoryMb,
          metric.tokensPerSec,
          metric.timestamp
        );

        db.prepare(
          `UPDATE experiments SET current_epoch = ?, current_step = ?, train_loss = ?, val_loss = COALESCE(?, val_loss) WHERE id = ?`
        ).run(epoch, step, metric.trainLoss, metric.valLoss, expId);

        job.emitter.emit('metric', metric);

        // Save checkpoint at epoch boundaries
        if (isLastStepInEpoch) {
          const ckptPath = path.join(artifactDir, 'checkpoints', `checkpoint-step-${step}.json`);
          const serialized = engine.serialize();
          const ckptJson = JSON.stringify(serialized);
          fs.writeFileSync(ckptPath, ckptJson, 'utf-8');
          const ckptBytes = Buffer.byteLength(ckptJson, 'utf-8');

          db.prepare(
            `INSERT INTO checkpoints (id, experiment_id, step, epoch, train_loss, val_loss, file_path, size_bytes, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
          ).run(
            `ckpt_${expId}_${step}`,
            expId,
            step,
            epoch,
            metric.trainLoss,
            valLoss,
            ckptPath,
            ckptBytes,
            new Date().toISOString()
          );

          appendLog(
            job,
            artifactDir,
            `Epoch ${epoch}/${config.epochs} | step=${step}/${totalSteps} | train_loss=${metric.trainLoss.toFixed(4)} | val_loss=${valLoss?.toFixed(4) ?? 'n/a'} | lr=${metric.learningRate.toExponential(2)}`
          );

          if (valLoss !== null && config.earlyStoppingPatience > 0) {
            if (valLoss < bestValLoss - 1e-4) {
              bestValLoss = valLoss;
              patienceCounter = 0;
            } else {
              patienceCounter++;
              if (patienceCounter >= config.earlyStoppingPatience && epoch >= 3) {
                appendLog(job, artifactDir, `Early stopping triggered at epoch ${epoch} (best val_loss=${bestValLoss.toFixed(4)}).`);
                break;
              }
            }
          }
        }

        // Yield briefly so SSE streams smoothly in real time
        await new Promise((r) => setTimeout(r, 80));
      }

      if (patienceCounter >= config.earlyStoppingPatience && epoch >= 3) {
        break;
      }
    }

    // Save final adapter, adapter_config.json, tokenizer.json, and metrics.json
    const adapterModelPath = path.join(artifactDir, 'adapter', 'adapter_model.json');
    const serialized = engine.serialize();
    const adapterJson = JSON.stringify(serialized);
    fs.writeFileSync(adapterModelPath, adapterJson, 'utf-8');
    const adapterSizeBytes = Buffer.byteLength(adapterJson, 'utf-8');

    fs.writeFileSync(
      path.join(artifactDir, 'adapter', 'adapter_config.json'),
      JSON.stringify(
        {
          peft_type: 'LORA',
          task_type: serialized.taskType === 'classification' ? 'SEQ_CLS' : 'CAUSAL_LM',
          base_model_name_or_path: modelId,
          r: config.loraR,
          lora_alpha: config.loraAlpha,
          lora_dropout: config.loraDropout,
          bias: config.bias,
          target_modules: config.targetModules,
          quantization: config.method === 'qlora' ? { bits: 4, quant_type: config.quantType, double_quant: config.doubleQuant } : null,
        },
        null,
        2
      ),
      'utf-8'
    );

    fs.writeFileSync(
      path.join(artifactDir, 'tokenizer', 'tokenizer.json'),
      JSON.stringify(serialized.tokenizer, null, 2),
      'utf-8'
    );

    fs.writeFileSync(
      path.join(artifactDir, 'metrics.json'),
      JSON.stringify(metricsHistory, null, 2),
      'utf-8'
    );

    const durationMs = Date.now() - startWall;
    const completedAt = new Date().toISOString();

    db.prepare(
      `UPDATE experiments SET status = ?, adapter_size_bytes = ?, duration_ms = ?, completed_at = ? WHERE id = ?`
    ).run('completed', adapterSizeBytes, durationMs, completedAt, expId);

    // Register model automatically in Model Registry
    const regId = `mdl_${expId}`;
    const safeName = `${config.experimentName.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${config.method}`;
    db.prepare(
      `INSERT INTO registered_models (
        id, workspace_id, name, version, base_model_id, base_model_name, method,
        experiment_id, dataset_id, dataset_name, adapter_path, adapter_size_bytes,
        trainable_params, status, training_config_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      regId,
      'ws_default',
      safeName,
      'v1.0.0',
      modelId,
      modelName,
      config.method,
      expId,
      config.datasetId,
      datasetName,
      adapterModelPath,
      adapterSizeBytes,
      trainableParams,
      'ready',
      JSON.stringify(config),
      completedAt
    );

    appendLog(
      job,
      artifactDir,
      `Training completed in ${(durationMs / 1000).toFixed(2)}s. Final train_loss=${lastTrainLoss.toFixed(4)}, val_loss=${lastValLoss?.toFixed(4) ?? 'n/a'}. Registered adapter ${regId}.`
    );
    job.emitter.emit('status', { status: 'completed', registeredModelId: regId });
    logAudit('COMPLETE_TRAINING', 'experiment', expId, `Saved adapter (${adapterSizeBytes} bytes) as ${safeName}`);
  } catch (err) {
    const msg = (err as Error).message;
    db.prepare(
      `UPDATE experiments SET status = ?, error_message = ?, duration_ms = ?, completed_at = ? WHERE id = ?`
    ).run('failed', msg, Date.now() - startWall, new Date().toISOString(), expId);
    appendLog(job, artifactDir, `ERROR: ${msg}`);
    job.emitter.emit('status', { status: 'failed', error: msg });
  } finally {
    setTimeout(() => activeJobs.delete(expId), 15000);
  }
}

export function pauseTrainingJob(experimentId: string): boolean {
  const job = activeJobs.get(experimentId);
  if (!job) return false;
  job.paused = true;
  db.prepare('UPDATE experiments SET status = ? WHERE id = ?').run('paused', experimentId);
  job.emitter.emit('status', { status: 'paused' });
  return true;
}

export function resumeTrainingJob(experimentId: string): boolean {
  const job = activeJobs.get(experimentId);
  if (!job) return false;
  job.paused = false;
  db.prepare('UPDATE experiments SET status = ? WHERE id = ?').run('running', experimentId);
  job.emitter.emit('status', { status: 'running' });
  return true;
}

export function cancelTrainingJob(experimentId: string): boolean {
  const job = activeJobs.get(experimentId);
  if (job) {
    job.cancelled = true;
    return true;
  }
  const row = db.prepare('SELECT status FROM experiments WHERE id = ?').get(experimentId) as
    | { status: ExperimentStatus }
    | undefined;
  if (row && (row.status === 'running' || row.status === 'paused' || row.status === 'preparing')) {
    db.prepare('UPDATE experiments SET status = ?, completed_at = ? WHERE id = ?').run(
      'cancelled',
      new Date().toISOString(),
      experimentId
    );
    return true;
  }
  return false;
}

export function getExperimentById(id: string): ExperimentRecord | null {
  const row = db.prepare('SELECT * FROM experiments WHERE id = ?').get(id) as Record<string, unknown> | undefined;
  if (!row) return null;
  return mapExperimentRow(row);
}

export function listExperiments(): ExperimentRecord[] {
  const rows = db
    .prepare('SELECT * FROM experiments ORDER BY created_at DESC')
    .all() as Record<string, unknown>[];
  return rows.map(mapExperimentRow);
}

function mapExperimentRow(row: Record<string, unknown>): ExperimentRecord {
  return {
    id: String(row.id),
    name: String(row.name),
    description: String(row.description ?? ''),
    datasetId: String(row.dataset_id),
    datasetName: String(row.dataset_name),
    modelId: String(row.model_id),
    modelName: String(row.model_name),
    method: row.method as ExperimentRecord['method'],
    status: row.status as ExperimentStatus,
    config: JSON.parse(String(row.config_json)) as TrainingConfig,
    currentEpoch: Number(row.current_epoch ?? 0),
    currentStep: Number(row.current_step ?? 0),
    totalSteps: Number(row.total_steps ?? 0),
    trainLoss: row.train_loss !== null && row.train_loss !== undefined ? Number(row.train_loss) : null,
    valLoss: row.val_loss !== null && row.val_loss !== undefined ? Number(row.val_loss) : null,
    trainableParams: Number(row.trainable_params ?? 0),
    totalParams: Number(row.total_params ?? 0),
    trainablePercent: Number(row.trainable_percent ?? 0),
    quantizationRmse:
      row.quantization_rmse !== null && row.quantization_rmse !== undefined
        ? Number(row.quantization_rmse)
        : null,
    adapterSizeBytes:
      row.adapter_size_bytes !== null && row.adapter_size_bytes !== undefined
        ? Number(row.adapter_size_bytes)
        : null,
    artifactDir: String(row.artifact_dir),
    errorMessage: row.error_message ? String(row.error_message) : null,
    durationMs: row.duration_ms !== null && row.duration_ms !== undefined ? Number(row.duration_ms) : null,
    startedAt: row.started_at ? String(row.started_at) : null,
    completedAt: row.completed_at ? String(row.completed_at) : null,
    createdAt: String(row.created_at),
  };
}

export function loadTrainedEngineFromAdapter(adapterPath: string): LoRANeuralEngine {
  if (!fs.existsSync(adapterPath)) {
    throw new Error(`Adapter file not found at ${adapterPath}`);
  }
  const serialized = JSON.parse(fs.readFileSync(adapterPath, 'utf-8'));
  return LoRANeuralEngine.deserialize(serialized);
}
