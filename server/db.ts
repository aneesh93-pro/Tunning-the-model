import { createRequire } from 'node:module';
import type { DatabaseSync as DatabaseSyncType } from 'node:sqlite';
import fs from 'node:fs';
import path from 'node:path';
import { installSqliteWarningFilter } from './suppressSqliteWarning.ts';
import {
  parseDatasetContent,
  preprocessAndSplitDataset,
  validateDatasetRows,
} from '../src/ml/datasetProcessor.ts';

installSqliteWarningFilter();
const require = createRequire(import.meta.url);
const { DatabaseSync } = require('node:sqlite') as {
  DatabaseSync: typeof DatabaseSyncType;
};

const IS_VERCEL = Boolean(process.env.VERCEL);
export const DATA_DIR = process.env.DATA_DIR
  ? path.resolve(process.env.DATA_DIR)
  : IS_VERCEL
    ? '/tmp/minimind-data'
    : path.resolve(process.cwd(), 'data');
export const DATASETS_DIR = path.join(DATA_DIR, 'datasets');
export const ARTIFACTS_DIR = process.env.ARTIFACTS_DIR
  ? path.resolve(process.env.ARTIFACTS_DIR)
  : IS_VERCEL
    ? '/tmp/minimind-artifacts'
    : path.resolve(process.cwd(), 'artifacts');

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(DATASETS_DIR, { recursive: true });
fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });

export const db: DatabaseSyncType = new DatabaseSync(path.join(DATA_DIR, 'minimind.db'));
db.exec('PRAGMA journal_mode = WAL;');
db.exec('PRAGMA foreign_keys = ON;');

db.exec(`
  CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    workspace_id TEXT NOT NULL,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    last_used_at TEXT,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    format TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    missing_count INTEGER NOT NULL,
    duplicate_count INTEGER NOT NULL,
    validation_report_json TEXT,
    splits_json TEXT,
    column_mapping_json TEXT,
    raw_path TEXT NOT NULL,
    processed_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    current_epoch INTEGER NOT NULL DEFAULT 0,
    current_step INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    train_loss REAL,
    val_loss REAL,
    trainable_params INTEGER NOT NULL DEFAULT 0,
    total_params INTEGER NOT NULL DEFAULT 0,
    trainable_percent REAL NOT NULL DEFAULT 0,
    quantization_rmse REAL,
    adapter_size_bytes INTEGER,
    artifact_dir TEXT NOT NULL,
    error_message TEXT,
    duration_ms INTEGER,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss REAL NOT NULL,
    val_loss REAL,
    learning_rate REAL NOT NULL,
    grad_norm REAL NOT NULL,
    memory_mb REAL NOT NULL,
    tokens_per_sec REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_metrics_exp ON training_metrics(experiment_id, step);

  CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss REAL NOT NULL,
    val_loss REAL,
    file_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS registered_models (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    base_model_id TEXT NOT NULL,
    base_model_name TEXT NOT NULL,
    method TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    adapter_path TEXT NOT NULL,
    adapter_size_bytes INTEGER NOT NULL,
    trainable_params INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    latest_accuracy REAL,
    latest_macro_f1 REAL,
    latest_perplexity REAL,
    training_config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    experiment_id TEXT,
    registered_model_id TEXT,
    model_name TEXT NOT NULL,
    base_model_id TEXT NOT NULL,
    method TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    test_sample_count INTEGER NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    macro_f1 REAL,
    weighted_f1 REAL,
    exact_match REAL,
    token_f1 REAL,
    perplexity REAL NOT NULL,
    avg_loss REAL NOT NULL,
    invalid_output_count INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL NOT NULL,
    confusion_matrix_json TEXT,
    samples_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS playground_conversations (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    model_label TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS playground_reports (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    model_ref TEXT NOT NULL,
    model_label TEXT NOT NULL,
    method TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    suite_mode TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    passed_count INTEGER NOT NULL,
    pass_rate REAL NOT NULL,
    hallucination_count INTEGER NOT NULL,
    hallucination_rate REAL NOT NULL,
    avg_score REAL NOT NULL,
    avg_latency_ms REAL NOT NULL,
    verdict TEXT NOT NULL,
    cases_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
`);

export function logAudit(
  action: string,
  resourceType: string,
  resourceId: string,
  detail: string,
  actor = 'ml-engineer@minimind.local',
  workspaceId = 'ws_default'
): void {
  db.prepare(
    `INSERT INTO audit_logs (workspace_id, actor, action, resource_type, resource_id, detail, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  ).run(workspaceId, actor, action, resourceType, resourceId, detail, new Date().toISOString());
}

function seedStarterEnvironment(): void {
  const now = new Date().toISOString();
  const wsExists = db.prepare('SELECT id FROM workspaces WHERE id = ?').get('ws_default');
  if (!wsExists) {
    db.prepare(
      `INSERT INTO workspaces (id, name, description, created_at) VALUES (?, ?, ?, ?)`
    ).run(
      'ws_default',
      'Core SLM Research Lab',
      'Primary fine-tuning workspace for LoRA & QLoRA adapter experiments.',
      now
    );

    db.prepare(
      `INSERT INTO users (id, name, email, role, workspace_id, created_at) VALUES (?, ?, ?, ?, ?, ?)`
    ).run('usr_lead', 'Lead ML Engineer', 'ml-engineer@minimind.local', 'admin', 'ws_default', now);
  }

  // Seed or repair starter datasets on disk if missing (NO fake experiments or fake metrics!)
  const bankingRow = db
    .prepare('SELECT id, raw_path, processed_path FROM datasets WHERE id = ?')
    .get('ds_banking_intents') as
    | { id: string; raw_path: string; processed_path: string | null }
    | undefined;
  const bankingNeedsSeed =
    !bankingRow ||
    !bankingRow.raw_path ||
    !fs.existsSync(bankingRow.raw_path) ||
    !bankingRow.processed_path ||
    !fs.existsSync(bankingRow.processed_path);

  if (bankingNeedsSeed) {
    const bankingRows = [
      { instruction: 'Classify the customer banking request.', input: 'I lost my debit card on the train this morning.', output: 'card_lost_or_stolen' },
      { instruction: 'Classify the customer banking request.', input: 'Please freeze my Visa credit card immediately.', output: 'card_lost_or_stolen' },
      { instruction: 'Classify the customer banking request.', input: 'My ATM card was swallowed by the machine.', output: 'card_lost_or_stolen' },
      { instruction: 'Classify the customer banking request.', input: 'Someone stole my wallet with my bank card inside.', output: 'card_lost_or_stolen' },
      { instruction: 'Classify the customer banking request.', input: 'Need to report a missing debit card and order replacement.', output: 'card_lost_or_stolen' },
      { instruction: 'Classify the customer banking request.', input: 'How do I temporarily lock my physical debit card?', output: 'card_lost_or_stolen' },
      { instruction: 'Classify the customer banking request.', input: 'Send $450 from checking to my savings account.', output: 'internal_transfer' },
      { instruction: 'Classify the customer banking request.', input: 'Transfer funds between my two personal accounts.', output: 'internal_transfer' },
      { instruction: 'Classify the customer banking request.', input: 'Move money into my high-yield savings vault.', output: 'internal_transfer' },
      { instruction: 'Classify the customer banking request.', input: 'How long does an ACH transfer take to settle?', output: 'internal_transfer' },
      { instruction: 'Classify the customer banking request.', input: 'Schedule a recurring monthly transfer to checking.', output: 'internal_transfer' },
      { instruction: 'Classify the customer banking request.', input: 'Cancel my pending wire transfer to vendor.', output: 'internal_transfer' },
      { instruction: 'Classify the customer banking request.', input: 'What is my current available checking balance?', output: 'balance_inquiry' },
      { instruction: 'Classify the customer banking request.', input: 'Check how much money is left in savings.', output: 'balance_inquiry' },
      { instruction: 'Classify the customer banking request.', input: 'Show my account balance after pending charges.', output: 'balance_inquiry' },
      { instruction: 'Classify the customer banking request.', input: 'Did my direct deposit paycheck arrive today?', output: 'balance_inquiry' },
      { instruction: 'Classify the customer banking request.', input: 'View total funds across all checking accounts.', output: 'balance_inquiry' },
      { instruction: 'Classify the customer banking request.', input: 'Why is my available balance lower than ledger balance?', output: 'balance_inquiry' },
      { instruction: 'Classify the customer banking request.', input: 'What is the international wire fee for EUR?', output: 'fees_and_rates' },
      { instruction: 'Classify the customer banking request.', input: 'Tell me the APY interest rate on savings.', output: 'fees_and_rates' },
      { instruction: 'Classify the customer banking request.', input: 'Was I charged an overdraft fee yesterday?', output: 'fees_and_rates' },
      { instruction: 'Classify the customer banking request.', input: 'Explain foreign transaction fees on my card.', output: 'fees_and_rates' },
      { instruction: 'Classify the customer banking request.', input: 'How do I qualify to waive the monthly maintenance fee?', output: 'fees_and_rates' },
      { instruction: 'Classify the customer banking request.', input: 'What APR applies to cash advances?', output: 'fees_and_rates' },
    ];

    const bankingJsonl = bankingRows.map((r) => JSON.stringify(r)).join('\n');
    const bankingRawPath = path.join(DATASETS_DIR, 'ds_banking_intents.jsonl');
    const bankingProcPath = path.join(DATASETS_DIR, 'ds_banking_intents.processed.json');
    fs.writeFileSync(bankingRawPath, bankingJsonl, 'utf-8');
    const parsedBanking = parseDatasetContent(bankingJsonl, 'jsonl');
    const validationBanking = validateDatasetRows(parsedBanking);
    const bundleBanking = preprocessAndSplitDataset(parsedBanking, { seed: 42 });
    fs.writeFileSync(bankingProcPath, JSON.stringify(bundleBanking, null, 2), 'utf-8');

    if (!bankingRow) {
      db.prepare(
        `INSERT INTO datasets (
          id, workspace_id, name, file_name, file_type, format, status,
          row_count, column_count, size_bytes, missing_count, duplicate_count,
          validation_report_json, splits_json, column_mapping_json, raw_path, processed_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        'ds_banking_intents',
        'ws_default',
        'Banking Intent Classification (24)',
        'banking_intents.jsonl',
        'jsonl',
        'instruction',
        'training_ready',
        bundleBanking.cleanedRows.length,
        validationBanking.columnCount,
        Buffer.byteLength(bankingJsonl, 'utf-8'),
        validationBanking.missingValuesTotal,
        validationBanking.duplicateCount,
        JSON.stringify(bundleBanking.report),
        JSON.stringify(bundleBanking.splits),
        JSON.stringify({ instruction: 'instruction', input: 'input', output: 'output' }),
        bankingRawPath,
        bankingProcPath,
        now,
        now
      );
    } else {
      db.prepare(
        `UPDATE datasets SET raw_path = ?, processed_path = ?, status = 'training_ready', splits_json = ?, updated_at = ? WHERE id = ?`
      ).run(bankingRawPath, bankingProcPath, JSON.stringify(bundleBanking.splits), now, 'ds_banking_intents');
    }
  }

  const chatRow = db
    .prepare('SELECT id, raw_path, processed_path FROM datasets WHERE id = ?')
    .get('ds_slm_engineering_chat') as
    | { id: string; raw_path: string; processed_path: string | null }
    | undefined;
  const chatNeedsSeed =
    !chatRow ||
    !chatRow.raw_path ||
    !fs.existsSync(chatRow.raw_path) ||
    !chatRow.processed_path ||
    !fs.existsSync(chatRow.processed_path);

  if (chatNeedsSeed) {
    const chatRows = [
      {
        messages: [
          { role: 'user', content: 'How do I check PyTorch CUDA availability?' },
          { role: 'assistant', content: 'Run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'What LoRA rank should I start with for 0.5B models?' },
          { role: 'assistant', content: 'Start with `r=8`, `lora_alpha=16`, and `lora_dropout=0.05` targeting `q_proj` and `v_proj`.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'How does QLoRA reduce VRAM usage?' },
          { role: 'assistant', content: 'QLoRA quantizes frozen base weights to 4-bit NormalFloat (NF4) with double quantization while training 16-bit LoRA adapters.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'Why should validation and test splits remain strictly separate?' },
          { role: 'assistant', content: 'Using test examples during hyperparameter tuning causes data leakage and inflates reported generalization metrics.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'How do I merge a LoRA adapter back into base weights?' },
          { role: 'assistant', content: 'Call `model = PeftModel.from_pretrained(base, adapter_dir); merged = model.merge_and_unload()`.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'What optimizer works best for QLoRA fine-tuning?' },
          { role: 'assistant', content: 'Use `paged_adamw_8bit` or `adamw_torch` with cosine learning rate decay and a 5–10% warmup ratio.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'How do I prevent overfitting on small instruction datasets?' },
          { role: 'assistant', content: 'Enable early stopping on validation loss, add `lora_dropout=0.05`, use weight decay `0.01`, and limit epochs to 3–5.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'Which target modules differ in Phi-3.5 compared to Qwen2.5?' },
          { role: 'assistant', content: 'Phi-3.5 uses fused `qkv_proj` and `gate_up_proj` modules instead of separate `q_proj`, `k_proj`, `v_proj`, and `gate_proj`.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'How is perplexity computed from cross-entropy loss?' },
          { role: 'assistant', content: 'Perplexity equals `exp(avg_cross_entropy_loss)`, measuring average branching uncertainty per token.' },
        ],
      },
      {
        messages: [
          { role: 'user', content: 'What does gradient accumulation do during SFT?' },
          { role: 'assistant', content: 'It sums gradients over `N` micro-batches before stepping the optimizer, simulating larger batch sizes within limited VRAM.' },
        ],
      },
    ];

    const chatJsonl = chatRows.map((r) => JSON.stringify(r)).join('\n');
    const chatRawPath = path.join(DATASETS_DIR, 'ds_slm_engineering_chat.jsonl');
    const chatProcPath = path.join(DATASETS_DIR, 'ds_slm_engineering_chat.processed.json');
    fs.writeFileSync(chatRawPath, chatJsonl, 'utf-8');
    const parsedChat = parseDatasetContent(chatJsonl, 'jsonl');
    const validationChat = validateDatasetRows(parsedChat);
    const bundleChat = preprocessAndSplitDataset(parsedChat, { seed: 42 });
    fs.writeFileSync(chatProcPath, JSON.stringify(bundleChat, null, 2), 'utf-8');

    if (!chatRow) {
      db.prepare(
        `INSERT INTO datasets (
          id, workspace_id, name, file_name, file_type, format, status,
          row_count, column_count, size_bytes, missing_count, duplicate_count,
          validation_report_json, splits_json, column_mapping_json, raw_path, processed_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        'ds_slm_engineering_chat',
        'ws_default',
        'SLM Engineering Chat (10)',
        'slm_engineering_chat.jsonl',
        'jsonl',
        'conversational',
        'training_ready',
        bundleChat.cleanedRows.length,
        validationChat.columnCount,
        Buffer.byteLength(chatJsonl, 'utf-8'),
        validationChat.missingValuesTotal,
        validationChat.duplicateCount,
        JSON.stringify(bundleChat.report),
        JSON.stringify(bundleChat.splits),
        JSON.stringify({ messages: 'messages' }),
        chatRawPath,
        chatProcPath,
        now,
        now
      );
    } else {
      db.prepare(
        `UPDATE datasets SET raw_path = ?, processed_path = ?, status = 'training_ready', splits_json = ?, updated_at = ? WHERE id = ?`
      ).run(chatRawPath, chatProcPath, JSON.stringify(bundleChat.splits), now, 'ds_slm_engineering_chat');
    }
  }
}

seedStarterEnvironment();
