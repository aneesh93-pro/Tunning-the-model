import type { DatasetRecord, RegisteredModelRecord } from '../shared/types';
import { formatBytes, formatNumber } from '../lib/utils';

export type DatasetBundleDocId = 'cleaned' | 'train' | 'val' | 'test' | 'raw' | 'zip';

export interface DatasetBundleDoc {
  id: DatasetBundleDocId;
  title: string;
  fileName: string;
  subtitle: string;
  description: string;
  rowCount: number;
  badge: string;
  previewTab?: 'raw' | 'cleaned' | 'train' | 'val' | 'test';
  exportUrl: string;
  accentHex: string;
}

export type AdapterBundleDocId = 'config' | 'weights' | 'script' | 'card' | 'eval';

export interface AdapterBundleDoc {
  id: AdapterBundleDocId;
  title: string;
  fileName: string;
  subtitle: string;
  description: string;
  previewCode: string;
  badge: string;
  accentHex: string;
}

export function buildDatasetBundleDocs(
  dataset: DatasetRecord,
  exportFormat: 'jsonl' | 'csv' | 'json'
): DatasetBundleDoc[] {
  const safeBase = dataset.name.toLowerCase().replace(/[^a-z0-9_-]+/g, '_') || 'dataset';

  if (dataset.status === 'training_ready' && dataset.splits) {
    const cleanedCount =
      dataset.splits.trainCount + dataset.splits.valCount + dataset.splits.testCount;
    return [
      {
        id: 'cleaned',
        title: 'Cleaned Corpus',
        fileName: `${safeBase}_cleaned.${exportFormat}`,
        subtitle: `${cleanedCount} rows · 0 duplicates`,
        description: `Deduplicated ${dataset.format} corpus ready for fine-tuning`,
        rowCount: cleanedCount,
        badge: exportFormat.toUpperCase(),
        previewTab: 'cleaned',
        exportUrl: `/api/datasets/${dataset.id}/export?split=cleaned&format=${exportFormat}`,
        accentHex: '#2563EB',
      },
      {
        id: 'train',
        title: 'Train Split',
        fileName: `${safeBase}_train.${exportFormat}`,
        subtitle: `${dataset.splits.trainCount} rows (${Math.round(dataset.splits.trainRatio * 100)}%)`,
        description: `Primary gradient optimization partition (${dataset.splits.leakageFree ? 'leak-free' : 'checked'})`,
        rowCount: dataset.splits.trainCount,
        badge: 'TRAIN',
        previewTab: 'train',
        exportUrl: `/api/datasets/${dataset.id}/export?split=train&format=${exportFormat}`,
        accentHex: '#059669',
      },
      {
        id: 'val',
        title: 'Validation Split',
        fileName: `${safeBase}_val.${exportFormat}`,
        subtitle: `${dataset.splits.valCount} rows (${Math.round(dataset.splits.valRatio * 100)}%)`,
        description: 'Epoch validation loss & early stopping telemetry partition',
        rowCount: dataset.splits.valCount,
        badge: 'VAL',
        previewTab: 'val',
        exportUrl: `/api/datasets/${dataset.id}/export?split=val&format=${exportFormat}`,
        accentHex: '#7C3AED',
      },
      {
        id: 'test',
        title: 'Test Split',
        fileName: `${safeBase}_test.${exportFormat}`,
        subtitle: `${dataset.splits.testCount} rows (${Math.round(dataset.splits.testRatio * 100)}%)`,
        description: 'Held-out benchmark rows for Evaluation & Comparison',
        rowCount: dataset.splits.testCount,
        badge: 'TEST',
        previewTab: 'test',
        exportUrl: `/api/datasets/${dataset.id}/export?split=test&format=${exportFormat}`,
        accentHex: '#D97706',
      },
      {
        id: 'raw',
        title: 'Raw Source',
        fileName: dataset.fileName,
        subtitle: `${dataset.rowCount} rows × ${dataset.columnCount} cols`,
        description: `Original uploaded ${dataset.fileType.toUpperCase()} artifact`,
        rowCount: dataset.rowCount,
        badge: dataset.fileType.toUpperCase(),
        previewTab: 'raw',
        exportUrl: `/api/datasets/${dataset.id}/export`,
        accentHex: '#475569',
      },
      {
        id: 'zip',
        title: 'Full ZIP Archive',
        fileName: `${safeBase}_bundle.zip`,
        subtitle: `${cleanedCount} rows + manifest.json`,
        description: 'Complete reproducible archive with all splits & schema report',
        rowCount: cleanedCount,
        badge: 'ZIP',
        exportUrl: `/api/datasets/${dataset.id}/export?split=all&format=zip`,
        accentHex: '#0D9488',
      },
    ];
  }

  return [
    {
      id: 'raw',
      title: 'Raw Dataset',
      fileName: dataset.fileName,
      subtitle: `${dataset.rowCount} rows × ${dataset.columnCount} cols`,
      description: `${dataset.fileType.toUpperCase()} · ${formatBytes(dataset.sizeBytes)}`,
      rowCount: dataset.rowCount,
      badge: dataset.fileType.toUpperCase(),
      previewTab: 'raw',
      exportUrl: `/api/datasets/${dataset.id}/export`,
      accentHex: '#2563EB',
    },
    {
      id: 'cleaned',
      title: 'Cleaned Preview',
      fileName: `${safeBase}_cleaned.${exportFormat}`,
      subtitle: `${dataset.missingCount} missing · ${dataset.duplicateCount} dupes`,
      description: 'Run "Preprocess & Split" to generate deduplicated Train/Val/Test splits',
      rowCount: dataset.rowCount,
      badge: dataset.status.toUpperCase(),
      previewTab: 'raw',
      exportUrl: `/api/datasets/${dataset.id}/export?split=cleaned&format=${exportFormat}`,
      accentHex: '#0EA5E9',
    },
    {
      id: 'zip',
      title: 'Export Source',
      fileName: `${safeBase}.${exportFormat}`,
      subtitle: `Format: ${exportFormat.toUpperCase()}`,
      description: `Download formatted ${exportFormat.toUpperCase()} export directly`,
      rowCount: dataset.rowCount,
      badge: exportFormat.toUpperCase(),
      exportUrl: `/api/datasets/${dataset.id}/export?split=cleaned&format=${exportFormat}`,
      accentHex: '#0D9488',
    },
  ];
}

export function buildAdapterBundleDocs(model: RegisteredModelRecord): AdapterBundleDoc[] {
  const cfg = model.trainingConfig;
  const accuracyText =
    model.latestAccuracy !== null ? `${(model.latestAccuracy * 100).toFixed(2)}%` : 'Pending eval';
  const f1Text =
    model.latestMacroF1 !== null ? `${(model.latestMacroF1 * 100).toFixed(2)}%` : 'Pending eval';
  const pplText = model.latestPerplexity !== null ? String(model.latestPerplexity) : 'Pending eval';

  return [
    {
      id: 'config',
      title: 'Adapter Config',
      fileName: 'adapter_config.json',
      subtitle: `r=${cfg.loraR} · α=${cfg.loraAlpha} · ${model.method.toUpperCase()}`,
      description: `Targets: ${cfg.targetModules.slice(0, 3).join(', ')}${cfg.targetModules.length > 3 ? '…' : ''}`,
      badge: 'JSON',
      accentHex: '#2563EB',
      previewCode: JSON.stringify(
        {
          peft_type: 'LORA',
          base_model_name_or_path: model.baseModelId,
          r: cfg.loraR,
          lora_alpha: cfg.loraAlpha,
          lora_dropout: cfg.loraDropout,
          target_modules: cfg.targetModules,
          bias: cfg.bias,
          quantization_bits: cfg.quantizationBits,
          quant_type: cfg.quantType,
        },
        null,
        2
      ),
    },
    {
      id: 'weights',
      title: 'Adapter Weights',
      fileName: 'adapter_model.safetensors',
      subtitle: `${formatBytes(model.adapterSizeBytes)} · ${formatNumber(model.trainableParams)} params`,
      description: `Trained on ${model.datasetName} (${cfg.epochs} epochs, lr=${cfg.learningRate})`,
      badge: 'WEIGHTS',
      accentHex: '#7C3AED',
      previewCode: [
        `# Artifact Binary Header`,
        `adapter_id: ${model.id}`,
        `version: ${model.version}`,
        `base_model: ${model.baseModelName} (${model.baseModelId})`,
        `trainable_params: ${model.trainableParams}`,
        `bundle_size: ${formatBytes(model.adapterSizeBytes)}`,
        `compute_dtype: ${cfg.computeDtype}`,
        `quantization: ${cfg.quantizationBits}-bit ${cfg.quantType.toUpperCase()}`,
      ].join('\n'),
    },
    {
      id: 'script',
      title: 'PEFT Script',
      fileName: 'train_peft.py',
      subtitle: `${cfg.optimizer.toUpperCase()} · ${cfg.scheduler} schedule`,
      description: `Reproducible Hugging Face PEFT + BitsAndBytes script`,
      badge: 'PYTHON',
      accentHex: '#059669',
      previewCode: [
        `from peft import LoraConfig, get_peft_model`,
        `from transformers import AutoModelForCausalLM`,
        ``,
        `lora_config = LoraConfig(`,
        `  r=${cfg.loraR},`,
        `  lora_alpha=${cfg.loraAlpha},`,
        `  lora_dropout=${cfg.loraDropout},`,
        `  target_modules=${JSON.stringify(cfg.targetModules)},`,
        `  task_type="CAUSAL_LM"`,
        `)`,
        `# Base: ${model.baseModelId} | Dataset: ${model.datasetName}`,
      ].join('\n'),
    },
    {
      id: 'card',
      title: 'Model Card',
      fileName: 'README.md',
      subtitle: `Acc: ${accuracyText} · F1: ${f1Text}`,
      description: `${model.baseModelName} · Click to open full Model Card & export`,
      badge: 'MD',
      accentHex: '#0EA5E9',
      previewCode: [
        `# ${model.name} (${model.version})`,
        ``,
        `- Base Model: ${model.baseModelName} (${model.baseModelId})`,
        `- Method: ${model.method.toUpperCase()}`,
        `- Dataset: ${model.datasetName}`,
        `- Trainable Params: ${formatNumber(model.trainableParams)}`,
        `- Latest Accuracy: ${accuracyText}`,
        `- Latest Macro F1: ${f1Text}`,
        `- Perplexity: ${pplText}`,
      ].join('\n'),
    },
    {
      id: 'eval',
      title: 'Eval Summary',
      fileName: 'eval_metrics.json',
      subtitle: `Status: ${model.status} · PPL ${pplText}`,
      description: `Deploy endpoint or launch Playground directly with reg:${model.id}`,
      badge: 'EVAL',
      accentHex: '#D97706',
      previewCode: JSON.stringify(
        {
          modelRef: `reg:${model.id}`,
          status: model.status,
          accuracy: model.latestAccuracy,
          macroF1: model.latestMacroF1,
          perplexity: model.latestPerplexity,
          inferenceEndpoint: `/api/inference/chat`,
        },
        null,
        2
      ),
    },
  ];
}
