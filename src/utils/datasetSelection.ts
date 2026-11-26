import type { DatasetRecord, TrainingMethod } from '../shared/types';

export const ACTIVE_DATASET_STORAGE_KEY = 'minimind_active_dataset_id';

export function readStoredActiveDatasetId(): string | null {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return null;
    const val = window.localStorage.getItem(ACTIVE_DATASET_STORAGE_KEY);
    return val && val.trim() ? val.trim() : null;
  } catch {
    return null;
  }
}

export function writeStoredActiveDatasetId(id: string | null | undefined): void {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return;
    if (id && id.trim()) {
      window.localStorage.setItem(ACTIVE_DATASET_STORAGE_KEY, id.trim());
    } else {
      window.localStorage.removeItem(ACTIVE_DATASET_STORAGE_KEY);
    }
  } catch {
    // Ignore storage quota or private browsing errors
  }
}

export function pickPreferredDatasetId(
  datasets: Pick<DatasetRecord, 'id' | 'status'>[],
  preferredId?: string | null
): string {
  if (datasets.length === 0) return '';
  if (preferredId && datasets.some((d) => d.id === preferredId)) {
    return preferredId;
  }
  const ready = datasets.find((d) => d.status === 'training_ready');
  if (ready) return ready.id;
  return datasets[0].id;
}

export function buildExperimentDefaultsFromDataset(
  dataset: Pick<DatasetRecord, 'name' | 'format' | 'rowCount' | 'columnCount'> | null | undefined,
  method: TrainingMethod
): { experimentName: string; description: string } {
  const methodLabel =
    method === 'qlora' ? 'QLoRA' : method === 'full' ? 'Full FT' : 'LoRA';
  if (!dataset) {
    return {
      experimentName: `Custom Dataset ${methodLabel} v1`,
      description: 'Supervised fine-tuning with low-rank adaptation on held-out split.',
    };
  }
  const cleanTitle =
    dataset.name.replace(/\s*\(\d+\)\s*$/, '').trim() || dataset.name;
  return {
    experimentName: `${cleanTitle} ${methodLabel} v1`,
    description: `Supervised ${methodLabel} fine-tuning on "${dataset.name}" (${dataset.rowCount} rows, ${dataset.columnCount} attributes, ${dataset.format} format).`,
  };
}

export interface PlaygroundDatasetPreset {
  label: string;
  prompt: string;
  choices: string;
}

const PROMPT_CANDIDATES = [
  'input',
  'prompt',
  'question',
  'query',
  'instruction',
  'text',
  'user',
  'context',
  'content',
  'message',
  'sentence',
  'review',
  'title',
];

const OUTPUT_CANDIDATES = [
  'output',
  'label',
  'class',
  'category',
  'intent',
  'answer',
  'response',
  'target',
  'completion',
  'sentiment',
  'status',
  'verdict',
];

function findColumnKey(row: Record<string, unknown>, candidates: string[]): string | null {
  const keys = Object.keys(row);
  for (const cand of candidates) {
    const match = keys.find((k) => k.trim().toLowerCase() === cand);
    if (match) return match;
  }
  return null;
}

export function buildPlaygroundPresetsFromDataset(
  dataset: Pick<DatasetRecord, 'name' | 'columnMapping'> | null | undefined,
  previewRows: Record<string, unknown>[]
): {
  suggestedChoices: string;
  quickPrompts: PlaygroundDatasetPreset[];
} {
  const fallbackProbe: PlaygroundDatasetPreset = {
    label: 'Anti-Hallucination Probe (JEV AI)',
    prompt:
      'Verify whether "JEV AI" is an approved intent class or registered adapter in this workspace.',
    choices: 'true, false, uncertain',
  };

  if (!dataset || previewRows.length === 0) {
    return {
      suggestedChoices: '',
      quickPrompts: [fallbackProbe],
    };
  }

  const firstRow = previewRows[0];
  const keys = Object.keys(firstRow);
  const outKey =
    (dataset.columnMapping?.output && keys.includes(dataset.columnMapping.output)
      ? dataset.columnMapping.output
      : null) ||
    findColumnKey(firstRow, OUTPUT_CANDIDATES) ||
    (keys.length >= 2 ? keys[keys.length - 1] : null);

  const inKey =
    (dataset.columnMapping?.input && keys.includes(dataset.columnMapping.input)
      ? dataset.columnMapping.input
      : null) ||
    (dataset.columnMapping?.instruction && keys.includes(dataset.columnMapping.instruction)
      ? dataset.columnMapping.instruction
      : null) ||
    findColumnKey(firstRow, PROMPT_CANDIDATES) ||
    keys[0] ||
    null;

  // Collect concise discrete labels if present
  const labelValues: string[] = [];
  const seenLabels = new Set<string>();
  if (outKey) {
    for (const row of previewRows) {
      const raw = row[outKey];
      if (raw === null || raw === undefined) continue;
      const val = String(raw).trim();
      if (!val || val.length > 42 || val.includes('\n')) continue;
      const norm = val.toLowerCase();
      if (!seenLabels.has(norm)) {
        seenLabels.add(norm);
        labelValues.push(val);
      }
    }
  }

  const discreteChoices =
    labelValues.length >= 2 && labelValues.length <= 16
      ? labelValues.join(', ')
      : '';

  const quickPrompts: PlaygroundDatasetPreset[] = [];
  const seenPrompts = new Set<string>();

  for (const row of previewRows) {
    if (quickPrompts.length >= 2) break;
    let promptText = '';
    if (Array.isArray(row.messages)) {
      const userMsg = (row.messages as Array<{ role?: string; content?: string }>).find(
        (m) => m?.role === 'user' && m?.content
      );
      promptText = userMsg?.content?.trim() ?? '';
    } else if (inKey && row[inKey] !== undefined && row[inKey] !== null) {
      promptText = String(row[inKey]).trim();
      if (
        dataset.columnMapping?.instruction &&
        inKey !== dataset.columnMapping.instruction &&
        row[dataset.columnMapping.instruction]
      ) {
        promptText = `${String(row[dataset.columnMapping.instruction]).trim()}\n${promptText}`;
      }
    } else {
      const pairs = Object.entries(row)
        .filter(([k, v]) => k !== outKey && v !== null && v !== undefined && String(v).trim() !== '')
        .slice(0, 3)
        .map(([k, v]) => `${k}: ${String(v).trim()}`);
      promptText = pairs.join(' | ');
    }

    if (!promptText || seenPrompts.has(promptText)) continue;
    seenPrompts.add(promptText);

    const shortLabel =
      promptText.length > 30 ? `${promptText.slice(0, 27).trim()}...` : promptText;
    quickPrompts.push({
      label: `Sample ${quickPrompts.length + 1}: ${shortLabel}`,
      prompt: promptText,
      choices: discreteChoices || 'supported, partially_supported, unsupported',
    });
  }

  quickPrompts.push(fallbackProbe);

  return {
    suggestedChoices: discreteChoices,
    quickPrompts,
  };
}
