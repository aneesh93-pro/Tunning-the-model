import type {
  DatasetColumnStats,
  DatasetFormat,
  DatasetSplits,
  DatasetValidationReport,
} from '../shared/types';

export interface RawRow {
  [key: string]: unknown;
}

export interface NormalizedExample {
  id: string;
  hash: string;
  format: DatasetFormat;
  promptText: string;
  targetText: string;
  messages?: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  raw: RawRow;
}

export interface ProcessedDatasetBundle {
  format: DatasetFormat;
  cleanedRows: RawRow[];
  examples: NormalizedExample[];
  train: NormalizedExample[];
  validation: NormalizedExample[];
  test: NormalizedExample[];
  splits: DatasetSplits;
  report: DatasetValidationReport;
}

/**
 * Deterministic 64-bit FNV-1a-style string hash for deduplication and leakage checks.
 */
export function stableHash(input: string): string {
  let h1 = 0xdeadbeef ^ input.length;
  let h2 = 0x41c6ce57 ^ input.length;
  for (let i = 0; i < input.length; i++) {
    const ch = input.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

/**
 * RFC-4180 compliant CSV parser handling quoted fields, escaped quotes, and newlines inside quotes.
 */
export function parseCSV(content: string): RawRow[] {
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentField = '';
  let inQuotes = false;

  const text = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];

    if (inQuotes) {
      if (char === '"' && next === '"') {
        currentField += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        currentField += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === ',') {
        currentRow.push(currentField);
        currentField = '';
      } else if (char === '\n') {
        currentRow.push(currentField);
        if (currentRow.some((cell) => cell.trim() !== '')) {
          rows.push(currentRow);
        }
        currentRow = [];
        currentField = '';
      } else {
        currentField += char;
      }
    }
  }

  if (currentField.length > 0 || currentRow.length > 0) {
    currentRow.push(currentField);
    if (currentRow.some((cell) => cell.trim() !== '')) {
      rows.push(currentRow);
    }
  }

  if (rows.length < 2) return [];
  const headers = rows[0].map((h) => h.trim());
  return rows.slice(1).map((cols) => {
    const obj: RawRow = {};
    headers.forEach((header, idx) => {
      const val = cols[idx] ?? '';
      obj[header || `column_${idx}`] = val.trim();
    });
    return obj;
  });
}

export function parseDatasetContent(
  content: string,
  fileType: 'csv' | 'json' | 'jsonl' | 'txt'
): RawRow[] {
  const trimmed = content.trim();
  if (!trimmed) return [];

  if (fileType === 'csv') {
    return parseCSV(trimmed);
  }

  if (fileType === 'json') {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is RawRow => typeof item === 'object' && item !== null);
    }
    if (typeof parsed === 'object' && parsed !== null && Array.isArray((parsed as RawRow).data)) {
      return (parsed as { data: RawRow[] }).data;
    }
    throw new Error('JSON dataset must be an array of objects or contain a "data" array.');
  }

  if (fileType === 'jsonl') {
    const lines = trimmed.split(/\r?\n/).filter((l) => l.trim().length > 0);
    return lines.map((line, index) => {
      try {
        const obj = JSON.parse(line);
        if (typeof obj !== 'object' || obj === null) {
          throw new Error(`Line ${index + 1} is not a JSON object.`);
        }
        return obj as RawRow;
      } catch (err) {
        throw new Error(`Invalid JSONL at line ${index + 1}: ${(err as Error).message}`);
      }
    });
  }

  // TXT: split by double newlines or non-empty lines into raw instruction/text records
  const blocks = trimmed.includes('\n\n')
    ? trimmed.split(/\n\n+/)
    : trimmed.split(/\r?\n/);
  return blocks
    .map((block) => block.trim())
    .filter(Boolean)
    .map((text) => ({ text }));
}

export function detectDatasetFormat(rows: RawRow[]): DatasetFormat {
  if (rows.length === 0) return 'raw_text';
  const sample = rows[0];
  if (Array.isArray(sample.messages)) {
    return 'conversational';
  }
  if (
    'instruction' in sample ||
    'output' in sample ||
    ('prompt' in sample && 'completion' in sample) ||
    ('input' in sample && 'label' in sample)
  ) {
    return 'instruction';
  }
  return 'raw_text';
}

function inferType(val: unknown): DatasetColumnStats['inferredType'] {
  if (val === null || val === undefined || val === '') return 'null';
  if (Array.isArray(val)) return 'array';
  if (typeof val === 'object') return 'object';
  if (typeof val === 'boolean') return 'boolean';
  if (typeof val === 'number') return 'number';
  if (typeof val === 'string' && !Number.isNaN(Number(val)) && val.trim() !== '') return 'number';
  return 'string';
}

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 3.8));
}

export function validateDatasetRows(
  rows: RawRow[],
  columnMapping?: {
    instruction?: string;
    input?: string;
    output?: string;
    messages?: string;
  } | null
): DatasetValidationReport {
  const format = detectDatasetFormat(rows);
  const warnings: string[] = [];
  const errors: string[] = [];

  if (rows.length === 0) {
    errors.push('Dataset contains zero rows.');
  } else if (rows.length < 6) {
    warnings.push('Dataset has fewer than 6 rows; train/val/test splits will be very small.');
  }

  const colNames = Array.from(
    new Set(rows.flatMap((row) => Object.keys(row)))
  );

  let missingValuesTotal = 0;
  const columns: DatasetColumnStats[] = colNames.map((name) => {
    let missingCount = 0;
    let totalLen = 0;
    const unique = new Set<string>();
    const typeCounts: Record<string, number> = {};
    const sampleValues: string[] = [];

    for (const row of rows) {
      const val = row[name];
      const t = inferType(val);
      typeCounts[t] = (typeCounts[t] ?? 0) + 1;

      if (val === null || val === undefined || (typeof val === 'string' && val.trim() === '')) {
        missingCount++;
      } else {
        const serialized = typeof val === 'string' ? val : JSON.stringify(val);
        totalLen += serialized.length;
        unique.add(serialized);
        if (sampleValues.length < 3 && !sampleValues.includes(serialized)) {
          sampleValues.push(serialized.slice(0, 80));
        }
      }
    }

    missingValuesTotal += missingCount;
    const dominantType = (Object.entries(typeCounts)
      .filter(([k]) => k !== 'null')
      .sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'string') as DatasetColumnStats['inferredType'];

    return {
      name,
      inferredType: dominantType,
      missingCount,
      uniqueCount: unique.size,
      avgLength: rows.length > missingCount ? Math.round(totalLen / (rows.length - missingCount)) : 0,
      sampleValues,
    };
  });

  // Duplicate detection by normalized JSON
  const seenHashes = new Set<string>();
  let duplicateCount = 0;
  let totalTokens = 0;
  let maxTokens = 0;

  for (const row of rows) {
    const serialized = JSON.stringify(row);
    const h = stableHash(serialized);
    if (seenHashes.has(h)) duplicateCount++;
    seenHashes.add(h);

    const tok = estimateTokens(serialized);
    totalTokens += tok;
    if (tok > maxTokens) maxTokens = tok;
  }

  // Format schema checks
  if (format === 'instruction') {
    const instKey = columnMapping?.instruction || (colNames.includes('instruction') ? 'instruction' : 'prompt');
    const outKey = columnMapping?.output || (colNames.includes('output') ? 'output' : colNames.includes('label') ? 'label' : 'completion');
    if (!colNames.includes(instKey) && !colNames.includes('input')) {
      errors.push(`Instruction dataset requires an "${instKey}" or "input" column.`);
    }
    if (!colNames.includes(outKey)) {
      errors.push(`Instruction dataset requires a target "${outKey}" column.`);
    }
  } else if (format === 'conversational') {
    const msgKey = columnMapping?.messages || 'messages';
    if (!colNames.includes(msgKey)) {
      errors.push(`Conversational dataset requires a "${msgKey}" array column.`);
    }
  } else if (!colNames.includes('text') && colNames.length === 0) {
    errors.push('Raw text dataset must contain at least one text column.');
  }

  if (duplicateCount > 0) {
    warnings.push(`Detected ${duplicateCount} duplicate record(s) that could cause train/test leakage.`);
  }
  if (missingValuesTotal > 0) {
    warnings.push(`Detected ${missingValuesTotal} missing cell value(s). Cleaning will drop incomplete targets.`);
  }

  return {
    valid: errors.length === 0,
    format,
    rowCount: rows.length,
    columnCount: colNames.length,
    missingValuesTotal,
    duplicateCount,
    leakageDetected: duplicateCount > 0,
    overlapCount: 0,
    avgTokenEstimate: rows.length > 0 ? Math.round(totalTokens / rows.length) : 0,
    maxTokenEstimate: maxTokens,
    warnings,
    errors,
    columns,
  };
}

export function normalizeRow(
  row: RawRow,
  index: number,
  format: DatasetFormat,
  mapping?: { instruction?: string; input?: string; output?: string; messages?: string } | null
): NormalizedExample | null {
  if (format === 'conversational') {
    const msgKey = mapping?.messages || 'messages';
    const rawMsgs = row[msgKey];
    if (!Array.isArray(rawMsgs) || rawMsgs.length < 2) return null;
    const cleanMsgs = rawMsgs
      .filter((m): m is { role: 'system' | 'user' | 'assistant'; content: string } =>
        Boolean(m && typeof m === 'object' && typeof (m as RawRow).content === 'string')
      )
      .map((m) => ({
        role: (m.role === 'system' || m.role === 'assistant' ? m.role : 'user') as
          | 'system'
          | 'user'
          | 'assistant',
        content: m.content.trim(),
      }))
      .filter((m) => m.content.length > 0);

    const lastAssistantIdx = [...cleanMsgs].reverse().findIndex((m) => m.role === 'assistant');
    if (lastAssistantIdx === -1) return null;
    const splitPoint = cleanMsgs.length - 1 - lastAssistantIdx;
    const promptMsgs = cleanMsgs.slice(0, splitPoint);
    const targetText = cleanMsgs[splitPoint].content;
    const promptText = promptMsgs.map((m) => `${m.role}: ${m.content}`).join('\n');
    if (!promptText || !targetText) return null;

    return {
      id: `ex_${index}`,
      hash: stableHash(`${promptText}|||${targetText}`),
      format,
      promptText,
      targetText,
      messages: cleanMsgs,
      raw: row,
    };
  }

  if (format === 'instruction') {
    const instKey = mapping?.instruction || ('instruction' in row ? 'instruction' : 'prompt');
    const inKey = mapping?.input || 'input';
    const outKey =
      mapping?.output ||
      ('output' in row ? 'output' : 'label' in row ? 'label' : 'completion');

    const instruction = String(row[instKey] ?? '').trim();
    const input = String(row[inKey] ?? '').trim();
    const targetText = String(row[outKey] ?? '').trim();

    if ((!instruction && !input) || !targetText) return null;
    const promptText = input ? `${instruction}\nInput: ${input}`.trim() : instruction;

    return {
      id: `ex_${index}`,
      hash: stableHash(`${promptText}|||${targetText}`),
      format,
      promptText,
      targetText,
      raw: row,
    };
  }

  // raw_text
  const firstCol = Object.keys(row)[0] || 'text';
  const text = String(row.text ?? row[firstCol] ?? '').trim();
  if (!text) return null;
  const splitIdx = Math.max(1, Math.floor(text.length * 0.65));
  const promptText = text.slice(0, splitIdx).trim();
  const targetText = text.slice(splitIdx).trim() || text;

  return {
    id: `ex_${index}`,
    hash: stableHash(text),
    format,
    promptText,
    targetText,
    raw: row,
  };
}

/**
 * Cleans, deduplicates, stratifies (when discrete labels exist), and splits dataset with zero leakage.
 */
export function preprocessAndSplitDataset(
  rows: RawRow[],
  options: {
    trainRatio?: number;
    valRatio?: number;
    testRatio?: number;
    deduplicate?: boolean;
    dropMissing?: boolean;
    seed?: number;
    columnMapping?: { instruction?: string; input?: string; output?: string; messages?: string } | null;
  } = {}
): ProcessedDatasetBundle {
  const trainRatio = options.trainRatio ?? 0.7;
  const valRatio = options.valRatio ?? 0.15;
  const testRatio = options.testRatio ?? 0.15;
  const deduplicate = options.deduplicate ?? true;
  const dropMissing = options.dropMissing ?? true;
  const seed = options.seed ?? 42;

  const format = detectDatasetFormat(rows);
  const seen = new Set<string>();
  const examples: NormalizedExample[] = [];
  const cleanedRows: RawRow[] = [];

  rows.forEach((row, idx) => {
    const normalized = normalizeRow(row, idx, format, options.columnMapping);
    if (!normalized) {
      if (!dropMissing) {
        cleanedRows.push(row);
      }
      return;
    }
    if (deduplicate && seen.has(normalized.hash)) {
      return;
    }
    seen.add(normalized.hash);
    examples.push(normalized);
    cleanedRows.push(row);
  });

  // Deterministic PRNG shuffle (Mulberry32)
  let s = seed >>> 0;
  const rand = () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // Group by target class for stratified splitting if <= 32 classes
  const labelGroups = new Map<string, NormalizedExample[]>();
  for (const ex of examples) {
    const group = labelGroups.get(ex.targetText) ?? [];
    group.push(ex);
    labelGroups.set(ex.targetText, group);
  }

  const train: NormalizedExample[] = [];
  const validation: NormalizedExample[] = [];
  const test: NormalizedExample[] = [];

  const shuffle = <T,>(arr: T[]): T[] => {
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  };

  if (labelGroups.size > 1 && labelGroups.size <= 32 && examples.length >= labelGroups.size * 3) {
    for (const group of labelGroups.values()) {
      const shuffled = shuffle(group);
      const n = shuffled.length;
      const nVal = n >= 3 ? Math.max(1, Math.round(n * valRatio)) : 0;
      const nTest = n >= 3 ? Math.max(1, Math.round(n * testRatio)) : 0;
      const nTrain = Math.max(1, n - nVal - nTest);

      train.push(...shuffled.slice(0, nTrain));
      validation.push(...shuffled.slice(nTrain, nTrain + nVal));
      test.push(...shuffled.slice(nTrain + nVal));
    }
  } else {
    const shuffled = shuffle(examples);
    const n = shuffled.length;
    const nTrain = Math.max(1, Math.round(n * trainRatio));
    const nVal = Math.max(n >= 3 ? 1 : 0, Math.round(n * valRatio));
    train.push(...shuffled.slice(0, nTrain));
    validation.push(...shuffled.slice(nTrain, nTrain + nVal));
    test.push(...shuffled.slice(nTrain + nVal));
  }

  // Guarantee non-empty validation/test when >= 3 examples exist
  if (validation.length === 0 && train.length > 2) {
    validation.push(train.pop()!);
  }
  if (test.length === 0 && train.length > 2) {
    test.push(train.pop()!);
  }

  // Strict leakage verification across splits
  const trainHashes = new Set(train.map((e) => e.hash));
  const valHashes = new Set(validation.map((e) => e.hash));
  let overlapCount = 0;
  for (const ex of validation) {
    if (trainHashes.has(ex.hash)) overlapCount++;
  }
  for (const ex of test) {
    if (trainHashes.has(ex.hash) || valHashes.has(ex.hash)) overlapCount++;
  }

  const report = validateDatasetRows(cleanedRows, options.columnMapping);
  report.overlapCount = overlapCount;
  report.leakageDetected = overlapCount > 0;

  return {
    format,
    cleanedRows,
    examples,
    train: shuffle(train),
    validation: shuffle(validation),
    test: shuffle(test),
    splits: {
      trainCount: train.length,
      valCount: validation.length,
      testCount: test.length,
      trainRatio,
      valRatio,
      testRatio,
      leakageFree: overlapCount === 0,
    },
    report,
  };
}

function escapeCsvCell(value: unknown): string {
  const raw =
    value === null || value === undefined
      ? ''
      : typeof value === 'object'
        ? JSON.stringify(value)
        : String(value);
  if (/[",\n\r]/.test(raw)) {
    return `"${raw.replace(/"/g, '""')}"`;
  }
  return raw;
}

export function rowsToJsonl(rows: RawRow[]): string {
  return rows.map((r) => JSON.stringify(r)).join('\n');
}

export function rowsToCsv(rows: RawRow[]): string {
  if (rows.length === 0) return '';
  const headers: string[] = [];
  const seen = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        headers.push(key);
      }
    }
  }
  const lines = [headers.map(escapeCsvCell).join(',')];
  for (const row of rows) {
    lines.push(headers.map((h) => escapeCsvCell(row[h])).join(','));
  }
  return lines.join('\n');
}

