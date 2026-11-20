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

function detectCsvDelimiter(text: string): string {
  const firstLine = text.split('\n', 1)[0] ?? '';
  let commaCount = 0;
  let tabCount = 0;
  let semiCount = 0;
  let inQuotes = false;
  for (let i = 0; i < firstLine.length; i++) {
    const ch = firstLine[i];
    if (ch === '"') inQuotes = !inQuotes;
    else if (!inQuotes) {
      if (ch === ',') commaCount++;
      else if (ch === '\t') tabCount++;
      else if (ch === ';') semiCount++;
    }
  }
  if (tabCount > commaCount && tabCount >= semiCount) return '\t';
  if (semiCount > commaCount && semiCount > tabCount) return ';';
  return ',';
}

/**
 * RFC-4180 compliant delimited parser handling quoted fields, escaped quotes, newlines inside quotes, and CSV/TSV/semicolon files.
 */
export function parseCSV(content: string, forcedDelimiter?: string): RawRow[] {
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentField = '';
  let inQuotes = false;

  const text = content.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const delimiter = forcedDelimiter || detectCsvDelimiter(text);

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
      } else if (char === delimiter) {
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

const JSON_WRAPPER_KEYS = [
  'data',
  'rows',
  'items',
  'records',
  'examples',
  'samples',
  'dataset',
  'conversations',
  'train',
  'validation',
  'test',
  'messages',
  'entries',
  'results',
] as const;

function coerceJsonArrayToRows(arr: unknown[]): RawRow[] {
  const rows: RawRow[] = [];
  for (const item of arr) {
    if (typeof item === 'object' && item !== null && !Array.isArray(item)) {
      rows.push(item as RawRow);
    } else if (typeof item === 'string' && item.trim().length > 0) {
      rows.push({ text: item.trim() });
    }
  }
  return rows;
}

function extractRowsFromParsedJson(parsed: unknown): RawRow[] | null {
  if (Array.isArray(parsed)) {
    return coerceJsonArrayToRows(parsed);
  }
  if (typeof parsed !== 'object' || parsed === null) {
    return null;
  }

  const record = parsed as Record<string, unknown>;

  // 1. Check standard wrapper array keys
  for (const key of JSON_WRAPPER_KEYS) {
    const val = record[key];
    if (Array.isArray(val) && val.length > 0) {
      // Special case: top-level `{ messages: [ {role, content}, ... ] }` single conversation object
      if (
        key === 'messages' &&
        val.every((m) => m && typeof m === 'object' && 'role' in (m as RawRow) && 'content' in (m as RawRow))
      ) {
        return [record];
      }
      const coerced = coerceJsonArrayToRows(val);
      if (coerced.length > 0) return coerced;
    }
  }

  // 2. Search nested wrapper objects up to 2 levels for arrays of records
  for (const val of Object.values(record)) {
    if (Array.isArray(val) && val.length > 0) {
      const coerced = coerceJsonArrayToRows(val);
      if (coerced.length > 0) return coerced;
    }
  }
  for (const val of Object.values(record)) {
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      for (const innerVal of Object.values(val as Record<string, unknown>)) {
        if (Array.isArray(innerVal) && innerVal.length > 0) {
          const coerced = coerceJsonArrayToRows(innerVal);
          if (coerced.length > 0) return coerced;
        }
      }
    }
  }

  // 3. Dictionary of keyed record objects e.g. { "0": {...}, "1": {...} }
  const values = Object.values(record);
  if (
    values.length >= 2 &&
    values.every((v) => v && typeof v === 'object' && !Array.isArray(v))
  ) {
    return values as RawRow[];
  }

  // 4. Single JSON object record (e.g. single instruction/prompt/text item)
  if (Object.keys(record).length > 0) {
    return [record];
  }

  return null;
}

function parseJsonLinesText(text: string): RawRow[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  return lines.map((line, index) => {
    try {
      const obj = JSON.parse(line);
      if (typeof obj === 'string' && obj.trim().length > 0) {
        return { text: obj.trim() };
      }
      if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) {
        throw new Error(`Line ${index + 1} is not a JSON object.`);
      }
      return obj as RawRow;
    } catch (err) {
      throw new Error(`Invalid JSONL at line ${index + 1}: ${(err as Error).message}`);
    }
  });
}

export function parseDatasetContent(
  content: string,
  fileType: 'csv' | 'json' | 'jsonl' | 'txt' | 'tsv' | 'ndjson'
): RawRow[] {
  const trimmed = content.replace(/^\uFEFF/, '').trim();
  if (!trimmed) return [];

  if (fileType === 'csv') {
    return parseCSV(trimmed);
  }
  if (fileType === 'tsv') {
    return parseCSV(trimmed, '\t');
  }

  if (fileType === 'json') {
    try {
      const parsed = JSON.parse(trimmed);
      const extracted = extractRowsFromParsedJson(parsed);
      if (extracted) return extracted;
    } catch (jsonErr) {
      // Fall back to NDJSON/JSONL when multiple JSON lines were saved with a .json extension
      const nonEmptyLines = trimmed.split(/\r?\n/).filter((l) => l.trim().length > 0);
      if (nonEmptyLines.length > 1 && nonEmptyLines.every((l) => l.trim().startsWith('{'))) {
        return parseJsonLinesText(trimmed);
      }
      throw new Error(`Invalid JSON file: ${(jsonErr as Error).message}`);
    }
    throw new Error('Could not find any dataset records in JSON file.');
  }

  if (fileType === 'jsonl' || fileType === 'ndjson') {
    return parseJsonLinesText(trimmed);
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

const PROMPT_KEY_CANDIDATES = ['instruction', 'prompt', 'question', 'query', 'task', 'title'] as const;
const INPUT_KEY_CANDIDATES = ['input', 'context', 'passage', 'document', 'user_input'] as const;
const OUTPUT_KEY_CANDIDATES = [
  'output',
  'response',
  'answer',
  'completion',
  'label',
  'target',
  'reply',
  'category',
  'class',
] as const;

function pickFirstMatchingKey(obj: Record<string, unknown>, candidates: readonly string[]): string | undefined {
  return candidates.find((k) => k in obj);
}

export function detectDatasetFormat(rows: RawRow[]): DatasetFormat {
  if (rows.length === 0) return 'raw_text';
  const sample = rows[0];
  if (Array.isArray(sample.messages) || Array.isArray(sample.conversations)) {
    return 'conversational';
  }
  const hasPromptLike =
    Boolean(pickFirstMatchingKey(sample, PROMPT_KEY_CANDIDATES)) ||
    Boolean(pickFirstMatchingKey(sample, INPUT_KEY_CANDIDATES));
  const hasOutputLike = Boolean(pickFirstMatchingKey(sample, OUTPUT_KEY_CANDIDATES));
  if (hasPromptLike && hasOutputLike) {
    return 'instruction';
  }
  if ('instruction' in sample || 'output' in sample) {
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
  const colObj = Object.fromEntries(colNames.map((c) => [c, true]));
  if (format === 'instruction') {
    const instKey =
      columnMapping?.instruction ||
      pickFirstMatchingKey(colObj, PROMPT_KEY_CANDIDATES) ||
      'instruction';
    const inKey =
      columnMapping?.input ||
      pickFirstMatchingKey(colObj, INPUT_KEY_CANDIDATES) ||
      'input';
    const outKey =
      columnMapping?.output ||
      pickFirstMatchingKey(colObj, OUTPUT_KEY_CANDIDATES) ||
      'output';
    if (!colNames.includes(instKey) && !colNames.includes(inKey)) {
      errors.push(`Instruction dataset requires an "${instKey}" or "${inKey}" column.`);
    }
    if (!colNames.includes(outKey)) {
      errors.push(`Instruction dataset requires a target "${outKey}" column.`);
    }
  } else if (format === 'conversational') {
    const msgKey =
      columnMapping?.messages ||
      (colNames.includes('messages') ? 'messages' : 'conversations');
    if (!colNames.includes(msgKey)) {
      errors.push(`Conversational dataset requires a "${msgKey}" array column.`);
    }
  } else if (colNames.length === 0) {
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
    const msgKey = mapping?.messages || ('messages' in row ? 'messages' : 'conversations');
    const rawMsgs = row[msgKey];
    if (!Array.isArray(rawMsgs) || rawMsgs.length < 2) return null;
    const cleanMsgs = rawMsgs
      .filter((m): m is { role?: string; from?: string; content?: string; value?: string } =>
        Boolean(
          m &&
            typeof m === 'object' &&
            (typeof (m as RawRow).content === 'string' || typeof (m as RawRow).value === 'string')
        )
      )
      .map((m) => {
        const rawRole = String(m.role ?? m.from ?? 'user').toLowerCase();
        const role: 'system' | 'user' | 'assistant' =
          rawRole === 'system'
            ? 'system'
            : rawRole === 'assistant' || rawRole === 'gpt' || rawRole === 'model'
              ? 'assistant'
              : 'user';
        return {
          role,
          content: String(m.content ?? m.value ?? '').trim(),
        };
      })
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
    const instKey =
      mapping?.instruction || pickFirstMatchingKey(row, PROMPT_KEY_CANDIDATES) || 'instruction';
    const inKey =
      mapping?.input || pickFirstMatchingKey(row, INPUT_KEY_CANDIDATES) || 'input';
    const outKey =
      mapping?.output || pickFirstMatchingKey(row, OUTPUT_KEY_CANDIDATES) || 'output';

    const instruction = String(row[instKey] ?? '').trim();
    const input = String(row[inKey] ?? '').trim();
    const targetText = String(row[outKey] ?? '').trim();

    if ((!instruction && !input) || !targetText) return null;
    const promptText =
      input && input !== instruction
        ? `${instruction}\nInput: ${input}`.trim()
        : instruction || input;

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
  const textCol =
    (typeof row.text === 'string' && 'text') ||
    (typeof row.content === 'string' && 'content') ||
    (typeof row.body === 'string' && 'body') ||
    (typeof row.document === 'string' && 'document') ||
    (typeof row.sentence === 'string' && 'sentence') ||
    Object.keys(row)[0] ||
    'text';
  const rawVal = row[textCol];
  const text =
    typeof rawVal === 'string'
      ? rawVal.trim()
      : rawVal !== null && rawVal !== undefined
        ? JSON.stringify(rawVal)
        : Object.values(row)
            .filter((v) => v !== null && v !== undefined && String(v).trim() !== '')
            .map((v) => (typeof v === 'string' ? v.trim() : JSON.stringify(v)))
            .join(' ');
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

