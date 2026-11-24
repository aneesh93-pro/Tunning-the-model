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

export interface ColumnMapping {
  instruction?: string;
  input?: string;
  output?: string;
  messages?: string;
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

function stripBomAndNormalizeEol(content: string): string {
  return content.replace(/^[\uFEFF\uFFFE]/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

function detectCsvDelimiter(text: string): string {
  const firstLine = text.split('\n', 1)[0] ?? '';
  let commaCount = 0;
  let tabCount = 0;
  let semiCount = 0;
  let pipeCount = 0;
  let inQuotes = false;
  for (let i = 0; i < firstLine.length; i++) {
    const ch = firstLine[i];
    if (ch === '"') inQuotes = !inQuotes;
    else if (!inQuotes) {
      if (ch === ',') commaCount++;
      else if (ch === '\t') tabCount++;
      else if (ch === ';') semiCount++;
      else if (ch === '|') pipeCount++;
    }
  }
  if (tabCount > commaCount && tabCount >= semiCount && tabCount >= pipeCount) return '\t';
  if (semiCount > commaCount && semiCount > tabCount && semiCount >= pipeCount) return ';';
  if (pipeCount > commaCount && pipeCount > tabCount && pipeCount > semiCount) return '|';
  return ',';
}

/**
 * RFC-4180 compliant delimited parser handling quoted fields, escaped quotes, newlines inside quotes,
 * variable column widths, and CSV/TSV/semicolon/pipe databases.
 */
export function parseCSV(content: string, forcedDelimiter?: string): RawRow[] {
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentField = '';
  let inQuotes = false;

  const text = stripBomAndNormalizeEol(content);
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

  if (rows.length === 0) return [];
  if (rows.length === 1) {
    // Single-line delimited file without separate header row: preserve every cell attribute cleanly
    const cells = rows[0].map((c) => c.trim()).filter(Boolean);
    if (cells.length === 0) return [];
    if (cells.length === 1) return [{ text: cells[0] }];
    const obj: RawRow = {};
    cells.forEach((cell, idx) => {
      obj[`column_${idx + 1}`] = cell;
    });
    return [obj];
  }

  const rawHeaders = rows[0].map((h, idx) => h.trim() || `column_${idx + 1}`);
  const maxCols = rows.reduce((max, r) => Math.max(max, r.length), rawHeaders.length);
  const headers: string[] = [];
  const usedHeaders = new Set<string>();
  for (let idx = 0; idx < maxCols; idx++) {
    let base = rawHeaders[idx] || `column_${idx + 1}`;
    if (usedHeaders.has(base)) {
      let suffix = 2;
      while (usedHeaders.has(`${base}_${suffix}`)) suffix++;
      base = `${base}_${suffix}`;
    }
    usedHeaders.add(base);
    headers.push(base);
  }

  return rows.slice(1).map((cols) => {
    const obj: RawRow = {};
    headers.forEach((header, idx) => {
      const val = cols[idx] ?? '';
      obj[header] = val.trim();
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
  'users',
  'table',
] as const;

function zipColumnarRows(columns: unknown[], rowArrays: unknown[]): RawRow[] {
  const headers = columns.map((c, idx) => String(c ?? '').trim() || `column_${idx + 1}`);
  const rows: RawRow[] = [];
  for (const rawRow of rowArrays) {
    if (!Array.isArray(rawRow)) continue;
    const obj: RawRow = {};
    const limit = Math.max(headers.length, rawRow.length);
    for (let i = 0; i < limit; i++) {
      const key = headers[i] || `column_${i + 1}`;
      obj[key] = rawRow[i] ?? '';
    }
    rows.push(obj);
  }
  return rows;
}

function coerceJsonArrayToRows(arr: unknown[], columnHints?: unknown[]): RawRow[] {
  if (arr.length === 0) return [];

  // Array of arrays (e.g., JSON tabular matrix export)
  if (arr.every((item) => Array.isArray(item))) {
    if (columnHints && columnHints.length > 0) {
      return zipColumnarRows(columnHints, arr);
    }
    if (arr.length >= 2) {
      return zipColumnarRows(arr[0] as unknown[], arr.slice(1));
    }
    if (arr.length === 1) {
      const cells = (arr[0] as unknown[]).map((v) => (v === null || v === undefined ? '' : v));
      const obj: RawRow = {};
      cells.forEach((val, idx) => {
        obj[`column_${idx + 1}`] = val;
      });
      return Object.keys(obj).length > 0 ? [obj] : [];
    }
  }

  const rows: RawRow[] = [];
  for (const item of arr) {
    if (typeof item === 'object' && item !== null && !Array.isArray(item)) {
      rows.push(item as RawRow);
    } else if (typeof item === 'string' && item.trim().length > 0) {
      rows.push({ text: item.trim() });
    } else if (typeof item === 'number' || typeof item === 'boolean') {
      rows.push({ value: item });
    }
  }
  return rows;
}

function tryTransposeColumnOrientObject(record: Record<string, unknown>): RawRow[] | null {
  const entries = Object.entries(record);
  if (entries.length < 2) return null;
  if (!entries.every(([, v]) => v && typeof v === 'object' && !Array.isArray(v))) return null;

  const firstKeys = Object.keys(entries[0][1] as Record<string, unknown>);
  if (firstKeys.length < 1) return null;
  const allMatch = entries.every(([, v]) => {
    const keys = Object.keys(v as Record<string, unknown>);
    return keys.length === firstKeys.length && firstKeys.every((k) => k in (v as Record<string, unknown>));
  });
  if (!allMatch) return null;

  // Distinguish pandas orient='columns' (column -> { rowId: val }) from dict of row objects
  const looksLikeColumnMaps = firstKeys.every((k) => /^\d+$/.test(k));
  if (!looksLikeColumnMaps) return null;

  return firstKeys.map((rowId) => {
    const row: RawRow = {};
    for (const [colName, colObj] of entries) {
      row[colName] = (colObj as Record<string, unknown>)[rowId];
    }
    return row;
  });
}

function extractRowsFromParsedJson(parsed: unknown): RawRow[] | null {
  if (Array.isArray(parsed)) {
    return coerceJsonArrayToRows(parsed);
  }
  if (typeof parsed !== 'object' || parsed === null) {
    if (typeof parsed === 'string' && parsed.trim()) return [{ text: parsed.trim() }];
    if (typeof parsed === 'number' || typeof parsed === 'boolean') return [{ value: parsed }];
    return null;
  }

  const record = parsed as Record<string, unknown>;
  const columnHints = Array.isArray(record.columns)
    ? record.columns
    : Array.isArray(record.headers)
      ? record.headers
      : Array.isArray(record.fields)
        ? record.fields
        : Array.isArray(record.schema)
          ? record.schema
          : undefined;

  // 1. Check standard wrapper array keys
  for (const key of JSON_WRAPPER_KEYS) {
    const val = record[key];
    if (Array.isArray(val) && val.length > 0) {
      if (
        key === 'messages' &&
        val.every((m) => m && typeof m === 'object' && 'role' in (m as RawRow) && 'content' in (m as RawRow))
      ) {
        return [record];
      }
      const coerced = coerceJsonArrayToRows(val, columnHints);
      if (coerced.length > 0) return coerced;
    }
  }

  // 2. Search nested wrapper objects up to 2 levels for arrays of records
  for (const val of Object.values(record)) {
    if (Array.isArray(val) && val.length > 0) {
      const coerced = coerceJsonArrayToRows(val, columnHints);
      if (coerced.length > 0) return coerced;
    }
  }
  for (const val of Object.values(record)) {
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      const innerRecord = val as Record<string, unknown>;
      const innerHints = Array.isArray(innerRecord.columns) ? innerRecord.columns : columnHints;
      for (const innerVal of Object.values(innerRecord)) {
        if (Array.isArray(innerVal) && innerVal.length > 0) {
          const coerced = coerceJsonArrayToRows(innerVal, innerHints);
          if (coerced.length > 0) return coerced;
        }
      }
    }
  }

  // 3. Column-oriented database export `{ col1: {"0": ...}, col2: {"0": ...} }`
  const transposed = tryTransposeColumnOrientObject(record);
  if (transposed && transposed.length > 0) {
    return transposed;
  }

  // 4. Dictionary of keyed record objects e.g. { "0": {...}, "1": {...} }
  const values = Object.values(record);
  if (
    values.length >= 2 &&
    values.every((v) => v && typeof v === 'object' && !Array.isArray(v))
  ) {
    return values as RawRow[];
  }

  // 5. Single JSON object record (preserves all attributes)
  if (Object.keys(record).length > 0) {
    return [record];
  }

  return null;
}

function parseLooseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch (firstErr) {
    // Retry after stripping trailing commas before } or ] (common hand-edited JSON databases)
    const relaxed = text.replace(/,\s*([}\]])/g, '$1');
    if (relaxed !== text) {
      return JSON.parse(relaxed);
    }
    throw firstErr;
  }
}

function parseJsonLinesText(text: string): RawRow[] {
  const lines = stripBomAndNormalizeEol(text)
    .split('\n')
    .filter((l) => l.trim().length > 0);
  return lines.map((line, index) => {
    try {
      const obj = parseLooseJson(line);
      if (typeof obj === 'string' && obj.trim().length > 0) {
        return { text: obj.trim() };
      }
      if (typeof obj === 'number' || typeof obj === 'boolean') {
        return { value: obj };
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
  const normalized = stripBomAndNormalizeEol(content).trim();
  if (!normalized) return [];

  if (fileType === 'csv') {
    return parseCSV(normalized);
  }
  if (fileType === 'tsv') {
    return parseCSV(normalized, '\t');
  }

  if (fileType === 'json') {
    try {
      const parsed = parseLooseJson(normalized);
      const extracted = extractRowsFromParsedJson(parsed);
      if (extracted && extracted.length > 0) return extracted;
    } catch (jsonErr) {
      // Fall back to NDJSON/JSONL when multiple JSON lines were saved with a .json extension
      const nonEmptyLines = normalized.split('\n').filter((l) => l.trim().length > 0);
      if (nonEmptyLines.length > 1 && nonEmptyLines.every((l) => l.trim().startsWith('{'))) {
        return parseJsonLinesText(normalized);
      }
      throw new Error(`Invalid JSON file: ${(jsonErr as Error).message}`);
    }
    throw new Error('Could not find any dataset records in JSON file.');
  }

  if (fileType === 'jsonl' || fileType === 'ndjson') {
    return parseJsonLinesText(normalized);
  }

  // TXT: also transparently handle JSON or JSONL saved with a .txt extension
  if (normalized.startsWith('{') || normalized.startsWith('[')) {
    try {
      const parsed = parseLooseJson(normalized);
      const extracted = extractRowsFromParsedJson(parsed);
      if (extracted && extracted.length > 0) return extracted;
    } catch {
      const nonEmptyLines = normalized.split('\n').filter((l) => l.trim().length > 0);
      if (nonEmptyLines.length > 1 && nonEmptyLines.every((l) => l.trim().startsWith('{'))) {
        try {
          return parseJsonLinesText(normalized);
        } catch {
          // fall through to plain text blocks
        }
      }
    }
  }

  // Plain TXT: split by double newlines or non-empty lines into raw text records
  const blocks = normalized.includes('\n\n')
    ? normalized.split(/\n\n+/)
    : normalized.split(/\n/);
  return blocks
    .map((block) => block.trim())
    .filter(Boolean)
    .map((text) => ({ text }));
}

const PROMPT_KEY_CANDIDATES = [
  'instruction',
  'prompt',
  'question',
  'query',
  'task',
  'title',
  'user',
  'subject',
  'input_text',
] as const;

const INPUT_KEY_CANDIDATES = [
  'input',
  'context',
  'passage',
  'document',
  'user_input',
  'description',
  'details',
  'body',
] as const;

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
  'intent',
  'sentiment',
  'classification',
  'summary',
  'result',
  'expected',
  'assistant',
] as const;

function resolveKey(obj: Record<string, unknown>, target?: string): string | undefined {
  if (!target) return undefined;
  if (target in obj) return target;
  const lower = target.trim().toLowerCase();
  return Object.keys(obj).find((k) => k.trim().toLowerCase() === lower);
}

function pickFirstMatchingKey(obj: Record<string, unknown>, candidates: readonly string[]): string | undefined {
  const exact = candidates.find((k) => k in obj);
  if (exact) return exact;
  const keys = Object.keys(obj);
  for (const candidate of candidates) {
    const found = keys.find((k) => k.trim().toLowerCase() === candidate);
    if (found) return found;
  }
  return undefined;
}

export function detectDatasetFormat(
  rows: RawRow[],
  columnMapping?: ColumnMapping | null
): DatasetFormat {
  if (rows.length === 0) return 'raw_text';

  const scanLimit = Math.min(rows.length, 50);
  const unionObj: Record<string, unknown> = {};
  for (let i = 0; i < scanLimit; i++) {
    for (const [k, v] of Object.entries(rows[i])) {
      if (!(k in unionObj) || unionObj[k] === undefined || unionObj[k] === null || unionObj[k] === '') {
        unionObj[k] = v;
      }
    }
  }

  if (columnMapping?.messages && resolveKey(unionObj, columnMapping.messages)) {
    return 'conversational';
  }
  const msgKey = resolveKey(unionObj, 'messages') ?? resolveKey(unionObj, 'conversations');
  if (msgKey && rows.slice(0, scanLimit).some((r) => Array.isArray(r[msgKey]))) {
    return 'conversational';
  }

  if (
    (columnMapping?.instruction || columnMapping?.input) &&
    columnMapping?.output &&
    resolveKey(unionObj, columnMapping.output) &&
    (resolveKey(unionObj, columnMapping.instruction) || resolveKey(unionObj, columnMapping.input))
  ) {
    return 'instruction';
  }

  const hasPromptLike =
    Boolean(pickFirstMatchingKey(unionObj, PROMPT_KEY_CANDIDATES)) ||
    Boolean(pickFirstMatchingKey(unionObj, INPUT_KEY_CANDIDATES));
  const hasOutputLike = Boolean(pickFirstMatchingKey(unionObj, OUTPUT_KEY_CANDIDATES));
  if (hasPromptLike && hasOutputLike) {
    return 'instruction';
  }
  if (resolveKey(unionObj, 'instruction') && resolveKey(unionObj, 'output')) {
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

function formatAttributeValue(val: unknown): string {
  if (val === null || val === undefined) return '';
  if (typeof val === 'string') return val.trim();
  if (typeof val === 'number' || typeof val === 'boolean') return String(val);
  return JSON.stringify(val);
}

export function validateDatasetRows(
  rows: RawRow[],
  columnMapping?: ColumnMapping | null
): DatasetValidationReport {
  const format = detectDatasetFormat(rows, columnMapping);
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

  // Explicit mapping check only flags when user requested a non-existent attribute
  const colObj = Object.fromEntries(colNames.map((c) => [c, true]));
  if (format === 'instruction') {
    if (columnMapping?.instruction && !resolveKey(colObj, columnMapping.instruction)) {
      errors.push(`Mapped instruction attribute "${columnMapping.instruction}" was not found in dataset.`);
    }
    if (columnMapping?.input && !resolveKey(colObj, columnMapping.input)) {
      errors.push(`Mapped input attribute "${columnMapping.input}" was not found in dataset.`);
    }
    if (columnMapping?.output && !resolveKey(colObj, columnMapping.output)) {
      errors.push(`Mapped target attribute "${columnMapping.output}" was not found in dataset.`);
    }
  } else if (format === 'conversational') {
    const msgKey =
      resolveKey(colObj, columnMapping?.messages) ||
      resolveKey(colObj, 'messages') ||
      resolveKey(colObj, 'conversations');
    if (!msgKey) {
      errors.push('Conversational dataset requires a "messages" or "conversations" array attribute.');
    }
  } else if (colNames.length === 0) {
    errors.push('Dataset must contain at least one attribute/column.');
  }

  if (duplicateCount > 0) {
    warnings.push(`Detected ${duplicateCount} duplicate record(s) that could cause train/test leakage.`);
  }
  if (missingValuesTotal > 0) {
    warnings.push(`Detected ${missingValuesTotal} missing cell value(s). Cleaning will preserve usable attributes.`);
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
  mapping?: ColumnMapping | null
): NormalizedExample | null {
  if (format === 'conversational') {
    const msgKey =
      resolveKey(row, mapping?.messages) ||
      resolveKey(row, 'messages') ||
      resolveKey(row, 'conversations') ||
      'messages';
    const rawMsgs = row[msgKey];
    if (!Array.isArray(rawMsgs) || rawMsgs.length < 2) return null;
    const cleanMsgs = rawMsgs
      .filter((m): m is { role?: string; from?: string; content?: string; value?: string; text?: string } =>
        Boolean(
          m &&
            typeof m === 'object' &&
            (typeof (m as RawRow).content === 'string' ||
              typeof (m as RawRow).value === 'string' ||
              typeof (m as RawRow).text === 'string')
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
          content: String(m.content ?? m.value ?? m.text ?? '').trim(),
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
      resolveKey(row, mapping?.instruction) ||
      pickFirstMatchingKey(row, PROMPT_KEY_CANDIDATES);
    const inKey =
      resolveKey(row, mapping?.input) ||
      pickFirstMatchingKey(row, INPUT_KEY_CANDIDATES);
    const outKey =
      resolveKey(row, mapping?.output) ||
      pickFirstMatchingKey(row, OUTPUT_KEY_CANDIDATES);

    const instruction = instKey ? formatAttributeValue(row[instKey]) : '';
    const input = inKey && inKey !== instKey ? formatAttributeValue(row[inKey]) : '';
    const targetText = outKey ? formatAttributeValue(row[outKey]) : '';

    // Preserve extra non-mapped attributes from database tables if present
    const consumedKeys = new Set([instKey, inKey, outKey].filter(Boolean));
    const extraPairs = Object.entries(row)
      .filter(([k, v]) => !consumedKeys.has(k) && formatAttributeValue(v) !== '')
      .map(([k, v]) => `${k}: ${formatAttributeValue(v)}`);

    if ((instruction || input) && targetText) {
      const basePrompt =
        input && input !== instruction
          ? `${instruction}\nInput: ${input}`.trim()
          : instruction || input;
      const promptText =
        extraPairs.length > 0 ? `${basePrompt}\nAttributes: ${extraPairs.join(' | ')}` : basePrompt;

      return {
        id: `ex_${index}`,
        hash: stableHash(`${promptText}|||${targetText}`),
        format,
        promptText,
        targetText,
        raw: row,
      };
    }
    // If row lacked mapped target in sparse instruction file, fall through to multi-attribute fallback below
  }

  // raw_text & arbitrary multi-attribute database rows: preserve EVERY non-empty attribute
  const nonEmptyEntries = Object.entries(row)
    .map(([key, val]) => [key, formatAttributeValue(val)] as const)
    .filter(([, val]) => val !== '');

  if (nonEmptyEntries.length === 0) return null;

  if (nonEmptyEntries.length === 1) {
    const [, singleVal] = nonEmptyEntries[0];
    const splitIdx = Math.max(1, Math.floor(singleVal.length * 0.65));
    const promptText = singleVal.slice(0, splitIdx).trim();
    const targetText = singleVal.slice(splitIdx).trim() || singleVal;
    return {
      id: `ex_${index}`,
      hash: stableHash(JSON.stringify(row)),
      format: 'raw_text',
      promptText,
      targetText,
      raw: row,
    };
  }

  // Multi-attribute CSV / JSON row: preserve all attributes cleanly across context and target
  const contextEntries = nonEmptyEntries.slice(0, -1);
  const [targetKey, targetVal] = nonEmptyEntries[nonEmptyEntries.length - 1];
  const promptText = contextEntries.map(([k, v]) => `${k}: ${v}`).join(' | ');
  const targetText = `${targetKey}: ${targetVal}`;

  return {
    id: `ex_${index}`,
    hash: stableHash(`${promptText}|||${targetText}`),
    format: format === 'instruction' ? 'instruction' : 'raw_text',
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
    columnMapping?: ColumnMapping | null;
  } = {}
): ProcessedDatasetBundle {
  const trainRatio = options.trainRatio ?? 0.7;
  const valRatio = options.valRatio ?? 0.15;
  const testRatio = options.testRatio ?? 0.15;
  const deduplicate = options.deduplicate ?? true;
  const dropMissing = options.dropMissing ?? true;
  const seed = options.seed ?? 42;

  const format = detectDatasetFormat(rows, options.columnMapping);
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

  if (examples.length >= 3 && labelGroups.size > 1 && labelGroups.size <= 32 && examples.length >= labelGroups.size * 3) {
    for (const group of labelGroups.values()) {
      const shuffled = shuffle(group);
      const gn = shuffled.length;
      const nVal = gn >= 3 && valRatio > 0 ? Math.max(1, Math.round(gn * valRatio)) : 0;
      const nTest = gn >= 3 && testRatio > 0 ? Math.max(1, Math.round(gn * testRatio)) : 0;
      const nTrain = Math.max(1, gn - nVal - nTest);

      train.push(...shuffled.slice(0, nTrain));
      validation.push(...shuffled.slice(nTrain, nTrain + nVal));
      test.push(...shuffled.slice(nTrain + nVal));
    }
  } else {
    const shuffled = shuffle(examples);
    const n = shuffled.length;
    if (n >= 3) {
      let nVal = valRatio > 0 ? Math.max(1, Math.round(n * valRatio)) : 0;
      let nTest = testRatio > 0 ? Math.max(1, Math.round(n * testRatio)) : 0;
      if (nVal + nTest >= n) {
        nVal = valRatio > 0 ? 1 : 0;
        nTest = testRatio > 0 ? 1 : 0;
      }
      const nTrain = Math.max(1, n - nVal - nTest);
      train.push(...shuffled.slice(0, nTrain));
      validation.push(...shuffled.slice(nTrain, nTrain + nVal));
      test.push(...shuffled.slice(nTrain + nVal));
    } else if (n === 2) {
      train.push(shuffled[0]);
      validation.push(shuffled[1]);
      test.push(shuffled[1]);
    } else if (n === 1) {
      train.push(shuffled[0]);
      validation.push(shuffled[0]);
      test.push(shuffled[0]);
    }
  }

  // Guarantee non-empty validation/test when >= 3 examples exist
  if (examples.length >= 3) {
    if (validation.length === 0 && train.length > 1) {
      validation.push(train.pop()!);
    }
    if (test.length === 0 && train.length > 1) {
      test.push(train.pop()!);
    }
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
