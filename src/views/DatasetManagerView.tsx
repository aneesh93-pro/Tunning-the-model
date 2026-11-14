import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  SlidersHorizontal,
  Trash2,
  Upload,
} from 'lucide-react';
import { api } from '../lib/api';
import { formatBytes } from '../lib/utils';
import type { DatasetRecord } from '../shared/types';
import type { NavSection } from './OverviewView';

interface DatasetManagerViewProps {
  onSelectForTraining: (datasetId: string) => void;
  onNavigate: (section: NavSection) => void;
}

const PIPELINE_STAGES = [
  'UPLOAD',
  'VALIDATE',
  'EXPLORE',
  'CLEAN',
  'FORMAT',
  'TOKENIZE',
  'TRAINING READY',
] as const;

export function DatasetManagerView({
  onSelectForTraining,
  onNavigate,
}: DatasetManagerViewProps) {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Upload form state
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteName, setPasteName] = useState('');
  const [pasteFileName, setPasteFileName] = useState('custom_dataset.jsonl');
  const [pasteContent, setPasteContent] = useState(
    '{"instruction":"Classify intent","input":"Reset my online banking password","output":"account_security"}\n{"instruction":"Classify intent","input":"Where is my debit card delivery","output":"card_delivery"}'
  );

  // Preprocess config state
  const [trainRatio, setTrainRatio] = useState(0.7);
  const [valRatio, setValRatio] = useState(0.15);
  const [testRatio, setTestRatio] = useState(0.15);
  const [deduplicate, setDeduplicate] = useState(true);
  const [dropMissing, setDropMissing] = useState(true);
  const [seed, setSeed] = useState(42);
  const [mapInstruction, setMapInstruction] = useState('');
  const [mapInput, setMapInput] = useState('');
  const [mapOutput, setMapOutput] = useState('');

  const loadDatasets = async (preserveId?: string) => {
    setLoading(true);
    try {
      const list = await api.getDatasets();
      setDatasets(list);
      const target = preserveId || selectedId || list[0]?.id || null;
      if (target && list.some((d) => d.id === target)) {
        setSelectedId(target);
        const detail = await api.getDatasetDetail(target);
        setPreviewRows(detail.previewRows);
        setMapInstruction(detail.dataset.columnMapping?.instruction || '');
        setMapInput(detail.dataset.columnMapping?.input || '');
        setMapOutput(detail.dataset.columnMapping?.output || '');
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadDatasets();
  }, []);

  const handleSelectDataset = async (id: string) => {
    setSelectedId(id);
    setError(null);
    setNotice(null);
    try {
      const detail = await api.getDatasetDetail(id);
      setPreviewRows(detail.previewRows);
      setMapInstruction(detail.dataset.columnMapping?.instruction || '');
      setMapInput(detail.dataset.columnMapping?.input || '');
      setMapOutput(detail.dataset.columnMapping?.output || '');
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const created = await api.uploadDatasetFile(file);
      setNotice(`Uploaded "${created.fileName}" (${created.rowCount} rows).`);
      await loadDatasets(created.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
      e.target.value = '';
    }
  };

  const handlePasteUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const created = await api.uploadDatasetText({
        name: pasteName.trim() || pasteFileName.replace(/\.[^.]+$/, ''),
        fileName: pasteFileName.trim(),
        content: pasteContent,
      });
      setShowPasteModal(false);
      setPasteName('');
      setNotice(`Created dataset "${created.name}" (${created.rowCount} rows).`);
      await loadDatasets(created.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const selected = datasets.find((d) => d.id === selectedId) ?? null;

  const handleValidate = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      const mapping =
        mapInstruction || mapInput || mapOutput
          ? {
              ...(mapInstruction ? { instruction: mapInstruction } : {}),
              ...(mapInput ? { input: mapInput } : {}),
              ...(mapOutput ? { output: mapOutput } : {}),
            }
          : undefined;
      await api.validateDataset(selected.id, mapping);
      setNotice('Schema validation completed.');
      await loadDatasets(selected.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const handlePreprocess = async () => {
    if (!selected) return;
    const sum = Number((trainRatio + valRatio + testRatio).toFixed(2));
    if (Math.abs(sum - 1) > 0.02) {
      setError(`Split ratios must sum to 1.00 (currently ${sum.toFixed(2)}).`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const mapping =
        mapInstruction || mapInput || mapOutput
          ? {
              ...(mapInstruction ? { instruction: mapInstruction } : {}),
              ...(mapInput ? { input: mapInput } : {}),
              ...(mapOutput ? { output: mapOutput } : {}),
            }
          : undefined;
      const updated = await api.preprocessDataset(selected.id, {
        trainRatio,
        valRatio,
        testRatio,
        deduplicate,
        dropMissing,
        seed,
        columnMapping: mapping,
      });
      setNotice(
        `Dataset training-ready: ${updated.splits?.trainCount} train / ${updated.splits?.valCount} val / ${updated.splits?.testCount} test (0 leakage overlap).`
      );
      await loadDatasets(selected.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (id: string) => {
    setBusy(true);
    try {
      await api.deleteDataset(id);
      setNotice('Dataset deleted.');
      await loadDatasets();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const activeStageIdx = !selected
    ? 0
    : selected.status === 'training_ready'
      ? 6
      : selected.status === 'validated'
        ? 3
        : 1;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Dataset Manager
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Upload CSV, JSON, JSONL, or TXT files, inspect statistics, prevent leakage, and build train/val/test splits.
          </p>
        </div>
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={() => setShowPasteModal(true)}
            className="px-3.5 py-2 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-white/[0.04] whitespace-nowrap"
          >
            Paste / Create JSONL
          </button>
          <label className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 cursor-pointer whitespace-nowrap">
            <Upload className="w-3.5 h-3.5" />
            Upload File
            <input
              type="file"
              accept=".csv,.json,.jsonl,.txt"
              onChange={handleFileUpload}
              className="sr-only"
              aria-label="Upload dataset file"
            />
          </label>
        </div>
      </div>

      {/* Preprocessing Workflow Stepper */}
      <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4 overflow-x-auto">
        <div className="flex items-center justify-between min-w-[680px] gap-2">
          {PIPELINE_STAGES.map((stage, idx) => {
            const done = idx <= activeStageIdx;
            return (
              <div key={stage} className="flex items-center gap-2 flex-1">
                <div
                  className={`flex items-center gap-2 text-[11px] font-mono font-medium whitespace-nowrap ${
                    done
                      ? 'text-blue-600 dark:text-blue-400'
                      : 'text-slate-400 dark:text-slate-500'
                  }`}
                >
                  <span
                    className={`w-5 h-5 rounded-md flex items-center justify-center text-[10px] border ${
                      done
                        ? 'border-blue-500/40 bg-blue-500/10 text-blue-600 dark:text-blue-400'
                        : 'border-slate-200 dark:border-white/[0.08]'
                    }`}
                  >
                    {idx + 1}
                  </span>
                  {stage}
                </div>
                {idx < PIPELINE_STAGES.length - 1 && (
                  <div
                    className={`h-px flex-1 ${
                      idx < activeStageIdx
                        ? 'bg-blue-500/50'
                        : 'bg-slate-200 dark:bg-white/[0.08]'
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
      {notice && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-xs text-emerald-700 dark:text-emerald-400 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{notice}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Dataset List */}
        <div className="lg:col-span-4 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden self-start">
          <div className="px-4 py-3 border-b border-slate-200 dark:border-white/[0.08] flex items-center justify-between">
            <h2 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
              Datasets ({datasets.length})
            </h2>
          </div>
          {loading ? (
            <div className="p-6 text-xs text-slate-500">Loading datasets...</div>
          ) : datasets.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-500">
              No datasets uploaded yet.
            </div>
          ) : (
            <ul className="divide-y divide-slate-200 dark:divide-white/[0.06]">
              {datasets.map((ds) => {
                const isActive = ds.id === selectedId;
                return (
                  <li key={ds.id}>
                    <button
                      type="button"
                      onClick={() => handleSelectDataset(ds.id)}
                      className={`w-full text-left px-4 py-3.5 transition-colors ${
                        isActive
                          ? 'bg-blue-500/[0.07] dark:bg-blue-500/[0.1]'
                          : 'hover:bg-slate-50 dark:hover:bg-white/[0.02]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                          {ds.name}
                        </span>
                        <span
                          className={`text-[11px] font-mono shrink-0 ${
                            ds.status === 'training_ready'
                              ? 'text-emerald-600 dark:text-emerald-400'
                              : 'text-amber-600 dark:text-amber-400'
                          }`}
                        >
                          {ds.status === 'training_ready' ? 'ready' : ds.status}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
                        {ds.fileType.toUpperCase()} · {ds.format} · {ds.rowCount} rows ·{' '}
                        {formatBytes(ds.sizeBytes)}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* Selected Dataset Workspace */}
        <div className="lg:col-span-8 space-y-6">
          {!selected ? (
            <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-10 text-center">
              <FileSpreadsheet className="w-8 h-8 text-slate-400 mx-auto mb-2" />
              <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                Select or upload a dataset to inspect and preprocess
              </p>
            </div>
          ) : (
            <>
              {/* Summary & Actions */}
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div>
                    <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
                      {selected.name}
                    </h2>
                    <p className="text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums mt-0.5">
                      {selected.fileName} · {selected.rowCount} rows × {selected.columnCount} cols ·{' '}
                      {selected.missingCount} missing · {selected.duplicateCount} duplicates
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 shrink-0">
                    <a
                      href={`/api/datasets/${selected.id}/export`}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Export
                    </a>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => handleDelete(selected.id)}
                      aria-label={`Delete ${selected.name}`}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-red-500/30 text-red-600 dark:text-red-400 hover:bg-red-500/10"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </button>
                    {selected.status === 'training_ready' && (
                      <button
                        type="button"
                        onClick={() => {
                          onSelectForTraining(selected.id);
                          onNavigate('studio');
                        }}
                        className="px-3.5 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500"
                      >
                        Use in Fine-Tuning Studio
                      </button>
                    )}
                  </div>
                </div>

                {/* Column Statistics */}
                {selected.validationReport && (
                  <div className="mt-5 overflow-x-auto border-t border-slate-200 dark:border-white/[0.06] pt-4">
                    <h3 className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2">
                      Column Schema &amp; Statistics
                    </h3>
                    <table className="w-full text-left text-xs">
                      <thead className="text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-white/[0.06]">
                        <tr>
                          <th className="py-2 pr-3 font-medium">Column</th>
                          <th className="py-2 pr-3 font-medium">Type</th>
                          <th className="py-2 pr-3 font-medium text-right">Unique</th>
                          <th className="py-2 pr-3 font-medium text-right">Missing</th>
                          <th className="py-2 pr-3 font-medium text-right">Avg Chars</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06] font-mono tabular-nums">
                        {selected.validationReport.columns.map((col) => (
                          <tr key={col.name}>
                            <td className="py-2 pr-3 text-slate-900 dark:text-slate-100">{col.name}</td>
                            <td className="py-2 pr-3 text-slate-500 dark:text-slate-400">{col.inferredType}</td>
                            <td className="py-2 pr-3 text-right">{col.uniqueCount}</td>
                            <td className="py-2 pr-3 text-right">{col.missingCount}</td>
                            <td className="py-2 pr-3 text-right">{col.avgLength}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Preprocessing & Leakage-Free Split Controls */}
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal className="w-4 h-4 text-blue-500" />
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      Clean, Format &amp; Split Configuration
                    </h3>
                  </div>
                  {selected.splits && (
                    <span className="text-xs font-mono tabular-nums text-emerald-600 dark:text-emerald-400">
                      {selected.splits.trainCount} train / {selected.splits.valCount} val /{' '}
                      {selected.splits.testCount} test · Leakage-Free: {selected.splits.leakageFree ? 'Verified' : 'Warning'}
                    </span>
                  )}
                </div>

                {selected.format === 'instruction' && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                        Instruction Column
                      </label>
                      <input
                        type="text"
                        value={mapInstruction}
                        onChange={(e) => setMapInstruction(e.target.value)}
                        placeholder="instruction"
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                        Input Column (optional)
                      </label>
                      <input
                        type="text"
                        value={mapInput}
                        onChange={(e) => setMapInput(e.target.value)}
                        placeholder="input"
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                        Target Output Column
                      </label>
                      <input
                        type="text"
                        value={mapOutput}
                        onChange={(e) => setMapOutput(e.target.value)}
                        placeholder="output"
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                      />
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Train Ratio
                    </label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.5"
                      max="0.9"
                      value={trainRatio}
                      onChange={(e) => setTrainRatio(Number(e.target.value))}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Validation Ratio
                    </label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.05"
                      max="0.4"
                      value={valRatio}
                      onChange={(e) => setValRatio(Number(e.target.value))}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Test Ratio
                    </label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.05"
                      max="0.4"
                      value={testRatio}
                      onChange={(e) => setTestRatio(Number(e.target.value))}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Split Seed
                    </label>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
                  <div className="flex items-center gap-5 text-xs">
                    <label className="inline-flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={deduplicate}
                        onChange={(e) => setDeduplicate(e.target.checked)}
                        className="rounded border-slate-300 dark:border-white/20"
                      />
                      Deduplicate records
                    </label>
                    <label className="inline-flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={dropMissing}
                        onChange={(e) => setDropMissing(e.target.checked)}
                        className="rounded border-slate-300 dark:border-white/20"
                      />
                      Drop incomplete rows
                    </label>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={handleValidate}
                      className="px-3.5 py-2 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                    >
                      Validate Schema
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={handlePreprocess}
                      className="px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
                    >
                      {busy ? 'Processing...' : 'Preprocess & Split'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Sample Row Preview */}
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
                <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08]">
                  <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                    Row Preview (First {previewRows.length} Records)
                  </h3>
                </div>
                <div className="overflow-x-auto max-h-72">
                  <table className="w-full text-left text-xs">
                    <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500 dark:text-slate-400">
                      <tr>
                        <th className="py-2 px-4 font-medium w-12">#</th>
                        <th className="py-2 px-4 font-medium">Payload</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06] font-mono">
                      {previewRows.map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-white/[0.02]">
                          <td className="py-2 px-4 text-slate-400 tabular-nums align-top">
                            {idx + 1}
                          </td>
                          <td className="py-2 px-4 text-slate-700 dark:text-slate-300 whitespace-pre-wrap break-words">
                            {JSON.stringify(row)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Paste / Create Dataset Modal */}
      {showPasteModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Create dataset from text"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        >
          <form
            onSubmit={handlePasteUpload}
            className="w-full max-w-xl rounded-xl border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] p-6 space-y-4"
          >
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Create Dataset from Raw Text / JSONL / CSV
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Dataset Name
                </label>
                <input
                  type="text"
                  required
                  value={pasteName}
                  onChange={(e) => setPasteName(e.target.value)}
                  placeholder="Support Intent Samples"
                  className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                  File Name (.jsonl, .json, .csv, .txt)
                </label>
                <input
                  type="text"
                  required
                  value={pasteFileName}
                  onChange={(e) => setPasteFileName(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                Dataset Content
              </label>
              <textarea
                rows={7}
                required
                value={pasteContent}
                onChange={(e) => setPasteContent(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
              />
            </div>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowPasteModal(false)}
                className="px-3.5 py-2 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1]"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={busy}
                className="px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500"
              >
                Save Dataset
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
