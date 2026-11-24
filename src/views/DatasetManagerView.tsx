import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Archive,
  CheckCircle2,
  Download,
  FileCode2,
  FileSpreadsheet,
  FlaskConical,
  Layers,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react';
import GlassIcons, { type GlassIconsItem } from '../components/reactbits/GlassIcons';
import Folder from '../components/reactbits/Folder';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { api } from '../lib/api';
import { useIsDark } from '../lib/useDarkMode';
import { formatBytes } from '../lib/utils';
import {
  buildDatasetBundleDocs,
  type DatasetBundleDocId,
} from '../utils/folderBundles';
import type { DatasetRecord } from '../shared/types';
import type { NavSection } from './OverviewView';

interface DatasetManagerViewProps {
  selectedDatasetId?: string | null;
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
  selectedDatasetId: externalSelectedDatasetId,
  onSelectForTraining,
  onNavigate,
}: DatasetManagerViewProps) {
  const isDark = useIsDark();
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(externalSelectedDatasetId ?? null);
  const [previewRows, setPreviewRows] = useState<Record<string, unknown>[]>([]);
  const [processedPreview, setProcessedPreview] = useState<{
    cleanedSample?: Record<string, unknown>[];
    trainSample: Record<string, unknown>[];
    valSample: Record<string, unknown>[];
    testSample: Record<string, unknown>[];
  } | null>(null);
  const [previewTab, setPreviewTab] = useState<'raw' | 'cleaned' | 'train' | 'val' | 'test'>('raw');
  const [previewMode, setPreviewMode] = useState<'table' | 'json'>('table');
  const [exportFormat, setExportFormat] = useState<'jsonl' | 'csv' | 'json'>('jsonl');
  const [dragActive, setDragActive] = useState(false);
  const [bundleFolderOpen, setBundleFolderOpen] = useState(false);
  const [bundleFolderLayout, setBundleFolderLayout] = useState<'spread' | 'row' | 'stack'>('spread');
  const [activeBundleDocId, setActiveBundleDocId] = useState<DatasetBundleDocId>('train');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  const triggerDownload = (url: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = '';
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const loadDatasets = async (preserveId?: string) => {
    setLoading(true);
    try {
      const list = await api.getDatasets();
      setDatasets(list);
      setLoading(false);
      const target =
        preserveId ||
        (selectedId && list.some((d) => d.id === selectedId)
          ? selectedId
          : externalSelectedDatasetId && list.some((d) => d.id === externalSelectedDatasetId)
            ? externalSelectedDatasetId
            : list[0]?.id) ||
        null;
      if (target && list.some((d) => d.id === target)) {
        setSelectedId(target);
        onSelectForTraining(target);
        const detail = await api.getDatasetDetail(target);
        setPreviewRows(detail.previewRows);
        setProcessedPreview(detail.processedPreview);
        setMapInstruction(detail.dataset.columnMapping?.instruction || '');
        setMapInput(detail.dataset.columnMapping?.input || '');
        setMapOutput(detail.dataset.columnMapping?.output || '');
      } else {
        setSelectedId(null);
        setPreviewRows([]);
        setProcessedPreview(null);
        setMapInstruction('');
        setMapInput('');
        setMapOutput('');
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
    onSelectForTraining(id);
    setError(null);
    setNotice(null);
    setPreviewTab('raw');
    try {
      const detail = await api.getDatasetDetail(id);
      setPreviewRows(detail.previewRows);
      setProcessedPreview(detail.processedPreview);
      setMapInstruction(detail.dataset.columnMapping?.instruction || '');
      setMapInput(detail.dataset.columnMapping?.input || '');
      setMapOutput(detail.dataset.columnMapping?.output || '');
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const uploadSelectedFile = async (file: File) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const created = await api.uploadDatasetFile(file);
      onSelectForTraining(created.id);
      setNotice(
        `Uploaded "${created.fileName}" (${created.rowCount} rows, ${created.columnCount} attributes) and set as active workspace dataset for Fine-Tuning Studio, Evaluation, Comparison, and Playground.`
      );
      await loadDatasets(created.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadSelectedFile(file);
    e.target.value = '';
  };

  const handleDropFile = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || busy) return;
    await uploadSelectedFile(file);
  };

  const handlePasteUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const trimmedFile = pasteFileName.trim() || 'custom_dataset.jsonl';
      const created = await api.uploadDatasetText({
        name: pasteName.trim() || trimmedFile.replace(/\.[^.]+$/, ''),
        fileName: trimmedFile,
        content: pasteContent,
      });
      setShowPasteModal(false);
      setPasteName('');
      onSelectForTraining(created.id);
      setNotice(
        `Created dataset "${created.name}" (${created.rowCount} rows, ${created.columnCount} attributes) and set as active workspace dataset.`
      );
      await loadDatasets(created.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const selected = datasets.find((d) => d.id === selectedId) ?? null;

  const buildMappingPayload = () => {
    const mapping: Record<string, string> = {};
    if (mapInstruction.trim()) mapping.instruction = mapInstruction.trim();
    if (mapInput.trim()) mapping.input = mapInput.trim();
    if (mapOutput.trim()) mapping.output = mapOutput.trim();
    return mapping;
  };

  const handleValidate = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await api.validateDataset(selected.id, buildMappingPayload());
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
      const updated = await api.preprocessDataset(selected.id, {
        trainRatio,
        valRatio,
        testRatio,
        deduplicate,
        dropMissing,
        seed,
        columnMapping: buildMappingPayload(),
      });
      onSelectForTraining(updated.id);
      setNotice(
        `Dataset training-ready: ${updated.splits?.trainCount} train / ${updated.splits?.valCount} val / ${updated.splits?.testCount} test (0 leakage overlap). Your processed dataset & splits are ready to use across Fine-Tuning Studio, Evaluation, Comparison, and Playground.`
      );
      setPreviewTab('cleaned');
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
    <div
      className="space-y-6"
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) {
          setDragActive(false);
        }
      }}
      onDrop={handleDropFile}
    >
      <div
        className={`flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 rounded-xl transition-colors ${
          dragActive
            ? 'ring-2 ring-blue-500/70 bg-blue-500/[0.06] p-3 -m-3'
            : ''
        }`}
      >
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Dataset Manager
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Upload or drop CSV, TSV, JSON, JSONL, NDJSON, or TXT files — all attributes present in your dataset are preserved for Fine-Tuning Studio.
          </p>
        </div>
        <div className="flex items-center gap-2.5 shrink-0">
          <LiquidButton
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setShowPasteModal(true)}
          >
            Paste / Create Dataset
          </LiquidButton>
          <LiquidButton
            type="button"
            variant="default"
            disabled={busy}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="w-3.5 h-3.5" />
            Upload File
          </LiquidButton>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.tsv,.json,.jsonl,.ndjson,.txt,text/csv,text/tab-separated-values,application/json,text/plain"
            onChange={handleFileUpload}
            className="sr-only"
            aria-label="Upload dataset file"
          />
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
        <div className="rounded-lg border border-red-500/35 bg-red-50/95 dark:bg-[#220F14]/95 backdrop-blur-md px-4 py-3 text-xs text-red-700 dark:text-red-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
      {notice && (
        <div className="rounded-lg border border-emerald-500/35 bg-emerald-50/95 dark:bg-[#0D211A]/95 backdrop-blur-md px-4 py-3 text-xs text-emerald-800 dark:text-emerald-300 flex items-center gap-2">
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
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/90 liquid-glass-surface p-5">
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
                    <LiquidButton
                      type="button"
                      variant="default"
                      onClick={() => {
                        onSelectForTraining(selected.id);
                        onNavigate('studio');
                      }}
                    >
                      Use in Fine-Tuning Studio
                    </LiquidButton>
                  </div>
                </div>

                {/* Column Statistics & Validation Notices */}
                {selected.validationReport && (
                  <div className="mt-5 overflow-x-auto border-t border-slate-200 dark:border-white/[0.06] pt-4 space-y-3">
                    {selected.validationReport.errors.length > 0 && (
                      <div className="rounded-lg border border-red-500/35 bg-red-50/90 dark:bg-[#220F14]/90 px-3 py-2 text-xs text-red-700 dark:text-red-300 space-y-0.5">
                        {selected.validationReport.errors.map((errMsg, idx) => (
                          <div key={idx}>{errMsg}</div>
                        ))}
                      </div>
                    )}
                    {selected.validationReport.warnings.length > 0 && (
                      <div className="rounded-lg border border-amber-500/30 bg-amber-50/80 dark:bg-amber-500/[0.08] px-3 py-2 text-xs text-amber-800 dark:text-amber-200 space-y-0.5">
                        {selected.validationReport.warnings.map((warnMsg, idx) => (
                          <div key={idx}>{warnMsg}</div>
                        ))}
                      </div>
                    )}
                    <h3 className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                      Column Schema &amp; Attributes ({selected.validationReport.columns.length})
                    </h3>
                    <table className="w-full text-left text-xs">
                      <thead className="text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-white/[0.06]">
                        <tr>
                          <th className="py-2 pr-3 font-medium">Attribute / Column</th>
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

                {selected.validationReport && selected.validationReport.columns.length > 0 && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                        Prompt / Instruction Attribute
                      </label>
                      <select
                        value={mapInstruction}
                        onChange={(e) => setMapInstruction(e.target.value)}
                        aria-label="Prompt or instruction attribute"
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono text-slate-900 dark:text-slate-100"
                      >
                        <option value="">Auto-detect (all attributes)</option>
                        {selected.validationReport.columns.map((col) => (
                          <option key={col.name} value={col.name}>
                            {col.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                        Context / Input Attribute (optional)
                      </label>
                      <select
                        value={mapInput}
                        onChange={(e) => setMapInput(e.target.value)}
                        aria-label="Context or input attribute"
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono text-slate-900 dark:text-slate-100"
                      >
                        <option value="">Auto-detect / optional</option>
                        {selected.validationReport.columns.map((col) => (
                          <option key={col.name} value={col.name}>
                            {col.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                        Target / Output Attribute
                      </label>
                      <select
                        value={mapOutput}
                        onChange={(e) => setMapOutput(e.target.value)}
                        aria-label="Target output attribute"
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono text-slate-900 dark:text-slate-100"
                      >
                        <option value="">Auto-detect (all attributes)</option>
                        {selected.validationReport.columns.map((col) => (
                          <option key={col.name} value={col.name}>
                            {col.name}
                          </option>
                        ))}
                      </select>
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
                      min="0.1"
                      max="1"
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
                      min="0"
                      max="0.5"
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
                      min="0"
                      max="0.5"
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
                  <div className="flex items-center gap-2.5">
                    <LiquidButton
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={busy}
                      onClick={handleValidate}
                    >
                      Validate Schema
                    </LiquidButton>
                    <LiquidButton
                      type="button"
                      variant="default"
                      disabled={busy}
                      onClick={handlePreprocess}
                    >
                      {busy ? 'Processing...' : 'Preprocess & Split'}
                    </LiquidButton>
                  </div>
                </div>
              </div>

              {/* Processed Dataset Output & Downloads */}
              <section
                aria-label="Processed dataset downloads"
                className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-5"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      Processed Dataset Output &amp; Export
                    </h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {selected.status === 'training_ready' && selected.splits
                        ? `All pipeline steps complete — download the cleaned corpus, isolated splits (${selected.splits.trainCount}/${selected.splits.valCount}/${selected.splits.testCount}), or a full ZIP archive.`
                        : 'Run "Preprocess & Split" above to clean, deduplicate, and unlock Train / Val / Test split & ZIP downloads.'}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400 mr-1">
                      Format:
                    </span>
                    {(['jsonl', 'csv', 'json'] as const).map((fmt) => (
                      <button
                        key={fmt}
                        type="button"
                        onClick={() => setExportFormat(fmt)}
                        className={`px-2.5 py-1 text-[11px] font-mono uppercase rounded-md border transition-colors ${
                          exportFormat === fmt
                            ? 'border-blue-500 bg-blue-600 text-white'
                            : 'border-slate-200 dark:border-white/[0.1] text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-white/[0.04]'
                        }`}
                      >
                        {fmt}
                      </button>
                    ))}
                  </div>
                </div>

                {(() => {
                  const bundleDocs = buildDatasetBundleDocs(selected, exportFormat);
                  const activeBundleDoc =
                    bundleDocs.find((d) => d.id === activeBundleDocId) ?? bundleDocs[0];
                  return (
                    <div className="rounded-xl border border-slate-200/80 dark:border-white/[0.08] bg-slate-50/80 dark:bg-[#090B10]/75 p-4 space-y-4">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                            Interactive Dataset Artifact Folder
                          </div>
                          <p className="text-[11px] text-slate-500 dark:text-slate-400">
                            Hover to peek, click to spread open, drag split files out, or click any file to preview &amp; download.
                          </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <div className="inline-flex rounded-lg border border-slate-200 dark:border-white/[0.1] bg-white/80 dark:bg-[#11141C] p-0.5">
                            {(['spread', 'row', 'stack'] as const).map((mode) => (
                              <button
                                key={mode}
                                type="button"
                                onClick={() => {
                                  setBundleFolderLayout(mode);
                                  setBundleFolderOpen(true);
                                }}
                                className={`px-2 py-1 text-[10px] font-mono uppercase rounded-md transition-colors ${
                                  bundleFolderLayout === mode
                                    ? 'bg-blue-600 text-white'
                                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                                }`}
                              >
                                {mode}
                              </button>
                            ))}
                          </div>
                          <button
                            type="button"
                            onClick={() => setBundleFolderOpen((prev) => !prev)}
                            className="px-2.5 py-1 text-[11px] font-medium rounded-lg border border-slate-200 dark:border-white/[0.12] bg-white dark:bg-[#11141C] text-slate-700 dark:text-slate-200 hover:border-blue-500/50 transition-colors"
                          >
                            {bundleFolderOpen ? 'Close Folder' : 'Open Folder'}
                          </button>
                        </div>
                      </div>

                      <div
                        data-testid="dataset-folder-stage"
                        className="relative flex items-center justify-center rounded-xl border border-slate-200/70 dark:border-white/[0.06] bg-gradient-to-b from-white/75 to-slate-100/75 dark:from-[#0E131D]/90 dark:to-[#080A0F]/95 pt-16 pb-10 px-6 min-h-[300px]"
                      >
                        <Folder
                          key={selected.id}
                          label={selected.status === 'training_ready' ? 'Dataset Splits' : `${selected.fileType.toUpperCase()} Bundle`}
                          mode="flap"
                          variant="glass"
                          layout={bundleFolderLayout}
                          color="#3B82F6"
                          paperColor="#ffffff"
                          size={1}
                          frost={0.28}
                          speed={1.45}
                          intro={false}
                          open={bundleFolderOpen}
                          onOpenChange={setBundleFolderOpen}
                          closeOnOutsideClick={false}
                          zoomOnClick={true}
                          theme={isDark ? 'dark' : 'light'}
                          onItemClick={(_entry, index) => {
                            const clicked = bundleDocs[index];
                            if (!clicked) return;
                            setActiveBundleDocId(clicked.id);
                            if (clicked.previewTab) {
                              setPreviewTab(clicked.previewTab);
                            }
                            if (clicked.id === 'zip') {
                              triggerDownload(clicked.exportUrl);
                            }
                          }}
                        >
                          {bundleDocs.map((doc) => (
                            <div
                              key={doc.id}
                              className="flex h-full flex-col justify-between overflow-hidden p-2 text-left"
                              style={{ backgroundColor: '#ffffff' }}
                            >
                              <div className="min-w-0">
                                <div
                                  className="inline-block rounded px-1.5 py-0.5 font-mono text-[8px] font-bold text-white"
                                  style={{ backgroundColor: doc.accentHex }}
                                >
                                  {doc.badge}
                                </div>
                                <div className="mt-1 truncate text-[10px] font-bold leading-tight text-slate-900">
                                  {doc.title}
                                </div>
                                <div className="mt-0.5 line-clamp-2 text-[8px] leading-snug text-slate-600">
                                  {doc.subtitle}
                                </div>
                              </div>
                              <div className="border-t border-slate-200 pt-1 font-mono text-[8px] text-slate-500 whitespace-nowrap truncate">
                                {doc.rowCount} rows
                              </div>
                            </div>
                          ))}
                        </Folder>
                      </div>

                      {activeBundleDoc && (
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-lg border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C] px-3.5 py-2.5">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <span
                                className="inline-block rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold text-white"
                                style={{ backgroundColor: activeBundleDoc.accentHex }}
                              >
                                {activeBundleDoc.badge}
                              </span>
                              <span className="text-xs font-semibold text-slate-900 dark:text-slate-100 truncate">
                                {activeBundleDoc.fileName}
                              </span>
                            </div>
                            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 truncate">
                              {activeBundleDoc.description}
                            </p>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            {activeBundleDoc.previewTab && (
                              <button
                                type="button"
                                onClick={() => setPreviewTab(activeBundleDoc.previewTab!)}
                                className="px-2.5 py-1 text-[11px] font-medium rounded-md border border-slate-200 dark:border-white/[0.1] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                              >
                                Preview Rows
                              </button>
                            )}
                            <a
                              href={activeBundleDoc.exportUrl}
                              className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded-md bg-blue-600 text-white hover:bg-blue-500"
                            >
                              <Download className="w-3 h-3" />
                              Download
                            </a>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()}

                {selected.status === 'training_ready' && selected.splits ? (
                  <>
                    <div className="pt-1 pb-6">
                      <GlassIcons
                        size={56}
                        gap={38}
                        columns={5}
                        roundness={0.52}
                        refraction={0.7}
                        bevel={0.5}
                        frost={0.22}
                        shine={0.85}
                        tint={isDark ? '#dfdfdf' : '#ffffff'}
                        tintOpacity={isDark ? 0.18 : 0.28}
                        plate="tilt"
                        tilt={24}
                        hover="lift"
                        labels="always"
                        intro={false}
                        theme={isDark ? 'dark' : 'light'}
                        items={[
                          {
                            id: 'dl-cleaned',
                            icon: <Sparkles />,
                            color: 'blue',
                            label: `Cleaned (${selected.rowCount - selected.duplicateCount - selected.missingCount > 0 ? selected.splits.trainCount + selected.splits.valCount + selected.splits.testCount : selected.rowCount})`,
                            onClick: () =>
                              triggerDownload(
                                `/api/datasets/${selected.id}/export?split=cleaned&format=${exportFormat}`
                              ),
                          },
                          {
                            id: 'dl-train',
                            icon: <FlaskConical />,
                            color: 'green',
                            label: `Train (${selected.splits.trainCount})`,
                            onClick: () =>
                              triggerDownload(
                                `/api/datasets/${selected.id}/export?split=train&format=${exportFormat}`
                              ),
                          },
                          {
                            id: 'dl-val',
                            icon: <Layers />,
                            color: 'purple',
                            label: `Val (${selected.splits.valCount})`,
                            onClick: () =>
                              triggerDownload(
                                `/api/datasets/${selected.id}/export?split=val&format=${exportFormat}`
                              ),
                          },
                          {
                            id: 'dl-test',
                            icon: <FileCode2 />,
                            color: 'orange',
                            label: `Test (${selected.splits.testCount})`,
                            onClick: () =>
                              triggerDownload(
                                `/api/datasets/${selected.id}/export?split=test&format=${exportFormat}`
                              ),
                          },
                          {
                            id: 'dl-zip',
                            icon: <Archive />,
                            color: 'teal',
                            label: 'Full ZIP Bundle',
                            onClick: () =>
                              triggerDownload(`/api/datasets/${selected.id}/export?split=all&format=zip`),
                          },
                        ] satisfies GlassIconsItem[]}
                      />
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-slate-200 dark:border-white/[0.06]">
                      <a
                        href={`/api/datasets/${selected.id}/export?split=cleaned&format=${exportFormat}`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Cleaned .{exportFormat}
                      </a>
                      <a
                        href={`/api/datasets/${selected.id}/export?split=train&format=${exportFormat}`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Train Split ({selected.splits.trainCount})
                      </a>
                      <a
                        href={`/api/datasets/${selected.id}/export?split=val&format=${exportFormat}`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Val Split ({selected.splits.valCount})
                      </a>
                      <a
                        href={`/api/datasets/${selected.id}/export?split=test&format=${exportFormat}`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Test Split ({selected.splits.testCount})
                      </a>
                      <a
                        href={`/api/datasets/${selected.id}/export?split=all&format=zip`}
                        className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-full border border-white/95 bg-white/85 hover:bg-white text-slate-900 dark:border-white/90 dark:bg-white/[0.86] dark:hover:bg-white dark:text-slate-950 backdrop-blur-xl shadow-[0_4px_18px_rgba(255,255,255,0.2),inset_0_1px_0_rgba(255,255,255,0.9)] transition-all ml-auto"
                      >
                        <Archive className="w-3.5 h-3.5" />
                        Download Complete Bundle (.zip)
                      </a>
                    </div>
                  </>
                ) : (
                  <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-dashed border-slate-200 dark:border-white/[0.1] p-4 text-xs text-slate-500 dark:text-slate-400">
                    <span>
                      Current status: <strong className="font-mono text-slate-700 dark:text-slate-200">{selected.status}</strong>. Click &ldquo;Preprocess &amp; Split&rdquo; to generate downloadable cleaned &amp; split datasets.
                    </span>
                    <a
                      href={`/api/datasets/${selected.id}/export?split=cleaned&format=${exportFormat}`}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download Raw as .{exportFormat}
                    </a>
                  </div>
                )}
              </section>

              {/* Sample Row Preview */}
              {(() => {
                const activePreviewRows =
                  previewTab === 'cleaned' && processedPreview?.cleanedSample
                    ? processedPreview.cleanedSample
                    : previewTab === 'train' && processedPreview?.trainSample
                      ? processedPreview.trainSample
                      : previewTab === 'val' && processedPreview?.valSample
                        ? processedPreview.valSample
                        : previewTab === 'test' && processedPreview?.testSample
                          ? processedPreview.testSample
                          : previewRows;
                const previewAttrKeys = Array.from(
                  new Set(activePreviewRows.flatMap((row) => Object.keys(row)))
                );
                const formatPreviewCell = (val: unknown): string => {
                  if (val === null || val === undefined || val === '') return '—';
                  if (typeof val === 'string') return val;
                  if (typeof val === 'number' || typeof val === 'boolean') return String(val);
                  return JSON.stringify(val);
                };
                return (
                  <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
                    <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
                      <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                        Row Preview ({activePreviewRows.length} Records{previewAttrKeys.length > 0 ? ` · ${previewAttrKeys.length} Attributes` : ''})
                      </h3>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <div className="inline-flex rounded-md border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] p-0.5 mr-1">
                          <button
                            type="button"
                            onClick={() => setPreviewMode('table')}
                            className={`px-2 py-0.5 text-[10px] font-mono rounded transition-colors ${
                              previewMode === 'table'
                                ? 'bg-blue-600 text-white'
                                : 'text-slate-600 dark:text-slate-400'
                            }`}
                          >
                            Attributes
                          </button>
                          <button
                            type="button"
                            onClick={() => setPreviewMode('json')}
                            className={`px-2 py-0.5 text-[10px] font-mono rounded transition-colors ${
                              previewMode === 'json'
                                ? 'bg-blue-600 text-white'
                                : 'text-slate-600 dark:text-slate-400'
                            }`}
                          >
                            Raw JSON
                          </button>
                        </div>
                        {(
                          [
                            { id: 'raw', label: 'Raw Input', enabled: true },
                            { id: 'cleaned', label: 'Cleaned', enabled: Boolean(processedPreview?.cleanedSample) },
                            { id: 'train', label: 'Train Split', enabled: Boolean(processedPreview?.trainSample) },
                            { id: 'val', label: 'Val Split', enabled: Boolean(processedPreview?.valSample) },
                            { id: 'test', label: 'Test Split', enabled: Boolean(processedPreview?.testSample) },
                          ] as const
                        )
                          .filter((t) => t.enabled)
                          .map((t) => (
                            <button
                              key={t.id}
                              type="button"
                              onClick={() => setPreviewTab(t.id)}
                              className={`px-2.5 py-1 text-[11px] font-medium rounded-md transition-colors ${
                                previewTab === t.id
                                  ? 'bg-blue-600 text-white'
                                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-white/[0.05]'
                              }`}
                            >
                              {t.label}
                            </button>
                          ))}
                      </div>
                    </div>
                    <div className="overflow-x-auto max-h-72">
                      {previewMode === 'table' && previewAttrKeys.length > 0 ? (
                        <table className="w-full text-left text-xs">
                          <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500 dark:text-slate-400 bg-slate-50/70 dark:bg-white/[0.02]">
                            <tr>
                              <th className="py-2 px-4 font-medium w-12">#</th>
                              {previewAttrKeys.map((attr) => (
                                <th key={attr} className="py-2 px-4 font-medium whitespace-nowrap">
                                  {attr}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06] font-mono">
                            {activePreviewRows.map((row, idx) => (
                              <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-white/[0.02]">
                                <td className="py-2 px-4 text-slate-400 tabular-nums align-top">
                                  {idx + 1}
                                </td>
                                {previewAttrKeys.map((attr) => (
                                  <td
                                    key={attr}
                                    className="py-2 px-4 text-slate-700 dark:text-slate-300 align-top max-w-xs break-words"
                                  >
                                    {formatPreviewCell(row[attr])}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <table className="w-full text-left text-xs">
                          <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500 dark:text-slate-400">
                            <tr>
                              <th className="py-2 px-4 font-medium w-12">#</th>
                              <th className="py-2 px-4 font-medium">Payload</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06] font-mono">
                            {activePreviewRows.map((row, idx) => (
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
                      )}
                    </div>
                  </div>
                );
              })()}
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
              Create Dataset from CSV / TSV / JSON / JSONL / TXT
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
                  File Name (.jsonl, .json, .ndjson, .csv, .tsv, .txt)
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
              <p className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400">
                Supports JSON arrays, wrapped JSON objects (<code className="font-mono">data</code>, <code className="font-mono">records</code>, <code className="font-mono">rows</code>, <code className="font-mono">items</code>, <code className="font-mono">examples</code>, <code className="font-mono">conversations</code>), single JSON objects, JSONL/NDJSON, CSV/TSV, and TXT.
              </p>
            </div>
            <div className="flex justify-end gap-2">
              <LiquidButton
                type="button"
                size="sm"
                variant="outline"
                onClick={() => setShowPasteModal(false)}
              >
                Cancel
              </LiquidButton>
              <LiquidButton type="submit" variant="default" disabled={busy}>
                Save Dataset
              </LiquidButton>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
