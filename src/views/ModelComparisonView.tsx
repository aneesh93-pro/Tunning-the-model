import { useEffect, useState } from 'react';
import { AlertTriangle, GitCompare, Play } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AiOrb } from '@/components/ui/ai-orb';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { mapJobRunnerToOrbState } from '@/utils/aiOrbState';
import { pickPreferredDatasetId } from '../utils/datasetSelection';
import { api } from '../lib/api';
import { formatBytes, formatDuration, formatNumber } from '../lib/utils';
import type { ComparisonModelMeta, DatasetRecord, EvaluationRunRecord } from '../shared/types';

const MODEL_GROUPS = [
  'Trained Adapter',
  'NVIDIA NIM',
  'Z.ai',
  'ChatGPT',
  'Claude',
  'Gemini',
  'DeepSeek',
  'Open SLM',
  'Llama',
  'Mistral',
];

function formatDeltaPts(a: number, b: number, higherIsBetter = true): { text: string; tone: string } {
  const diff = b - a;
  if (Math.abs(diff) < 0.05) return { text: '0.0', tone: 'text-slate-400' };
  const improved = higherIsBetter ? diff > 0 : diff < 0;
  return {
    text: `${diff > 0 ? '+' : ''}${diff.toFixed(1)}`,
    tone: improved ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400',
  };
}

interface ModelComparisonViewProps {
  initialDatasetId?: string | null;
  onSelectDataset?: (datasetId: string) => void;
}

export function ModelComparisonView({
  initialDatasetId,
  onSelectDataset,
}: ModelComparisonViewProps) {
  const [models, setModels] = useState<Array<{ ref: string; label: string; aiTool?: string }>>([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [leftRef, setLeftRef] = useState('');
  const [rightRef, setRightRef] = useState('');
  const [datasetId, setDatasetId] = useState('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<{
    left: {
      evaluation: EvaluationRunRecord;
      meta: ComparisonModelMeta;
    };
    right: {
      evaluation: EvaluationRunRecord;
      meta: ComparisonModelMeta;
    };
    sampleDiffs: Array<{
      index: number;
      prompt: string;
      groundTruth: string;
      leftPrediction: string;
      leftCorrect: boolean;
      leftConfidence: number;
      rightPrediction: string;
      rightCorrect: boolean;
      rightConfidence: number;
    }>;
  } | null>(null);

  useEffect(() => {
    Promise.all([api.getInferenceModels(), api.getDatasets()])
      .then(([mRes, dsList]) => {
        setModels(mRes.models);
        setDatasets(dsList);
        const baseDefault =
          mRes.models.find((m) => m.ref.startsWith('base:'))?.ref || mRes.models[0]?.ref || '';
        const tunedDefault =
          mRes.models.find((m) => m.ref.startsWith('reg:'))?.ref ||
          mRes.models.find((m) => m.ref !== baseDefault)?.ref ||
          baseDefault;
        setLeftRef(baseDefault);
        setRightRef(tunedDefault);
        const preferred = pickPreferredDatasetId(dsList, initialDatasetId);
        if (preferred) setDatasetId(preferred);
      })
      .catch((err) => {
        setError((err as Error).message);
      });
  }, []);

  useEffect(() => {
    if (!initialDatasetId) return;
    if (datasets.some((d) => d.id === initialDatasetId)) {
      setDatasetId(initialDatasetId);
    } else {
      api
        .getDatasets()
        .then((dsList) => {
          setDatasets(dsList);
          const resolved = pickPreferredDatasetId(dsList, initialDatasetId);
          if (resolved) setDatasetId(resolved);
        })
        .catch(() => undefined);
    }
  }, [initialDatasetId, datasets]);

  const handleCompare = async () => {
    if (!leftRef || !rightRef || !datasetId) return;
    setRunning(true);
    setError(null);
    try {
      const res = await api.runComparison(leftRef, rightRef, datasetId);
      setComparison(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const renderGroupedOptions = () =>
    MODEL_GROUPS.map((group) => {
      const items = models.filter((m) => (m.aiTool ?? 'Open SLM') === group);
      if (items.length === 0) return null;
      return (
        <optgroup key={group} label={group}>
          {items.map((m) => (
            <option key={m.ref} value={m.ref}>
              {m.label}
            </option>
          ))}
        </optgroup>
      );
    });

  const chartData = comparison
    ? [
        {
          metric: comparison.left.evaluation.taskType === 'classification' ? 'Accuracy %' : 'Token F1 %',
          Left: Number(
            (
              (comparison.left.evaluation.accuracy ??
                comparison.left.evaluation.tokenF1 ??
                0) * 100
            ).toFixed(1)
          ),
          Right: Number(
            (
              (comparison.right.evaluation.accuracy ??
                comparison.right.evaluation.tokenF1 ??
                0) * 100
            ).toFixed(1)
          ),
        },
        {
          metric: 'Macro F1 / EM %',
          Left: Number(
            (
              (comparison.left.evaluation.macroF1 ??
                comparison.left.evaluation.exactMatch ??
                0) * 100
            ).toFixed(1)
          ),
          Right: Number(
            (
              (comparison.right.evaluation.macroF1 ??
                comparison.right.evaluation.exactMatch ??
                0) * 100
            ).toFixed(1)
          ),
        },
        {
          metric: 'Confidence %',
          Left: Number(
            (
              (comparison.left.evaluation.samples.reduce((s, x) => s + x.confidence, 0) /
                Math.max(1, comparison.left.evaluation.samples.length)) *
              100
            ).toFixed(1)
          ),
          Right: Number(
            (
              (comparison.right.evaluation.samples.reduce((s, x) => s + x.confidence, 0) /
                Math.max(1, comparison.right.evaluation.samples.length)) *
              100
            ).toFixed(1)
          ),
        },
      ]
    : [];

  const leftAccPct = comparison
    ? ((comparison.left.evaluation.accuracy ?? comparison.left.evaluation.exactMatch ?? 0) * 100)
    : 0;
  const rightAccPct = comparison
    ? ((comparison.right.evaluation.accuracy ?? comparison.right.evaluation.exactMatch ?? 0) * 100)
    : 0;
  const leftF1Pct = comparison
    ? ((comparison.left.evaluation.macroF1 ?? comparison.left.evaluation.tokenF1 ?? 0) * 100)
    : 0;
  const rightF1Pct = comparison
    ? ((comparison.right.evaluation.macroF1 ?? comparison.right.evaluation.tokenF1 ?? 0) * 100)
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Model Comparison
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Run deterministic side-by-side benchmarks across Base Models, NVIDIA NIM, Z.ai GLM, Frontier APIs, and LoRA/QLoRA Adapters with real accuracy and memory footprint deltas.
          </p>
        </div>
        <AiOrb
          state={mapJobRunnerToOrbState({
            error,
            running,
            hasResult: Boolean(comparison),
          })}
          size="sm"
          showLabel
          label={running ? 'Comparing' : error ? 'Comparison error' : comparison ? 'Benchmark ready' : 'Idle'}
        />
      </div>

      <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5 grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Baseline / Model A
          </label>
          <select
            aria-label="Baseline or Model A"
            value={leftRef}
            onChange={(e) => setLeftRef(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {renderGroupedOptions()}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Candidate / Model B
          </label>
          <select
            aria-label="Candidate or Model B"
            value={rightRef}
            onChange={(e) => setRightRef(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {renderGroupedOptions()}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Held-Out Test Dataset
          </label>
          <select
            aria-label="Held-Out Test Dataset"
            value={datasetId}
            onChange={(e) => {
              setDatasetId(e.target.value);
              onSelectDataset?.(e.target.value);
            }}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} (
                {d.splits?.testCount
                  ? `${d.splits.testCount} test rows`
                  : `${d.rowCount} rows · auto-split`}
                )
              </option>
            ))}
          </select>
        </div>
        <LiquidButton
          type="button"
          variant="default"
          disabled={running || !leftRef || !rightRef || !datasetId}
          onClick={handleCompare}
          className="w-full justify-center"
        >
          <Play className="w-3.5 h-3.5" />
          {running ? 'Comparing...' : 'Run Comparison'}
        </LiquidButton>
      </div>

      {running && (
        <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] px-4 py-3">
          <AiOrb
            state="tool-result"
            size="md"
            showLabel
            label="Evaluating Model A vs Model B across identical held-out samples..."
          />
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-500/35 bg-red-50/95 dark:bg-[#220F14]/95 backdrop-blur-md px-4 py-3 text-xs text-red-700 dark:text-red-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!comparison ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-12 text-center">
          <GitCompare className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
            Select two models and run a side-by-side comparison
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="liquid-glass-surface lg:col-span-5 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5">
              <div className="mb-3">
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Metric Comparison Chart
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Compare Model B (blue) vs. Model A (slate) on the exact same held-out prompts. Dotted line shows the 80% benchmark.
                </p>
              </div>
              <div className="h-72 w-full min-w-0">
                <ResponsiveContainer
                  width="100%"
                  height={288}
                  minWidth={0}
                  minHeight={288}
                  initialDimension={{ width: 420, height: 288 }}
                >
                  <BarChart data={chartData} margin={{ top: 10, right: 12, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                    <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <YAxis
                      domain={[0, 100]}
                      width={44}
                      tickFormatter={(v) => `${v}%`}
                      tick={{ fontSize: 11, fill: '#94A3B8' }}
                    />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const row = payload[0].payload as { metric: string; Left: number; Right: number };
                        const delta = row.Right - row.Left;
                        return (
                          <div className="rounded-lg border border-white/15 bg-[#11141C]/95 px-3 py-2 text-xs text-slate-100 shadow-lg backdrop-blur-md space-y-1">
                            <div className="font-semibold text-slate-200">{label}</div>
                            <div className="font-mono text-slate-300">
                              Model A: <span className="text-slate-100 font-semibold">{row.Left}%</span>
                            </div>
                            <div className="font-mono text-blue-300">
                              Model B: <span className="text-blue-200 font-semibold">{row.Right}%</span>
                            </div>
                            <div
                              className={`font-mono text-[11px] pt-0.5 border-t border-white/10 ${
                                delta >= 0 ? 'text-emerald-400' : 'text-amber-400'
                              }`}
                            >
                              Delta (B - A): {delta > 0 ? '+' : ''}
                              {delta.toFixed(1)} pts
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <ReferenceLine
                      y={80}
                      stroke="#10B981"
                      strokeDasharray="4 4"
                      label={{
                        value: '80% Benchmark',
                        position: 'insideTopRight',
                        fill: '#10B981',
                        fontSize: 10,
                      }}
                    />
                    <Bar
                      dataKey="Left"
                      name={comparison.left.evaluation.modelName}
                      fill="#64748B"
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={false}
                    />
                    <Bar
                      dataKey="Right"
                      name={comparison.right.evaluation.modelName}
                      fill="#3B82F6"
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={false}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="liquid-glass-surface lg:col-span-7 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08] flex items-center justify-between text-xs font-semibold">
                <span>Measured Metrics &amp; Footprint</span>
                <span className="font-mono font-normal text-slate-500">
                  {comparison.left.evaluation.testSampleCount} held-out test samples
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono tabular-nums">
                  <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500">
                    <tr>
                      <th className="py-2.5 px-4 font-medium">Dimension</th>
                      <th className="py-2.5 px-3 font-medium text-right max-w-[140px] truncate" title={comparison.left.evaluation.modelName}>
                        Model A
                      </th>
                      <th className="py-2.5 px-3 font-medium text-right max-w-[140px] truncate" title={comparison.right.evaluation.modelName}>
                        Model B
                      </th>
                      <th className="py-2.5 px-4 font-medium text-right">Delta (B-A)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Accuracy / Exact Match</td>
                      <td className="py-2 px-3 text-right">{leftAccPct.toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right font-semibold text-blue-600 dark:text-blue-400">
                        {rightAccPct.toFixed(1)}%
                      </td>
                      <td className={`py-2 px-4 text-right ${formatDeltaPts(leftAccPct, rightAccPct, true).tone}`}>
                        {formatDeltaPts(leftAccPct, rightAccPct, true).text}%
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Macro F1 / Token F1</td>
                      <td className="py-2 px-3 text-right">{leftF1Pct.toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right">{rightF1Pct.toFixed(1)}%</td>
                      <td className={`py-2 px-4 text-right ${formatDeltaPts(leftF1Pct, rightF1Pct, true).tone}`}>
                        {formatDeltaPts(leftF1Pct, rightF1Pct, true).text}%
                      </td>
                    </tr>
                    {comparison.left.evaluation.precision !== null && (
                      <tr>
                        <td className="py-2 px-4 text-slate-500 font-sans">Precision / Recall</td>
                        <td className="py-2 px-3 text-right">
                          {((comparison.left.evaluation.precision ?? 0) * 100).toFixed(1)}% /{' '}
                          {((comparison.left.evaluation.recall ?? 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 px-3 text-right">
                          {((comparison.right.evaluation.precision ?? 0) * 100).toFixed(1)}% /{' '}
                          {((comparison.right.evaluation.recall ?? 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 px-4 text-right text-slate-400">—</td>
                      </tr>
                    )}
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Perplexity</td>
                      <td className="py-2 px-3 text-right">{comparison.left.evaluation.perplexity}</td>
                      <td className="py-2 px-3 text-right">{comparison.right.evaluation.perplexity}</td>
                      <td
                        className={`py-2 px-4 text-right ${
                          formatDeltaPts(
                            comparison.left.evaluation.perplexity,
                            comparison.right.evaluation.perplexity,
                            false
                          ).tone
                        }`}
                      >
                        {formatDeltaPts(
                          comparison.left.evaluation.perplexity,
                          comparison.right.evaluation.perplexity,
                          false
                        ).text}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Avg Test Loss</td>
                      <td className="py-2 px-3 text-right">{comparison.left.evaluation.avgLoss.toFixed(4)}</td>
                      <td className="py-2 px-3 text-right">{comparison.right.evaluation.avgLoss.toFixed(4)}</td>
                      <td
                        className={`py-2 px-4 text-right ${
                          comparison.right.evaluation.avgLoss <= comparison.left.evaluation.avgLoss
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-amber-600 dark:text-amber-400'
                        }`}
                      >
                        {(comparison.right.evaluation.avgLoss - comparison.left.evaluation.avgLoss > 0 ? '+' : '')}
                        {(comparison.right.evaluation.avgLoss - comparison.left.evaluation.avgLoss).toFixed(3)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Inference Latency</td>
                      <td className="py-2 px-3 text-right">{comparison.left.evaluation.avgLatencyMs} ms</td>
                      <td className="py-2 px-3 text-right">{comparison.right.evaluation.avgLatencyMs} ms</td>
                      <td
                        className={`py-2 px-4 text-right ${
                          formatDeltaPts(
                            comparison.left.evaluation.avgLatencyMs,
                            comparison.right.evaluation.avgLatencyMs,
                            false
                          ).tone
                        }`}
                      >
                        {formatDeltaPts(
                          comparison.left.evaluation.avgLatencyMs,
                          comparison.right.evaluation.avgLatencyMs,
                          false
                        ).text} ms
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Training Duration</td>
                      <td className="py-2 px-3 text-right">{formatDuration(comparison.left.meta.durationMs)}</td>
                      <td className="py-2 px-3 text-right">{formatDuration(comparison.right.meta.durationMs)}</td>
                      <td className="py-2 px-4 text-right text-slate-400">—</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Adapter Size</td>
                      <td className="py-2 px-3 text-right">{formatBytes(comparison.left.meta.adapterSizeBytes)}</td>
                      <td className="py-2 px-3 text-right">{formatBytes(comparison.right.meta.adapterSizeBytes)}</td>
                      <td className="py-2 px-4 text-right text-slate-400">—</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Trainable Params</td>
                      <td className="py-2 px-3 text-right">{formatNumber(comparison.left.meta.trainableParams)}</td>
                      <td className="py-2 px-3 text-right">{formatNumber(comparison.right.meta.trainableParams)}</td>
                      <td className="py-2 px-4 text-right text-slate-400">—</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Total Architecture Params</td>
                      <td className="py-2 px-3 text-right">
                        {comparison.left.meta.parameterCountLabel || formatNumber(comparison.left.meta.totalParams)}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {comparison.right.meta.parameterCountLabel || formatNumber(comparison.right.meta.totalParams)}
                      </td>
                      <td className="py-2 px-4 text-right text-slate-400">—</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">VRAM / Memory Footprint</td>
                      <td className="py-2 px-3 text-right">{comparison.left.meta.estimatedMemoryGb} GB</td>
                      <td className="py-2 px-3 text-right">{comparison.right.meta.estimatedMemoryGb} GB</td>
                      <td
                        className={`py-2 px-4 text-right ${
                          formatDeltaPts(
                            comparison.left.meta.estimatedMemoryGb,
                            comparison.right.meta.estimatedMemoryGb,
                            false
                          ).tone
                        }`}
                      >
                        {formatDeltaPts(
                          comparison.left.meta.estimatedMemoryGb,
                          comparison.right.meta.estimatedMemoryGb,
                          false
                        ).text} GB
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-4 text-slate-500 font-sans">Precision &amp; Context</td>
                      <td className="py-2 px-3 text-right">
                        {comparison.left.meta.precision} · {formatNumber(comparison.left.meta.contextLength)}
                      </td>
                      <td className="py-2 px-3 text-right">
                        {comparison.right.meta.precision} · {formatNumber(comparison.right.meta.contextLength)}
                      </td>
                      <td className="py-2 px-4 text-right text-slate-400 uppercase">
                        {comparison.left.meta.architecture} / {comparison.right.meta.architecture}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 overflow-hidden">
            <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08] text-xs font-semibold">
              Sample Prediction Differences ({comparison.sampleDiffs.length} test rows)
            </div>
            <div className="overflow-x-auto max-h-80">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500">
                  <tr>
                    <th className="py-2 px-4 font-medium">Prompt</th>
                    <th className="py-2 px-4 font-medium">Ground Truth</th>
                    <th className="py-2 px-4 font-medium">Model A</th>
                    <th className="py-2 px-4 font-medium">Model B</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                  {comparison.sampleDiffs.map((row) => (
                    <tr key={row.index}>
                      <td className="py-2 px-4 max-w-xs truncate">{row.prompt}</td>
                      <td className="py-2 px-4 font-mono">{row.groundTruth}</td>
                      <td
                        className={`py-2 px-4 font-mono ${
                          row.leftCorrect ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'
                        }`}
                      >
                        {row.leftPrediction}{' '}
                        <span className="text-[10px] text-slate-400">
                          ({(row.leftConfidence * 100).toFixed(0)}%)
                        </span>
                      </td>
                      <td
                        className={`py-2 px-4 font-mono ${
                          row.rightCorrect ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'
                        }`}
                      >
                        {row.rightPrediction}{' '}
                        <span className="text-[10px] text-slate-400">
                          ({(row.rightConfidence * 100).toFixed(0)}%)
                        </span>
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
  );
}

