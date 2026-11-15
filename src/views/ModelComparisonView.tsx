import { useEffect, useState } from 'react';
import { AlertTriangle, GitCompare, Play } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { LatticeLoader } from '../components/reactbits/LatticeLoader';
import { api } from '../lib/api';
import { formatBytes, formatDuration, formatNumber } from '../lib/utils';
import type { DatasetRecord, EvaluationRunRecord } from '../shared/types';

export function ModelComparisonView() {
  const [models, setModels] = useState<Array<{ ref: string; label: string }>>([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [leftRef, setLeftRef] = useState('');
  const [rightRef, setRightRef] = useState('');
  const [datasetId, setDatasetId] = useState('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<{
    left: {
      evaluation: EvaluationRunRecord;
      meta: {
        durationMs: number;
        adapterSizeBytes: number;
        trainableParams: number;
        estimatedMemoryGb: number;
      };
    };
    right: {
      evaluation: EvaluationRunRecord;
      meta: {
        durationMs: number;
        adapterSizeBytes: number;
        trainableParams: number;
        estimatedMemoryGb: number;
      };
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
    Promise.all([api.getInferenceModels(), api.getDatasets()]).then(([mRes, dsList]) => {
      setModels(mRes.models);
      const ready = dsList.filter((d) => d.status === 'training_ready');
      setDatasets(ready);
      const baseDefault = mRes.models.find((m) => m.ref.startsWith('base:'))?.ref || mRes.models[0]?.ref || '';
      const tunedDefault = mRes.models.find((m) => m.ref.startsWith('reg:'))?.ref || mRes.models[1]?.ref || baseDefault;
      setLeftRef(baseDefault);
      setRightRef(tunedDefault);
      if (ready.length > 0) setDatasetId(ready[0].id);
    }).catch((err) => {
      setError((err as Error).message);
    });
  }, []);

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
      ]
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
          Model Comparison
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Run paired side-by-side benchmarks comparing Base vs LoRA, Base vs QLoRA, or LoRA vs QLoRA on identical test inputs.
        </p>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Baseline / Model A
          </label>
          <select
            value={leftRef}
            onChange={(e) => setLeftRef(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {models.map((m) => (
              <option key={m.ref} value={m.ref}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Candidate / Model B
          </label>
          <select
            value={rightRef}
            onChange={(e) => setRightRef(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {models.map((m) => (
              <option key={m.ref} value={m.ref}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Held-Out Test Dataset
          </label>
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          disabled={running || !leftRef || !rightRef || !datasetId}
          onClick={handleCompare}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
        >
          <Play className="w-3.5 h-3.5" />
          {running ? 'Comparing...' : 'Run Comparison'}
        </button>
      </div>

      {running && (
        <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] px-4 py-3">
          <LatticeLoader
            pattern="ripple"
            grid={3}
            cellSize={5}
            gap={2}
            fontSize={12}
            color="#3B82F6"
            status="working"
            label="Evaluating Model A vs Model B across identical held-out samples..."
            showTimer
          />
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
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
            <div className="lg:col-span-6 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">
                Metric Comparison Chart
              </h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                    <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#11141C',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        fontSize: '12px',
                        color: '#F8FAFC',
                      }}
                      itemStyle={{ color: '#E2E8F0' }}
                      labelStyle={{ color: '#94A3B8', marginBottom: '4px' }}
                    />
                    <Legend />
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

            <div className="lg:col-span-6 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08] text-xs font-semibold">
                Measured Metrics &amp; Footprint
              </div>
              <table className="w-full text-left text-xs font-mono tabular-nums">
                <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500">
                  <tr>
                    <th className="py-2.5 px-4 font-medium">Dimension</th>
                    <th className="py-2.5 px-4 font-medium text-right">Model A</th>
                    <th className="py-2.5 px-4 font-medium text-right">Model B</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                  <tr>
                    <td className="py-2 px-4 text-slate-500">Accuracy / Exact Match</td>
                    <td className="py-2 px-4 text-right">
                      {(((comparison.left.evaluation.accuracy ?? comparison.left.evaluation.exactMatch) ?? 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-4 text-right font-semibold text-blue-600 dark:text-blue-400">
                      {(((comparison.right.evaluation.accuracy ?? comparison.right.evaluation.exactMatch) ?? 0) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 text-slate-500">Macro F1 / Token F1</td>
                    <td className="py-2 px-4 text-right">
                      {(((comparison.left.evaluation.macroF1 ?? comparison.left.evaluation.tokenF1) ?? 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-4 text-right">
                      {(((comparison.right.evaluation.macroF1 ?? comparison.right.evaluation.tokenF1) ?? 0) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 text-slate-500">Perplexity</td>
                    <td className="py-2 px-4 text-right">{comparison.left.evaluation.perplexity}</td>
                    <td className="py-2 px-4 text-right">{comparison.right.evaluation.perplexity}</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 text-slate-500">Inference Latency</td>
                    <td className="py-2 px-4 text-right">{comparison.left.evaluation.avgLatencyMs} ms</td>
                    <td className="py-2 px-4 text-right">{comparison.right.evaluation.avgLatencyMs} ms</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 text-slate-500">Training Duration</td>
                    <td className="py-2 px-4 text-right">{formatDuration(comparison.left.meta.durationMs)}</td>
                    <td className="py-2 px-4 text-right">{formatDuration(comparison.right.meta.durationMs)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 text-slate-500">Adapter Size</td>
                    <td className="py-2 px-4 text-right">{formatBytes(comparison.left.meta.adapterSizeBytes)}</td>
                    <td className="py-2 px-4 text-right">{formatBytes(comparison.right.meta.adapterSizeBytes)}</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 text-slate-500">Trainable Params</td>
                    <td className="py-2 px-4 text-right">{formatNumber(comparison.left.meta.trainableParams)}</td>
                    <td className="py-2 px-4 text-right">{formatNumber(comparison.right.meta.trainableParams)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
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
                        {row.leftPrediction}
                      </td>
                      <td
                        className={`py-2 px-4 font-mono ${
                          row.rightCorrect ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'
                        }`}
                      >
                        {row.rightPrediction}
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
