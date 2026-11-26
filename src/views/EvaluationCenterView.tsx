import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Play,
  ShieldAlert,
  Wrench,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { analyzeEvaluationRun } from '../ml/evaluationDiagnostics';
import type { DatasetRecord, EvaluationRunRecord } from '../shared/types';
import type { NavSection } from './OverviewView';

interface EvaluationCenterViewProps {
  initialDatasetId?: string | null;
  onSelectDataset?: (datasetId: string) => void;
  onNavigate?: (section: NavSection) => void;
}

export function EvaluationCenterView({
  initialDatasetId,
  onSelectDataset,
  onNavigate,
}: EvaluationCenterViewProps) {
  const [models, setModels] = useState<Array<{ ref: string; label: string; aiTool?: string }>>([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationRunRecord[]>([]);
  const [selectedModelRef, setSelectedModelRef] = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [activeEval, setActiveEval] = useState<EvaluationRunRecord | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const [mRes, dsList, evalList] = await Promise.all([
        api.getInferenceModels(),
        api.getDatasets(),
        api.getEvaluations(),
      ]);
      setModels(mRes.models);
      setDatasets(dsList);
      setEvaluations(evalList);
      if (!selectedModelRef && mRes.models.length > 0) setSelectedModelRef(mRes.models[0].ref);
      const preferredDatasetId = pickPreferredDatasetId(
        dsList,
        initialDatasetId || selectedDatasetId
      );
      if (preferredDatasetId && preferredDatasetId !== selectedDatasetId) {
        setSelectedDatasetId(preferredDatasetId);
      }
      if (!activeEval && evalList.length > 0) setActiveEval(evalList[0]);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  useEffect(() => {
    if (!initialDatasetId) return;
    if (datasets.some((d) => d.id === initialDatasetId)) {
      setSelectedDatasetId(initialDatasetId);
    } else {
      api
        .getDatasets()
        .then((dsList) => {
          setDatasets(dsList);
          const resolved = pickPreferredDatasetId(dsList, initialDatasetId);
          if (resolved) setSelectedDatasetId(resolved);
        })
        .catch(() => undefined);
    }
  }, [initialDatasetId, datasets]);

  const handleRunEval = async () => {
    if (!selectedModelRef || !selectedDatasetId) return;
    setRunning(true);
    setError(null);
    try {
      const res = await api.runEvaluation(selectedModelRef, selectedDatasetId);
      setActiveEval(res);
      await loadData();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const diagnosis = useMemo(
    () => (activeEval ? analyzeEvaluationRun(activeEval) : null),
    [activeEval]
  );

  const scoreChartData = useMemo(() => {
    if (!activeEval) return [];
    const passRate =
      activeEval.testSampleCount > 0
        ? ((activeEval.testSampleCount - activeEval.invalidOutputCount) / activeEval.testSampleCount) * 100
        : 100;
    const avgConfidence =
      activeEval.samples.length > 0
        ? (activeEval.samples.reduce((sum, s) => sum + s.confidence, 0) / activeEval.samples.length) * 100
        : 0;

    if (activeEval.taskType === 'classification') {
      return [
        {
          metric: 'Accuracy',
          value: Number(((activeEval.accuracy ?? 0) * 100).toFixed(1)),
          detail: 'Share of held-out prompts predicted accurately',
          color: '#3B82F6',
        },
        {
          metric: 'Macro F1',
          value: Number(((activeEval.macroF1 ?? 0) * 100).toFixed(1)),
          detail: 'Balanced F1 across all classes equally',
          color: '#0EA5E9',
        },
        {
          metric: 'Weighted F1',
          value: Number(((activeEval.weightedF1 ?? 0) * 100).toFixed(1)),
          detail: 'Class-support weighted F1 score',
          color: '#6366F1',
        },
        {
          metric: 'Precision',
          value: Number(((activeEval.precision ?? 0) * 100).toFixed(1)),
          detail: 'How often positive predictions are right',
          color: '#10B981',
        },
        {
          metric: 'Valid Format',
          value: Number(passRate.toFixed(1)),
          detail: 'Outputs passing schema / label constraints',
          color: '#14B8A6',
        },
      ];
    }
    return [
      {
        metric: 'Exact Match',
        value: Number(((activeEval.exactMatch ?? 0) * 100).toFixed(1)),
        detail: 'Exact string matches against target completion',
        color: '#3B82F6',
      },
      {
        metric: 'Token F1',
        value: Number(((activeEval.tokenF1 ?? 0) * 100).toFixed(1)),
        detail: 'Word/token overlap similarity with ground truth',
        color: '#0EA5E9',
      },
      {
        metric: 'Confidence',
        value: Number(avgConfidence.toFixed(1)),
        detail: 'Mean model certainty across test prompts',
        color: '#6366F1',
      },
      {
        metric: 'Valid Format',
        value: Number(passRate.toFixed(1)),
        detail: 'Outputs passing length and schema checks',
        color: '#10B981',
      },
    ];
  }, [activeEval]);

  const modelGroups = [
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

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Evaluation Center
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Benchmark models strictly on held-out test splits, diagnose misclassifications and perplexity spikes, and inspect actionable steps to rectify errors.
          </p>
        </div>
        <AiOrb
          state={mapJobRunnerToOrbState({
            error,
            running,
            hasResult: Boolean(activeEval),
          })}
          size="sm"
          showLabel
          label={running ? 'Evaluating' : error ? 'Eval error' : activeEval ? 'Report ready' : 'Idle'}
        />
      </div>

      {/* Runner Bar */}
      <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5 flex flex-col md:flex-row md:items-end gap-4">
        <div className="flex-1">
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Model / Fine-Tuned Adapter
          </label>
          <select
            aria-label="Model or Fine-Tuned Adapter"
            value={selectedModelRef}
            onChange={(e) => setSelectedModelRef(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {modelGroups.map((group) => {
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
            })}
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Held-Out Test Dataset
          </label>
          <select
            aria-label="Held-Out Test Dataset"
            value={selectedDatasetId}
            onChange={(e) => {
              setSelectedDatasetId(e.target.value);
              onSelectDataset?.(e.target.value);
            }}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} (
                {d.splits?.testCount
                  ? `${d.splits.testCount} test samples`
                  : `${d.rowCount} rows · auto-split`}
                )
              </option>
            ))}
          </select>
        </div>
        <LiquidButton
          type="button"
          variant="default"
          disabled={running || !selectedModelRef || !selectedDatasetId}
          onClick={handleRunEval}
        >
          <Play className="w-3.5 h-3.5" />
          {running ? 'Evaluating...' : 'Run Evaluation'}
        </LiquidButton>
      </div>

      {running && (
        <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] px-4 py-3 flex items-center justify-between">
          <AiOrb
            state="tool-call"
            size="md"
            showLabel
            label="Running held-out test inference & computing F1 / perplexity..."
          />
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-500/35 bg-red-50/95 dark:bg-[#220F14]/95 backdrop-blur-md px-4 py-3 text-xs text-red-700 dark:text-red-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!activeEval ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-12 text-center">
          <BarChart3 className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
            No evaluation reports generated yet
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Select a model and test dataset above to compute accuracy, F1, perplexity, confusion matrix, and error rectification diagnostics.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* History Selector */}
          <div className="liquid-glass-surface lg:col-span-3 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 overflow-hidden self-start">
            <div className="px-4 py-3 border-b border-slate-200 dark:border-white/[0.08] text-xs font-semibold">
              Evaluation Runs ({evaluations.length})
            </div>
            <ul className="divide-y divide-slate-200 dark:divide-white/[0.06] max-h-[520px] overflow-y-auto">
              {evaluations.map((ev) => (
                <li key={ev.id}>
                  <button
                    type="button"
                    onClick={() => setActiveEval(ev)}
                    className={`w-full text-left px-4 py-3 text-xs ${
                      ev.id === activeEval.id
                        ? 'bg-blue-500/[0.08]'
                        : 'hover:bg-slate-50 dark:hover:bg-white/[0.02]'
                    }`}
                  >
                    <div className="font-medium text-slate-900 dark:text-slate-100 truncate">
                      {ev.modelName}
                    </div>
                    <div className="text-slate-500 font-mono tabular-nums mt-0.5">
                      {ev.accuracy !== null
                        ? `Acc ${(ev.accuracy * 100).toFixed(1)}%`
                        : `TokenF1 ${((ev.tokenF1 ?? 0) * 100).toFixed(1)}%`}{' '}
                      · PPL {ev.perplexity}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {/* Active Evaluation Report */}
          <div className="lg:col-span-9 space-y-6">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {activeEval.taskType === 'classification' ? (
                <>
                  <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                    <div className="text-xs text-slate-500">Accuracy</div>
                    <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                      {((activeEval.accuracy ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                    <div className="text-xs text-slate-500">Macro F1</div>
                    <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                      {((activeEval.macroF1 ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                    <div className="text-xs text-slate-500">Weighted F1</div>
                    <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                      {((activeEval.weightedF1 ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                    <div className="text-xs text-slate-500">Precision / Recall</div>
                    <div className="mt-1 text-base font-semibold font-mono tabular-nums">
                      {((activeEval.precision ?? 0) * 100).toFixed(1)}% /{' '}
                      {((activeEval.recall ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                    <div className="text-xs text-slate-500">Exact Match</div>
                    <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                      {((activeEval.exactMatch ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                    <div className="text-xs text-slate-500">Token F1</div>
                    <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                      {((activeEval.tokenF1 ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                </>
              )}

              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                <div className="text-xs text-slate-500">Perplexity</div>
                <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                  {activeEval.perplexity}
                </div>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                <div className="text-xs text-slate-500">Avg Test Loss</div>
                <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                  {activeEval.avgLoss.toFixed(4)}
                </div>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                <div className="text-xs text-slate-500">Avg Latency</div>
                <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                  {activeEval.avgLatencyMs} ms
                </div>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-4">
                <div className="text-xs text-slate-500">Invalid Outputs</div>
                <div className="mt-1 text-xl font-semibold font-mono tabular-nums">
                  {activeEval.invalidOutputCount} / {activeEval.testSampleCount}
                </div>
              </div>
            </div>

            {/* Evaluation Score Breakdown Chart */}
            <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    Evaluation Score Breakdown (%)
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Higher bars indicate stronger held-out test quality; the dashed green line marks the 75% benchmark target.
                  </p>
                </div>
                <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                  {activeEval.modelName} · {activeEval.testSampleCount} test rows
                </span>
              </div>
              <div className="h-60 w-full min-w-0">
                <ResponsiveContainer
                  width="100%"
                  height={240}
                  minWidth={0}
                  minHeight={240}
                  initialDimension={{ width: 480, height: 240 }}
                >
                  <BarChart data={scoreChartData} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                    <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <YAxis
                      domain={[0, 100]}
                      width={44}
                      tickFormatter={(v) => `${v}%`}
                      tick={{ fontSize: 11, fill: '#94A3B8' }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const row = payload[0].payload as {
                          metric: string;
                          value: number;
                          detail: string;
                        };
                        return (
                          <div className="rounded-lg border border-white/15 bg-[#11141C]/95 px-3 py-2 text-xs text-slate-100 shadow-lg backdrop-blur-md">
                            <div className="font-semibold text-blue-300">
                              {row.metric}: {row.value}%
                            </div>
                            <div className="mt-0.5 text-[11px] text-slate-300 max-w-xs">
                              {row.detail}
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <ReferenceLine
                      y={75}
                      stroke="#10B981"
                      strokeDasharray="4 4"
                      label={{
                        value: '75% Target',
                        position: 'insideTopLeft',
                        fill: '#10B981',
                        fontSize: 10,
                      }}
                    />
                    <Bar
                      dataKey="value"
                      name="Measured Score (%)"
                      radius={[5, 5, 0, 0]}
                      isAnimationActive={false}
                    >
                      {scoreChartData.map((entry) => (
                        <Cell key={entry.metric} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Error Analysis & Way to Rectify Panel */}
            {diagnosis && (
              <section
                aria-label="Error analysis and rectification"
                className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5 space-y-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-white/[0.06] pb-3">
                  <div className="flex items-center gap-2.5">
                    <div
                      className={`p-2 rounded-lg ${
                        diagnosis.overallHealth === 'critical'
                          ? 'bg-red-500/10 text-red-600 dark:text-red-400'
                          : diagnosis.overallHealth === 'warning'
                            ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
                            : 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                      }`}
                    >
                      {diagnosis.overallHealth === 'healthy' ? (
                        <CheckCircle2 className="w-4 h-4" />
                      ) : (
                        <ShieldAlert className="w-4 h-4" />
                      )}
                    </div>
                    <div>
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                        Error Analysis &amp; How to Rectify
                      </h2>
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        {diagnosis.summary}
                      </p>
                    </div>
                  </div>
                  <span
                    className={`px-2.5 py-1 rounded-md text-[11px] font-mono font-semibold uppercase ${
                      diagnosis.overallHealth === 'critical'
                        ? 'bg-red-500/15 text-red-600 dark:text-red-400 border border-red-500/30'
                        : diagnosis.overallHealth === 'warning'
                          ? 'bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/30'
                          : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border border-emerald-500/30'
                    }`}
                  >
                    {diagnosis.overallHealth} · {diagnosis.mismatchCount} errors
                  </span>
                </div>

                {diagnosis.topConfusions.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="font-medium text-slate-600 dark:text-slate-300">
                      Top Confusion Pairs:
                    </span>
                    {diagnosis.topConfusions.map((pair, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 rounded-md border border-red-500/25 bg-red-500/10 font-mono text-[11px] text-red-600 dark:text-red-300"
                      >
                        {pair.actual} → {pair.predicted} ({pair.count}×)
                      </span>
                    ))}
                  </div>
                )}

                <div className="space-y-3">
                  {diagnosis.diagnostics.map((diag) => (
                    <div
                      key={diag.id}
                      className={`rounded-lg border p-4 space-y-2 ${
                        diag.severity === 'critical'
                          ? 'border-red-500/30 bg-red-500/[0.05]'
                          : diag.severity === 'warning'
                            ? 'border-amber-500/30 bg-amber-500/[0.05]'
                            : 'border-blue-500/25 bg-blue-500/[0.04]'
                      }`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
                          <Wrench className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                          <span>{diag.title}</span>
                        </div>
                        <span className="text-[10px] font-mono uppercase text-slate-500 dark:text-slate-400">
                          {diag.severity}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                        <strong className="text-slate-800 dark:text-slate-100">Why it happened:</strong>{' '}
                        {diag.explanation}
                      </p>
                      <div className="rounded-md border border-slate-200/80 dark:border-white/[0.08] bg-white/80 dark:bg-[#090B10]/70 p-2.5 text-xs text-slate-700 dark:text-slate-200 leading-relaxed">
                        <strong className="text-blue-600 dark:text-blue-400">How to rectify:</strong>{' '}
                        {diag.rectification}
                      </div>
                      {diag.recommendedAction && onNavigate && (
                        <div className="pt-1">
                          <LiquidButton
                            type="button"
                            size="sm"
                            variant="outline"
                            onClick={() => onNavigate(diag.recommendedAction!.targetSection)}
                            className="text-blue-600 dark:text-blue-400 border-blue-500/35"
                          >
                            {diag.recommendedAction.label}
                            <ArrowRight className="w-3.5 h-3.5" />
                          </LiquidButton>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {diagnosis.hardestSamples.length > 0 && (
                  <div className="pt-2 border-t border-slate-200 dark:border-white/[0.06] space-y-2">
                    <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                      Highest-Loss / Failing Samples ({diagnosis.hardestSamples.length})
                    </div>
                    <div className="space-y-1.5">
                      {diagnosis.hardestSamples.map((s) => (
                        <div
                          key={s.index}
                          className="rounded-lg border border-slate-200 dark:border-white/[0.06] bg-slate-50 dark:bg-[#090B10] px-3 py-2 text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                        >
                          <div className="min-w-0 flex-1">
                            <span className="font-mono text-slate-400 mr-2">#{s.index + 1}</span>
                            <span className="text-slate-800 dark:text-slate-200 truncate inline-block max-w-md align-bottom">
                              {s.prompt}
                            </span>
                          </div>
                          <div className="font-mono text-[11px] shrink-0 flex items-center gap-2">
                            <span className="text-emerald-600 dark:text-emerald-400">
                              gt:{s.groundTruth}
                            </span>
                            <span className="text-slate-400">vs</span>
                            <span className="text-red-500">pred:{s.prediction}</span>
                            <span className="text-slate-500">loss {s.loss.toFixed(2)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* Confusion Matrix */}
            {activeEval.confusionMatrix && (
              <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5 overflow-x-auto">
                <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                  Confusion Matrix (Rows = Ground Truth, Columns = Predicted)
                </h3>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 mb-3">
                  Blue diagonal cells count accurate predictions; red off-diagonal cells highlight which labels the model confused.
                </p>
                <table className="text-xs font-mono tabular-nums border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-slate-500">Actual \ Pred</th>
                      {activeEval.confusionMatrix.labels.map((lbl) => (
                        <th key={lbl} className="p-2 text-center text-slate-500">
                          {lbl}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {activeEval.confusionMatrix.labels.map((rowLbl, rIdx) => (
                      <tr key={rowLbl}>
                        <td className="p-2 font-medium text-slate-700 dark:text-slate-300">
                          {rowLbl}
                        </td>
                        {activeEval.confusionMatrix!.matrix[rIdx].map((val, cIdx) => (
                          <td
                            key={cIdx}
                            className={`p-2 text-center border border-slate-200 dark:border-white/[0.06] ${
                              rIdx === cIdx && val > 0
                                ? 'bg-blue-500/20 text-blue-600 dark:text-blue-300 font-semibold'
                                : val > 0
                                  ? 'bg-red-500/15 text-red-600 dark:text-red-300'
                                  : 'text-slate-400'
                            }`}
                          >
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Per-sample inspection */}
            <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08] text-xs font-semibold">
                Test Sample Predictions ({activeEval.samples.length})
              </div>
              <div className="overflow-x-auto max-h-80">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500">
                    <tr>
                      <th className="py-2 px-4 font-medium">Prompt</th>
                      <th className="py-2 px-4 font-medium">Ground Truth</th>
                      <th className="py-2 px-4 font-medium">Prediction</th>
                      <th className="py-2 px-4 font-medium text-right">Conf</th>
                      <th className="py-2 px-4 font-medium text-right">Result</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                    {activeEval.samples.map((s) => (
                      <tr key={s.index}>
                        <td className="py-2 px-4 max-w-xs truncate text-slate-700 dark:text-slate-300">
                          {s.prompt}
                        </td>
                        <td className="py-2 px-4 font-mono">{s.groundTruth}</td>
                        <td className="py-2 px-4 font-mono">{s.prediction}</td>
                        <td className="py-2 px-4 text-right font-mono tabular-nums">
                          {(s.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 px-4 text-right font-medium">
                          <span
                            className={
                              s.correct
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-red-600 dark:text-red-400'
                            }
                          >
                            {s.correct ? 'Pass' : 'Mismatch'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

