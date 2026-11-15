import { useEffect, useState } from 'react';
import { AlertTriangle, BarChart3, Play } from 'lucide-react';
import { LatticeLoader } from '../components/reactbits/LatticeLoader';
import { api } from '../lib/api';
import type { DatasetRecord, EvaluationRunRecord } from '../shared/types';

export function EvaluationCenterView() {
  const [models, setModels] = useState<Array<{ ref: string; label: string }>>([]);
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
      const readyDs = dsList.filter((d) => d.status === 'training_ready');
      setDatasets(readyDs);
      setEvaluations(evalList);
      if (!selectedModelRef && mRes.models.length > 0) setSelectedModelRef(mRes.models[0].ref);
      if (!selectedDatasetId && readyDs.length > 0) setSelectedDatasetId(readyDs[0].id);
      if (!activeEval && evalList.length > 0) setActiveEval(evalList[0]);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
          Evaluation Center
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Benchmark models strictly on held-out test splits. Validation and test sets remain isolated to prevent leakage.
        </p>
      </div>

      {/* Runner Bar */}
      <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 flex flex-col md:flex-row md:items-end gap-4">
        <div className="flex-1">
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Model / Fine-Tuned Adapter
          </label>
          <select
            value={selectedModelRef}
            onChange={(e) => setSelectedModelRef(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {models.map((m) => (
              <option key={m.ref} value={m.ref}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
            Held-Out Test Dataset
          </label>
          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.splits?.testCount ?? 0} test samples)
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          disabled={running || !selectedModelRef || !selectedDatasetId}
          onClick={handleRunEval}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 whitespace-nowrap"
        >
          <Play className="w-3.5 h-3.5" />
          {running ? 'Evaluating...' : 'Run Evaluation'}
        </button>
      </div>

      {running && (
        <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] px-4 py-3 flex items-center justify-between">
          <LatticeLoader
            pattern="ripple"
            grid={3}
            cellSize={5}
            gap={2}
            fontSize={12}
            color="#3B82F6"
            status="working"
            label="Running held-out test inference & computing F1 / perplexity..."
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

      {!activeEval ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-12 text-center">
          <BarChart3 className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
            No evaluation reports generated yet
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Select a model and test dataset above to compute accuracy, F1, perplexity, and confusion matrix.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* History Selector */}
          <div className="lg:col-span-3 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden self-start">
            <div className="px-4 py-3 border-b border-slate-200 dark:border-white/[0.08] text-xs font-semibold">
              Evaluation Runs ({evaluations.length})
            </div>
            <ul className="divide-y divide-slate-200 dark:divide-white/[0.06]">
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

            {/* Confusion Matrix */}
            {activeEval.confusionMatrix && (
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 overflow-x-auto">
                <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100 mb-3">
                  Confusion Matrix (Rows = Ground Truth, Columns = Predicted)
                </h3>
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
            <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
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
