import { useEffect, useState } from 'react';
import { AlertTriangle, Database, FlaskConical, History, Search } from 'lucide-react';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { api } from '../lib/api';
import { formatDuration, formatNumber } from '../lib/utils';
import type { DatasetRecord, ExperimentRecord } from '../shared/types';
import type { NavSection } from './OverviewView';

interface ExperimentHistoryViewProps {
  activeDataset?: DatasetRecord | null;
  onOpenMonitor: (experimentId: string) => void;
  onNavigate?: (section: NavSection) => void;
}

export function ExperimentHistoryView({
  activeDataset,
  onOpenMonitor,
  onNavigate,
}: ExperimentHistoryViewProps) {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [query, setQuery] = useState('');
  const [methodFilter, setMethodFilter] = useState<string>('all');
  const [onlyActiveDataset, setOnlyActiveDataset] = useState(false);
  const [selected, setSelected] = useState<ExperimentRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getExperiments()
      .then((list) => {
        setExperiments(list);
        setError(null);
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  const filtered = experiments.filter((exp) => {
    const matchesQuery =
      !query.trim() ||
      exp.name.toLowerCase().includes(query.toLowerCase()) ||
      exp.modelName.toLowerCase().includes(query.toLowerCase()) ||
      exp.datasetName.toLowerCase().includes(query.toLowerCase());
    const matchesMethod = methodFilter === 'all' || exp.method === methodFilter;
    const matchesDataset =
      !onlyActiveDataset || !activeDataset || exp.datasetId === activeDataset.id;
    return matchesQuery && matchesMethod && matchesDataset;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Experiment History
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Full reproducible ledger of training runs, seeds, hyperparameter configs, and final losses.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {activeDataset && (
            <button
              type="button"
              onClick={() => setOnlyActiveDataset((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-3 py-2 text-xs rounded-lg border transition-colors ${
                onlyActiveDataset
                  ? 'border-blue-500 bg-blue-500/15 text-blue-600 dark:text-blue-400 font-medium'
                  : 'border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-white/[0.04]'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              Active: {activeDataset.name}
            </button>
          )}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search experiments..."
              className="pl-8 pr-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C]"
            />
          </div>
          <select
            aria-label="Filter by method"
            value={methodFilter}
            onChange={(e) => setMethodFilter(e.target.value)}
            className="px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C]"
          >
            <option value="all">All Methods</option>
            <option value="lora">LoRA</option>
            <option value="qlora">QLoRA</option>
            <option value="full">Full FT</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-12 text-center space-y-3">
          <History className="w-8 h-8 text-slate-400 mx-auto" />
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
            {onlyActiveDataset && activeDataset
              ? `No experiments logged yet for "${activeDataset.name}"`
              : 'No matching experiments'}
          </p>
          {onlyActiveDataset && activeDataset && onNavigate && (
            <div className="flex items-center justify-center gap-2">
              <LiquidButton type="button" size="sm" variant="default" onClick={() => onNavigate('studio')}>
                <FlaskConical className="w-3.5 h-3.5" />
                Train on {activeDataset.name}
              </LiquidButton>
              <button
                type="button"
                onClick={() => setOnlyActiveDataset(false)}
                className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] text-slate-600 dark:text-slate-300"
              >
                Show All Runs
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 dark:border-white/[0.08] text-slate-500">
                <tr>
                  <th className="py-3 px-4 font-medium">Experiment</th>
                  <th className="py-3 px-4 font-medium">Model / Method</th>
                  <th className="py-3 px-4 font-medium">Status</th>
                  <th className="py-3 px-4 font-medium text-right">Train / Val Loss</th>
                  <th className="py-3 px-4 font-medium text-right">Duration</th>
                  <th className="py-3 px-4 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                {filtered.map((exp) => (
                  <tr
                    key={exp.id}
                    onClick={() => setSelected(exp)}
                    className={`cursor-pointer ${
                      selected?.id === exp.id
                        ? 'bg-blue-500/[0.06]'
                        : 'hover:bg-slate-50 dark:hover:bg-white/[0.02]'
                    }`}
                  >
                    <td className="py-3 px-4">
                      <div className="font-medium text-slate-900 dark:text-slate-100">
                        {exp.name}
                      </div>
                      <div className="text-[11px] text-slate-500">{exp.datasetName}</div>
                    </td>
                    <td className="py-3 px-4 font-mono">
                      <div>{exp.modelName}</div>
                      <div className="text-[11px] uppercase text-slate-500">
                        {exp.method} · r={exp.config.loraR} · seed={exp.config.seed}
                      </div>
                    </td>
                    <td className="py-3 px-4 font-medium capitalize">{exp.status}</td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums">
                      {exp.trainLoss !== null ? exp.trainLoss.toFixed(4) : '—'} /{' '}
                      {exp.valLoss !== null ? exp.valLoss.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono tabular-nums">
                      {formatDuration(exp.durationMs)}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onOpenMonitor(exp.id);
                        }}
                        className="text-blue-600 dark:text-blue-400 hover:underline"
                      >
                        Monitor
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="lg:col-span-4 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 self-start">
            <h2 className="text-xs font-semibold text-slate-900 dark:text-slate-100 mb-3">
              Reproducibility Manifest
            </h2>
            {!selected ? (
              <p className="text-xs text-slate-500">
                Click an experiment row to inspect saved seed, parameters, and artifact directory.
              </p>
            ) : (
              <div className="space-y-3 text-xs">
                <div>
                  <div className="text-slate-500">Artifact Directory</div>
                  <div className="font-mono text-[11px] break-all mt-0.5">{selected.artifactDir}</div>
                </div>
                <div>
                  <div className="text-slate-500">Trainable Parameters</div>
                  <div className="font-mono tabular-nums mt-0.5">
                    {formatNumber(selected.trainableParams)} ({selected.trainablePercent}%)
                  </div>
                </div>
                {selected.quantizationRmse !== null && (
                  <div>
                    <div className="text-slate-500">4-Bit NF4 Quantization RMSE</div>
                    <div className="font-mono tabular-nums mt-0.5">
                      {selected.quantizationRmse.toFixed(6)}
                    </div>
                  </div>
                )}
                <pre className="p-3 rounded-lg bg-slate-100 dark:bg-[#090B10] text-[11px] font-mono overflow-x-auto max-h-64">
                  {JSON.stringify(selected.config, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
