import { useEffect, useRef, useState } from 'react';
import {
  Download,
  Pause,
  Play,
  Square,
  Terminal,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { LatticeLoader } from '../components/reactbits/LatticeLoader';
import { api } from '../lib/api';
import { formatBytes, formatDuration } from '../lib/utils';
import type {
  ExperimentRecord,
  HardwareTelemetry,
  TrainingMetricPoint,
} from '../shared/types';

interface TrainingMonitorViewProps {
  activeExperimentId: string | null;
  onSelectExperiment: (id: string) => void;
}

export function TrainingMonitorView({
  activeExperimentId,
  onSelectExperiment,
}: TrainingMonitorViewProps) {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [selectedExp, setSelectedExp] = useState<ExperimentRecord | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetricPoint[]>([]);
  const [checkpoints, setCheckpoints] = useState<
    Array<{
      id: string;
      step: number;
      epoch: number;
      trainLoss: number;
      valLoss: number | null;
      filePath: string;
      sizeBytes: number;
      createdAt: string;
    }>
  >([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [hardware, setHardware] = useState<HardwareTelemetry | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  const loadListAndDetail = async (targetId?: string | null) => {
    const list = await api.getExperiments();
    setExperiments(list);
    const id = targetId || activeExperimentId || list[0]?.id || null;
    if (!id) {
      setSelectedExp(null);
      setMetrics([]);
      setCheckpoints([]);
      setLogs([]);
      return;
    }
    if (id !== activeExperimentId) onSelectExperiment(id);
    const detail = await api.getExperimentDetail(id);
    setSelectedExp(detail.experiment);
    setMetrics(detail.metrics);
    setCheckpoints(detail.checkpoints);
    setLogs(detail.logs);
    setHardware(detail.hardware);
  };

  useEffect(() => {
    void loadListAndDetail(activeExperimentId);
  }, [activeExperimentId]);

  // Connect live SSE stream while experiment is active
  useEffect(() => {
    if (!selectedExp) return;
    if (
      selectedExp.status !== 'preparing' &&
      selectedExp.status !== 'running' &&
      selectedExp.status !== 'paused'
    ) {
      return;
    }

    const es = new EventSource(`/api/training/${selectedExp.id}/stream`);

    es.addEventListener('metric', (e) => {
      const point = JSON.parse((e as MessageEvent).data) as TrainingMetricPoint;
      setMetrics((prev) => {
        if (prev.some((p) => p.step === point.step)) return prev;
        return [...prev, point];
      });
      setSelectedExp((prev) =>
        prev
          ? {
              ...prev,
              currentEpoch: point.epoch,
              currentStep: point.step,
              trainLoss: point.trainLoss,
              valLoss: point.valLoss ?? prev.valLoss,
            }
          : prev
      );
    });

    es.addEventListener('log', (e) => {
      const { line } = JSON.parse((e as MessageEvent).data) as { line: string };
      setLogs((prev) => [...prev, line]);
    });

    es.addEventListener('status', (e) => {
      const payload = JSON.parse((e as MessageEvent).data) as { status: ExperimentRecord['status'] };
      setSelectedExp((prev) => (prev ? { ...prev, status: payload.status } : prev));
      if (
        payload.status === 'completed' ||
        payload.status === 'failed' ||
        payload.status === 'cancelled'
      ) {
        es.close();
        void loadListAndDetail(selectedExp.id);
      }
    });

    return () => {
      es.close();
    };
  }, [selectedExp?.id, selectedExp?.status]);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs.length]);

  const progressPct =
    selectedExp && selectedExp.totalSteps > 0
      ? Math.min(100, Math.round((selectedExp.currentStep / selectedExp.totalSteps) * 100))
      : 0;

  const latestMetric = metrics[metrics.length - 1];
  const elapsedMs = selectedExp?.durationMs
    ? selectedExp.durationMs
    : selectedExp?.startedAt
      ? Date.now() - new Date(selectedExp.startedAt).getTime()
      : 0;

  const etaMs =
    selectedExp &&
    selectedExp.status === 'running' &&
    selectedExp.currentStep > 0 &&
    selectedExp.totalSteps > selectedExp.currentStep
      ? Math.round(
          (elapsedMs / selectedExp.currentStep) *
            (selectedExp.totalSteps - selectedExp.currentStep)
        )
      : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Training Monitor
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Live Server-Sent Events (SSE) telemetry for loss convergence, learning rate schedule, checkpoints, and hardware usage.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {experiments.length > 0 && (
            <select
              aria-label="Select experiment to monitor"
              value={selectedExp?.id || ''}
              onChange={(e) => onSelectExperiment(e.target.value)}
              className="px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C]"
            >
              {experiments.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  {exp.name} ({exp.status})
                </option>
              ))}
            </select>
          )}
          {selectedExp && (
            <a
              href={`/api/training/${selectedExp.id}/logs?download=1`}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] hover:bg-slate-50 dark:hover:bg-white/[0.04]"
            >
              <Download className="w-3.5 h-3.5" />
              Download Logs
            </a>
          )}
        </div>
      </div>

      {!selectedExp ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-12 text-center">
          <Terminal className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
            No training runs to monitor
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Launch an experiment from Fine-Tuning Studio to stream real-time step metrics.
          </p>
        </div>
      ) : (
        <>
          {/* Live Status & Controls */}
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <div className="flex items-center gap-3 text-xs font-mono text-slate-500 dark:text-slate-400">
                  <span>{selectedExp.modelName}</span>
                  <span>·</span>
                  <span className="uppercase">{selectedExp.method}</span>
                  <span>·</span>
                  <span>{selectedExp.datasetName}</span>
                </div>
                <h2 className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                  {selectedExp.name}
                </h2>
              </div>

              <div className="flex items-center gap-3 shrink-0">
                <LatticeLoader
                  pattern="orbit"
                  grid={3}
                  cellSize={5}
                  gap={2}
                  fontSize={12}
                  color="#3B82F6"
                  doneColor="#10B981"
                  errorColor="#EF4444"
                  label={selectedExp.status === 'preparing' ? 'Preparing' : selectedExp.status === 'paused' ? 'Paused' : 'Training'}
                  doneLabel="Completed"
                  errorLabel={selectedExp.status === 'cancelled' ? 'Cancelled' : 'Failed'}
                  status={
                    selectedExp.status === 'completed'
                      ? 'done'
                      : selectedExp.status === 'failed' || selectedExp.status === 'cancelled'
                        ? 'error'
                        : 'working'
                  }
                />
                {selectedExp.status === 'running' && (
                  <button
                    type="button"
                    onClick={() => api.pauseTraining(selectedExp.id).then(() => loadListAndDetail(selectedExp.id))}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] hover:bg-slate-50 dark:hover:bg-white/[0.04]"
                  >
                    <Pause className="w-3.5 h-3.5" />
                    Pause
                  </button>
                )}
                {selectedExp.status === 'paused' && (
                  <button
                    type="button"
                    onClick={() => api.resumeTraining(selectedExp.id).then(() => loadListAndDetail(selectedExp.id))}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500"
                  >
                    <Play className="w-3.5 h-3.5" />
                    Resume
                  </button>
                )}
                {(selectedExp.status === 'running' ||
                  selectedExp.status === 'paused' ||
                  selectedExp.status === 'preparing') && (
                  <button
                    type="button"
                    onClick={() => api.cancelTraining(selectedExp.id).then(() => loadListAndDetail(selectedExp.id))}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-red-500/30 text-red-600 dark:text-red-400 hover:bg-red-500/10"
                  >
                    <Square className="w-3.5 h-3.5" />
                    Stop
                  </button>
                )}
              </div>
            </div>

            {/* Progress Bar */}
            <div>
              <div className="flex items-center justify-between text-xs font-mono tabular-nums mb-1.5">
                <span className="capitalize font-medium text-blue-600 dark:text-blue-400">
                  Status: {selectedExp.status}
                </span>
                <span className="text-slate-500 dark:text-slate-400">
                  Step {selectedExp.currentStep}/{selectedExp.totalSteps} ({progressPct}%) · Epoch{' '}
                  {selectedExp.currentEpoch}/{selectedExp.config.epochs}
                </span>
              </div>
              <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-white/[0.06] overflow-hidden">
                <div
                  className="h-full bg-blue-600 transition-all duration-200"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>

            {/* KPI Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 pt-2 border-t border-slate-200 dark:border-white/[0.06] text-xs">
              <div>
                <div className="text-slate-500 dark:text-slate-400">Train Loss</div>
                <div className="mt-0.5 text-sm font-mono tabular-nums font-semibold">
                  {selectedExp.trainLoss !== null ? selectedExp.trainLoss.toFixed(4) : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Val Loss</div>
                <div className="mt-0.5 text-sm font-mono tabular-nums font-semibold">
                  {selectedExp.valLoss !== null ? selectedExp.valLoss.toFixed(4) : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Learning Rate</div>
                <div className="mt-0.5 text-sm font-mono tabular-nums font-semibold">
                  {latestMetric ? latestMetric.learningRate.toExponential(2) : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Elapsed / ETA</div>
                <div className="mt-0.5 text-sm font-mono tabular-nums font-semibold">
                  {formatDuration(elapsedMs)} {etaMs !== null ? `/ ~${formatDuration(etaMs)}` : ''}
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">Process Memory</div>
                <div className="mt-0.5 text-sm font-mono tabular-nums font-semibold">
                  {latestMetric ? `${latestMetric.memoryMb} MB` : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500 dark:text-slate-400">GPU Telemetry</div>
                <div className="mt-0.5 text-sm font-mono tabular-nums font-semibold">
                  {hardware?.gpuAvailable ? `${hardware.gpuUtilizationPct}%` : 'Unavailable'}
                </div>
              </div>
            </div>
          </div>

          {/* Live Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">
                Training &amp; Validation Loss
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                    <XAxis dataKey="step" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <YAxis tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#11141C',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="trainLoss"
                      name="Train Loss"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="valLoss"
                      name="Val Loss"
                      stroke="#10B981"
                      strokeWidth={2}
                      connectNulls
                      dot={{ r: 3 }}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">
                Learning Rate Schedule &amp; Process Memory (MB)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                    <XAxis dataKey="step" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <YAxis yAxisId="left" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#11141C',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="learningRate"
                      name="Learning Rate"
                      stroke="#0EA5E9"
                      fill="rgba(14,165,233,0.12)"
                      isAnimationActive={false}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="memoryMb"
                      name="Memory (MB)"
                      stroke="#A855F7"
                      dot={false}
                      isAnimationActive={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Checkpoints & Streaming Logs */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08]">
                <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                  Saved Checkpoints ({checkpoints.length})
                </h3>
              </div>
              {checkpoints.length === 0 ? (
                <div className="p-6 text-xs text-slate-500">
                  Checkpoints are written at epoch boundaries.
                </div>
              ) : (
                <table className="w-full text-left text-xs font-mono tabular-nums">
                  <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500">
                    <tr>
                      <th className="py-2 px-4 font-medium">Step</th>
                      <th className="py-2 px-4 font-medium">Epoch</th>
                      <th className="py-2 px-4 font-medium text-right">Val Loss</th>
                      <th className="py-2 px-4 font-medium text-right">Size</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                    {checkpoints.map((c) => (
                      <tr key={c.id}>
                        <td className="py-2 px-4">{c.step}</td>
                        <td className="py-2 px-4">{c.epoch}</td>
                        <td className="py-2 px-4 text-right">
                          {c.valLoss !== null ? c.valLoss.toFixed(4) : '—'}
                        </td>
                        <td className="py-2 px-4 text-right">{formatBytes(c.sizeBytes)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            <div className="lg:col-span-7 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-[#0B0E14] text-slate-200 overflow-hidden">
              <div className="px-4 py-2.5 border-b border-white/[0.08] flex items-center justify-between text-xs font-mono">
                <span>Training Logs</span>
                <span className="text-slate-400">{logs.length} lines</span>
              </div>
              <div
                ref={logContainerRef}
                className="p-4 h-64 overflow-y-auto font-mono text-[11px] leading-relaxed space-y-1"
              >
                {logs.length === 0 ? (
                  <div className="text-slate-500">Waiting for worker logs...</div>
                ) : (
                  logs.map((line, i) => (
                    <div key={i} className="whitespace-pre-wrap break-all">
                      {line}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
