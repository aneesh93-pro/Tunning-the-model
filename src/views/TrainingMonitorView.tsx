import { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Database,
  Download,
  FlaskConical,
  Pause,
  Play,
  Square,
  Terminal,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AiOrb } from '@/components/ui/ai-orb';
import { LiquidButton, MetalButton } from '@/components/ui/liquid-glass-button';
import { mapExperimentStatusToOrbState } from '@/utils/aiOrbState';
import { api } from '../lib/api';
import { formatBytes, formatDuration } from '../lib/utils';
import type {
  DatasetRecord,
  ExperimentRecord,
  HardwareTelemetry,
  TrainingMetricPoint,
} from '../shared/types';
import type { NavSection } from './OverviewView';

interface TrainingMonitorViewProps {
  activeExperimentId: string | null;
  activeDataset?: DatasetRecord | null;
  onSelectExperiment: (id: string) => void;
  onNavigate?: (section: NavSection) => void;
}

export function TrainingMonitorView({
  activeExperimentId,
  activeDataset,
  onSelectExperiment,
  onNavigate,
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const skipNextFetchIdRef = useRef<string | null>(null);

  const loadListAndDetail = async (targetId?: string | null) => {
    setLoading((prev) => (selectedExp ? prev : true));
    try {
      const list = await api.getExperiments();
      setExperiments(list);
      const id = targetId || activeExperimentId || list[0]?.id || null;
      if (!id) {
        setSelectedExp(null);
        setMetrics([]);
        setCheckpoints([]);
        setLogs([]);
        setError(null);
        return;
      }
      const detail = await api.getExperimentDetail(id);
      setSelectedExp(detail.experiment);
      setMetrics(detail.metrics);
      setCheckpoints(detail.checkpoints);
      setLogs(detail.logs);
      setHardware(detail.hardware);
      setError(null);
      if (!activeExperimentId && id !== activeExperimentId) {
        skipNextFetchIdRef.current = id;
        onSelectExperiment(id);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeExperimentId && skipNextFetchIdRef.current === activeExperimentId) {
      skipNextFetchIdRef.current = null;
      return;
    }
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

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {activeDataset && (
        <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] px-4 py-2.5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs">
          <div className="flex items-center gap-2 min-w-0">
            <Database className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
            <span className="text-slate-700 dark:text-slate-200 truncate">
              {selectedExp && selectedExp.datasetId === activeDataset.id ? (
                <>
                  Monitoring active workspace dataset:{' '}
                  <strong className="font-semibold text-slate-900 dark:text-slate-100">
                    {activeDataset.name}
                  </strong>
                </>
              ) : (
                <>
                  Active workspace dataset:{' '}
                  <strong className="font-semibold text-slate-900 dark:text-slate-100">
                    {activeDataset.name}
                  </strong>
                  {selectedExp ? ` (currently viewing run on ${selectedExp.datasetName})` : ''}
                </>
              )}
            </span>
          </div>
          {onNavigate && (!selectedExp || selectedExp.datasetId !== activeDataset.id) && (
            <button
              type="button"
              onClick={() => onNavigate('studio')}
              className="inline-flex items-center gap-1.5 text-blue-600 dark:text-blue-400 font-medium hover:underline shrink-0 self-start sm:self-auto"
            >
              <FlaskConical className="w-3.5 h-3.5" />
              Train on {activeDataset.name}
            </button>
          )}
        </div>
      )}

      {loading && !selectedExp ? (
        <div className="space-y-4 animate-pulse">
          <div className="h-28 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C]" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-64 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C]" />
            <div className="h-64 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C]" />
          </div>
        </div>
      ) : !selectedExp ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-12 text-center space-y-3">
          <Terminal className="w-8 h-8 text-slate-400 mx-auto" />
          <div>
            <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
              No training runs to monitor
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {activeDataset
                ? `Launch an experiment on "${activeDataset.name}" from Fine-Tuning Studio to stream real-time step metrics.`
                : 'Launch an experiment from Fine-Tuning Studio to stream real-time step metrics.'}
            </p>
          </div>
          {onNavigate && (
            <div className="pt-1">
              <LiquidButton type="button" size="sm" variant="default" onClick={() => onNavigate('studio')}>
                <FlaskConical className="w-3.5 h-3.5" />
                Open Fine-Tuning Studio
              </LiquidButton>
            </div>
          )}
        </div>
      ) : (
        <>
          {/* Live Status & Controls */}
          <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5 space-y-4">
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
                <AiOrb
                  state={mapExperimentStatusToOrbState(selectedExp.status)}
                  size="md"
                  showLabel
                  label={
                    selectedExp.status === 'preparing'
                      ? 'Preparing'
                      : selectedExp.status === 'running'
                        ? 'Training'
                        : selectedExp.status === 'paused'
                          ? 'Paused'
                          : selectedExp.status === 'completed'
                            ? 'Completed'
                            : selectedExp.status === 'cancelled'
                              ? 'Cancelled'
                              : 'Failed'
                  }
                />
                {selectedExp.status === 'running' && (
                  <LiquidButton
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      api
                        .pauseTraining(selectedExp.id)
                        .then(() => loadListAndDetail(selectedExp.id))
                        .catch((err) => setError((err as Error).message))
                    }
                  >
                    <Pause className="w-3.5 h-3.5" />
                    Pause
                  </LiquidButton>
                )}
                {selectedExp.status === 'paused' && (
                  <MetalButton
                    type="button"
                    variant="success"
                    onClick={() =>
                      api
                        .resumeTraining(selectedExp.id)
                        .then(() => loadListAndDetail(selectedExp.id))
                        .catch((err) => setError((err as Error).message))
                    }
                  >
                    <Play className="w-3.5 h-3.5" />
                    Resume
                  </MetalButton>
                )}
                {(selectedExp.status === 'running' ||
                  selectedExp.status === 'paused' ||
                  selectedExp.status === 'preparing') && (
                  <MetalButton
                    type="button"
                    variant="error"
                    onClick={() =>
                      api
                        .cancelTraining(selectedExp.id)
                        .then(() => loadListAndDetail(selectedExp.id))
                        .catch((err) => setError((err as Error).message))
                    }
                  >
                    <Square className="w-3.5 h-3.5" />
                    Stop
                  </MetalButton>
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
            <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5">
              <div className="mb-3">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Training &amp; Validation Loss
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Lower loss = better fit. Both Train (blue) and Validation (green) should decrease together; rising Val Loss signals overfitting.
                </p>
              </div>
              <div className="h-64 w-full min-w-0">
                <ResponsiveContainer
                  width="100%"
                  height={256}
                  minWidth={0}
                  minHeight={256}
                  initialDimension={{ width: 480, height: 256 }}
                >
                  <LineChart data={metrics} margin={{ top: 8, right: 12, left: 0, bottom: 2 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                    <XAxis
                      dataKey="step"
                      tickFormatter={(v) => `S${v}`}
                      tick={{ fontSize: 11, fill: '#94A3B8' }}
                    />
                    <YAxis width={42} tick={{ fontSize: 11, fill: '#94A3B8' }} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const pt = payload[0].payload as TrainingMetricPoint;
                        return (
                          <div className="rounded-lg border border-white/15 bg-[#11141C]/95 px-3 py-2 text-xs text-slate-100 shadow-lg backdrop-blur-md space-y-0.5">
                            <div className="font-semibold text-slate-200">
                              Step {pt.step} · Epoch {pt.epoch}
                            </div>
                            <div className="font-mono text-blue-300">
                              Train Loss: {pt.trainLoss.toFixed(4)}
                            </div>
                            {pt.valLoss !== null && (
                              <div className="font-mono text-emerald-400">
                                Val Loss: {pt.valLoss.toFixed(4)}
                              </div>
                            )}
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
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

            <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5">
              <div className="mb-3">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Learning Rate Schedule &amp; Process Memory (MB)
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Blue curve shows warmup then cosine decay for smooth convergence; purple line tracks trainer RAM footprint.
                </p>
              </div>
              <div className="h-64 w-full min-w-0">
                <ResponsiveContainer
                  width="100%"
                  height={256}
                  minWidth={0}
                  minHeight={256}
                  initialDimension={{ width: 480, height: 256 }}
                >
                  <AreaChart data={metrics} margin={{ top: 8, right: 10, left: 0, bottom: 2 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                    <XAxis
                      dataKey="step"
                      tickFormatter={(v) => `S${v}`}
                      tick={{ fontSize: 11, fill: '#94A3B8' }}
                    />
                    <YAxis
                      yAxisId="left"
                      width={52}
                      tickFormatter={(v) => Number(v).toExponential(0)}
                      tick={{ fontSize: 10, fill: '#94A3B8' }}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      width={46}
                      tickFormatter={(v) => `${v}MB`}
                      tick={{ fontSize: 10, fill: '#94A3B8' }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const pt = payload[0].payload as TrainingMetricPoint;
                        return (
                          <div className="rounded-lg border border-white/15 bg-[#11141C]/95 px-3 py-2 text-xs text-slate-100 shadow-lg backdrop-blur-md space-y-0.5">
                            <div className="font-semibold text-slate-200">
                              Step {pt.step} · {pt.tokensPerSec} tok/s
                            </div>
                            <div className="font-mono text-sky-300">
                              Learning Rate: {pt.learningRate.toExponential(2)}
                            </div>
                            <div className="font-mono text-purple-300">
                              Process RSS: {pt.memoryMb} MB
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
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
            <div className="liquid-glass-surface lg:col-span-5 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 overflow-hidden">
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
                  <div className="text-slate-500">
                    {selectedExp.status === 'completed' ||
                    selectedExp.status === 'failed' ||
                    selectedExp.status === 'cancelled'
                      ? `No persisted log lines for this ${selectedExp.status} run.`
                      : 'Waiting for worker logs...'}
                  </div>
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
