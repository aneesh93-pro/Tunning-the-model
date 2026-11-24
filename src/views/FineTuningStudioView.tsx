import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Cpu, Layers, Play, ShieldAlert } from 'lucide-react';
import { RubberSegment } from '../components/reactbits/RubberSegment';
import { AiOrb } from '@/components/ui/ai-orb';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { api } from '../lib/api';
import { useIsDark } from '../lib/useDarkMode';
import { pickPreferredDatasetId } from '../utils/datasetSelection';
import { estimateTrainingMemory, getModelById, PRETRAINED_MODELS } from '../ml/modelCatalog';
import { formatDuration, formatNumber } from '../lib/utils';
import type {
  DatasetRecord,
  HardwareTelemetry,
  TrainingConfig,
  TrainingMethod,
} from '../shared/types';
import type { NavSection } from './OverviewView';

interface FineTuningStudioViewProps {
  initialDatasetId: string | null;
  onSelectDataset?: (datasetId: string) => void;
  initialModelId: string;
  onTrainingStarted: (experimentId: string) => void;
  onNavigate: (section: NavSection) => void;
}

export function FineTuningStudioView({
  initialDatasetId,
  onSelectDataset,
  initialModelId,
  onTrainingStarted,
}: FineTuningStudioViewProps) {
  const isDark = useIsDark();
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [hardware, setHardware] = useState<HardwareTelemetry | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [experimentName, setExperimentName] = useState('Banking Intent LoRA v1');
  const [description, setDescription] = useState(
    'Supervised fine-tuning with low-rank adaptation on held-out split.'
  );
  const [datasetId, setDatasetId] = useState(initialDatasetId || '');
  const [modelId, setModelId] = useState(initialModelId || PRETRAINED_MODELS[1].id);
  const [method, setMethod] = useState<TrainingMethod>('lora');

  // LoRA defaults (r=8, alpha=16, dropout=0.05, bias=none)
  const [loraR, setLoraR] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [loraDropout, setLoraDropout] = useState(0.05);
  const [bias, setBias] = useState<'none' | 'all' | 'lora_only'>('none');
  const [targetModules, setTargetModules] = useState<string[]>(['q_proj', 'v_proj']);

  // QLoRA defaults
  const [quantType, setQuantType] = useState<'nf4' | 'fp4'>('nf4');
  const [doubleQuant, setDoubleQuant] = useState(true);
  const [computeDtype, setComputeDtype] = useState<'bfloat16' | 'float16' | 'float32'>('bfloat16');
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true);

  // Optimization defaults
  const [learningRate, setLearningRate] = useState(0.035);
  const [epochs, setEpochs] = useState(14);
  const [batchSize, setBatchSize] = useState(4);
  const [gradientAccumulation, setGradientAccumulation] = useState(1);
  const [maxSeqLength, setMaxSeqLength] = useState(256);
  const [weightDecay, setWeightDecay] = useState(0.005);
  const [warmupRatio, setWarmupRatio] = useState(0.1);
  const [optimizer, setOptimizer] = useState<'adamw' | 'sgd'>('adamw');
  const [scheduler, setScheduler] = useState<'cosine' | 'linear' | 'constant'>('cosine');
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(4);
  const [seed, setSeed] = useState(42);

  useEffect(() => {
    Promise.all([api.getDatasets(), api.getAvailableModels()])
      .then(([ds, avail]) => {
        setDatasets(ds);
        setHardware(avail.hardware);
        const preferred = pickPreferredDatasetId(ds, initialDatasetId || datasetId);
        if (preferred) setDatasetId(preferred);
      })
      .catch((err) => {
        setError((err as Error).message);
      });

    const hwInterval = window.setInterval(() => {
      api
        .getHardware()
        .then(setHardware)
        .catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(hwInterval);
  }, []);

  useEffect(() => {
    if (initialDatasetId) setDatasetId(initialDatasetId);
  }, [initialDatasetId]);

  useEffect(() => {
    if (initialModelId) setModelId(initialModelId);
  }, [initialModelId]);

  const modelSpec = useMemo(
    () => getModelById(modelId) || PRETRAINED_MODELS[0],
    [modelId]
  );

  // Synchronize architecture-specific target modules whenever base model changes
  useEffect(() => {
    const valid = targetModules.filter((m) => modelSpec.targetModules.includes(m));
    if (valid.length === 0) {
      setTargetModules(modelSpec.targetModules.slice(0, 2));
    } else if (valid.length !== targetModules.length) {
      setTargetModules(valid);
    }
  }, [modelSpec.id]);

  const memoryEstimate = useMemo(
    () =>
      estimateTrainingMemory(
        modelSpec,
        method,
        loraR,
        targetModules,
        batchSize,
        maxSeqLength,
        gradientCheckpointing,
        hardware?.totalMemoryGb ?? 12
      ),
    [modelSpec, method, loraR, targetModules, batchSize, maxSeqLength, gradientCheckpointing, hardware]
  );

  const selectedDataset = datasets.find((d) => d.id === datasetId);

  const scheduleEstimate = useMemo(() => {
    const trainRows = selectedDataset?.splits?.trainCount ?? selectedDataset?.rowCount ?? 12;
    const effectiveBatch = Math.max(1, batchSize * Math.max(1, gradientAccumulation));
    const stepsPerEpoch = Math.max(1, Math.ceil(trainRows / effectiveBatch));
    const totalSteps = stepsPerEpoch * Math.max(1, epochs);
    const msPerStep =
      (method === 'qlora' ? 210 : method === 'full' ? 285 : 165) *
      Math.max(1, Math.log2(modelSpec.parameterBillions + 1.5) * 0.55);
    const estimatedRuntimeMs = Math.round(totalSteps * msPerStep);
    return { trainRows, stepsPerEpoch, totalSteps, estimatedRuntimeMs };
  }, [selectedDataset, batchSize, gradientAccumulation, epochs, method, modelSpec.parameterBillions]);

  const toggleModule = (mod: string) => {
    setTargetModules((prev) =>
      prev.includes(mod) ? (prev.length > 1 ? prev.filter((x) => x !== mod) : prev) : [...prev, mod]
    );
  };

  const handleLaunch = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const cfg: TrainingConfig = {
        experimentName: experimentName.trim(),
        description: description.trim(),
        datasetId,
        modelId: modelSpec.id,
        method,
        loraR,
        loraAlpha,
        loraDropout,
        targetModules,
        bias,
        quantizationBits: method === 'qlora' ? 4 : 16,
        quantType,
        doubleQuant,
        computeDtype,
        gradientCheckpointing,
        learningRate,
        epochs,
        batchSize,
        gradientAccumulation,
        maxSeqLength,
        weightDecay,
        warmupRatio,
        optimizer,
        scheduler,
        earlyStoppingPatience,
        seed,
      };
      const exp = await api.startTraining(cfg);
      setShowConfirm(false);
      onTrainingStarted(exp.id);
    } catch (err) {
      setError((err as Error).message);
      setShowConfirm(false);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Fine-Tuning Studio
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Configure LoRA, 4-bit NF4 QLoRA, optimizer schedules, and hardware memory constraints before launching training.
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <AiOrb
            state={submitting ? 'tool-call' : error ? 'error' : showConfirm ? 'waiting-approval' : 'idle'}
            size="sm"
            showLabel
            label={
              submitting
                ? 'Starting job'
                : error
                  ? 'Config invalid'
                  : showConfirm
                    ? 'Awaiting launch'
                    : 'Config ready'
            }
          />
          <LiquidButton
            type="button"
            variant="default"
            size="lg"
            onClick={() => {
              if (!datasetId) {
                setError('Select a preprocessed dataset first.');
                return;
              }
              if (!experimentName.trim()) {
                setError('Experiment name is required.');
                return;
              }
              if (
                (method !== 'full' && (!Number.isFinite(loraR) || loraR < 1 || !Number.isFinite(loraAlpha) || loraAlpha < 1)) ||
                !Number.isFinite(learningRate) ||
                learningRate <= 0 ||
                !Number.isFinite(epochs) ||
                epochs < 1 ||
                !Number.isFinite(batchSize) ||
                batchSize < 1 ||
                !Number.isFinite(maxSeqLength) ||
                maxSeqLength < 64
              ) {
                setError('Enter valid positive numeric hyperparameters (Rank, Alpha, LR, Epochs, Batch Size, Seq Length).');
                return;
              }
              setError(null);
              setShowConfirm(true);
            }}
            className="shrink-0 whitespace-nowrap"
          >
            <Play className="w-3.5 h-3.5" />
            Start Training
          </LiquidButton>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Config Column */}
        <div className="lg:col-span-8 space-y-6">
          {/* Experiment, Dataset & Model */}
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              1. Experiment Target &amp; Method
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Experiment Name
                </label>
                <input
                  type="text"
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Training Dataset
                </label>
                <select
                  aria-label="Training Dataset"
                  value={datasetId}
                  onChange={(e) => {
                    setDatasetId(e.target.value);
                    onSelectDataset?.(e.target.value);
                  }}
                  className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.status === 'training_ready' ? 'ready' : `${d.rowCount} rows · auto-split`})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Pretrained Base Model
                </label>
                <select
                  aria-label="Pretrained Base Model"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
                >
                  <optgroup label="Recommended Local CPU / QLoRA SLMs">
                    {PRETRAINED_MODELS.filter(
                      (m) => m.trainableLocally !== false && m.aiTool !== 'NVIDIA NIM' && m.aiTool !== 'Z.ai'
                    ).map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} ({m.parameterCount})
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="NVIDIA NIM Models">
                    {PRETRAINED_MODELS.filter((m) => m.aiTool === 'NVIDIA NIM').map((m) => (
                      <option key={m.id} value={m.id}>
                        [NVIDIA NIM] {m.name} ({m.parameterCount})
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Z.ai (GLM) Models">
                    {PRETRAINED_MODELS.filter((m) => m.aiTool === 'Z.ai').map((m) => (
                      <option key={m.id} value={m.id}>
                        [Z.ai] {m.name} ({m.parameterCount})
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Frontier & Open-Weight Architectures">
                    {PRETRAINED_MODELS.filter(
                      (m) => m.trainableLocally === false && m.aiTool !== 'NVIDIA NIM' && m.aiTool !== 'Z.ai'
                    ).map((m) => (
                      <option key={m.id} value={m.id}>
                        [{m.aiTool ?? m.source}] {m.name} ({m.parameterCount})
                      </option>
                    ))}
                  </optgroup>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Fine-Tuning Method
                </label>
                <RubberSegment
                  aria-label="Fine-Tuning Method"
                  equalSlots
                  size="sm"
                  radius={8}
                  inset={3}
                  value={method}
                  onChange={(val) => setMethod(val as TrainingMethod)}
                  trackColor={isDark ? '#090B10' : '#F1F5F9'}
                  thumbColor="#2563EB"
                  textColor={isDark ? '#94A3B8' : '#475569'}
                  activeTextColor="#FFFFFF"
                  className="w-full border border-slate-200 dark:border-white/[0.08] font-mono"
                  items={[
                    { value: 'lora', label: 'LoRA' },
                    { value: 'qlora', label: 'QLoRA' },
                    { value: 'full', label: 'Full' },
                  ]}
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                Experiment Hypothesis / Notes
              </label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
              />
            </div>
          </div>

          {/* LoRA / QLoRA Hyperparameters */}
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              2. LoRA &amp; Quantization Configuration
            </h2>
            {method !== 'full' && (
              <>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Rank (r)
                    </label>
                    <input
                      type="number"
                      min="2"
                      max="64"
                      value={loraR}
                      onChange={(e) => setLoraR(Number(e.target.value))}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      LoRA Alpha
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="128"
                      value={loraAlpha}
                      onChange={(e) => setLoraAlpha(Number(e.target.value))}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      LoRA Dropout
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.5"
                      value={loraDropout}
                      onChange={(e) => setLoraDropout(Number(e.target.value))}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                      Bias
                    </label>
                    <select
                      value={bias}
                      onChange={(e) => setBias(e.target.value as 'none' | 'all' | 'lora_only')}
                      className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                    >
                      <option value="none">none</option>
                      <option value="lora_only">lora_only</option>
                      <option value="all">all</option>
                    </select>
                  </div>
                </div>

                <div>
                  <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-2">
                    Architecture-Verified Target Modules ({modelSpec.architecture})
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {modelSpec.targetModules.map((mod) => {
                      const active = targetModules.includes(mod);
                      return (
                        <button
                          key={mod}
                          type="button"
                          onClick={() => toggleModule(mod)}
                          className={`px-2.5 py-1.5 text-xs font-mono rounded-md border transition-colors ${
                            active
                              ? 'border-blue-500 bg-blue-500/15 text-blue-600 dark:text-blue-400'
                              : 'border-slate-200 dark:border-white/[0.1] text-slate-600 dark:text-slate-400'
                          }`}
                        >
                          {mod}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </>
            )}

            {method === 'qlora' && (
              <div className="pt-3 border-t border-slate-200 dark:border-white/[0.06] grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                    4-Bit Quantization Type
                  </label>
                  <select
                    value={quantType}
                    onChange={(e) => setQuantType(e.target.value as 'nf4' | 'fp4')}
                    className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                  >
                    <option value="nf4">NF4 (NormalFloat4)</option>
                    <option value="fp4">FP4 (Float4)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                    Compute Dtype
                  </label>
                  <select
                    value={computeDtype}
                    onChange={(e) =>
                      setComputeDtype(e.target.value as 'bfloat16' | 'float16' | 'float32')
                    }
                    className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                  >
                    <option value="bfloat16">bfloat16</option>
                    <option value="float16">float16</option>
                    <option value="float32">float32</option>
                  </select>
                </div>
                <div className="flex flex-col justify-end gap-1.5 text-xs">
                  <label className="inline-flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={doubleQuant}
                      onChange={(e) => setDoubleQuant(e.target.checked)}
                    />
                    Double Quantization (8-bit scales)
                  </label>
                  <label className="inline-flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={gradientCheckpointing}
                      onChange={(e) => setGradientCheckpointing(e.target.checked)}
                    />
                    Gradient Checkpointing
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Optimization Schedule */}
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              3. Optimizer, Schedule &amp; Reproducibility
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.001"
                  min="0.0001"
                  max="0.2"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Epochs
                </label>
                <input
                  type="number"
                  min="1"
                  max="40"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  min="1"
                  max="32"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Grad Accumulation
                </label>
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={gradientAccumulation}
                  onChange={(e) => setGradientAccumulation(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Max Seq Length
                </label>
                <input
                  type="number"
                  step="64"
                  min="64"
                  max="2048"
                  value={maxSeqLength}
                  onChange={(e) => setMaxSeqLength(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Weight Decay
                </label>
                <input
                  type="number"
                  step="0.001"
                  min="0"
                  max="0.2"
                  value={weightDecay}
                  onChange={(e) => setWeightDecay(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Warmup Ratio
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="0.4"
                  value={warmupRatio}
                  onChange={(e) => setWarmupRatio(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Random Seed
                </label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Optimizer
                </label>
                <select
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value as 'adamw' | 'sgd')}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                >
                  <option value="adamw">AdamW</option>
                  <option value="sgd">SGD</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  LR Scheduler
                </label>
                <select
                  value={scheduler}
                  onChange={(e) =>
                    setScheduler(e.target.value as 'cosine' | 'linear' | 'constant')
                  }
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
                >
                  <option value="cosine">Cosine Decay</option>
                  <option value="linear">Linear Decay</option>
                  <option value="constant">Constant</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Early Stopping Patience
                </label>
                <input
                  type="number"
                  min="0"
                  max="20"
                  value={earlyStoppingPatience}
                  onChange={(e) => setEarlyStoppingPatience(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono tabular-nums"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Summary & Hardware Check Column */}
        <div className="lg:col-span-4 space-y-5">
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Hardware Compatibility &amp; Memory
              </h2>
              <Cpu className="w-4 h-4 text-blue-500" />
            </div>

            <dl className="space-y-2 text-xs">
              <div className="flex justify-between">
                <dt className="text-slate-500 dark:text-slate-400">Reference Base Weights</dt>
                <dd className="font-mono tabular-nums">{memoryEstimate.weightsGb} GB</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500 dark:text-slate-400">Adapter + Gradients</dt>
                <dd className="font-mono tabular-nums">{memoryEstimate.adapterAndGradsGb} GB</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500 dark:text-slate-400">Optimizer States</dt>
                <dd className="font-mono tabular-nums">{memoryEstimate.optimizerStatesGb} GB</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500 dark:text-slate-400">Activations</dt>
                <dd className="font-mono tabular-nums">{memoryEstimate.activationsGb} GB</dd>
              </div>
              <div className="flex justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06] font-semibold">
                <dt>Estimated GPU Footprint</dt>
                <dd className="font-mono tabular-nums text-blue-600 dark:text-blue-400">
                  {memoryEstimate.totalEstimatedGb} GB
                </dd>
              </div>
            </dl>

            <div className="pt-2 border-t border-slate-200 dark:border-white/[0.06] space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500 dark:text-slate-400">Reference Trainable Params</span>
                <span className="font-mono tabular-nums">
                  {formatNumber(memoryEstimate.trainableParams)} ({memoryEstimate.trainablePct}%)
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500 dark:text-slate-400">Host Environment</span>
                <span className="font-mono">
                  {hardware?.gpuAvailable ? hardware.gpuName : `${hardware?.cpuCores ?? 4} CPU / ${hardware?.totalMemoryGb ?? 12} GB RAM`}
                </span>
              </div>
            </div>

            {memoryEstimate.warnings.length > 0 && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 space-y-1.5 text-xs text-amber-700 dark:text-amber-300">
                <div className="flex items-center gap-1.5 font-medium">
                  <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
                  Hardware Compatibility Notes
                </div>
                {memoryEstimate.warnings.map((w, i) => (
                  <p key={i} className="leading-relaxed">
                    {w}
                  </p>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-400" />
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Pre-Launch Training Summary
              </h2>
            </div>
            <dl className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <dt className="text-slate-500">Dataset</dt>
                <dd className="font-medium truncate max-w-[180px]">{selectedDataset?.name ?? 'None'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Splits (Train/Val/Test)</dt>
                <dd className="font-mono tabular-nums">
                  {selectedDataset?.splits
                    ? `${selectedDataset.splits.trainCount}/${selectedDataset.splits.valCount}/${selectedDataset.splits.testCount}`
                    : 'Not split'}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Base Model</dt>
                <dd className="font-mono">{modelSpec.name}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Method</dt>
                <dd className="font-mono uppercase">{method}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">LoRA (r / alpha)</dt>
                <dd className="font-mono tabular-nums">
                  {loraR} / {loraAlpha}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Estimated Steps</dt>
                <dd className="font-mono tabular-nums">
                  {scheduleEstimate.totalSteps} steps ({scheduleEstimate.stepsPerEpoch}/epoch)
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Estimated Runtime</dt>
                <dd className="font-mono tabular-nums text-blue-600 dark:text-blue-400">
                  ~{formatDuration(scheduleEstimate.estimatedRuntimeMs)}
                </dd>
              </div>
              {hardware && (
                <div className="flex justify-between">
                  <dt className="text-slate-500">Live Free RAM / RSS</dt>
                  <dd className="font-mono tabular-nums">
                    {hardware.freeMemoryGb} GB free · {hardware.processMemoryMb} MB
                  </dd>
                </div>
              )}
            </dl>
          </div>
        </div>
      </div>

      {/* Confirmation Modal */}
      {showConfirm && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Confirm training launch"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        >
          <div className="w-full max-w-md rounded-xl border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] p-6 space-y-4">
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Launch Fine-Tuning Job?
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              This will initialize <span className="font-mono">{modelSpec.name}</span> with{' '}
              <span className="font-mono uppercase">{method}</span> (r={loraR}, alpha={loraAlpha}),
              save step checkpoints and final adapter artifacts under{' '}
              <span className="font-mono">artifacts/</span>, and stream live loss telemetry.
            </p>
            <div className="flex justify-end gap-2.5 pt-2">
              <LiquidButton
                type="button"
                size="sm"
                variant="outline"
                disabled={submitting}
                onClick={() => setShowConfirm(false)}
              >
                Cancel
              </LiquidButton>
              <LiquidButton
                type="button"
                variant="default"
                disabled={submitting}
                onClick={handleLaunch}
              >
                {submitting ? 'Launching...' : 'Confirm & Start Training'}
              </LiquidButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
