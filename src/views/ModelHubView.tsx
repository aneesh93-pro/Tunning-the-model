import { useEffect, useState } from 'react';
import { CheckCircle2, Cpu, Search, ShieldCheck } from 'lucide-react';
import { DecryptedText } from '../components/reactbits/DecryptedText';
import { SpotlightCard } from '../components/reactbits/SpotlightCard';
import { api } from '../lib/api';
import type { HardwareTelemetry, ModelSpec } from '../shared/types';
import type { NavSection } from './OverviewView';

interface ModelHubViewProps {
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  onNavigate: (section: NavSection) => void;
}

export function ModelHubView({
  selectedModelId,
  onSelectModel,
  onNavigate,
}: ModelHubViewProps) {
  const [models, setModels] = useState<ModelSpec[]>([]);
  const [hardware, setHardware] = useState<HardwareTelemetry | null>(null);
  const [customRepo, setCustomRepo] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [validationResult, setValidationResult] = useState<{
    valid: boolean;
    reachable: boolean;
    repoId: string;
    architecture: string;
    recommendedModules: string[];
    downloads?: number | null;
    note?: string;
  } | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    api.getAvailableModels().then((res) => {
      setModels(res.models);
      setHardware(res.hardware);
    });
  }, []);

  const handleValidateRepo = async (repoId: string) => {
    setChecking(true);
    try {
      const res = await api.validateHfModel(repoId);
      setValidationResult(res);
    } catch (err) {
      setValidationResult({
        valid: false,
        reachable: false,
        repoId,
        architecture: 'unknown',
        recommendedModules: [],
        note: (err as Error).message,
      });
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Model Hub
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Select a pretrained Small Language Model, inspect architecture-specific LoRA target modules, and verify compatibility.
          </p>
        </div>
      </div>

      {/* Hugging Face Repository Validator */}
      <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5">
              Verify Hugging Face Model Compatibility
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={customRepo}
                onChange={(e) => setCustomRepo(e.target.value)}
                placeholder="org/model-name"
                className="flex-1 px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
              />
              <button
                type="button"
                disabled={checking}
                onClick={() => handleValidateRepo(customRepo)}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 whitespace-nowrap"
              >
                <Search className="w-3.5 h-3.5" />
                {checking ? 'Inspecting...' : 'Validate Repo'}
              </button>
            </div>
          </div>
        </div>
        {validationResult && (
          <div className="mt-3 pt-3 border-t border-slate-200 dark:border-white/[0.06] text-xs flex flex-wrap items-center gap-4">
            <span
              className={`inline-flex items-center gap-1.5 font-medium ${
                validationResult.valid
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-red-600 dark:text-red-400'
              }`}
            >
              <ShieldCheck className="w-4 h-4" />
              {validationResult.valid ? 'Compatible Repository' : 'Unverified Repository'}
            </span>
            <span className="font-mono text-slate-600 dark:text-slate-300">
              {validationResult.repoId} · arch: {validationResult.architecture}
            </span>
            {validationResult.recommendedModules.length > 0 && (
              <span className="font-mono text-slate-500 dark:text-slate-400">
                modules: {validationResult.recommendedModules.join(', ')}
              </span>
            )}
            {validationResult.note && (
              <span className="text-slate-500 dark:text-slate-400">{validationResult.note}</span>
            )}
          </div>
        )}
      </div>

      {/* Pretrained SLM Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {models.map((m) => {
          const isSelected = m.id === selectedModelId;
          const totalRam = hardware?.totalMemoryGb ?? 12;
          const qloraFits = m.estimatedQloraMemoryGb <= totalRam;

          return (
            <SpotlightCard
              key={m.id}
              className={`p-5 ${
                isSelected ? 'ring-1 ring-blue-500 border-blue-500/60' : ''
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs font-mono text-blue-600 dark:text-blue-400">
                    <DecryptedText
                      text={m.repoId}
                      animateOn="hover"
                      speed={28}
                      maxIterations={8}
                      scrambleColor="#38BDF8"
                    />
                  </div>
                  <h2 className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                    {m.name}
                  </h2>
                </div>
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400 shrink-0">
                  {m.parameterCount} · {m.license}
                </span>
              </div>

              <p className="mt-2.5 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                {m.description}
              </p>

              <dl className="mt-4 grid grid-cols-2 gap-y-2 gap-x-4 text-xs border-t border-slate-200 dark:border-white/[0.06] pt-3">
                <div>
                  <dt className="text-slate-500 dark:text-slate-400">Architecture</dt>
                  <dd className="font-mono text-slate-900 dark:text-slate-100">
                    {m.architecture} · {m.numLayers}L / d={m.hiddenSize}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500 dark:text-slate-400">Context / Template</dt>
                  <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                    {m.contextLength.toLocaleString()} · {m.chatTemplate}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500 dark:text-slate-400">Memory (FP16 / QLoRA)</dt>
                  <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                    {m.estimatedFp16MemoryGb} GB / {m.estimatedQloraMemoryGb} GB
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500 dark:text-slate-400">Methods</dt>
                  <dd className="font-mono uppercase text-slate-900 dark:text-slate-100">
                    {m.supportedMethods.join(' · ')}
                  </dd>
                </div>
              </dl>

              <div className="mt-3 pt-3 border-t border-slate-200 dark:border-white/[0.06]">
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mb-1">
                  Inspectable LoRA Target Modules
                </div>
                <div className="text-xs font-mono text-slate-700 dark:text-slate-300">
                  {m.targetModules.join(', ')}
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                  <Cpu className="w-3.5 h-3.5" />
                  {qloraFits ? 'QLoRA memory compatible' : 'High VRAM recommended'}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    onSelectModel(m.id);
                    onNavigate('studio');
                  }}
                  className={`inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-lg transition-colors ${
                    isSelected
                      ? 'bg-blue-600 text-white hover:bg-blue-500'
                      : 'border border-slate-200 dark:border-white/[0.1] text-slate-800 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-white/[0.04]'
                  }`}
                >
                  {isSelected && <CheckCircle2 className="w-3.5 h-3.5" />}
                  Configure Fine-Tuning
                </button>
              </div>
            </SpotlightCard>
          );
        })}
      </div>
    </div>
  );
}
