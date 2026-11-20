import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Globe,
  Layers,
  MessageSquare,
  Search,
  ShieldCheck,
  Sparkles,
  SquareTerminal,
} from 'lucide-react';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { DecryptedText } from '../components/reactbits/DecryptedText';
import GlassIcons, { type GlassIconsItem } from '../components/reactbits/GlassIcons';
import { SpotlightCard } from '../components/reactbits/SpotlightCard';
import { api } from '../lib/api';
import { useIsDark } from '../lib/useDarkMode';
import type { AiToolCategory, HardwareTelemetry, ModelSpec } from '../shared/types';
import type { NavSection } from './OverviewView';

interface ModelHubViewProps {
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  onTestInPlayground?: (modelId: string) => void;
  onNavigate: (section: NavSection) => void;
}

type FilterTool = 'all' | AiToolCategory | 'open-weights';

const TOOL_FILTERS: Array<{
  id: FilterTool;
  label: string;
  color: string;
  icon: typeof Globe;
}> = [
  { id: 'all', label: 'All AI Tools', color: 'graphite', icon: Globe },
  { id: 'ChatGPT', label: 'ChatGPT', color: 'teal', icon: MessageSquare },
  { id: 'Claude', label: 'Claude', color: 'orange', icon: BrainCircuit },
  { id: 'Gemini', label: 'Gemini', color: 'blue', icon: Sparkles },
  { id: 'DeepSeek', label: 'DeepSeek', color: 'indigo', icon: Bot },
  { id: 'NVIDIA NIM', label: 'NVIDIA NIM', color: 'green', icon: Cpu },
  {
    id: 'Z.ai',
    label: 'Z.ai (GLM)',
    color: 'linear-gradient(135deg, hsl(192, 90%, 54%), hsl(218, 86%, 48%))',
    icon: Sparkles,
  },
  { id: 'Open SLM', label: 'Open SLMs', color: 'purple', icon: Cpu },
  { id: 'open-weights', label: 'Llama & Mistral', color: 'silver', icon: Layers },
];

export function ModelHubView({
  selectedModelId,
  onSelectModel,
  onTestInPlayground,
  onNavigate,
}: ModelHubViewProps) {
  const isDark = useIsDark();
  const [models, setModels] = useState<ModelSpec[]>([]);
  const [hardware, setHardware] = useState<HardwareTelemetry | null>(null);
  const [activeTool, setActiveTool] = useState<FilterTool>('all');
  const [searchQuery, setSearchQuery] = useState('');
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
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getAvailableModels()
      .then((res) => {
        setModels(res.models);
        setHardware(res.hardware);
        setError(null);
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  const filteredModels = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return models.filter((m) => {
      const tool = m.aiTool ?? 'Open SLM';
      const matchesTool =
        activeTool === 'all'
          ? true
          : activeTool === 'open-weights'
            ? tool === 'Llama' || tool === 'Mistral'
            : tool === activeTool;
      if (!matchesTool) return false;
      if (!q) return true;
      return (
        m.name.toLowerCase().includes(q) ||
        m.repoId.toLowerCase().includes(q) ||
        (m.provider ?? m.source).toLowerCase().includes(q) ||
        m.description.toLowerCase().includes(q)
      );
    });
  }, [models, activeTool, searchQuery]);

  const glassFilterItems: GlassIconsItem[] = useMemo(
    () =>
      TOOL_FILTERS.map((f) => {
        const IconComp = f.icon;
        return {
          id: f.id,
          label: f.label,
          color: f.color,
          icon: <IconComp />,
          active: activeTool === f.id,
          onClick: () => setActiveTool(f.id),
        };
      }),
    [activeTool]
  );

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
            AI Tools &amp; Model Hub
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Browse flagship AI tool models (ChatGPT, Claude, Gemini, DeepSeek, NVIDIA NIM, Z.ai GLM) and local open-weight SLMs — test any model directly in Playground or configure LoRA fine-tuning.
          </p>
        </div>
        <div className="text-xs font-mono text-slate-500 dark:text-slate-400 shrink-0">
          Showing {filteredModels.length} / {models.length} models
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* AI Tool Provider Filter Bar using ReactBits GlassIcons */}
      <SpotlightCard
        spotlightColor={isDark ? 'rgba(59, 130, 246, 0.14)' : 'rgba(37, 99, 235, 0.08)'}
        className="p-5 sm:p-6 space-y-5"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Filter by AI Tool &amp; Model Family
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Click a glass icon to filter models by ChatGPT, Claude, Gemini, DeepSeek, NVIDIA NIM, Z.ai GLM, or Open SLMs
            </p>
          </div>
          <input
            type="search"
            aria-label="Search models by name or provider"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search GPT-4o, Claude, Gemini, Nemotron, GLM..."
            className="w-full sm:w-80 px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          />
        </div>

        <div className="pt-1 pb-6">
          <GlassIcons
            items={glassFilterItems}
            size={62}
            gap={44}
            columns={5}
            roundness={0.55}
            refraction={0.75}
            bevel={0.55}
            frost={0.22}
            shine={0.85}
            tint={isDark ? '#dfdfdf' : '#ffffff'}
            tintOpacity={isDark ? 0.18 : 0.28}
            plate="tilt"
            tilt={26}
            hover="lift"
            labels="always"
            intro={false}
            theme={isDark ? 'dark' : 'light'}
          />
        </div>
      </SpotlightCard>

      {/* Hugging Face Repository Validator */}
      <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex-1">
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5">
              Verify Hugging Face or Provider Model Identifier
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={customRepo}
                onChange={(e) => setCustomRepo(e.target.value)}
                placeholder="org/model-name"
                className="flex-1 px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
              />
              <LiquidButton
                type="button"
                variant="default"
                disabled={checking}
                onClick={() => handleValidateRepo(customRepo)}
                className="whitespace-nowrap"
              >
                <Search className="w-3.5 h-3.5" />
                {checking ? 'Inspecting...' : 'Validate Repo'}
              </LiquidButton>
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

      {/* Model Grid */}
      {filteredModels.length === 0 ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-10 text-center text-xs text-slate-500 dark:text-slate-400">
          No models match the active filter. Clear search or select &ldquo;All AI Tools&rdquo;.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {filteredModels.map((m) => {
            const isSelected = m.id === selectedModelId;
            const totalRam = hardware?.totalMemoryGb ?? 12;
            const qloraFits = m.estimatedQloraMemoryGb <= totalRam;
            const isLocal = m.trainableLocally !== false;

            return (
              <SpotlightCard
                key={m.id}
                className={`p-5 ${
                  isSelected ? 'ring-1 ring-blue-500 border-blue-500/60' : ''
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-blue-600 dark:text-blue-400">
                      <span>{m.provider ?? m.source}</span>
                      <span className="text-slate-400">·</span>
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
                    <dt className="text-slate-500 dark:text-slate-400">Architecture &amp; Tier</dt>
                    <dd className="font-mono text-slate-900 dark:text-slate-100">
                      {m.architecture} · {m.modelTier ?? 'SLM'}
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

                <div className="mt-4 flex flex-wrap items-center justify-between gap-2.5">
                  <span className="inline-flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                    <Cpu className="w-3.5 h-3.5" />
                    {isLocal
                      ? qloraFits
                        ? 'Local CPU/QLoRA Ready'
                        : 'High VRAM recommended'
                      : 'Playground & Eval Benchmark Ready'}
                  </span>
                  <div className="flex flex-wrap items-center gap-2">
                    {onTestInPlayground && (
                      <LiquidButton
                        type="button"
                        size="sm"
                        variant="default"
                        onClick={() => onTestInPlayground(m.id)}
                        className="whitespace-nowrap"
                      >
                        <SquareTerminal className="w-3.5 h-3.5" />
                        Test in Playground
                      </LiquidButton>
                    )}
                    <LiquidButton
                      type="button"
                      size="sm"
                      variant={isSelected ? 'default' : 'outline'}
                      onClick={() => {
                        onSelectModel(m.id);
                        onNavigate('studio');
                      }}
                      className="text-xs whitespace-nowrap"
                    >
                      {isSelected && <CheckCircle2 className="w-3.5 h-3.5" />}
                      Configure Fine-Tuning
                    </LiquidButton>
                  </div>
                </div>
              </SpotlightCard>
            );
          })}
        </div>
      )}
    </div>
  );
}
