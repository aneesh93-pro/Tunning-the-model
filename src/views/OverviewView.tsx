import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Box,
  Check,
  Compass,
  Copy,
  Cpu,
  Database,
  Download,
  FileText,
  FlaskConical,
  HardDrive,
  Layers,
  Mail,
  Pause,
  Play,
  RotateCcw,
  SquareTerminal,
  Target,
  Upload,
  UserCheck,
  X,
} from 'lucide-react';
import { motion } from 'motion/react';
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
import { CountUp } from '../components/reactbits/CountUp';
import { DecryptedText } from '../components/reactbits/DecryptedText';
import { DitherVeil } from '../components/reactbits/DitherVeil';
import { DotGrid } from '../components/reactbits/DotGrid';
import GlassIcons, { type GlassIconsItem } from '../components/reactbits/GlassIcons';
import type {
  ModelViewerEnvironment,
  ModelViewerHandle,
} from '../components/reactbits/ModelViewer';
import type {
  PaperCrumpleReleaseBehavior,
  PaperCrumpleState,
} from '../components/reactbits/PaperCrumple';

const ModelViewer = lazy(() => import('../components/reactbits/ModelViewer'));
const PaperCrumple = lazy(() => import('../components/reactbits/PaperCrumple'));
import ProfileCard from '../components/reactbits/ProfileCard';
import { RubberSegment } from '../components/reactbits/RubberSegment';
import { ShinyText } from '../components/reactbits/ShinyText';
import { SpotlightCard } from '../components/reactbits/SpotlightCard';
import { TextLoop } from '../components/reactbits/TextLoop';
import { TextPressure } from '../components/reactbits/TextPressure';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import type { OverviewResponse } from '../lib/api';
import {
  renderMiniMindBriefBackDataUrl,
  renderMiniMindBriefFrontDataUrl,
} from '../lib/minimindPaperAsset';
import { useIsDark } from '../lib/useDarkMode';
import { formatDuration, formatNumber } from '../lib/utils';

export type NavSection =
  | 'overview'
  | 'datasets'
  | 'models'
  | 'studio'
  | 'monitor'
  | 'playground'
  | 'evaluation'
  | 'comparison'
  | 'registry'
  | 'history'
  | 'settings';

interface OverviewViewProps {
  data: OverviewResponse | null;
  loading: boolean;
  onNavigate: (section: NavSection, experimentId?: string) => void;
  onOpenPlayground?: (modelRef: string) => void;
}

export function OverviewView({ data, loading, onNavigate, onOpenPlayground }: OverviewViewProps) {
  const isDark = useIsDark();
  const [releaseBehavior, setReleaseBehavior] =
    useState<PaperCrumpleReleaseBehavior>('creased');
  const [paperState, setPaperState] = useState<PaperCrumpleState>('flat');
  const [paperResetKey, setPaperResetKey] = useState(0);

  const modelViewerRef = useRef<ModelViewerHandle>(null);
  const [iconEnvironment, setIconEnvironment] =
    useState<ModelViewerEnvironment>('studio');
  const [iconAutoRotate, setIconAutoRotate] = useState(true);
  const [iconLoaded, setIconLoaded] = useState(false);
  const [contactOpen, setContactOpen] = useState(false);
  const [emailCopied, setEmailCopied] = useState(false);

  const engineerEmail = 'aneeshkancharla2007@gmail.com';
  const engineerPhoto =
    '/assets/aistudio/BF87E7A0-2FE4-4B72-87E7-0BF358535B01__1__4fc2fed6.jpg';

  useEffect(() => {
    if (!contactOpen) return undefined;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setContactOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [contactOpen]);

  const handleCopyEmail = async () => {
    try {
      await navigator.clipboard.writeText(engineerEmail);
      setEmailCopied(true);
      setTimeout(() => setEmailCopied(false), 2200);
    } catch {
      setEmailCopied(false);
    }
  };

  const handleCaptureIcon = () => {
    const dataUrl = modelViewerRef.current?.capture('image/png');
    if (!dataUrl) return;
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = 'minimind-ai-icon.png';
    link.click();
  };

  const briefFrontUrl = useMemo(
    () => renderMiniMindBriefFrontDataUrl(data?.stats),
    [
      data?.stats?.totalExperiments,
      data?.stats?.totalDatasets,
      data?.stats?.totalPretrainedModels,
    ]
  );
  const briefBackUrl = useMemo(() => renderMiniMindBriefBackDataUrl(), []);

  if (loading && !data) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-44 rounded-xl bg-slate-200 dark:bg-[#11141C]" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-28 rounded-xl bg-slate-200 dark:bg-[#11141C]" />
          ))}
        </div>
      </div>
    );
  }

  const stats = data?.stats;
  const hw = data?.hardware;
  const recentExperiments = data?.recentExperiments ?? [];
  const recentModels = data?.recentModels ?? [];
  const recentEvaluations = data?.recentEvaluations ?? [];

  const evalChartData = recentEvaluations
    .slice(0, 6)
    .reverse()
    .map((ev) => ({
      name: ev.modelName.slice(0, 18),
      score:
        ev.accuracy !== null
          ? Number((ev.accuracy * 100).toFixed(1))
          : ev.tokenF1 !== null
            ? Number((ev.tokenF1 * 100).toFixed(1))
            : 0,
      perplexity: ev.perplexity,
    }));

  const expActivityMap = new Map<
    string,
    { date: string; completed: number; failed: number; other: number }
  >();
  for (const exp of recentExperiments) {
    const day = exp.createdAt.slice(0, 10);
    const cur = expActivityMap.get(day) ?? { date: day, completed: 0, failed: 0, other: 0 };
    if (exp.status === 'completed') cur.completed++;
    else if (exp.status === 'failed') cur.failed++;
    else cur.other++;
    expActivityMap.set(day, cur);
  }
  const activityData = [...expActivityMap.values()];

  return (
    <div className="space-y-6">
      {/* Hero Showcase: TextLoop Website Heading + Interactive DitherVeil Neural Canvas */}
      <div className="liquid-glass-surface liquid-glass-hero relative overflow-hidden rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88">
        <DotGrid />
        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 items-stretch">
          <div className="lg:col-span-7 p-5 sm:p-6 flex flex-col justify-between gap-4">
            <div className="space-y-2.5">
              <div className="rounded-lg border border-slate-200/80 dark:border-white/[0.08] bg-slate-50/80 dark:bg-[#090B10]/80 overflow-hidden h-16 sm:h-20">
                <TextLoop
                  text="MiniMind AI"
                  shape="wave"
                  viewBox="0 165 1200 190"
                  speed={85}
                  direction="forward"
                  separator="✦"
                  curviness={34}
                  tilt={0}
                  fontSize={42}
                  fontWeight={800}
                  letterSpacing={3}
                  uppercase
                  ribbon
                  ribbonColor={isDark ? '#2563EB' : '#1D4ED8'}
                  ribbonWidth={72}
                  edgeFade={12}
                  pauseOnHover
                  className="h-full"
                />
              </div>

              <div className="pt-1 space-y-2">
                <p className="text-xs font-medium">
                  <ShinyText
                    text="Custom Small Language Model Studio"
                    trigger="loop"
                    speed={3.2}
                    delay={1.5}
                    color={isDark ? '#60A5FA' : '#1D4ED8'}
                    shineColor={isDark ? '#BAE6FD' : '#3B82F6'}
                  />
                </p>
                <div className="w-full py-1 pr-2">
                  <TextPressure
                    text="MiniMind AI"
                    flex={false}
                    alpha={false}
                    stroke={false}
                    width
                    weight
                    italic
                    radius={0.55}
                    smoothing={0.45}
                    idle="wave"
                    idleSpeed={1}
                    clickRipple
                    textColor={isDark ? '#F8FAFC' : '#0F172A'}
                    pressureColor={isDark ? '#60A5FA' : '#2563EB'}
                    minFontSize={22}
                    maxFontSize={50}
                    className="text-left tracking-[0.06em]"
                  />
                </div>
                <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 max-w-2xl">
                  Fine-Tuning Console — preprocess instruction datasets, train LoRA &amp; 4-bit NF4 QLoRA adapters, run structured boolean/choice/score inference, and benchmark fine-tuned models on held-out test splits.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <LiquidButton
                type="button"
                variant="outline"
                size="sm"
                onClick={() => onNavigate('datasets')}
              >
                <Upload className="w-3.5 h-3.5" />
                Upload Dataset
              </LiquidButton>
              <LiquidButton
                type="button"
                variant="default"
                size="lg"
                onClick={() => onNavigate('studio')}
              >
                <Play className="w-3.5 h-3.5" />
                Start Fine-Tuning
              </LiquidButton>
            </div>
          </div>

          <div className="lg:col-span-5 relative min-h-[200px] lg:min-h-[240px] border-t lg:border-t-0 lg:border-l border-slate-200 dark:border-white/[0.08] bg-[#090B10]">
            <DitherVeil
              src="/minimind-neural-matrix.png"
              fit="cover"
              pattern="floyd"
              pixelSize={2}
              levels={2}
              palette="duotone"
              inkColor={isDark ? '#090B10' : '#0F172A'}
              paperColor={isDark ? '#DBEAFE' : '#EFF6FF'}
              rimColor="#3B82F6"
              rim={0.08}
              contrast={1.2}
              revealRadius={165}
              softness={0.6}
              linger={1}
              wander={false}
              clickBurst
            />
            <div className="pointer-events-none absolute bottom-2.5 right-3 rounded-md bg-black/55 px-2.5 py-1 text-[10px] font-mono text-slate-200 backdrop-blur-sm">
              LoRA Tensor Field · Hover or click to reveal
            </div>
          </div>
        </div>
      </div>

      {/* Animated Website Description & Main Aim Intro Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        >
          <SpotlightCard
            spotlightColor={isDark ? 'rgba(59, 130, 246, 0.18)' : 'rgba(37, 99, 235, 0.10)'}
            className="h-full p-5"
          >
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 shrink-0">
                <Compass className="w-4 h-4" />
              </div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                <DecryptedText
                  text="About MiniMind AI & Getting Started"
                  animateOn="hover"
                  maxIterations={8}
                  speed={28}
                  scrambleColor="#3B82F6"
                />
              </h2>
            </div>
            <p className="mt-3 text-xs sm:text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              MiniMind AI is a full-stack workspace built for experimenting with Language Models end to end. Begin by uploading, cleaning, and exporting split datasets in <strong className="font-medium text-slate-900 dark:text-slate-100">Dataset Manager</strong>, pick compact open SLMs (SmolLM2, Qwen2.5, Phi-3.5) or test flagship AI models (<strong className="font-medium text-slate-900 dark:text-slate-100">NVIDIA NIM, Z.ai GLM, ChatGPT, Claude, Gemini, DeepSeek</strong>) in <strong className="font-medium text-slate-900 dark:text-slate-100">Model Hub</strong> &amp; <strong className="font-medium text-slate-900 dark:text-slate-100">Model Playground</strong>, and launch LoRA / 4-bit NF4 QLoRA fine-tuning in the <strong className="font-medium text-slate-900 dark:text-slate-100">Fine-Tuning Studio</strong>.
            </p>
          </SpotlightCard>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
        >
          <SpotlightCard
            spotlightColor={isDark ? 'rgba(6, 182, 212, 0.18)' : 'rgba(14, 165, 233, 0.10)'}
            className="h-full p-5"
          >
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 shrink-0">
                <Target className="w-4 h-4" />
              </div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                <DecryptedText
                  text="Main Aim of This Platform"
                  animateOn="hover"
                  maxIterations={8}
                  speed={28}
                  scrambleColor="#06B6D4"
                />
              </h2>
            </div>
            <p className="mt-3 text-xs sm:text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              Our primary aim is to make parameter-efficient LLM customization transparent, verifiable, and reliable—replacing black-box heuristics with real training telemetry, held-out test benchmarks, and structured <span className="font-mono text-[11px] sm:text-xs text-blue-600 dark:text-blue-400">boolean / choice / score</span> inference so engineers, students, and researchers can detect hallucinations early, compare base vs. adapter gains, and export ready-to-deploy LoRA weights.
            </p>
          </SpotlightCard>
        </motion.div>
      </div>

      {/* Practical Uses & Example Workflows */}
      <SpotlightCard
        spotlightColor={isDark ? 'rgba(59, 130, 246, 0.14)' : 'rgba(37, 99, 235, 0.08)'}
        className="p-5 sm:p-6"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 shrink-0">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                <DecryptedText
                  text="How to Use MiniMind AI — Practical Example Workflows"
                  animateOn="hover"
                  maxIterations={8}
                  speed={28}
                  scrambleColor="#3B82F6"
                />
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Real end-to-end ways engineers, students, and ML teams use this studio
              </p>
            </div>
          </div>
          <LiquidButton
            type="button"
            size="sm"
            variant="outline"
            onClick={() => onNavigate('studio')}
          >
            Launch Fine-Tuning Studio
            <ArrowRight className="w-3.5 h-3.5" />
          </LiquidButton>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              step: 'Example 01 · Intent Classification (LoRA)',
              title: 'Banking Support Triage Adapter',
              description:
                'Take the included 24-row Banking Intent Classification dataset, validate & split it into Train/Val/Test in Dataset Manager, and fine-tune Qwen2.5-0.5B with LoRA (r=8).',
              sample:
                'Input: "I lost my debit card on the train." → Label: card_lost_or_stolen',
              cta: 'Open Fine-Tuning Studio',
              target: 'studio' as NavSection,
              icon: FlaskConical,
            },
            {
              step: 'Example 02 · Anti-Hallucination Audit',
              title: 'Structured Boolean / Choice / Score Verification',
              description:
                'Test any base model or fine-tuned adapter in Model Playground using deterministic boolean, choice, and score modes to catch fabricated terms (like "JEV AI") and download a full model test report.',
              sample:
                'Mode: choice + score → Validates grounded options, flags hallucinations & computes pass rate',
              cta: 'Run Test in Playground',
              target: 'playground' as NavSection,
              icon: SquareTerminal,
            },
            {
              step: 'Example 03 · 4-Bit NF4 QLoRA Chat',
              title: 'Domain Engineering Assistant on CPU/Low-VRAM',
              description:
                'Upload custom conversational JSONL/CSV data (or use SLM Engineering Chat), select 4-bit NF4 QLoRA on SmolLM2 or Phi-3.5, and monitor live step loss, grad norm, and tokens/sec.',
              sample:
                'Trains only ~0.5%–2% adapter weights while base parameters stay NF4 quantized',
              cta: 'Manage Datasets',
              target: 'datasets' as NavSection,
              icon: Database,
            },
            {
              step: 'Example 04 · Benchmark & Export',
              title: 'Compare Base vs. Adapter & Export PEFT Code',
              description:
                'Score held-out test splits in Evaluation Center (Accuracy, Macro F1, Perplexity), compare base vs. adapter side by side, and download weights + train_peft.py from Model Registry.',
              sample:
                'Outputs confusion matrix, base-vs-tuned delta chart & ready-to-run GPU training script',
              cta: 'Open Evaluation Center',
              target: 'evaluation' as NavSection,
              icon: Activity,
            },
          ].map((ex) => (
            <div
              key={ex.title}
              className="rounded-xl border border-slate-200/80 dark:border-white/[0.08] bg-slate-50/75 dark:bg-[#090B10]/70 p-4 flex flex-col justify-between gap-3"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-mono font-medium text-blue-600 dark:text-blue-400">
                    {ex.step}
                  </span>
                  <ex.icon className="w-4 h-4 text-slate-400 shrink-0" />
                </div>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  {ex.title}
                </h3>
                <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                  {ex.description}
                </p>
                <div className="rounded-lg border border-slate-200/70 dark:border-white/[0.06] bg-white/80 dark:bg-[#11141C] px-3 py-2 font-mono text-[11px] text-slate-700 dark:text-slate-300">
                  {ex.sample}
                </div>
              </div>
              <div>
                <LiquidButton
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => onNavigate(ex.target)}
                >
                  {ex.cta}
                  <ArrowRight className="w-3.5 h-3.5" />
                </LiquidButton>
              </div>
            </div>
          ))}
        </div>
      </SpotlightCard>

      {/* AI Engineer Profile Showcase (ReactBits ProfileCard) */}
      <SpotlightCard
        spotlightColor={isDark ? 'rgba(59, 130, 246, 0.16)' : 'rgba(37, 99, 235, 0.09)'}
        className="p-5 sm:p-6"
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          <div className="lg:col-span-5 flex justify-center">
            <ProfileCard
              name="KANCHARLA ANEESH"
              title="AI ENGINEER"
              handle="kancharla.aneesh"
              status="Online"
              statusColor="#22c55e"
              contactText="Contact"
              avatarUrl={engineerPhoto}
              miniAvatarUrl={engineerPhoto}
              photoPosition="center 12%"
              holo={0.55}
              glare={0.4}
              tiltStrength={12}
              parallax={8}
              radius={16}
              enableTilt
              intro
              showUserInfo
              enableMobileTilt
              theme={isDark ? 'dark' : 'light'}
              onContactClick={() => setContactOpen(true)}
            />
          </div>

          <div className="lg:col-span-7 space-y-5">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 shrink-0">
                <UserCheck className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  <DecryptedText
                    text="AI Engineer Profile & Workspace Architect"
                    animateOn="hover"
                    maxIterations={8}
                    speed={28}
                    scrambleColor="#3B82F6"
                  />
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  KANCHARLA ANEESH · AI ENGINEER · Custom SLM Fine-Tuning &amp; Evaluation
                </p>
              </div>
            </div>

            <p className="text-xs sm:text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              Focused on designing parameter-efficient Small Language Model pipelines, deterministic LoRA &amp; 4-bit NF4 QLoRA training workflows, and hallucination-resistant evaluation harnesses with structured <span className="font-mono text-[11px] sm:text-xs text-blue-600 dark:text-blue-400">boolean / choice / score</span> model reporting.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-3 border-t border-slate-200/80 dark:border-white/[0.06] text-xs">
              <div className="rounded-lg border border-slate-200/80 dark:border-white/[0.07] bg-slate-50/70 dark:bg-[#090B10]/60 p-3.5">
                <div className="font-medium text-slate-900 dark:text-slate-100">
                  Core Engineering Stack
                </div>
                <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                  LoRA / QLoRA NF4 quantization, causal SLM training, token-level evaluation, and PEFT export automation.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200/80 dark:border-white/[0.07] bg-slate-50/70 dark:bg-[#090B10]/60 p-3.5">
                <div className="font-medium text-slate-900 dark:text-slate-100">
                  Verification &amp; Reliability
                </div>
                <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                  Structured inference testing across SmolLM2, Qwen2.5, and Phi-3.5 adapters with downloadable model reports.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2.5 pt-1">
              <LiquidButton
                type="button"
                variant="default"
                onClick={() => setContactOpen(true)}
              >
                <Mail className="w-3.5 h-3.5" />
                Contact AI Engineer
              </LiquidButton>
              <LiquidButton
                type="button"
                variant="outline"
                size="sm"
                onClick={() => onNavigate('playground')}
              >
                <SquareTerminal className="w-3.5 h-3.5" />
                Test Models in Playground
              </LiquidButton>
            </div>
          </div>
        </div>
      </SpotlightCard>

      {/* 3D MiniMind AI Emblem Showcase (ReactBits ModelViewer) */}
      <SpotlightCard
        spotlightColor={isDark ? 'rgba(59, 130, 246, 0.16)' : 'rgba(37, 99, 235, 0.09)'}
        className="p-5"
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          <div className="lg:col-span-6 relative rounded-lg border border-slate-200/80 dark:border-white/[0.08] bg-slate-100/70 dark:bg-[#090B10]/85 overflow-hidden">
            <Suspense
              fallback={
                <div className="h-[390px] w-full flex items-center justify-center text-xs font-mono text-slate-500">
                  Loading 3D Emblem...
                </div>
              }
            >
              <ModelViewer
                ref={modelViewerRef}
                url="/minimind-icon.glb"
                width="100%"
                height={390}
                environment={iconEnvironment}
                environmentIntensity={isDark ? 1.15 : 1.0}
                exposure={isDark ? 1.05 : 1.0}
                intro="rise"
                autoRotate={iconAutoRotate}
                autoRotateSpeed={0.55}
                autoRotateDelay={1.2}
                dragRotate
                hoverTilt={0.18}
                float={0.28}
                shadow
                shadowOpacity={isDark ? 0.55 : 0.28}
                fit={0.8}
                startAngle={-28}
                tilt={12}
                onModelLoaded={() => setIconLoaded(true)}
              />
            </Suspense>
            <div className="pointer-events-none absolute bottom-2.5 right-3 rounded-md bg-black/55 px-2.5 py-1 text-[10px] font-mono text-slate-200 backdrop-blur-sm">
              {iconLoaded ? '3D Emblem · Drag to orbit · Double-click to reset' : 'Loading 3D Emblem...'}
            </div>
          </div>

          <div className="lg:col-span-6 space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 shrink-0">
                <Box className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  MiniMind AI 3D Website Icon
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Sculpted LoRA &amp; NF4 neural core emblem representing the platform
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-5 gap-y-3.5 text-xs pt-3 border-t border-slate-200/80 dark:border-white/[0.06]">
              <div>
                <div className="font-medium text-slate-900 dark:text-slate-100">
                  Frozen Base Slab (W₀)
                </div>
                <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                  Anodized foundation plate representing frozen pretrained SLM weights across SmolLM2, Qwen2.5, and Phi-3.5.
                </p>
              </div>
              <div>
                <div className="font-medium text-slate-900 dark:text-slate-100">
                  Dual LoRA Matrix Rings (A · B)
                </div>
                <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                  Interlocking cobalt and cyan orbital rings embodying trainable low-rank adaptation &amp; 4-bit QLoRA deltas.
                </p>
              </div>
              <div>
                <div className="font-medium text-slate-900 dark:text-slate-100">
                  NF4 Core &amp; 4 Workflow Nodes
                </div>
                <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                  Faceted violet center linked to Datasets, Training, Playground, and Evaluation stages.
                </p>
              </div>
              <div>
                <div className="font-medium text-slate-900 dark:text-slate-100">
                  Grounded Verification Halo
                </div>
                <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
                  Titanium crown above the core symbolizing boolean/choice/score hallucination checks.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <RubberSegment
                aria-label="3D icon lighting environment"
                value={iconEnvironment}
                onChange={(val) => setIconEnvironment(val as ModelViewerEnvironment)}
                size="sm"
                trackColor={isDark ? '#171B26' : '#E2E8F0'}
                thumbColor="#2563EB"
                textColor={isDark ? '#94A3B8' : '#475569'}
                activeTextColor="#FFFFFF"
                items={[
                  { value: 'studio', label: 'Studio' },
                  { value: 'night', label: 'Night' },
                  { value: 'city', label: 'City' },
                  { value: 'sunset', label: 'Sunset' },
                ]}
              />

              <div className="flex flex-wrap items-center gap-2">
                <LiquidButton
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setIconAutoRotate((r) => !r)}
                >
                  {iconAutoRotate ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {iconAutoRotate ? 'Pause Spin' : 'Auto Spin'}
                </LiquidButton>
                <LiquidButton
                  type="button"
                  variant="default"
                  onClick={handleCaptureIcon}
                  disabled={!iconLoaded}
                >
                  <Download className="w-3.5 h-3.5" />
                  Export Icon PNG
                </LiquidButton>
              </div>
            </div>
          </div>
        </div>
      </SpotlightCard>

      {/* Interactive Platform Brief Sheet (ReactBits PaperCrumple) */}
      <SpotlightCard
        spotlightColor={isDark ? 'rgba(59, 130, 246, 0.14)' : 'rgba(37, 99, 235, 0.08)'}
        className="p-5"
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          <div className="lg:col-span-5 space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-violet-500/10 text-violet-600 dark:text-violet-400 shrink-0">
                <FileText className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Interactive Platform Field Brief
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  MiniMind AI architecture &amp; workflow specification card
                </p>
              </div>
            </div>

            <p className="text-xs sm:text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              This printed specification sheet summarizes the end-to-end MiniMind AI fine-tuning architecture—from leak-free dataset splitting and 4-bit NF4 QLoRA adaptation to anti-hallucination playground scoring and adapter export. Press and hold or drag the sheet to crumple it in 3D, or use keyboard controls (<span className="font-mono text-[11px]">Space</span> / <span className="font-mono text-[11px]">Esc</span>).
            </p>

            <div className="space-y-3 pt-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-medium text-slate-600 dark:text-slate-300">
                  Release Behavior
                </span>
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                  Sheet state: <span className="text-blue-600 dark:text-blue-400">{paperState}</span>
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                <RubberSegment
                  aria-label="Paper release behavior"
                  value={releaseBehavior}
                  onChange={(val) => setReleaseBehavior(val as PaperCrumpleReleaseBehavior)}
                  size="sm"
                  trackColor={isDark ? '#171B26' : '#E2E8F0'}
                  thumbColor="#2563EB"
                  textColor={isDark ? '#94A3B8' : '#475569'}
                  activeTextColor="#FFFFFF"
                  items={[
                    { value: 'creased', label: 'Creased' },
                    { value: 'restore', label: 'Restore' },
                    { value: 'stay', label: 'Stay' },
                  ]}
                />

                <button
                  type="button"
                  onClick={() => setPaperResetKey((k) => k + 1)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-white/[0.03] text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-white/[0.06] transition-colors whitespace-nowrap"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Reset Sheet
                </button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-7 relative rounded-lg border border-slate-200/80 dark:border-white/[0.08] bg-slate-100/70 dark:bg-[#090B10]/80 overflow-hidden">
            {briefFrontUrl && (
              <Suspense
                fallback={
                  <div className="h-[430px] w-full flex items-center justify-center text-xs font-mono text-slate-500">
                    Loading Interactive Brief Sheet...
                  </div>
                }
              >
                <PaperCrumple
                  src={briefFrontUrl}
                  backSrc={briefBackUrl}
                  alt="MiniMind AI Platform Architecture and Workflow Brief Sheet"
                  width={310}
                  height={392}
                  sceneHeight={430}
                  imageFit="cover"
                  releaseBehavior={releaseBehavior}
                  crumpleAmount={0.85}
                  crumpleDuration={0.55}
                  releaseDuration={0.4}
                  foldCount={6}
                  foldSharpness={0.6}
                  wrinkleDepth={0.65}
                  creaseStrength={0.22}
                  paperColor={isDark ? '#1e2433' : '#f4f0e8'}
                  paperTexture={0.08}
                  lightIntensity={1.8}
                  shadowOpacity={0.1}
                  draggable
                  returnToOrigin
                  detail={48}
                  resetKey={paperResetKey}
                  onStateChange={setPaperState}
                />
              </Suspense>
            )}
            <div className="pointer-events-none absolute bottom-2.5 right-3 rounded-md bg-black/55 px-2.5 py-1 text-[10px] font-mono text-slate-200 backdrop-blur-sm">
              Hold &amp; drag sheet · 3D Paper Physics
            </div>
          </div>
        </div>
      </SpotlightCard>

      {/* Primary KPI Grid (real backend metrics) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <SpotlightCard className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Total Experiments
            </span>
            <FlaskConical className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-50 font-mono tabular-nums">
            <CountUp to={stats?.totalExperiments ?? 0} trigger="load" duration={0.75} separator="," />
          </div>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            {stats?.completedRuns ?? 0} completed · {stats?.failedRuns ?? 0} failed
          </p>
        </SpotlightCard>

        <SpotlightCard className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Registered Adapters
            </span>
            <Layers className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-50 font-mono tabular-nums">
            <CountUp to={stats?.totalRegisteredModels ?? 0} trigger="load" duration={0.75} separator="," />
          </div>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            {stats?.totalPretrainedModels ?? 25} base models · {stats?.totalDatasets ?? 0} datasets
          </p>
        </SpotlightCard>

        <SpotlightCard className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Trainable Parameters
            </span>
            <Activity className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-50 font-mono tabular-nums">
            <CountUp to={stats?.totalTrainableParams ?? 0} trigger="load" duration={0.85} separator="," />
          </div>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            Avg duration: {formatDuration(stats?.avgDurationMs)}
          </p>
        </SpotlightCard>

        <SpotlightCard className="p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Latest Model Performance
            </span>
            <Cpu className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-50 font-mono tabular-nums">
            {stats?.latestPerformance ? (
              stats.latestPerformance.accuracy !== null ? (
                `${(stats.latestPerformance.accuracy * 100).toFixed(1)}%`
              ) : (
                `PPL ${stats.latestPerformance.perplexity}`
              )
            ) : (
              'No runs yet'
            )}
          </div>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 truncate">
            {stats?.latestPerformance
              ? `${stats.latestPerformance.modelName} · PPL ${stats.latestPerformance.perplexity}`
              : 'Run evaluation after training'}
          </p>
        </SpotlightCard>
      </div>

      {/* Quick Studio Modules — ReactBits GlassIcons */}
      <SpotlightCard
        spotlightColor={isDark ? 'rgba(59, 130, 246, 0.12)' : 'rgba(37, 99, 235, 0.07)'}
        className="p-5 sm:p-6"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 mb-6">
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Interactive Studio Modules
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Select a workspace module — preprocess &amp; download datasets, test NVIDIA NIM / Z.ai / ChatGPT / Claude / Gemini / DeepSeek, or train LoRA adapters
            </p>
          </div>
          <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
            {stats?.totalPretrainedModels ?? 25} models · {stats?.totalDatasets ?? 0} datasets
          </span>
        </div>

        <div className="py-2 pb-6">
          <GlassIcons
            size={68}
            gap={52}
            columns={6}
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
            items={[
              {
                id: 'datasets',
                icon: <Database />,
                color: 'blue',
                label: 'Dataset Manager',
                onClick: () => onNavigate('datasets'),
              },
              {
                id: 'models',
                icon: <Cpu />,
                color: 'purple',
                label: 'AI Model Hub',
                onClick: () => onNavigate('models'),
              },
              {
                id: 'studio',
                icon: <FlaskConical />,
                color: 'indigo',
                label: 'LoRA Studio',
                onClick: () => onNavigate('studio'),
              },
              {
                id: 'playground',
                icon: <SquareTerminal />,
                color: 'teal',
                label: 'Playground',
                onClick: () => onNavigate('playground'),
              },
              {
                id: 'evaluation',
                icon: <Activity />,
                color: 'orange',
                label: 'Evaluation',
                onClick: () => onNavigate('evaluation'),
              },
              {
                id: 'registry',
                icon: <Layers />,
                color: 'green',
                label: 'Model Registry',
                onClick: () => onNavigate('registry'),
              },
            ] satisfies GlassIconsItem[]}
          />
        </div>
      </SpotlightCard>

      {/* Charts Row: Training Activity & Model Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Training Activity
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Daily completed (blue) vs. failed (red) training runs — tracks workspace reliability over time.
              </p>
            </div>
          </div>
          {activityData.length === 0 ? (
            <div className="h-56 flex flex-col items-center justify-center text-center border border-dashed border-slate-200 dark:border-white/[0.08] rounded-lg p-6">
              <FlaskConical className="w-6 h-6 text-slate-400 mb-2" />
              <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                No training runs recorded yet
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-xs">
                Launch a LoRA or QLoRA experiment in Fine-Tuning Studio to track real runs over time.
              </p>
              <LiquidButton
                type="button"
                size="sm"
                variant="default"
                onClick={() => onNavigate('studio')}
                className="mt-3"
              >
                New Experiment
              </LiquidButton>
            </div>
          ) : (
            <div className="h-56 w-full min-w-0">
              <ResponsiveContainer
                width="100%"
                height={224}
                minWidth={0}
                minHeight={224}
                initialDimension={{ width: 480, height: 224 }}
              >
                <BarChart data={activityData} margin={{ top: 8, right: 12, left: 0, bottom: 2 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                  <YAxis allowDecimals={false} width={36} tick={{ fontSize: 11, fill: '#94A3B8' }} />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null;
                      const row = payload[0].payload as { date: string; completed: number; failed: number };
                      return (
                        <div className="rounded-lg border border-white/15 bg-[#11141C]/95 px-3 py-2 text-xs text-slate-100 shadow-lg backdrop-blur-md space-y-0.5">
                          <div className="font-semibold text-slate-200">Date: {label}</div>
                          <div className="font-mono text-blue-300">Completed runs: {row.completed}</div>
                          <div className="font-mono text-rose-400">Failed runs: {row.failed}</div>
                        </div>
                      );
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar
                    dataKey="completed"
                    name="Completed Runs"
                    fill="#3B82F6"
                    radius={[4, 4, 0, 0]}
                    isAnimationActive={false}
                  />
                  <Bar
                    dataKey="failed"
                    name="Failed Runs"
                    fill="#DC2626"
                    radius={[4, 4, 0, 0]}
                    isAnimationActive={false}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="liquid-glass-surface rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/88 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Model Performance
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Held-out test accuracy / token F1 (%) — higher bars mean better predictions; 80% is strong quality.
              </p>
            </div>
          </div>
          {evalChartData.length === 0 ? (
            <div className="h-56 flex flex-col items-center justify-center text-center border border-dashed border-slate-200 dark:border-white/[0.08] rounded-lg p-6">
              <Activity className="w-6 h-6 text-slate-400 mb-2" />
              <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                No evaluation benchmarks yet
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-xs">
                Evaluate a base model or trained adapter against a test split to plot real metrics.
              </p>
              <LiquidButton
                type="button"
                size="sm"
                variant="outline"
                onClick={() => onNavigate('evaluation')}
                className="mt-3"
              >
                Open Evaluation Center
              </LiquidButton>
            </div>
          ) : (
            <div className="h-56 w-full min-w-0">
              <ResponsiveContainer
                width="100%"
                height={224}
                minWidth={0}
                minHeight={224}
                initialDimension={{ width: 480, height: 224 }}
              >
                <BarChart data={evalChartData} margin={{ top: 8, right: 12, left: 0, bottom: 2 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                  <YAxis
                    domain={[0, 100]}
                    width={42}
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fontSize: 11, fill: '#94A3B8' }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const row = payload[0].payload as { name: string; score: number; perplexity: number };
                      return (
                        <div className="rounded-lg border border-white/15 bg-[#11141C]/95 px-3 py-2 text-xs text-slate-100 shadow-lg backdrop-blur-md space-y-0.5">
                          <div className="font-semibold text-cyan-300">{row.name}</div>
                          <div className="font-mono text-slate-100">Accuracy / F1 Score: {row.score}%</div>
                          <div className="font-mono text-[11px] text-slate-300">
                            Perplexity: {row.perplexity} (lower = more confident)
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
                      value: '80% Target',
                      position: 'insideTopRight',
                      fill: '#10B981',
                      fontSize: 10,
                    }}
                  />
                  <Bar
                    dataKey="score"
                    name="Accuracy / Token F1 (%)"
                    fill="#0EA5E9"
                    radius={[4, 4, 0, 0]}
                    isAnimationActive={false}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Grid: Recent Experiments, Recent Models & Hardware Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-200 dark:border-white/[0.08] flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Recent Experiments
            </h2>
            <button
              type="button"
              onClick={() => onNavigate('history')}
              className="text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline"
            >
              View all
            </button>
          </div>
          {recentExperiments.length === 0 ? (
            <div className="p-8 text-center text-sm text-slate-500 dark:text-slate-400">
              No experiments have been launched yet. Configure a run in Fine-Tuning Studio.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500 dark:text-slate-400">
                  <tr>
                    <th className="py-2.5 px-4 font-medium">Experiment</th>
                    <th className="py-2.5 px-4 font-medium">Model</th>
                    <th className="py-2.5 px-4 font-medium">Method</th>
                    <th className="py-2.5 px-4 font-medium">Status</th>
                    <th className="py-2.5 px-4 font-medium text-right">Loss</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                  {recentExperiments.map((exp) => (
                    <tr
                      key={exp.id}
                      onClick={() => onNavigate('monitor', exp.id)}
                      className="hover:bg-slate-50 dark:hover:bg-white/[0.02] cursor-pointer"
                    >
                      <td className="py-2.5 px-4 font-medium text-slate-900 dark:text-slate-100">
                        {exp.name}
                      </td>
                      <td className="py-2.5 px-4 text-slate-600 dark:text-slate-300">
                        {exp.modelName}
                      </td>
                      <td className="py-2.5 px-4 font-mono uppercase text-slate-600 dark:text-slate-300">
                        {exp.method}
                      </td>
                      <td className="py-2.5 px-4">
                        <span
                          className={
                            exp.status === 'completed'
                              ? 'text-emerald-600 dark:text-emerald-400 font-medium'
                              : exp.status === 'failed'
                                ? 'text-red-600 dark:text-red-400 font-medium'
                                : 'text-blue-600 dark:text-blue-400 font-medium'
                          }
                        >
                          {exp.status}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums text-slate-700 dark:text-slate-200">
                        {exp.trainLoss !== null ? exp.trainLoss.toFixed(4) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Hardware Monitoring
              </h2>
              <HardDrive className="w-4 h-4 text-slate-400" />
            </div>
            <dl className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <dt className="text-slate-500 dark:text-slate-400">CUDA GPU</dt>
                <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                  {hw?.gpuAvailable ? hw.gpuName : 'Unavailable'}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-slate-500 dark:text-slate-400">GPU Utilization / Temp</dt>
                <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                  {hw?.gpuAvailable
                    ? `${hw.gpuUtilizationPct}% · ${hw.gpuTemperatureC}°C`
                    : 'Unavailable'}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-slate-500 dark:text-slate-400">Host CPU Cores</dt>
                <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                  {hw ? `${hw.cpuCores} vCPU` : '—'}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-slate-500 dark:text-slate-400">System RAM Free</dt>
                <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                  {hw ? `${hw.freeMemoryGb} / ${hw.totalMemoryGb} GB` : '—'}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-slate-500 dark:text-slate-400">Trainer Process RSS</dt>
                <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                  {hw ? `${hw.processMemoryMb} MB` : '—'}
                </dd>
              </div>
            </dl>
            <p className="mt-3 pt-3 border-t border-slate-200 dark:border-white/[0.06] text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
              {hw?.note}
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Recent Saved Adapters
              </h2>
              <button
                type="button"
                onClick={() => onNavigate('registry')}
                className="text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline"
              >
                Registry
              </button>
            </div>
            {recentModels.length === 0 ? (
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Completed training runs automatically register adapters here.
              </p>
            ) : (
              <ul className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                {recentModels.map((m) => (
                  <li key={m.id} className="py-2 flex items-center justify-between gap-2 text-xs">
                    <div className="min-w-0">
                      <div className="font-medium text-slate-900 dark:text-slate-100 truncate">
                        {m.name}
                      </div>
                      <div className="text-slate-500 dark:text-slate-400 font-mono truncate">
                        {m.method.toUpperCase()} · {formatNumber(m.trainableParams)} params
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        onOpenPlayground
                          ? onOpenPlayground(`reg:${m.id}`)
                          : onNavigate('playground')
                      }
                      className="text-blue-600 dark:text-blue-400 hover:underline shrink-0"
                    >
                      Test
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      {/* AI Engineer Contact Dialog */}
      {contactOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-labelledby="engineer-contact-title"
          onClick={() => setContactOpen(false)}
        >
          <div
            className="w-full max-w-md rounded-2xl border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] p-6 shadow-2xl space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <img
                  src={engineerPhoto}
                  alt="KANCHARLA ANEESH"
                  className="w-12 h-12 rounded-full object-cover object-[center_12%] border border-slate-200 dark:border-white/[0.12] shrink-0"
                />
                <div className="min-w-0">
                  <h3
                    id="engineer-contact-title"
                    className="text-sm font-semibold text-slate-900 dark:text-slate-50 truncate"
                  >
                    KANCHARLA ANEESH
                  </h3>
                  <p className="text-xs font-mono text-blue-600 dark:text-blue-400">
                    AI ENGINEER
                  </p>
                </div>
              </div>
              <button
                type="button"
                aria-label="Close contact dialog"
                onClick={() => setContactOpen(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-white/[0.06]"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3.5 space-y-2 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-500 dark:text-slate-400">Role</span>
                <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
                  AI ENGINEER
                </span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-500 dark:text-slate-400">Handle</span>
                <span className="font-mono text-slate-800 dark:text-slate-200">
                  @kancharla.aneesh
                </span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-500 dark:text-slate-400">Email</span>
                <span className="font-mono text-slate-900 dark:text-slate-100 truncate">
                  {engineerEmail}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2.5 pt-1">
              <LiquidButton
                type="button"
                size="sm"
                variant="outline"
                onClick={handleCopyEmail}
              >
                {emailCopied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-500" />
                    Copied Email
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    Copy Email
                  </>
                )}
              </LiquidButton>
              <a
                href={`mailto:${engineerEmail}?subject=MiniMind%20AI%20Studio`}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-full border border-white/95 bg-white/85 hover:bg-white text-slate-900 dark:border-white/90 dark:bg-white/[0.86] dark:hover:bg-white dark:text-slate-950 backdrop-blur-xl shadow-[0_4px_18px_rgba(255,255,255,0.22),inset_0_1px_0_rgba(255,255,255,0.9)] transition-all"
              >
                <Mail className="w-3.5 h-3.5" />
                Send Email
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
