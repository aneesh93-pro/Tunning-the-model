import { Suspense, lazy, useEffect, useRef, useState } from 'react';
import {
  Activity,
  BarChart3,
  Bell,
  Cpu,
  Database,
  FlaskConical,
  GitCompare,
  History,
  Layers,
  LayoutDashboard,
  Menu,
  Moon,
  Plus,
  Settings,
  SquareTerminal,
  Sun,
  Upload,
  X,
} from 'lucide-react';
import { ClickSpark } from './components/reactbits/ClickSpark';
import Dock from './components/reactbits/Dock';
import GradientASCII from './components/reactbits/GradientASCII';
import { SiteFooter } from './components/SiteFooter';
import { WelcomeIntro } from './components/WelcomeIntro';
import { AiOrb, orbStateFor } from '@/components/ui/ai-orb';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { UserCursor } from '@/components/ui/user-cursor';
import { api, type OverviewResponse } from './lib/api';
import type { DatasetRecord } from './shared/types';
import {
  pickPreferredDatasetId,
  readStoredActiveDatasetId,
  writeStoredActiveDatasetId,
} from './utils/datasetSelection';
import { OverviewView, type NavSection } from './views/OverviewView';

const DatasetManagerView = lazy(() =>
  import('./views/DatasetManagerView').then((m) => ({ default: m.DatasetManagerView })),
);
const EvaluationCenterView = lazy(() =>
  import('./views/EvaluationCenterView').then((m) => ({ default: m.EvaluationCenterView })),
);
const ExperimentHistoryView = lazy(() =>
  import('./views/ExperimentHistoryView').then((m) => ({ default: m.ExperimentHistoryView })),
);
const FineTuningStudioView = lazy(() =>
  import('./views/FineTuningStudioView').then((m) => ({ default: m.FineTuningStudioView })),
);
const ModelComparisonView = lazy(() =>
  import('./views/ModelComparisonView').then((m) => ({ default: m.ModelComparisonView })),
);
const ModelHubView = lazy(() =>
  import('./views/ModelHubView').then((m) => ({ default: m.ModelHubView })),
);
const ModelPlaygroundView = lazy(() =>
  import('./views/ModelPlaygroundView').then((m) => ({ default: m.ModelPlaygroundView })),
);
const ModelRegistryView = lazy(() =>
  import('./views/ModelRegistryView').then((m) => ({ default: m.ModelRegistryView })),
);
const SettingsView = lazy(() =>
  import('./views/SettingsView').then((m) => ({ default: m.SettingsView })),
);
const TrainingMonitorView = lazy(() =>
  import('./views/TrainingMonitorView').then((m) => ({ default: m.TrainingMonitorView })),
);

const NAV_ITEMS: Array<{ id: NavSection; label: string; icon: typeof LayoutDashboard }> = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'datasets', label: 'Dataset Manager', icon: Database },
  { id: 'models', label: 'Model Hub', icon: Cpu },
  { id: 'studio', label: 'Fine-Tuning Studio', icon: FlaskConical },
  { id: 'monitor', label: 'Training Monitor', icon: Activity },
  { id: 'playground', label: 'Model Playground', icon: SquareTerminal },
  { id: 'evaluation', label: 'Evaluation Center', icon: BarChart3 },
  { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
  { id: 'registry', label: 'Model Registry', icon: Layers },
  { id: 'history', label: 'Experiment History', icon: History },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export default function App() {
  const [section, setSection] = useState<NavSection>('overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [workspaceName, setWorkspaceName] = useState('Core SLM Research Lab');

  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [selectedDatasetId, setSelectedDatasetIdState] = useState<string | null>(() =>
    readStoredActiveDatasetId()
  );
  const [selectedModelId, setSelectedModelId] = useState<string>('qwen2.5-0.5b-instruct');
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(null);
  const [playgroundModelRef, setPlaygroundModelRef] = useState<string | null>(null);
  const [showNotifications, setShowNotifications] = useState(false);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [importingDataset, setImportingDataset] = useState(false);
  const notificationsRef = useRef<HTMLDivElement>(null);
  const headerFileInputRef = useRef<HTMLInputElement | null>(null);

  const selectActiveDataset = (id: string | null) => {
    const clean = id && id.trim() ? id.trim() : null;
    setSelectedDatasetIdState(clean);
    writeStoredActiveDatasetId(clean);
  };

  const loadWorkspaceData = async (preferredDatasetId?: string | null) => {
    setOverviewLoading(true);
    try {
      const [ov, settings, dsList] = await Promise.all([
        api.getOverview(),
        api.getSettings(),
        api.getDatasets(),
      ]);
      setOverview(ov);
      setWorkspaceName(settings.workspace.name);
      setDatasets(dsList);
      setSelectedDatasetIdState((prev) => {
        const resolved = pickPreferredDatasetId(dsList, preferredDatasetId ?? prev);
        const nextId = resolved || null;
        writeStoredActiveDatasetId(nextId);
        return nextId;
      });
      setOverviewError(null);
    } catch (err) {
      setOverviewError((err as Error).message);
    } finally {
      setOverviewLoading(false);
    }
  };

  useEffect(() => {
    void loadWorkspaceData();
  }, [section]);

  const handleGlobalFileImport = async (file: File) => {
    setImportingDataset(true);
    setOverviewError(null);
    try {
      const created = await api.uploadDatasetFile(file);
      selectActiveDataset(created.id);
      await loadWorkspaceData(created.id);
    } catch (err) {
      setOverviewError((err as Error).message);
    } finally {
      setImportingDataset(false);
    }
  };

  useEffect(() => {
    const timer = window.setInterval(() => {
      api
        .getHardware()
        .then((hw) => {
          setOverview((prev) => (prev ? { ...prev, hardware: hw } : prev));
        })
        .catch(() => undefined);
    }, 15000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!showNotifications) return undefined;
    const handleOutside = (e: PointerEvent) => {
      if (notificationsRef.current && !notificationsRef.current.contains(e.target as Node)) {
        setShowNotifications(false);
      }
    };
    document.addEventListener('pointerdown', handleOutside);
    return () => document.removeEventListener('pointerdown', handleOutside);
  }, [showNotifications]);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  const handleNavigate = (target: NavSection, expId?: string) => {
    if (expId) setActiveExperimentId(expId);
    setSection(target);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'auto' });
  };

  const activeNavLabel = NAV_ITEMS.find((n) => n.id === section)?.label ?? 'Overview';
  const activeDataset =
    datasets.find((d) => d.id === selectedDatasetId) ?? datasets[0] ?? null;
  const notifications = overview?.recentExperiments.slice(0, 4) ?? [];
  const runningJobsCount =
    overview?.recentExperiments.filter((exp) => exp.status === 'running').length ?? 0;
  const runtimeOrbState = orbStateFor({
    error: Boolean(overviewError),
    thinking: runningJobsCount > 0 || importingDataset,
  });

  return (
    <ClickSpark
      sparkColor={['#3B82F6', '#06B6D4', '#8B5CF6']}
      sparkSize={8}
      sparkRadius={16}
      sparkCount={8}
      duration={340}
      className="min-h-screen"
    >
    <UserCursor
      fullScreen
      name={`Aneesh · ${activeNavLabel}`}
      color="#000000"
      directionAwareTilt
      labelTiltStrength={8}
      pressScale={0.9}
      trigger="hover"
      hideNativeCursor
      zIndex={70}
    />
    <WelcomeIntro theme={theme} />
    <div className="min-h-screen flex bg-[#F8FAFC] dark:bg-[#090B10] text-slate-900 dark:text-slate-100 relative overflow-x-clip">
      {/* Global Interactive GradientASCII Horizon Background */}
      <div
        aria-hidden="true"
        className="fixed inset-0 z-0 pointer-events-none"
      >
        <GradientASCII
          colors={
            theme === 'dark'
              ? ['#1e3fd8', '#38bdf8', '#8b5cf6', '#a5d8ff']
              : ['#1d4ed8', '#0284c7', '#7c3aed', '#38bdf8']
          }
          shape="horizon"
          characters=" =+xX0S8#@"
          cellSize={18}
          density={0.78}
          glyphOpacity={theme === 'dark' ? 0.62 : 0.38}
          gradientOpacity={theme === 'dark' ? 0.72 : 0.36}
          glow={theme === 'dark' ? 0.14 : 0.05}
          flicker={0.18}
          softness={0.55}
          flow={0.48}
          speed={0.85}
          grain={0.06}
          mouseInteraction
          scrambleRadius={0.1}
          clickEffect="wave"
          intro
          lightMode={theme === 'light'}
          dpr={1.5}
          className="opacity-45 dark:opacity-50"
        />
        <div className="absolute inset-x-0 top-0 h-60 bg-gradient-to-b from-[#F8FAFC]/95 via-[#F8FAFC]/70 to-transparent dark:from-[#090B10]/95 dark:via-[#090B10]/70 dark:to-transparent" />
      </div>

      {/* Desktop Left Dock Rail */}
      <aside
        aria-label="Primary navigation"
        className="hidden md:flex flex-col shrink-0 w-[86px] border-r border-slate-200/80 dark:border-white/[0.08] bg-white/80 dark:bg-[#0C0F16]/80 backdrop-blur-xl sticky top-0 h-screen overflow-visible z-30"
      >
        <div
          title={`MiniMind AI · ${workspaceName}`}
          className="h-14 border-b border-slate-200/80 dark:border-white/[0.08] flex items-center justify-center shrink-0"
        >
          <button
            type="button"
            aria-label="MiniMind AI Home"
            onClick={() => handleNavigate('overview')}
            className="w-10 h-10 rounded-xl border border-slate-200 dark:border-white/[0.12] bg-white/85 dark:bg-white/[0.06] backdrop-blur-md flex items-center justify-center hover:border-blue-500/50 transition-all shadow-sm overflow-hidden"
          >
            <img
              src="/minimind-favicon.svg"
              alt="MiniMind AI"
              className="w-7 h-7 object-contain"
            />
          </button>
        </div>

        <nav
          aria-label="Workspace Dock"
          className="relative flex-1 overflow-visible"
        >
          <Dock
            position="left"
            theme={theme}
            baseItemSize={40}
            magnification={56}
            distance={140}
            panelHeight={54}
            gap={6}
            roundness={0.55}
            showLabels
            showIndicators
            showBadges
            accentColor="#3B82F6"
            className="!left-1/2 !-translate-x-1/2"
            items={NAV_ITEMS.map((item) => ({
              icon: <item.icon className="w-[18px] h-[18px]" />,
              label: item.label,
              active: section === item.id,
              badge: item.id === 'monitor' && runningJobsCount > 0 ? runningJobsCount : undefined,
              onClick: () => handleNavigate(item.id),
            }))}
          />
        </nav>

        <div
          title={
            overview?.hardware.gpuAvailable
              ? overview.hardware.gpuName ?? 'GPU Engine'
              : `${overview?.hardware.cpuCores ?? 4} vCPU · CPU Engine`
          }
          className="py-2.5 border-t border-slate-200/80 dark:border-white/[0.08] flex flex-col items-center justify-center text-[10px] font-mono text-slate-500 dark:text-slate-400 shrink-0"
        >
          <span className="uppercase tracking-wider">
            {overview?.hardware.gpuAvailable ? 'GPU' : 'CPU'}
          </span>
        </div>
      </aside>

      {/* Mobile Sidebar Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          <div
            className="fixed inset-0 bg-black/50"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative z-10 w-64 bg-white dark:bg-[#0C0F16] border-r border-slate-200 dark:border-white/[0.08] flex flex-col">
            <div className="h-14 px-4 border-b border-slate-200 dark:border-white/[0.08] flex items-center justify-between">
              <span className="font-display font-semibold text-sm">MiniMind AI</span>
              <button
                type="button"
                aria-label="Close navigation menu"
                onClick={() => setMobileMenuOpen(false)}
                className="p-1.5 rounded-md text-slate-500"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleNavigate(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium ${
                    section === item.id
                      ? 'bg-blue-600/10 text-blue-600 dark:text-blue-400'
                      : 'text-slate-600 dark:text-slate-400'
                  }`}
                >
                  <item.icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 relative z-10">
        {/* Top Bar */}
        <header className="h-14 px-3 sm:px-6 border-b border-slate-200/80 dark:border-white/[0.08] bg-white/80 dark:bg-[#0C0F16]/80 backdrop-blur-xl flex items-center justify-between gap-2 sm:gap-4 sticky top-0 z-30">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <button
              type="button"
              aria-label="Open navigation menu"
              onClick={() => setMobileMenuOpen(true)}
              className="md:hidden p-1.5 rounded-md text-slate-500 hover:bg-slate-100 dark:hover:bg-white/[0.05] shrink-0"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-2 text-xs min-w-0">
              <span className="text-slate-500 dark:text-slate-400 hidden sm:inline truncate">
                {workspaceName}
              </span>
              <span className="text-slate-400 hidden sm:inline">/</span>
              <span className="font-medium text-slate-900 dark:text-slate-100 truncate">
                {activeNavLabel}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-1.5 sm:gap-2.5 shrink-0">
            {datasets.length > 0 && (
              <div className="hidden md:flex items-center gap-1.5 rounded-lg border border-slate-200/90 dark:border-white/[0.1] bg-slate-50/90 dark:bg-[#11141C]/90 px-2 py-1">
                <Database className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                <select
                  aria-label="Active workspace dataset"
                  value={activeDataset?.id ?? ''}
                  onChange={(e) => selectActiveDataset(e.target.value)}
                  className="max-w-[165px] truncate bg-transparent text-xs font-medium text-slate-800 dark:text-slate-100 focus:outline-none"
                >
                  {datasets.map((ds) => (
                    <option key={ds.id} value={ds.id} className="bg-white dark:bg-[#11141C]">
                      {ds.name} ({ds.rowCount})
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  title="Import dataset to use across all modules"
                  aria-label="Quick import dataset"
                  disabled={importingDataset}
                  onClick={() => headerFileInputRef.current?.click()}
                  className="p-1 rounded-md text-slate-500 hover:text-blue-600 dark:text-slate-400 dark:hover:text-blue-400 hover:bg-white dark:hover:bg-white/[0.06] transition-colors"
                >
                  <Upload className="w-3.5 h-3.5" />
                </button>
                <input
                  ref={headerFileInputRef}
                  type="file"
                  accept=".csv,.tsv,.json,.jsonl,.ndjson,.txt,text/csv,text/tab-separated-values,application/json,text/plain"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void handleGlobalFileImport(f);
                    e.target.value = '';
                  }}
                  className="sr-only"
                  aria-label="Quick import dataset file"
                />
              </div>
            )}

            <div
              title={
                overviewError
                  ? 'Runtime attention required'
                  : runningJobsCount > 0
                    ? `${runningJobsCount} active fine-tuning run(s)`
                    : 'Model runtime idle and ready'
              }
              className="inline-flex items-center gap-2"
            >
              <AiOrb state={runtimeOrbState} size="sm" decorative />
              <span className="hidden lg:inline text-xs font-mono text-slate-500 dark:text-slate-400 whitespace-nowrap">
                {overview?.hardware.gpuAvailable
                  ? `GPU: ${overview.hardware.gpuName}`
                  : 'Runtime: CPU LoRA Engine'}
              </span>
            </div>

            <div ref={notificationsRef} className="relative">
              <button
                type="button"
                aria-label="Toggle notifications"
                aria-expanded={showNotifications}
                onClick={() => setShowNotifications((s) => !s)}
                className="p-1.5 sm:p-2 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-white/[0.05]"
              >
                <Bell className="w-4 h-4" />
              </button>
              {showNotifications && (
                <div className="absolute right-0 mt-2 w-72 rounded-xl border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] p-3 shadow-lg z-40">
                  <div className="text-xs font-semibold mb-2">Recent Activity</div>
                  {notifications.length === 0 ? (
                    <p className="text-xs text-slate-500">No training notifications yet.</p>
                  ) : (
                    <ul className="divide-y divide-slate-200 dark:divide-white/[0.06] text-xs">
                      {notifications.map((exp) => (
                        <li key={exp.id}>
                          <button
                            type="button"
                            onClick={() => {
                              setShowNotifications(false);
                              handleNavigate('monitor', exp.id);
                            }}
                            className="w-full text-left py-2 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                          >
                            <div className="font-medium truncate">{exp.name}</div>
                            <div className="text-[11px] font-mono text-slate-500">
                              {exp.status} · {exp.modelName}
                            </div>
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>

            <button
              type="button"
              aria-label="Toggle color theme"
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              className="p-1.5 sm:p-2 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-white/[0.05]"
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            <LiquidButton
              type="button"
              variant="default"
              size="sm"
              onClick={() => handleNavigate('studio')}
              className="shrink-0"
            >
              <Plus className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">New Experiment</span>
              <span className="sm:hidden">New</span>
            </LiquidButton>
          </div>
        </header>

        {/* Active Viewport */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1440px] w-full mx-auto">
          {overviewError && (
            <div className="mb-4 flex items-center justify-between gap-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-xs text-red-600 dark:text-red-400">
              <span>{overviewError}</span>
              <button
                type="button"
                onClick={() => setOverviewError(null)}
                className="font-medium hover:underline"
              >
                Dismiss
              </button>
            </div>
          )}
          <Suspense
            fallback={
              <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/70 dark:bg-[#11141C]/70 p-8 text-xs font-mono text-slate-500 dark:text-slate-400">
                Loading workspace module...
              </div>
            }
          >
            {section === 'overview' && (
              <OverviewView
                data={overview}
                loading={overviewLoading}
                activeDataset={activeDataset}
                datasets={datasets}
                onSelectDataset={(id) => selectActiveDataset(id)}
                onImportDataset={handleGlobalFileImport}
                onNavigate={handleNavigate}
                onOpenPlayground={(ref) => {
                  setPlaygroundModelRef(ref);
                  handleNavigate('playground');
                }}
              />
            )}
            {section === 'datasets' && (
              <DatasetManagerView
                selectedDatasetId={selectedDatasetId}
                onSelectForTraining={(id) => selectActiveDataset(id || null)}
                onDatasetsChanged={(preferredId) => void loadWorkspaceData(preferredId)}
                onNavigate={handleNavigate}
              />
            )}
            {section === 'models' && (
              <ModelHubView
                selectedModelId={selectedModelId}
                activeDataset={activeDataset}
                onSelectModel={(id) => setSelectedModelId(id)}
                onTestInPlayground={(modelId) => {
                  setPlaygroundModelRef(`base:${modelId}`);
                  handleNavigate('playground');
                }}
                onNavigate={handleNavigate}
              />
            )}
            {section === 'studio' && (
              <FineTuningStudioView
                initialDatasetId={selectedDatasetId}
                onSelectDataset={(id) => selectActiveDataset(id || null)}
                onDatasetImported={(ds) => {
                  selectActiveDataset(ds.id);
                  void loadWorkspaceData(ds.id);
                }}
                initialModelId={selectedModelId}
                onTrainingStarted={(expId) => handleNavigate('monitor', expId)}
                onNavigate={handleNavigate}
              />
            )}
            {section === 'monitor' && (
              <TrainingMonitorView
                activeExperimentId={activeExperimentId}
                activeDataset={activeDataset}
                onSelectExperiment={(id) => setActiveExperimentId(id)}
                onNavigate={handleNavigate}
              />
            )}
            {section === 'playground' && (
              <ModelPlaygroundView
                initialModelRef={playgroundModelRef}
                initialDatasetId={selectedDatasetId}
                onSelectDataset={(id) => selectActiveDataset(id || null)}
              />
            )}
            {section === 'evaluation' && (
              <EvaluationCenterView
                initialDatasetId={selectedDatasetId}
                onSelectDataset={(id) => selectActiveDataset(id || null)}
                onNavigate={handleNavigate}
              />
            )}
            {section === 'comparison' && (
              <ModelComparisonView
                initialDatasetId={selectedDatasetId}
                onSelectDataset={(id) => selectActiveDataset(id || null)}
              />
            )}
            {section === 'registry' && (
              <ModelRegistryView
                activeDataset={activeDataset}
                onOpenPlayground={(ref) => {
                  setPlaygroundModelRef(ref);
                  handleNavigate('playground');
                }}
                onNavigate={handleNavigate}
              />
            )}
            {section === 'history' && (
              <ExperimentHistoryView
                activeDataset={activeDataset}
                onOpenMonitor={(id) => handleNavigate('monitor', id)}
                onNavigate={handleNavigate}
              />
            )}
            {section === 'settings' && (
              <SettingsView
                theme={theme}
                onToggleTheme={setTheme}
                onWorkspaceUpdated={loadWorkspaceData}
              />
            )}
          </Suspense>
        </main>

        <SiteFooter onNavigate={handleNavigate} />
      </div>
    </div>
    </ClickSpark>
  );
}
