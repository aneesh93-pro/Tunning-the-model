import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Check,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileText,
  FlaskConical,
  FolderOpen,
  Layers,
  Rocket,
  Trash2,
} from 'lucide-react';
import Folder from '../components/reactbits/Folder';
import { LiquidButton, MetalButton } from '@/components/ui/liquid-glass-button';
import { api } from '../lib/api';
import { useIsDark } from '../lib/useDarkMode';
import { formatBytes, formatNumber } from '../lib/utils';
import {
  buildAdapterBundleDocs,
  type AdapterBundleDocId,
} from '../utils/folderBundles';
import type { DatasetRecord, RegisteredModelRecord } from '../shared/types';
import type { NavSection } from './OverviewView';

interface ModelRegistryViewProps {
  activeDataset?: DatasetRecord | null;
  onOpenPlayground: (modelRef: string) => void;
  onNavigate: (section: NavSection) => void;
}

export function ModelRegistryView({
  activeDataset,
  onOpenPlayground,
  onNavigate,
}: ModelRegistryViewProps) {
  const isDark = useIsDark();
  const [models, setModels] = useState<RegisteredModelRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [inspectedId, setInspectedId] = useState<string | null>(null);
  const [bundleOpen, setBundleOpen] = useState(true);
  const [bundleLayout, setBundleLayout] = useState<'spread' | 'row' | 'stack'>('spread');
  const [selectedDocId, setSelectedDocId] = useState<AdapterBundleDocId>('config');
  const [copiedCode, setCopiedCode] = useState(false);
  const [selectedCard, setSelectedCard] = useState<RegisteredModelRecord | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<RegisteredModelRecord | null>(null);
  const [deployModal, setDeployModal] = useState<RegisteredModelRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadRegistry = async () => {
    setLoading(true);
    try {
      const list = await api.getRegistryModels();
      setModels(list);
      setInspectedId((prev) => (prev && list.some((m) => m.id === prev) ? prev : list[0]?.id ?? null));
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadRegistry();
  }, []);

  const inspectedModel = models.find((m) => m.id === inspectedId) ?? models[0] ?? null;

  const handleCopySnippet = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 1800);
    } catch {
      setCopiedCode(false);
    }
  };

  const handleToggleDeploy = async (m: RegisteredModelRecord) => {
    const next = m.status === 'deployed' ? 'ready' : 'deployed';
    try {
      const updated = await api.deployModel(m.id, next);
      setModels((prev) => prev.map((x) => (x.id === m.id ? updated : x)));
      setError(null);
      if (next === 'deployed') setDeployModal(updated);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleDelete = async () => {
    if (!confirmDelete) return;
    try {
      await api.deleteRegisteredModel(confirmDelete.id);
      setConfirmDelete(null);
      await loadRegistry();
    } catch (err) {
      setError((err as Error).message);
      setConfirmDelete(null);
    }
  };

  const handleExportModelCard = (m: RegisteredModelRecord) => {
    const md = `# ${m.name} (${m.version})

## Model Details
- **Base Model**: ${m.baseModelName} (\`${m.baseModelId}\`)
- **Method**: ${m.method.toUpperCase()}
- **Dataset**: ${m.datasetName}
- **Trainable Parameters**: ${formatNumber(m.trainableParams)}
- **Adapter Size**: ${formatBytes(m.adapterSizeBytes)}
- **Created**: ${m.createdAt}

## Evaluation Metrics
- **Accuracy / Exact Match**: ${m.latestAccuracy !== null ? `${(m.latestAccuracy * 100).toFixed(2)}%` : 'Not evaluated'}
- **Macro F1 / Token F1**: ${m.latestMacroF1 !== null ? `${(m.latestMacroF1 * 100).toFixed(2)}%` : 'Not evaluated'}
- **Perplexity**: ${m.latestPerplexity ?? 'Not evaluated'}

## Training Configuration
\`\`\`json
${JSON.stringify(m.trainingConfig, null, 2)}
\`\`\`
`;
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${m.name}-model-card.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Model Registry
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Manage trained LoRA &amp; QLoRA adapters, download reproducible artifact bundles, export model cards, or deploy inference endpoints.
          </p>
        </div>
        <LiquidButton
          type="button"
          variant="default"
          onClick={() => onNavigate('studio')}
          className="shrink-0"
        >
          Train New Adapter
        </LiquidButton>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {activeDataset && (() => {
        const matchingCount = models.filter((m) => m.datasetId === activeDataset.id).length;
        return (
          <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.06] px-4 py-2.5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 min-w-0">
              <Database className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
              <span className="text-slate-700 dark:text-slate-200 truncate">
                Active Dataset:{' '}
                <strong className="font-semibold text-slate-900 dark:text-slate-100">
                  {activeDataset.name}
                </strong>{' '}
                — {matchingCount} registered {matchingCount === 1 ? 'adapter' : 'adapters'} trained on this dataset.
              </span>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('studio')}
              className="inline-flex items-center gap-1.5 text-blue-600 dark:text-blue-400 font-medium hover:underline shrink-0 self-start sm:self-auto"
            >
              <FlaskConical className="w-3.5 h-3.5" />
              Train Adapter on {activeDataset.name}
            </button>
          </div>
        );
      })()}

      {loading ? (
        <div className="p-8 text-xs text-slate-500">Loading registered models...</div>
      ) : models.length === 0 ? (
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-12 text-center">
          <Layers className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
            No saved adapters in Model Registry
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Completed training runs automatically register versioned adapters here.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {inspectedModel && (() => {
            const bundleDocs = buildAdapterBundleDocs(inspectedModel);
            const activeDoc =
              bundleDocs.find((d) => d.id === selectedDocId) ?? bundleDocs[0];
            return (
              <section
                aria-label="Adapter Artifact Bundle Inspector"
                className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/95 dark:bg-[#11141C]/95 p-5 space-y-5"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 shrink-0">
                      <FolderOpen className="w-4 h-4" />
                    </div>
                    <div>
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                        Adapter Artifact Bundle — {inspectedModel.name} ({inspectedModel.version})
                      </h2>
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        Flip open the glass PEFT binder, spread or drag artifact pages, and click any file to inspect its manifest &amp; weights.
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 shrink-0">
                    <div className="inline-flex rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] p-0.5">
                      {(['spread', 'row', 'stack'] as const).map((layoutMode) => (
                        <button
                          key={layoutMode}
                          type="button"
                          onClick={() => {
                            setBundleLayout(layoutMode);
                            setBundleOpen(true);
                          }}
                          className={`px-2 py-1 text-[10px] font-mono uppercase rounded-md transition-colors ${
                            bundleLayout === layoutMode
                              ? 'bg-indigo-600 text-white'
                              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                          }`}
                        >
                          {layoutMode}
                        </button>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() => setBundleOpen((prev) => !prev)}
                      className="px-2.5 py-1 text-[11px] font-medium rounded-lg border border-slate-200 dark:border-white/[0.12] bg-slate-50 dark:bg-white/[0.04] text-slate-700 dark:text-slate-200 hover:border-indigo-500/50 transition-colors"
                    >
                      {bundleOpen ? 'Close Cover' : 'Open Cover'}
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
                  <div
                    data-testid="registry-folder-stage"
                    className="lg:col-span-6 relative flex items-center justify-center rounded-xl border border-slate-200/70 dark:border-white/[0.06] bg-gradient-to-br from-slate-50/90 to-slate-100/80 dark:from-[#0C1018]/95 dark:to-[#080A0F]/95 py-12 px-8 min-h-[340px]"
                  >
                    <Folder
                      key={inspectedModel.id}
                      label={`${inspectedModel.method.toUpperCase()} Bundle`}
                      mode="book"
                      variant="glass"
                      layout={bundleLayout}
                      color="#6366F1"
                      paperColor="#ffffff"
                      size={0.85}
                      frost={0.28}
                      speed={1.55}
                      intro={false}
                      open={bundleOpen}
                      onOpenChange={setBundleOpen}
                      closeOnOutsideClick={false}
                      zoomOnClick={true}
                      theme={isDark ? 'dark' : 'light'}
                      onItemClick={(_entry, index) => {
                        const doc = bundleDocs[index];
                        if (!doc) return;
                        setSelectedDocId(doc.id);
                        if (doc.id === 'card') {
                          setSelectedCard(inspectedModel);
                        }
                      }}
                    >
                      {bundleDocs.map((doc) => (
                        <div
                          key={doc.id}
                          className="flex h-full flex-col justify-between overflow-hidden p-2 text-left"
                          style={{ backgroundColor: '#ffffff' }}
                        >
                          <div className="min-w-0">
                            <div
                              className="inline-block rounded px-1.5 py-0.5 font-mono text-[8px] font-bold text-white"
                              style={{ backgroundColor: doc.accentHex }}
                            >
                              {doc.badge}
                            </div>
                            <div className="mt-1 truncate text-[10px] font-bold leading-tight text-slate-900">
                              {doc.title}
                            </div>
                            <div className="mt-0.5 truncate font-mono text-[8px] text-slate-500">
                              {doc.fileName}
                            </div>
                            <div className="mt-1 line-clamp-2 text-[8px] leading-snug text-slate-600">
                              {doc.subtitle}
                            </div>
                          </div>
                          <div className="border-t border-slate-200 pt-1 font-mono text-[8px] text-slate-500 whitespace-nowrap truncate">
                            Click to inspect
                          </div>
                        </div>
                      ))}
                    </Folder>
                  </div>

                  <div className="lg:col-span-6 space-y-3">
                    <div className="flex flex-wrap gap-1.5">
                      {bundleDocs.map((doc) => (
                        <button
                          key={doc.id}
                          type="button"
                          onClick={() => setSelectedDocId(doc.id)}
                          className={`px-2.5 py-1 text-[11px] font-mono rounded-md border transition-colors ${
                            activeDoc.id === doc.id
                              ? 'border-indigo-500 bg-indigo-600 text-white'
                              : 'border-slate-200 dark:border-white/[0.1] text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-white/[0.04]'
                          }`}
                        >
                          {doc.fileName}
                        </button>
                      ))}
                    </div>

                    <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] overflow-hidden">
                      <div className="px-3.5 py-2 border-b border-slate-200 dark:border-white/[0.06] flex items-center justify-between gap-2">
                        <span className="font-mono text-[11px] font-medium text-slate-700 dark:text-slate-200 truncate">
                          {activeDoc.fileName} — {activeDoc.subtitle}
                        </span>
                        <button
                          type="button"
                          onClick={() => handleCopySnippet(activeDoc.previewCode)}
                          className="inline-flex items-center gap-1 text-[11px] text-slate-500 hover:text-indigo-500 shrink-0"
                        >
                          {copiedCode ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-500" />
                              Copied
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              Copy
                            </>
                          )}
                        </button>
                      </div>
                      <pre className="p-3.5 text-[11px] font-mono text-slate-800 dark:text-slate-200 overflow-x-auto max-h-48">
                        {activeDoc.previewCode}
                      </pre>
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-1">
                      <LiquidButton
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={() => setSelectedCard(inspectedModel)}
                      >
                        <FileText className="w-3.5 h-3.5" />
                        Open Full Model Card
                      </LiquidButton>
                      <LiquidButton
                        type="button"
                        size="sm"
                        variant="default"
                        onClick={() => onOpenPlayground(`reg:${inspectedModel.id}`)}
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                        Launch Playground
                      </LiquidButton>
                      <MetalButton
                        type="button"
                        variant="gold"
                        onClick={() => {
                          const link = document.createElement('a');
                          link.href = `/api/models/${inspectedModel.id}/download`;
                          link.click();
                        }}
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download .zip ({formatBytes(inspectedModel.adapterSizeBytes)})
                      </MetalButton>
                    </div>
                  </div>
                </div>
              </section>
            );
          })()}

          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 dark:border-white/[0.08] text-slate-500">
                <tr>
                  <th className="py-3 px-4 font-medium">Adapter Name</th>
                  <th className="py-3 px-4 font-medium">Base Model</th>
                  <th className="py-3 px-4 font-medium">Method</th>
                  <th className="py-3 px-4 font-medium">Dataset</th>
                  <th className="py-3 px-4 font-medium text-right">Size</th>
                  <th className="py-3 px-4 font-medium text-right">Latest Metric</th>
                  <th className="py-3 px-4 font-medium">Status</th>
                  <th className="py-3 px-4 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
                {models.map((m) => {
                  const isInspected = m.id === inspectedModel?.id;
                  return (
                    <tr
                      key={m.id}
                      className={
                        isInspected
                          ? 'bg-indigo-500/[0.06] dark:bg-indigo-500/[0.09]'
                          : 'hover:bg-slate-50 dark:hover:bg-white/[0.02]'
                      }
                    >
                      <td className="py-3 px-4">
                        <button
                          type="button"
                          onClick={() => {
                            setInspectedId(m.id);
                            setBundleOpen(true);
                          }}
                          className="text-left group"
                        >
                          <div className="font-medium text-slate-900 dark:text-slate-100 group-hover:text-indigo-500 transition-colors">
                            {m.name}
                          </div>
                          <div className="font-mono text-[11px] text-slate-500">
                            {m.version} · {formatNumber(m.trainableParams)} params
                          </div>
                        </button>
                      </td>
                      <td className="py-3 px-4 text-slate-600 dark:text-slate-300">{m.baseModelName}</td>
                      <td className="py-3 px-4 font-mono uppercase">{m.method}</td>
                      <td className="py-3 px-4 text-slate-600 dark:text-slate-300">
                        <div className="flex items-center gap-1.5">
                          <span>{m.datasetName}</span>
                          {activeDataset && m.datasetId === activeDataset.id && (
                            <span className="rounded px-1.5 py-0.5 text-[10px] font-mono bg-blue-500/10 text-blue-600 dark:text-blue-400">
                              active
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums whitespace-nowrap">
                        {formatBytes(m.adapterSizeBytes)}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums whitespace-nowrap">
                        {m.latestAccuracy !== null
                          ? `${(m.latestAccuracy * 100).toFixed(1)}%`
                          : m.latestPerplexity !== null
                            ? `PPL ${m.latestPerplexity}`
                            : 'Not evaluated'}
                      </td>
                      <td className="py-3 px-4 font-mono capitalize">
                        <span
                          className={
                            m.status === 'deployed'
                              ? 'text-emerald-600 dark:text-emerald-400 font-medium'
                              : 'text-slate-500 dark:text-slate-400'
                          }
                        >
                          {m.status}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            type="button"
                            onClick={() => {
                              setInspectedId(m.id);
                              setBundleOpen(true);
                            }}
                            className="inline-flex items-center gap-1 text-indigo-600 dark:text-indigo-400 hover:underline whitespace-nowrap"
                          >
                            <FolderOpen className="w-3 h-3" />
                            Inspect
                          </button>
                          <button
                            type="button"
                            onClick={() => onOpenPlayground(`reg:${m.id}`)}
                            className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline whitespace-nowrap"
                          >
                            <ExternalLink className="w-3 h-3" />
                            Playground
                          </button>
                          <button
                            type="button"
                            onClick={() => handleToggleDeploy(m)}
                            className="inline-flex items-center gap-1 text-slate-700 dark:text-slate-200 hover:text-blue-500 whitespace-nowrap"
                          >
                            <Rocket className="w-3 h-3" />
                            {m.status === 'deployed' ? 'Undeploy' : 'Deploy'}
                          </button>
                          <button
                            type="button"
                            onClick={() => setSelectedCard(m)}
                            className="inline-flex items-center gap-1 text-slate-700 dark:text-slate-200 hover:text-blue-500 whitespace-nowrap"
                          >
                            <FileText className="w-3 h-3" />
                            Card
                          </button>
                          <a
                            href={`/api/models/${m.id}/download`}
                            className="inline-flex items-center gap-1 text-slate-700 dark:text-slate-200 hover:text-blue-500 whitespace-nowrap"
                          >
                            <Download className="w-3 h-3" />
                            .zip
                          </a>
                          <button
                            type="button"
                            aria-label={`Delete ${m.name}`}
                            onClick={() => setConfirmDelete(m)}
                            className="text-slate-400 hover:text-red-500"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Card Modal */}
      {selectedCard && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        >
          <div className="w-full max-w-lg rounded-xl border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">{selectedCard.name} — Model Card</h2>
              <span className="font-mono text-xs text-slate-500">{selectedCard.version}</span>
            </div>
            <pre className="p-3 rounded-lg bg-slate-100 dark:bg-[#090B10] text-[11px] font-mono overflow-x-auto max-h-72">
              {JSON.stringify(selectedCard.trainingConfig, null, 2)}
            </pre>
            <div className="flex justify-end gap-2">
              <LiquidButton
                type="button"
                size="sm"
                variant="outline"
                onClick={() => handleExportModelCard(selectedCard)}
              >
                Export README.md
              </LiquidButton>
              <LiquidButton
                type="button"
                size="sm"
                variant="default"
                onClick={() => setSelectedCard(null)}
              >
                Close
              </LiquidButton>
            </div>
          </div>
        </div>
      )}

      {/* Inference API Snippet Modal */}
      {deployModal && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        >
          <div className="w-full max-w-lg rounded-xl border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] p-6 space-y-4">
            <h2 className="text-base font-semibold">
              Inference Endpoint Active: {deployModal.name}
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Send streaming inference requests directly against this registered adapter via POST /api/inference/chat:
            </p>
            <pre className="p-3 rounded-lg bg-[#090B10] text-slate-200 text-[11px] font-mono overflow-x-auto">
{`curl -N -X POST ${window.location.origin}/api/inference/chat \\
  -H "Content-Type: application/json" \\
  -d '{"modelRef":"reg:${deployModal.id}","prompt":"I lost my debit card"}'`}
            </pre>
            <div className="flex justify-end">
              <LiquidButton
                type="button"
                size="sm"
                variant="default"
                onClick={() => setDeployModal(null)}
              >
                Done
              </LiquidButton>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation */}
      {confirmDelete && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        >
          <div className="w-full max-w-sm rounded-xl border border-slate-200 dark:border-white/[0.1] bg-white dark:bg-[#11141C] p-6 space-y-4">
            <h2 className="text-base font-semibold">Delete Registered Adapter?</h2>
            <p className="text-xs text-slate-500">
              Remove <span className="font-mono">{confirmDelete.name}</span> from the Model Registry?
            </p>
            <div className="flex justify-end gap-2">
              <LiquidButton
                type="button"
                size="sm"
                variant="outline"
                onClick={() => setConfirmDelete(null)}
              >
                Cancel
              </LiquidButton>
              <button
                type="button"
                onClick={handleDelete}
                className="px-3.5 py-2 text-xs rounded-lg bg-red-600 text-white"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
