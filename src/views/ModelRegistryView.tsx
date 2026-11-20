import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Download,
  ExternalLink,
  FileText,
  Layers,
  Rocket,
  Trash2,
} from 'lucide-react';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { api } from '../lib/api';
import { formatBytes, formatNumber } from '../lib/utils';
import type { RegisteredModelRecord } from '../shared/types';
import type { NavSection } from './OverviewView';

interface ModelRegistryViewProps {
  onOpenPlayground: (modelRef: string) => void;
  onNavigate: (section: NavSection) => void;
}

export function ModelRegistryView({
  onOpenPlayground,
  onNavigate,
}: ModelRegistryViewProps) {
  const [models, setModels] = useState<RegisteredModelRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCard, setSelectedCard] = useState<RegisteredModelRecord | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<RegisteredModelRecord | null>(null);
  const [deployModal, setDeployModal] = useState<RegisteredModelRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadRegistry = async () => {
    setLoading(true);
    try {
      const list = await api.getRegistryModels();
      setModels(list);
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
              {models.map((m) => (
                <tr key={m.id} className="hover:bg-slate-50 dark:hover:bg-white/[0.02]">
                  <td className="py-3 px-4">
                    <div className="font-medium text-slate-900 dark:text-slate-100">{m.name}</div>
                    <div className="font-mono text-[11px] text-slate-500">
                      {m.version} · {formatNumber(m.trainableParams)} params
                    </div>
                  </td>
                  <td className="py-3 px-4 text-slate-600 dark:text-slate-300">{m.baseModelName}</td>
                  <td className="py-3 px-4 font-mono uppercase">{m.method}</td>
                  <td className="py-3 px-4 text-slate-600 dark:text-slate-300">{m.datasetName}</td>
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
              ))}
            </tbody>
          </table>
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
              <button
                type="button"
                onClick={() => setConfirmDelete(null)}
                className="px-3.5 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1]"
              >
                Cancel
              </button>
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
