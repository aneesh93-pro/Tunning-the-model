import React, { useEffect, useState } from 'react';
import { AlertTriangle, Key, Shield, Trash2 } from 'lucide-react';
import { RubberSegment } from '../components/reactbits/RubberSegment';
import { api } from '../lib/api';
import { useIsDark } from '../lib/useDarkMode';
import type { HardwareTelemetry } from '../shared/types';

interface SettingsViewProps {
  theme: 'dark' | 'light';
  onToggleTheme: (theme: 'dark' | 'light') => void;
  onWorkspaceUpdated: () => void;
}

export function SettingsView({
  theme,
  onToggleTheme,
  onWorkspaceUpdated,
}: SettingsViewProps) {
  const isDark = useIsDark();
  const [wsName, setWsName] = useState('');
  const [wsDesc, setWsDesc] = useState('');
  const [apiKeys, setApiKeys] = useState<
    Array<{ id: string; name: string; prefix: string; role: string; created_at: string }>
  >([]);
  const [auditLogs, setAuditLogs] = useState<
    Array<{
      id: number;
      actor: string;
      action: string;
      resource_type: string;
      resource_id: string;
      detail: string;
      created_at: string;
    }>
  >([]);
  const [hardware, setHardware] = useState<HardwareTelemetry | null>(null);
  const [geminiConfigured, setGeminiConfigured] = useState(false);

  const [newKeyName, setNewKeyName] = useState('');
  const [newKeyRole, setNewKeyRole] = useState('engineer');
  const [revealedSecret, setRevealedSecret] = useState<string | null>(null);
  const [savedNotice, setSavedNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadSettings = async () => {
    try {
      const s = await api.getSettings();
      setWsName(s.workspace.name);
      setWsDesc(s.workspace.description);
      setApiKeys(s.apiKeys);
      setAuditLogs(s.auditLogs);
      setHardware(s.hardware);
      setGeminiConfigured(s.geminiConfigured);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    void loadSettings();
  }, []);

  const handleSaveWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!wsName.trim()) {
      setError('Workspace name is required.');
      return;
    }
    try {
      await api.updateWorkspace(wsName.trim(), wsDesc);
      setError(null);
      setSavedNotice('Workspace settings saved.');
      onWorkspaceUpdated();
      setTimeout(() => setSavedNotice(null), 2500);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKeyName.trim()) {
      setError('Token label is required.');
      return;
    }
    try {
      const res = await api.createApiKey(newKeyName.trim(), newKeyRole);
      setRevealedSecret(res.secretToken);
      setNewKeyName('');
      setError(null);
      await loadSettings();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
          Settings &amp; Security
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Configure workspace metadata, appearance, RBAC API tokens, runtime diagnostics, and audit trails.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Workspace & Theme */}
        <form
          onSubmit={handleSaveWorkspace}
          className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4"
        >
          <h2 className="text-sm font-semibold">Workspace &amp; Appearance</h2>
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
              Workspace Name
            </label>
            <input
              type="text"
              value={wsName}
              onChange={(e) => setWsName(e.target.value)}
              className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
              Description
            </label>
            <input
              type="text"
              value={wsDesc}
              onChange={(e) => setWsDesc(e.target.value)}
              className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5">
              Interface Theme
            </label>
            <RubberSegment
              aria-label="Interface Theme"
              size="sm"
              radius={8}
              inset={3}
              value={theme}
              onChange={(val) => onToggleTheme(val as 'dark' | 'light')}
              trackColor={isDark ? '#090B10' : '#F1F5F9'}
              thumbColor="#2563EB"
              textColor={isDark ? '#94A3B8' : '#475569'}
              activeTextColor="#FFFFFF"
              className="border border-slate-200 dark:border-white/[0.08]"
              items={[
                { value: 'dark', label: 'Dark Matte' },
                { value: 'light', label: 'Clean Light' },
              ]}
            />
          </div>
          <div className="flex items-center justify-between pt-2">
            <span className="text-xs text-emerald-600 dark:text-emerald-400">
              {savedNotice || ''}
            </span>
            <button
              type="submit"
              className="px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500"
            >
              Save Workspace
            </button>
          </div>
        </form>

        {/* Runtime & GPU Diagnostics */}
        <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-blue-500" />
            <h2 className="text-sm font-semibold">Runtime &amp; Deployment Diagnostics</h2>
          </div>
          <dl className="space-y-2 text-xs">
            <div className="flex justify-between">
              <dt className="text-slate-500">Execution Engine</dt>
              <dd className="font-mono">{hardware?.executionBackend}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">CUDA Accelerator</dt>
              <dd className="font-mono">
                {hardware?.gpuAvailable ? hardware.gpuName : 'Unavailable (CPU sandbox)'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">Inference Model Backing</dt>
              <dd className="font-mono">
                {geminiConfigured ? 'LoRA Adapter + Gemini Server API' : 'Local LoRA Adapter'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-slate-500">Python GPU Stack Export</dt>
              <dd className="font-mono">/backend (FastAPI + PEFT + PostgreSQL)</dd>
            </div>
          </dl>
          <p className="pt-3 border-t border-slate-200 dark:border-white/[0.06] text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
            Every experiment exports a standalone PyTorch + Hugging Face PEFT + BitsAndBytes{' '}
            <code className="font-mono">train_peft.py</code> script alongside the trained local adapter weights for seamless GPU worker scaling.
          </p>
        </div>
      </div>

      {/* RBAC API Keys */}
      <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Key className="w-4 h-4 text-blue-500" />
          <h2 className="text-sm font-semibold">Inference API Tokens (RBAC)</h2>
        </div>

        <form onSubmit={handleCreateKey} className="flex flex-col sm:flex-row gap-2.5">
          <input
            type="text"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            placeholder="Token label (e.g. staging-inference-worker)"
            className="flex-1 px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
          />
          <select
            value={newKeyRole}
            onChange={(e) => setNewKeyRole(e.target.value)}
            className="px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] font-mono"
          >
            <option value="admin">admin</option>
            <option value="engineer">engineer</option>
            <option value="viewer">viewer</option>
          </select>
          <button
            type="submit"
            className="px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-500 whitespace-nowrap"
          >
            Generate Token
          </button>
        </form>

        {revealedSecret && (
          <div className="p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-xs font-mono break-all">
            Copy this token now (shown once): <strong>{revealedSecret}</strong>
          </div>
        )}

        {apiKeys.length > 0 && (
          <table className="w-full text-left text-xs font-mono">
            <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500">
              <tr>
                <th className="py-2 pr-3 font-medium">Name</th>
                <th className="py-2 pr-3 font-medium">Prefix</th>
                <th className="py-2 pr-3 font-medium">Role</th>
                <th className="py-2 text-right font-medium">Revoke</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06]">
              {apiKeys.map((k) => (
                <tr key={k.id}>
                  <td className="py-2 pr-3">{k.name}</td>
                  <td className="py-2 pr-3">{k.prefix}</td>
                  <td className="py-2 pr-3">{k.role}</td>
                  <td className="py-2 text-right">
                    <button
                      type="button"
                      onClick={() =>
                        api
                          .revokeApiKey(k.id)
                          .then(loadSettings)
                          .catch((err) => setError((err as Error).message))
                      }
                      className="text-red-500 hover:underline inline-flex items-center gap-1"
                    >
                      <Trash2 className="w-3 h-3" /> Revoke
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Audit Trail */}
      <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-200 dark:border-white/[0.08] text-xs font-semibold">
          Security &amp; Experiment Audit Trail
        </div>
        <div className="overflow-x-auto max-h-64">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-200 dark:border-white/[0.06] text-slate-500">
              <tr>
                <th className="py-2 px-4 font-medium">Timestamp</th>
                <th className="py-2 px-4 font-medium">Action</th>
                <th className="py-2 px-4 font-medium">Resource</th>
                <th className="py-2 px-4 font-medium">Detail</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06] font-mono">
              {auditLogs.map((log) => (
                <tr key={log.id}>
                  <td className="py-2 px-4 text-slate-500 whitespace-nowrap">
                    {log.created_at.slice(0, 19).replace('T', ' ')}
                  </td>
                  <td className="py-2 px-4">{log.action}</td>
                  <td className="py-2 px-4">
                    {log.resource_type}/{log.resource_id}
                  </td>
                  <td className="py-2 px-4 text-slate-600 dark:text-slate-300">{log.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
