import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowUp,
  Check,
  CheckCircle2,
  ClipboardCheck,
  Copy,
  Download,
  FileSpreadsheet,
  FileText,
  Layers,
  MessageSquare,
  Play,
  Send,
  ShieldCheck,
  Trash2,
  XCircle,
} from 'lucide-react';
import Markdown from 'react-markdown';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { AquariumChatCanvas } from '../components/reactbits/AquariumChatCanvas';
import { LatticeLoader } from '../components/reactbits/LatticeLoader';
import { RubberSegment } from '../components/reactbits/RubberSegment';
import { api } from '../lib/api';
import { useIsDark } from '../lib/useDarkMode';
import { summarizePlaygroundBatchReports } from '../ml/structuredInference';
import type {
  DatasetRecord,
  PlaygroundBatchTestSummary,
  PlaygroundMessage,
  PlaygroundTestReport,
  StructuredInferenceOutput,
} from '../shared/types';

const MODEL_GROUPS = [
  'Trained Adapter',
  'ChatGPT',
  'Claude',
  'Gemini',
  'DeepSeek',
  'NVIDIA NIM',
  'Z.ai',
  'Open SLM',
  'Llama',
  'Mistral',
] as const;

interface ModelPlaygroundViewProps {
  initialModelRef: string | null;
}

const CHOICE_PRESETS: Array<{ label: string; value: string }> = [
  { label: 'Auto', value: '' },
  { label: 'Boolean', value: 'true, false, uncertain' },
  {
    label: 'Banking',
    value: 'card_lost_or_stolen, internal_transfer, balance_inquiry, fees_and_rates',
  },
];

const QUICK_PROMPTS = [
  {
    label: 'Card Lost Intent',
    prompt: 'I lost my debit card on the train this morning. Freeze it right away.',
    choices: 'card_lost_or_stolen, internal_transfer, balance_inquiry, fees_and_rates',
  },
  {
    label: 'Anti-Hallucination Probe (JEV AI)',
    prompt: 'Verify whether "JEV AI" is an approved intent class or registered adapter in this workspace.',
    choices: 'true, false, uncertain',
  },
];

function downloadTextFile(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function buildMarkdownReport(report: PlaygroundTestReport): string {
  const header = [
    `# MiniMind AI — Model Playground Diagnostic Report`,
    ``,
    `- **Report ID:** \`${report.id}\``,
    `- **Model:** ${report.modelLabel} (\`${report.modelRef}\`)`,
    `- **Dataset:** ${report.datasetName}`,
    `- **Suite Mode:** ${report.suiteMode}`,
    `- **Verdict:** **${report.verdict.toUpperCase()}**`,
    `- **Pass Rate:** ${(report.passRate * 100).toFixed(1)}% (${report.passedCount}/${report.totalCases})`,
    `- **Hallucination Rate:** ${(report.hallucinationRate * 100).toFixed(1)}% (${report.hallucinationCount} flagged)`,
    `- **Mean Confidence Score:** ${report.avgScore.toFixed(4)}`,
    `- **Avg Latency:** ${report.avgLatencyMs.toFixed(2)} ms`,
    `- **Generated:** ${report.createdAt}`,
    ``,
    `## Case Breakdown`,
    ``,
    `| # | Category | Expected | Boolean | Choice | Score | Status | Prompt |`,
    `|---:|---|---|---:|---|---:|---|---|`,
  ];

  const rows = report.cases.map(
    (c) =>
      `| ${c.index + 1} | ${c.category} | \`${c.expectedChoice}\` (${String(c.expectedBoolean)}) | \`${String(c.booleanValue)}\` | \`${c.choice}\` | ${c.score.toFixed(4)} | ${c.hallucinationFlag ? 'HALLUCINATION' : c.passed ? 'PASS' : 'FAIL'} | ${c.prompt.replace(/\|/g, '\\|').slice(0, 72)} |`
  );

  return [...header, ...rows].join('\n');
}

function buildBatchCsvReport(
  rows: Array<{
    rank: number | null;
    modelLabel: string;
    modelRef: string;
    aiTool: string;
    method: string;
    report: PlaygroundTestReport | null;
  }>
): string {
  const header = [
    'Rank',
    'Model',
    'ModelRef',
    'Family',
    'Method',
    'Dataset',
    'SuiteMode',
    'Verdict',
    'PassRatePct',
    'PassedCases',
    'TotalCases',
    'Hallucinations',
    'HallucinationRatePct',
    'MeanScore',
    'AvgLatencyMs',
  ].join(',');

  const escapeCsv = (val: string) => `"${val.replace(/"/g, '""')}"`;

  const lines = rows.map((row) => {
    const r = row.report;
    return [
      row.rank ?? '-',
      escapeCsv(row.modelLabel),
      escapeCsv(row.modelRef),
      escapeCsv(row.aiTool),
      escapeCsv(row.method),
      escapeCsv(r?.datasetName ?? 'Not tested'),
      escapeCsv(r?.suiteMode ?? '-'),
      escapeCsv(r ? r.verdict.toUpperCase() : 'UNTESTED'),
      r ? (r.passRate * 100).toFixed(1) : '-',
      r ? r.passedCount : '-',
      r ? r.totalCases : '-',
      r ? r.hallucinationCount : '-',
      r ? (r.hallucinationRate * 100).toFixed(1) : '-',
      r ? r.avgScore.toFixed(4) : '-',
      r ? r.avgLatencyMs.toFixed(2) : '-',
    ].join(',');
  });

  return [header, ...lines].join('\n');
}

function buildBatchMarkdownReport(
  rows: Array<{
    rank: number | null;
    modelLabel: string;
    modelRef: string;
    aiTool: string;
    method: string;
    report: PlaygroundTestReport | null;
  }>,
  summary: PlaygroundBatchTestSummary
): string {
  const header = [
    `# MiniMind AI — All Playground Models Benchmark Report`,
    ``,
    `- **Models Tested:** ${summary.totalModels}`,
    `- **Verdict Breakdown:** ${summary.passedModels} PASS · ${summary.reviewModels} REVIEW · ${summary.failedModels} FAIL`,
    `- **Mean Pass Rate:** ${(summary.avgPassRate * 100).toFixed(1)}%`,
    `- **Total Hallucinations Flagged:** ${summary.totalHallucinations}`,
    `- **Mean Confidence Score:** ${summary.avgScore.toFixed(4)}`,
    `- **Mean Latency:** ${summary.avgLatencyMs.toFixed(2)} ms`,
    ``,
    `| Rank | Model | Family | Verdict | Pass Rate | Hallucinations | Mean Score | Avg Latency |`,
    `|---:|---|---|---|---:|---:|---:|---:|`,
  ];

  const lines = rows.map((row) => {
    const r = row.report;
    if (!r) {
      return `| - | ${row.modelLabel} | ${row.aiTool} | UNTESTED | - | - | - | - |`;
    }
    return `| ${row.rank ?? '-'} | ${row.modelLabel} | ${row.aiTool} | **${r.verdict.toUpperCase()}** | ${(r.passRate * 100).toFixed(1)}% (${r.passedCount}/${r.totalCases}) | ${r.hallucinationCount} (${(r.hallucinationRate * 100).toFixed(1)}%) | ${r.avgScore.toFixed(4)} | ${r.avgLatencyMs.toFixed(2)} ms |`;
  });

  return [...header, ...lines].join('\n');
}

export function ModelPlaygroundView({ initialModelRef }: ModelPlaygroundViewProps) {
  const isDark = useIsDark();
  const [models, setModels] = useState<
    Array<{
      ref: string;
      label: string;
      method: string;
      baseModelId: string;
      aiTool?: string;
      provider?: string;
    }>
  >([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [selectedRef, setSelectedRef] = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [suiteMode, setSuiteMode] =
    useState<PlaygroundTestReport['suiteMode']>('dataset_and_probes');
  const [viewMode, setViewMode] = useState<'chat' | 'report' | 'all'>('chat');
  const [toolFilter, setToolFilter] = useState<string>('All');

  const [systemPrompt, setSystemPrompt] = useState(
    'You are a grounded domain-adapted Small Language Model. Return verified boolean, choice, and confidence score without fabricating entities.'
  );
  const [candidateChoicesInput, setCandidateChoicesInput] = useState('');
  const [temperature, setTemperature] = useState(0.2);
  const [topP, setTopP] = useState(0.9);
  const [maxNewTokens, setMaxNewTokens] = useState(256);

  const [messages, setMessages] = useState<PlaygroundMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const [conversations, setConversations] = useState<
    Array<{
      id: string;
      modelRef: string;
      modelLabel: string;
      systemPrompt: string;
      messages: PlaygroundMessage[];
      updatedAt: string;
    }>
  >([]);
  const [activeConvId, setActiveConvId] = useState<string | undefined>(undefined);

  const [reports, setReports] = useState<PlaygroundTestReport[]>([]);
  const [activeReport, setActiveReport] = useState<PlaygroundTestReport | null>(null);
  const [testingModel, setTestingModel] = useState(false);
  const [testingAllModels, setTestingAllModels] = useState(false);
  const [testingRef, setTestingRef] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.getInferenceModels(),
      api.getConversations(),
      api.getDatasets(),
      api.getPlaygroundTestReports(),
    ]).then(([mRes, convs, dsList, repList]) => {
      setModels(mRes.models);
      setConversations(convs);
      const readyDs = dsList.filter((d) => d.status === 'training_ready');
      setDatasets(readyDs.length > 0 ? readyDs : dsList);
      if (readyDs.length > 0) setSelectedDatasetId(readyDs[0].id);
      else if (dsList.length > 0) setSelectedDatasetId(dsList[0].id);

      setReports(repList);
      if (repList.length > 0) setActiveReport(repList[0]);

      if (initialModelRef && mRes.models.some((m) => m.ref === initialModelRef)) {
        setSelectedRef(initialModelRef);
      } else if (mRes.models.length > 0) {
        setSelectedRef(mRes.models[0].ref);
      }
    }).catch((err) => {
      setReportError((err as Error).message);
    });
  }, [initialModelRef]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || !selectedRef || streaming) return;

    const userMsg: PlaygroundMessage = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };
    const assistantId = `msg_${Date.now() + 1}`;
    const assistantMsg: PlaygroundMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    };

    const nextHistory = [...messages, userMsg];
    setMessages([...nextHistory, assistantMsg]);
    setInput('');
    setStreaming(true);

    let finalContent = '';
    let finalLatency: number | undefined;
    let finalTokens: number | undefined;
    let finalPredClass: string | undefined;
    let finalConf: number | undefined;
    let finalStructured: StructuredInferenceOutput | undefined;

    let streamHadError = false;
    try {
      const response = await fetch('/api/inference/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelRef: selectedRef,
          prompt: text,
          systemPrompt,
          choices: candidateChoicesInput.trim() || undefined,
          history: nextHistory.map((m) => ({ role: m.role, content: m.content })),
          temperature,
          topP,
          maxNewTokens,
        }),
      });

      if (!response.ok) {
        let msg = `Inference request failed (${response.status})`;
        try {
          const errJson = (await response.json()) as { error?: string };
          if (errJson?.error) msg = errJson.error;
        } catch {
          // ignore non-JSON body
        }
        throw new Error(msg);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (reader) {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';

          for (const chunk of lines) {
            if (!chunk.startsWith('data: ')) continue;
            const data = JSON.parse(chunk.slice(6)) as {
              type: string;
              text?: string;
              error?: string;
              meta?: {
                predictedClass?: string;
                confidence?: number;
                latencyMs?: number;
                tokensGenerated?: number;
                structured?: StructuredInferenceOutput;
              };
            };

            if (data.type === 'meta' && data.meta) {
              finalPredClass = data.meta.predictedClass;
              finalConf = data.meta.confidence;
              finalStructured = data.meta.structured;
            } else if (data.type === 'token' && data.text) {
              finalContent += data.text;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: finalContent,
                        predictedClass: finalPredClass,
                        confidence: finalConf,
                        structured: finalStructured,
                      }
                    : m
                )
              );
            } else if (data.type === 'done' && data.meta) {
              finalLatency = data.meta.latencyMs;
              finalTokens = data.meta.tokensGenerated;
              finalPredClass = data.meta.predictedClass ?? finalPredClass;
              finalConf = data.meta.confidence ?? finalConf;
              finalStructured = data.meta.structured ?? finalStructured;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: finalContent,
                        latencyMs: finalLatency,
                        tokensGenerated: finalTokens,
                        predictedClass: finalPredClass,
                        confidence: finalConf,
                        structured: finalStructured,
                      }
                    : m
                )
              );
            } else if (data.type === 'error') {
              streamHadError = true;
              finalContent = `Error: ${data.error}`;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, content: finalContent } : m))
              );
            }
          }
        }
      }

      if (!streamHadError && finalContent.trim()) {
        const modelLabel = models.find((m) => m.ref === selectedRef)?.label || selectedRef;
        const saved = await api.saveConversation({
          id: activeConvId,
          modelRef: selectedRef,
          modelLabel,
          systemPrompt,
          messages: [
            ...nextHistory,
            {
              ...assistantMsg,
              content: finalContent,
              latencyMs: finalLatency,
              tokensGenerated: finalTokens,
              predictedClass: finalPredClass,
              confidence: finalConf,
              structured: finalStructured,
            },
          ],
        });
        setActiveConvId(saved.id);
        const updatedConvs = await api.getConversations();
        setConversations(updatedConvs);
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: `Error: ${(err as Error).message}` } : m
        )
      );
    } finally {
      setStreaming(false);
    }
  };

  const handleRunModelReport = async (overrideModelRef?: string, openReportTab = true) => {
    const targetRef = overrideModelRef ?? selectedRef;
    if (!targetRef || testingModel || testingAllModels) return;
    setTestingModel(true);
    setTestingRef(targetRef);
    setReportError(null);
    try {
      const report = await api.runPlaygroundTestReport({
        modelRef: targetRef,
        datasetId: selectedDatasetId || undefined,
        suiteMode,
      });
      setActiveReport(report);
      setReports((prev) => [report, ...prev.filter((r) => r.id !== report.id)]);
      if (openReportTab) {
        setViewMode('report');
      }
    } catch (err) {
      setReportError((err as Error).message);
    } finally {
      setTestingModel(false);
      setTestingRef(null);
    }
  };

  const handleRunAllModelsReport = async () => {
    if (models.length === 0 || testingAllModels || testingModel) return;
    setTestingAllModels(true);
    setReportError(null);
    try {
      const result = await api.runPlaygroundTestAllReports({
        modelRefs: models.map((m) => m.ref),
        datasetId: selectedDatasetId || undefined,
        suiteMode,
      });
      setReports((prev) => {
        const incomingIds = new Set(result.reports.map((r) => r.id));
        return [...result.reports, ...prev.filter((r) => !incomingIds.has(r.id))];
      });
      if (result.reports.length > 0) {
        const matchSelected = result.reports.find((r) => r.modelRef === selectedRef);
        setActiveReport(matchSelected ?? result.reports[0]);
      }
      setViewMode('all');
    } catch (err) {
      setReportError((err as Error).message);
    } finally {
      setTestingAllModels(false);
    }
  };

  const latestReportByModelRef = useMemo(() => {
    const map = new Map<string, PlaygroundTestReport>();
    for (const r of reports) {
      if (!map.has(r.modelRef)) {
        map.set(r.modelRef, r);
      }
    }
    return map;
  }, [reports]);

  const { filteredLeaderboardRows, batchSummary } = useMemo(() => {
    const matchingModels = models.filter(
      (m) => toolFilter === 'All' || (m.aiTool ?? 'Open SLM') === toolFilter
    );
    const testedReports: PlaygroundTestReport[] = [];
    const modelByRef = new Map(matchingModels.map((m) => [m.ref, m]));

    for (const m of matchingModels) {
      const rep = latestReportByModelRef.get(m.ref);
      if (rep) testedReports.push(rep);
    }

    const { sortedReports, summary } = summarizePlaygroundBatchReports(testedReports);
    const testedRefs = new Set(sortedReports.map((r) => r.modelRef));

    const rankedRows = sortedReports.map((rep, idx) => {
      const m = modelByRef.get(rep.modelRef);
      return {
        rank: idx + 1,
        modelRef: rep.modelRef,
        modelLabel: m?.label ?? rep.modelLabel,
        aiTool: m?.aiTool ?? 'Open SLM',
        method: m?.method ?? rep.method,
        report: rep as PlaygroundTestReport | null,
      };
    });

    const untestedRows = matchingModels
      .filter((m) => !testedRefs.has(m.ref))
      .map((m) => ({
        rank: null,
        modelRef: m.ref,
        modelLabel: m.label,
        aiTool: m.aiTool ?? 'Open SLM',
        method: m.method,
        report: null as PlaygroundTestReport | null,
      }));

    return {
      filteredLeaderboardRows: [...rankedRows, ...untestedRows],
      batchSummary: summary,
    };
  }, [models, toolFilter, latestReportByModelRef]);

  const handleCopy = (id: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
            Model Playground &amp; Grounded Verification
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Structured boolean, choice, and score inference with anti-hallucination verification and per-model diagnostic reports.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <div className="max-w-full overflow-x-auto">
            <RubberSegment
              aria-label="Playground view mode"
              value={viewMode}
              size="sm"
              equalSlots={false}
              className="w-max"
              trackColor={isDark ? '#171B26' : '#E2E8F0'}
              thumbColor="#2563EB"
              textColor={isDark ? '#94A3B8' : '#475569'}
              activeTextColor="#FFFFFF"
              onChange={(val) => setViewMode(val as 'chat' | 'report' | 'all')}
              items={[
                { value: 'chat', label: 'Structured Chat' },
                { value: 'report', label: `Model Report (${reports.length})` },
                { value: 'all', label: `All Models (${models.length})` },
              ]}
            />
          </div>
          <LiquidButton
            type="button"
            variant="default"
            size="sm"
            onClick={() => handleRunModelReport()}
            disabled={!selectedRef || testingModel || testingAllModels}
            className="whitespace-nowrap"
          >
            <ClipboardCheck className="w-3.5 h-3.5" />
            {testingModel ? 'Testing Model...' : 'Test Model & Generate Report'}
          </LiquidButton>
          <LiquidButton
            type="button"
            variant="default"
            size="sm"
            onClick={handleRunAllModelsReport}
            disabled={models.length === 0 || testingAllModels || testingModel}
            className="whitespace-nowrap"
          >
            <Layers className="w-3.5 h-3.5" />
            {testingAllModels
              ? `Testing All (${models.length})...`
              : `Test All Models (${models.length})`}
          </LiquidButton>
          <LiquidButton
            type="button"
            size="sm"
            variant="outline"
            onClick={() => {
              setMessages([]);
              setActiveConvId(undefined);
            }}
            className="shrink-0 whitespace-nowrap"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Chat
          </LiquidButton>
        </div>
      </div>

      {reportError && (
        <div className="rounded-lg border border-red-500/35 bg-red-50/95 dark:bg-[#220F14]/95 backdrop-blur-md px-4 py-2.5 text-xs text-red-700 dark:text-red-300">
          {reportError}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Parameters, Diagnostic Runner & Saved Sessions */}
        <div className="lg:col-span-4 min-w-0 space-y-5">
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-4">
            <h2 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
              Inference &amp; Verification Config
            </h2>
            <div>
              <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                Active Model / Adapter
              </label>
              <select
                aria-label="Active Model or Adapter"
                value={selectedRef}
                onChange={(e) => setSelectedRef(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
              >
                {MODEL_GROUPS.map((group) => {
                  const groupItems = models.filter((m) => (m.aiTool ?? 'Open SLM') === group);
                  if (groupItems.length === 0) return null;
                  return (
                    <optgroup key={group} label={group}>
                      {groupItems.map((m) => (
                        <option key={m.ref} value={m.ref}>
                          {m.label}
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {[
                  { ref: 'base:gpt-4o', label: 'GPT-4o' },
                  { ref: 'base:claude-3.7-sonnet', label: 'Claude 3.7 Sonnet' },
                  { ref: 'base:claude-3-opus', label: 'Claude 3 Opus' },
                  { ref: 'base:gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
                  { ref: 'base:deepseek-r1', label: 'DeepSeek-R1' },
                  { ref: 'base:nvidia-llama-3.1-nemotron-70b', label: 'Nemotron-70B' },
                  { ref: 'base:zai-glm-4.5', label: 'GLM-4.5' },
                ]
                  .filter((q) => models.some((m) => m.ref === q.ref))
                  .map((q) => (
                    <button
                      key={q.ref}
                      type="button"
                      onClick={() => setSelectedRef(q.ref)}
                      className={`px-2 py-1 text-[10px] font-medium rounded-md border transition-colors ${
                        selectedRef === q.ref
                          ? 'border-blue-500 bg-blue-500/15 text-blue-600 dark:text-blue-400'
                          : 'border-slate-200 dark:border-white/[0.08] text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-white/[0.04]'
                      }`}
                    >
                      {q.label}
                    </button>
                  ))}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                  Allowed Choices (comma-separated)
                </label>
                <div className="flex items-center gap-1.5">
                  {CHOICE_PRESETS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => setCandidateChoicesInput(p.value)}
                      className="text-[10px] font-medium text-blue-600 dark:text-blue-400 hover:underline"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
              <input
                type="text"
                value={candidateChoicesInput}
                onChange={(e) => setCandidateChoicesInput(e.target.value)}
                placeholder="Auto from dataset / boolean prompt"
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
              />
            </div>

            <div>
              <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                System Prompt
              </label>
              <textarea
                rows={2}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
              />
            </div>

            <div className="space-y-2.5 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-slate-500">Temperature</span>
                  <span className="font-mono tabular-nums">{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-slate-500">Top-p</span>
                  <span className="font-mono tabular-nums">{topP}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-slate-500">Max New Tokens</span>
                  <span className="font-mono tabular-nums">{maxNewTokens}</span>
                </div>
                <input
                  type="range"
                  min="32"
                  max="1024"
                  step="32"
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* Model Diagnostic Benchmark Suite */}
          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Diagnostic Test Suite
              </h2>
              <ShieldCheck className="w-4 h-4 text-blue-500" />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                Benchmark Dataset
              </label>
              <select
                aria-label="Benchmark Dataset"
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1">
                Suite Scope
              </label>
              <select
                aria-label="Suite Scope"
                value={suiteMode}
                onChange={(e) => setSuiteMode(e.target.value as PlaygroundTestReport['suiteMode'])}
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
              >
                <option value="dataset_and_probes">Held-Out Split + Anti-Hallucination</option>
                <option value="dataset_only">Held-Out Test Split Only</option>
                <option value="anti_hallucination">Anti-Hallucination Probes Only</option>
              </select>
            </div>
            <LiquidButton
              type="button"
              size="sm"
              variant="default"
              onClick={() => handleRunModelReport()}
              disabled={!selectedRef || testingModel || testingAllModels}
              className="w-full"
            >
              <Play className="w-3.5 h-3.5" />
              {testingModel ? 'Running Suite...' : 'Run Benchmark & View Report'}
            </LiquidButton>
            <LiquidButton
              type="button"
              size="sm"
              variant="outline"
              onClick={handleRunAllModelsReport}
              disabled={models.length === 0 || testingAllModels || testingModel}
              className="w-full"
            >
              <Layers className="w-3.5 h-3.5" />
              {testingAllModels
                ? `Testing All ${models.length} Models...`
                : `Test Every Model (${models.length})`}
            </LiquidButton>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5">
            <h2 className="text-xs font-semibold text-slate-900 dark:text-slate-100 mb-3">
              Saved Sessions ({conversations.length})
            </h2>
            {conversations.length === 0 ? (
              <p className="text-xs text-slate-500">Send a prompt to save conversation history.</p>
            ) : (
              <ul className="divide-y divide-slate-200 dark:divide-white/[0.06] text-xs">
                {conversations.map((c) => (
                  <li key={c.id} className="py-2 flex items-center justify-between gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setActiveConvId(c.id);
                        setSelectedRef(c.modelRef);
                        setSystemPrompt(c.systemPrompt);
                        setMessages(c.messages);
                        setViewMode('chat');
                      }}
                      className="text-left min-w-0 flex-1 hover:text-blue-500"
                    >
                      <div className="font-medium truncate">{c.modelLabel}</div>
                      <div className="text-[11px] text-slate-500 truncate">
                        {c.messages[0]?.content || 'Empty thread'}
                      </div>
                    </button>
                    <button
                      type="button"
                      aria-label={`Delete conversation ${c.id}`}
                      onClick={() =>
                        api
                          .deleteConversation(c.id)
                          .then(() => {
                            setConversations((prev) => prev.filter((x) => x.id !== c.id));
                            if (activeConvId === c.id) {
                              setActiveConvId(undefined);
                              setMessages([]);
                            }
                          })
                          .catch((err) => setReportError((err as Error).message))
                      }
                      className="text-slate-400 hover:text-red-500"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Right Panel: Structured Chat OR Diagnostic Model Report OR All Models Matrix */}
        <div className="lg:col-span-8 min-w-0">
          {viewMode === 'chat' ? (
            <div className="relative flex flex-col rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] overflow-hidden h-[660px]">
              <div className="relative z-10 backdrop-blur-md bg-white/80 dark:bg-[#0C0F16]/80 px-5 py-3 border-b border-slate-200/70 dark:border-white/[0.08] flex items-center justify-between text-xs">
                <span className="font-medium text-slate-900 dark:text-slate-100 truncate">
                  {models.find((m) => m.ref === selectedRef)?.label || 'Select model'}
                </span>
                {streaming ? (
                  <LatticeLoader
                    pattern="pulse"
                    grid={3}
                    cellSize={4}
                    gap={2}
                    fontSize={12}
                    color="#3B82F6"
                    status="working"
                    label="Verifying output"
                  />
                ) : (
                  <span className="font-mono text-slate-600 dark:text-slate-300">
                    Boolean · Choice · Score
                  </span>
                )}
              </div>

              <div className="relative flex-1 min-h-0 flex flex-col">
                <div className="absolute inset-0 z-0">
                  <AquariumChatCanvas darkTheme={isDark} dimmed={messages.length > 0} />
                </div>

                {messages.length === 0 ? (
                  <div className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 text-center pointer-events-none">
                    <h2 className="text-2xl sm:text-3xl font-normal tracking-tight text-slate-800 dark:text-slate-100 drop-shadow-sm mb-5 select-none">
                      What&apos;s on your mind?
                    </h2>

                    <form
                      onSubmit={handleSend}
                      className="pointer-events-auto w-full max-w-md rounded-full border border-white/60 dark:border-white/25 bg-white/45 dark:bg-slate-900/45 backdrop-blur-md shadow-[0_8px_32px_rgba(15,23,42,0.12)] px-4 py-1.5 flex items-center gap-2"
                    >
                      <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask anything"
                        aria-label="Ask anything"
                        className="flex-1 bg-transparent text-sm text-slate-800 dark:text-slate-100 placeholder:text-slate-700/80 dark:placeholder:text-slate-300/80 focus:outline-none px-2 py-1"
                      />
                      <button
                        type="submit"
                        aria-label="Send query"
                        disabled={streaming || !input.trim() || !selectedRef}
                        className="w-8 h-8 rounded-full bg-white/85 hover:bg-white text-slate-900 border border-white shadow-[inset_0_1px_1px_rgba(255,255,255,0.95),0_4px_12px_rgba(15,23,42,0.14)] backdrop-blur-md flex items-center justify-center transition-all disabled:opacity-45 shrink-0"
                      >
                        <ArrowUp className="w-4 h-4" />
                      </button>
                    </form>

                    <p className="mt-3 text-[11px] font-medium text-slate-800/85 dark:text-slate-200/85 select-none">
                      Move through the water, or tap it.
                    </p>

                    <div className="pointer-events-auto mt-5 flex flex-wrap justify-center gap-1.5 max-w-lg">
                      {QUICK_PROMPTS.map((qp) => (
                        <button
                          key={qp.label}
                          type="button"
                          onClick={() => {
                            setInput(qp.prompt);
                            setCandidateChoicesInput(qp.choices);
                          }}
                          className="rounded-full backdrop-blur-md bg-white/60 dark:bg-slate-900/60 border border-white/60 dark:border-white/15 px-3 py-1 text-[11px] font-medium text-slate-800 dark:text-slate-100 hover:bg-white/80 dark:hover:bg-slate-900/80 transition-colors"
                        >
                          {qp.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="relative z-10 flex-1 overflow-y-auto p-5 space-y-4 pointer-events-auto">
                      {messages.map((m) => (
                        <div
                          key={m.id}
                          className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
                        >
                          <div
                            className={`max-w-[88%] rounded-xl px-4 py-3 text-xs leading-relaxed shadow-sm ${
                              m.role === 'user'
                                ? 'bg-blue-600/90 backdrop-blur-md text-white'
                                : 'border border-white/40 dark:border-white/15 bg-white/88 dark:bg-[#090B10]/88 backdrop-blur-md text-slate-800 dark:text-slate-200'
                            }`}
                          >
                            {m.role === 'assistant' && m.structured && (
                              <div className="mb-3 grid grid-cols-3 gap-2 rounded-lg border border-slate-200/80 dark:border-white/[0.08] bg-white/90 dark:bg-[#11141C]/90 p-2.5 font-mono tabular-nums">
                                <div>
                                  <div className="text-[10px] uppercase text-slate-500">boolean</div>
                                  <div
                                    className={`mt-0.5 text-xs font-semibold ${
                                      m.structured.booleanValue
                                        ? 'text-emerald-600 dark:text-emerald-400'
                                        : 'text-amber-600 dark:text-amber-400'
                                    }`}
                                  >
                                    {String(m.structured.booleanValue)}
                                  </div>
                                </div>
                                <div className="min-w-0">
                                  <div className="text-[10px] uppercase text-slate-500">choice</div>
                                  <div className="mt-0.5 text-xs font-semibold text-blue-600 dark:text-blue-400 truncate">
                                    {m.structured.choice}
                                  </div>
                                </div>
                                <div>
                                  <div className="text-[10px] uppercase text-slate-500">score</div>
                                  <div className="mt-0.5 text-xs font-semibold text-slate-900 dark:text-slate-100">
                                    {m.structured.score.toFixed(4)}
                                  </div>
                                </div>
                              </div>
                            )}

                            {m.role === 'assistant' && m.structured?.hallucinationFlag && (
                              <div className="mb-2 flex items-center gap-1.5 text-[11px] font-medium text-red-600 dark:text-red-400">
                                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                                Potential hallucination flagged outside candidate choices
                              </div>
                            )}

                            <div className="prose prose-sm dark:prose-invert max-w-none prose-pre:bg-black/40 prose-pre:p-2 prose-pre:rounded">
                              <Markdown>{m.content || (streaming ? '...' : '')}</Markdown>
                            </div>
                          </div>

                          {m.role === 'assistant' && m.content && (
                            <div className="mt-1.5 inline-flex flex-wrap items-center gap-3 rounded-full bg-white/70 dark:bg-slate-900/70 backdrop-blur-md px-2.5 py-0.5 text-[11px] font-mono tabular-nums text-slate-700 dark:text-slate-300">
                              {m.structured && (
                                <span>
                                  {m.structured.grounded ? 'Grounded' : 'Flagged'} ·{' '}
                                  {m.structured.choicesConsidered.length} choices
                                </span>
                              )}
                              {m.latencyMs !== undefined && <span>{m.latencyMs} ms</span>}
                              {m.tokensGenerated !== undefined && <span>{m.tokensGenerated} tok</span>}
                              <button
                                type="button"
                                onClick={() => handleCopy(m.id, m.content)}
                                className="inline-flex items-center gap-1 hover:text-slate-950 dark:hover:text-white"
                              >
                                {copiedId === m.id ? (
                                  <>
                                    <Check className="w-3 h-3" /> Copied
                                  </>
                                ) : (
                                  <>
                                    <Copy className="w-3 h-3" /> Copy
                                  </>
                                )}
                              </button>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>

                    <form
                      onSubmit={handleSend}
                      className="relative z-10 p-3.5 border-t border-white/30 dark:border-white/[0.08] bg-white/75 dark:bg-[#0C0F16]/80 backdrop-blur-md flex items-center gap-2.5 pointer-events-auto"
                    >
                      <div className="flex-1 flex items-center rounded-full border border-white/60 dark:border-white/20 bg-white/65 dark:bg-[#090B10]/75 px-3.5 py-1.5">
                        <input
                          type="text"
                          value={input}
                          onChange={(e) => setInput(e.target.value)}
                          placeholder="Ask anything"
                          aria-label="Ask anything"
                          className="flex-1 bg-transparent text-xs text-slate-900 dark:text-slate-100 placeholder:text-slate-500 focus:outline-none"
                        />
                      </div>
                      <LiquidButton
                        type="submit"
                        variant="default"
                        size="sm"
                        disabled={streaming || !input.trim() || !selectedRef}
                        className="rounded-full px-4"
                      >
                        <Send className="w-3.5 h-3.5" />
                        Send
                      </LiquidButton>
                    </form>
                  </>
                )}
              </div>
            </div>
          ) : viewMode === 'report' ? (
            <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-200 dark:border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    Playground Model Diagnostic Report
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {activeReport
                      ? `${activeReport.modelLabel} · ${activeReport.datasetName}`
                      : 'Run a test suite on any Playground model to generate a verification report.'}
                  </p>
                </div>
                {activeReport && (
                  <div className="flex flex-wrap items-center gap-2">
                    <LiquidButton
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        downloadTextFile(
                          `${activeReport.id}-report.md`,
                          buildMarkdownReport(activeReport),
                          'text/markdown'
                        )
                      }
                    >
                      <FileText className="w-3.5 h-3.5" />
                      Export Markdown
                    </LiquidButton>
                    <LiquidButton
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        downloadTextFile(
                          `${activeReport.id}-report.json`,
                          JSON.stringify(activeReport, null, 2),
                          'application/json'
                        )
                      }
                    >
                      <Download className="w-3.5 h-3.5" />
                      Export JSON
                    </LiquidButton>
                  </div>
                )}
              </div>

              {reports.length > 0 && (
                <div className="flex flex-wrap items-center gap-2 min-w-0">
                  <label className="text-xs text-slate-500 shrink-0">Saved Reports:</label>
                  <select
                    aria-label="Saved Model Reports"
                    value={activeReport?.id ?? ''}
                    onChange={(e) => {
                      const found = reports.find((r) => r.id === e.target.value);
                      if (found) setActiveReport(found);
                    }}
                    className="flex-1 min-w-0 w-full px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10]"
                  >
                    {reports.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.modelLabel} — {(r.passRate * 100).toFixed(1)}% pass ({r.verdict.toUpperCase()})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {!activeReport ? (
                <div className="py-16 text-center space-y-3 border border-dashed border-slate-200 dark:border-white/[0.08] rounded-lg">
                  <ClipboardCheck className="w-7 h-7 text-slate-400 mx-auto" />
                  <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    No diagnostic report selected yet
                  </p>
                  <p className="text-xs text-slate-500 max-w-md mx-auto">
                    Select any base model or LoRA adapter on the left and click &ldquo;Test Model &amp; Generate Report&rdquo; to verify boolean, choice, score accuracy and anti-hallucination behavior.
                  </p>
                </div>
              ) : (
                <>
                  {/* Aggregate Report Metrics */}
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 font-mono tabular-nums">
                    <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                      <div className="text-[11px] text-slate-500">Verdict</div>
                      <div
                        className={`mt-1 text-base font-semibold uppercase ${
                          activeReport.verdict === 'pass'
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : activeReport.verdict === 'review'
                              ? 'text-amber-600 dark:text-amber-400'
                              : 'text-red-600 dark:text-red-400'
                        }`}
                      >
                        {activeReport.verdict}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                      <div className="text-[11px] text-slate-500">Pass Rate</div>
                      <div className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                        {(activeReport.passRate * 100).toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {activeReport.passedCount}/{activeReport.totalCases} cases
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                      <div className="text-[11px] text-slate-500">Mean Score</div>
                      <div className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                        {activeReport.avgScore.toFixed(4)}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                      <div className="text-[11px] text-slate-500">Hallucinations</div>
                      <div
                        className={`mt-1 text-base font-semibold ${
                          activeReport.hallucinationCount === 0
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}
                      >
                        {activeReport.hallucinationCount}
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {(activeReport.hallucinationRate * 100).toFixed(1)}% rate
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                      <div className="text-[11px] text-slate-500">Avg Latency</div>
                      <div className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                        {activeReport.avgLatencyMs.toFixed(2)} ms
                      </div>
                    </div>
                  </div>

                  {/* Per-Case Verification Table */}
                  <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-white/[0.08]">
                    <table className="w-full text-left text-xs">
                      <thead className="border-b border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] text-slate-500 dark:text-slate-400">
                        <tr>
                          <th className="py-2.5 px-2.5 font-medium w-8">#</th>
                          <th className="py-2.5 px-2.5 font-medium">Prompt</th>
                          <th className="py-2.5 px-2.5 font-medium">Expected</th>
                          <th className="py-2.5 px-2.5 font-medium">Boolean</th>
                          <th className="py-2.5 px-2.5 font-medium">Choice</th>
                          <th className="py-2.5 px-2.5 font-medium text-right">Score</th>
                          <th className="py-2.5 px-2.5 font-medium text-right">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06] font-mono tabular-nums">
                        {activeReport.cases.map((c) => (
                          <tr key={c.index} className="hover:bg-slate-50 dark:hover:bg-white/[0.02]">
                            <td className="py-2.5 px-2.5 text-slate-500">{c.index + 1}</td>
                            <td className="py-2.5 px-2.5 font-sans max-w-[165px]">
                              <div className="truncate text-slate-800 dark:text-slate-200" title={c.prompt}>
                                {c.prompt}
                              </div>
                              <div className="text-[10px] font-mono text-slate-500">
                                {c.category === 'anti_hallucination' ? 'anti-hallucination' : 'dataset-split'}
                              </div>
                            </td>
                            <td className="py-2.5 px-2.5 text-slate-600 dark:text-slate-300 max-w-[125px] truncate" title={c.expectedChoice}>
                              {c.expectedChoice}
                            </td>
                            <td className="py-2.5 px-2.5">
                              <span
                                className={
                                  c.booleanValue
                                    ? 'text-emerald-600 dark:text-emerald-400'
                                    : 'text-amber-600 dark:text-amber-400'
                                }
                              >
                                {String(c.booleanValue)}
                              </span>
                            </td>
                            <td className="py-2.5 px-2.5 text-blue-600 dark:text-blue-400 max-w-[125px] truncate" title={c.choice}>
                              {c.choice}
                            </td>
                            <td className="py-2.5 px-2.5 text-right text-slate-900 dark:text-slate-100">
                              {c.score.toFixed(4)}
                            </td>
                            <td className="py-2.5 px-2.5 text-right whitespace-nowrap">
                              {c.hallucinationFlag ? (
                                <span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400">
                                  <AlertTriangle className="w-3.5 h-3.5" />
                                  Flagged
                                </span>
                              ) : c.passed ? (
                                <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                                  <CheckCircle2 className="w-3.5 h-3.5" />
                                  Pass
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400">
                                  <XCircle className="w-3.5 h-3.5" />
                                  Fail
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white dark:bg-[#11141C] p-5 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-200 dark:border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    All Playground Models Benchmark Matrix ({models.length} Models)
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Every trained LoRA adapter and base AI model present in Model Playground ranked by pass rate, anti-hallucination grounding, score, and latency.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2 shrink-0">
                  <LiquidButton
                    type="button"
                    variant="default"
                    size="sm"
                    onClick={handleRunAllModelsReport}
                    disabled={models.length === 0 || testingAllModels || testingModel}
                  >
                    <Play className="w-3.5 h-3.5" />
                    {testingAllModels ? 'Running Sweep...' : 'Run Test All Models'}
                  </LiquidButton>
                  <LiquidButton
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={filteredLeaderboardRows.length === 0}
                    onClick={() =>
                      downloadTextFile(
                        `minimind-all-models-benchmark-${Date.now()}.csv`,
                        buildBatchCsvReport(filteredLeaderboardRows),
                        'text/csv'
                      )
                    }
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    Export CSV
                  </LiquidButton>
                  <LiquidButton
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={filteredLeaderboardRows.length === 0}
                    onClick={() =>
                      downloadTextFile(
                        `minimind-all-models-benchmark-${Date.now()}.md`,
                        buildBatchMarkdownReport(filteredLeaderboardRows, batchSummary),
                        'text/markdown'
                      )
                    }
                  >
                    <FileText className="w-3.5 h-3.5" />
                    Export Markdown
                  </LiquidButton>
                </div>
              </div>

              {/* Aggregate Batch Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono tabular-nums">
                <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                  <div className="text-[11px] text-slate-500">Models Tested</div>
                  <div className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                    {batchSummary.totalModels}/{filteredLeaderboardRows.length}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    {batchSummary.passedModels} pass · {batchSummary.reviewModels} review · {batchSummary.failedModels} fail
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                  <div className="text-[11px] text-slate-500">Mean Pass Rate</div>
                  <div className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                    {batchSummary.totalModels > 0 ? `${(batchSummary.avgPassRate * 100).toFixed(1)}%` : '—'}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    Mean score {batchSummary.totalModels > 0 ? batchSummary.avgScore.toFixed(4) : '—'}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                  <div className="text-[11px] text-slate-500">Hallucinations Flagged</div>
                  <div
                    className={`mt-1 text-base font-semibold ${
                      batchSummary.totalHallucinations === 0
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}
                  >
                    {batchSummary.totalHallucinations}
                  </div>
                  <div className="text-[10px] text-slate-500">Across tested suite</div>
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] p-3">
                  <div className="text-[11px] text-slate-500">Mean Latency</div>
                  <div className="mt-1 text-base font-semibold text-slate-900 dark:text-slate-100">
                    {batchSummary.totalModels > 0 ? `${batchSummary.avgLatencyMs.toFixed(2)} ms` : '—'}
                  </div>
                  <div className="text-[10px] text-slate-500">Per verification case</div>
                </div>
              </div>

              {/* AI Tool / Family Filter Bar */}
              <div className="flex flex-wrap items-center gap-1.5">
                {(['All', ...MODEL_GROUPS] as const).map((group) => {
                  const count =
                    group === 'All'
                      ? models.length
                      : models.filter((m) => (m.aiTool ?? 'Open SLM') === group).length;
                  if (group !== 'All' && count === 0) return null;
                  const active = toolFilter === group;
                  return (
                    <button
                      key={group}
                      type="button"
                      onClick={() => setToolFilter(group)}
                      className={`px-2.5 py-1 text-[11px] font-medium rounded-md border transition-colors ${
                        active
                          ? 'border-blue-500 bg-blue-600 text-white'
                          : 'border-slate-200 dark:border-white/[0.08] text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-white/[0.04]'
                      }`}
                    >
                      {group} ({count})
                    </button>
                  );
                })}
              </div>

              {/* All Models Ranked Table */}
              <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-white/[0.08]">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-200 dark:border-white/[0.08] bg-slate-50 dark:bg-[#090B10] text-slate-500 dark:text-slate-400">
                    <tr>
                      <th className="py-2.5 px-2 font-medium w-8">#</th>
                      <th className="py-2.5 px-2 font-medium">Model</th>
                      <th className="py-2.5 px-2 font-medium">Family</th>
                      <th className="py-2.5 px-2 font-medium">Verdict</th>
                      <th className="py-2.5 px-2 font-medium text-right">Pass Rate</th>
                      <th className="py-2.5 px-2 font-medium text-right">Halluc.</th>
                      <th className="py-2.5 px-2 font-medium text-right">Score</th>
                      <th className="py-2.5 px-2 font-medium text-right">Latency</th>
                      <th className="py-2.5 px-2 font-medium text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-white/[0.06] font-mono tabular-nums">
                    {filteredLeaderboardRows.map((row) => {
                      const r = row.report;
                      const isRowTesting = testingRef === row.modelRef || testingAllModels;
                      return (
                        <tr
                          key={row.modelRef}
                          className={`hover:bg-slate-50 dark:hover:bg-white/[0.02] ${
                            selectedRef === row.modelRef ? 'bg-blue-500/[0.04]' : ''
                          }`}
                        >
                          <td className="py-2 px-2 text-slate-500">
                            {row.rank !== null ? row.rank : '—'}
                          </td>
                          <td className="py-2 px-2 font-sans max-w-[165px]">
                            <div
                              className="font-medium text-slate-900 dark:text-slate-100 truncate"
                              title={row.modelLabel}
                            >
                              {row.modelLabel}
                            </div>
                            <div className="text-[10px] font-mono text-slate-500 truncate">
                              {row.method}
                            </div>
                          </td>
                          <td className="py-2 px-2 font-sans text-slate-600 dark:text-slate-300 whitespace-nowrap">
                            {row.aiTool}
                          </td>
                          <td className="py-2 px-2 whitespace-nowrap">
                            {!r ? (
                              <span className="text-slate-400">UNTESTED</span>
                            ) : r.verdict === 'pass' ? (
                              <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                                PASS
                              </span>
                            ) : r.verdict === 'review' ? (
                              <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400 font-semibold">
                                <AlertTriangle className="w-3.5 h-3.5" />
                                REVIEW
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400 font-semibold">
                                <XCircle className="w-3.5 h-3.5" />
                                FAIL
                              </span>
                            )}
                          </td>
                          <td className="py-2 px-2 text-right whitespace-nowrap">
                            {r ? (
                              <span>
                                {(r.passRate * 100).toFixed(1)}%{' '}
                                <span className="text-[10px] text-slate-500">
                                  ({r.passedCount}/{r.totalCases})
                                </span>
                              </span>
                            ) : (
                              <span className="text-slate-400">—</span>
                            )}
                          </td>
                          <td className="py-2 px-2 text-right whitespace-nowrap">
                            {r ? (
                              <span
                                className={
                                  r.hallucinationCount === 0
                                    ? 'text-emerald-600 dark:text-emerald-400'
                                    : 'text-red-600 dark:text-red-400'
                                }
                              >
                                {r.hallucinationCount} ({(r.hallucinationRate * 100).toFixed(0)}%)
                              </span>
                            ) : (
                              <span className="text-slate-400">—</span>
                            )}
                          </td>
                          <td className="py-2 px-2 text-right">
                            {r ? r.avgScore.toFixed(4) : <span className="text-slate-400">—</span>}
                          </td>
                          <td className="py-2 px-2 text-right whitespace-nowrap">
                            {r ? (
                              `${r.avgLatencyMs.toFixed(2)} ms`
                            ) : (
                              <span className="text-slate-400">—</span>
                            )}
                          </td>
                          <td className="py-2 px-2 text-right whitespace-nowrap">
                            <div className="inline-flex items-center gap-1 font-sans">
                              {r ? (
                                <button
                                  type="button"
                                  title="View Case-by-Case Report"
                                  onClick={() => {
                                    setSelectedRef(row.modelRef);
                                    setActiveReport(r);
                                    setViewMode('report');
                                  }}
                                  className="px-2 py-1 text-[11px] font-medium rounded border border-blue-500/30 text-blue-600 dark:text-blue-400 hover:bg-blue-500/10"
                                >
                                  Report
                                </button>
                              ) : (
                                <button
                                  type="button"
                                  disabled={isRowTesting || testingModel}
                                  onClick={() => handleRunModelReport(row.modelRef, false)}
                                  className="px-2 py-1 text-[11px] font-medium rounded border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/10 disabled:opacity-50"
                                >
                                  {testingRef === row.modelRef ? 'Testing...' : 'Test'}
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedRef(row.modelRef);
                                  setViewMode('chat');
                                }}
                                aria-label={`Chat with ${row.modelLabel}`}
                                title="Open in Structured Chat"
                                className="p-1 rounded border border-slate-200 dark:border-white/[0.1] text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-white/[0.06]"
                              >
                                <MessageSquare className="w-3 h-3" />
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
        </div>
      </div>
    </div>
  );
}
