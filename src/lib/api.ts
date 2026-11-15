import type {
  DatasetRecord,
  EvaluationRunRecord,
  ExperimentRecord,
  HardwareTelemetry,
  ModelSpec,
  PlaygroundMessage,
  PlaygroundTestReport,
  RegisteredModelRecord,
  TrainingConfig,
  TrainingMetricPoint,
} from '../shared/types';

async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.error) msg = data.error;
    } catch {
      // ignore json parse failure
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export interface OverviewResponse {
  stats: {
    totalExperiments: number;
    completedRuns: number;
    failedRuns: number;
    totalDatasets: number;
    totalRegisteredModels: number;
    totalPretrainedModels: number;
    avgDurationMs: number | null;
    totalTrainableParams: number;
    latestPerformance: {
      modelName: string;
      accuracy: number | null;
      macroF1: number | null;
      perplexity: number;
      taskType: 'classification' | 'generation';
    } | null;
  };
  hardware: HardwareTelemetry;
  recentExperiments: ExperimentRecord[];
  recentModels: RegisteredModelRecord[];
  recentEvaluations: EvaluationRunRecord[];
}

export const api = {
  getOverview: () => apiFetch<OverviewResponse>('/api/overview'),
  getDatasets: () => apiFetch<DatasetRecord[]>('/api/datasets'),
  getDatasetDetail: (id: string) =>
    apiFetch<{
      dataset: DatasetRecord;
      previewRows: Record<string, unknown>[];
      processedPreview: {
        cleanedSample?: Record<string, unknown>[];
        trainSample: Record<string, unknown>[];
        valSample: Record<string, unknown>[];
        testSample: Record<string, unknown>[];
      } | null;
    }>(`/api/datasets/${id}`),
  uploadDatasetFile: async (file: File, name?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (name) form.append('name', name);
    return apiFetch<DatasetRecord>('/api/datasets/upload', { method: 'POST', body: form });
  },
  uploadDatasetText: (payload: { name: string; fileName: string; content: string }) =>
    apiFetch<DatasetRecord>('/api/datasets/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  validateDataset: (id: string, columnMapping?: Record<string, string>) =>
    apiFetch<DatasetRecord>(`/api/datasets/${id}/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ columnMapping }),
    }),
  preprocessDataset: (
    id: string,
    payload: {
      trainRatio: number;
      valRatio: number;
      testRatio: number;
      deduplicate: boolean;
      dropMissing: boolean;
      seed: number;
      columnMapping?: Record<string, string>;
    }
  ) =>
    apiFetch<DatasetRecord>(`/api/datasets/${id}/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  deleteDataset: (id: string) =>
    apiFetch<{ deleted: boolean }>(`/api/datasets/${id}`, { method: 'DELETE' }),

  getAvailableModels: () =>
    apiFetch<{ models: ModelSpec[]; hardware: HardwareTelemetry }>('/api/models/available'),
  validateHfModel: (repoId: string) =>
    apiFetch<{
      valid: boolean;
      reachable: boolean;
      repoId: string;
      downloads?: number | null;
      likes?: number | null;
      pipelineTag?: string;
      architecture: string;
      recommendedModules: string[];
      note?: string;
    }>('/api/models/validate-hf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoId }),
    }),
  getRegistryModels: () => apiFetch<RegisteredModelRecord[]>('/api/models/registry'),
  deployModel: (id: string, status: 'ready' | 'deployed') =>
    apiFetch<RegisteredModelRecord>(`/api/models/${id}/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    }),
  deleteRegisteredModel: (id: string) =>
    apiFetch<{ deleted: boolean }>(`/api/models/${id}`, { method: 'DELETE' }),

  startTraining: (config: TrainingConfig) =>
    apiFetch<ExperimentRecord>('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }),
  getExperiments: () => apiFetch<ExperimentRecord[]>('/api/training'),
  getExperimentDetail: (id: string) =>
    apiFetch<{
      experiment: ExperimentRecord;
      metrics: TrainingMetricPoint[];
      checkpoints: Array<{
        id: string;
        step: number;
        epoch: number;
        trainLoss: number;
        valLoss: number | null;
        filePath: string;
        sizeBytes: number;
        createdAt: string;
      }>;
      logs: string[];
      hardware: HardwareTelemetry;
    }>(`/api/training/${id}`),
  pauseTraining: (id: string) =>
    apiFetch<{ paused: boolean }>(`/api/training/${id}/pause`, { method: 'POST' }),
  resumeTraining: (id: string) =>
    apiFetch<{ resumed: boolean }>(`/api/training/${id}/resume`, { method: 'POST' }),
  cancelTraining: (id: string) =>
    apiFetch<{ cancelled: boolean }>(`/api/training/${id}/cancel`, { method: 'POST' }),

  runEvaluation: (modelRef: string, datasetId: string) =>
    apiFetch<EvaluationRunRecord>('/api/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelRef, datasetId }),
    }),
  getEvaluations: () => apiFetch<EvaluationRunRecord[]>('/api/evaluation'),
  runComparison: (leftModelRef: string, rightModelRef: string, datasetId: string) =>
    apiFetch<{
      left: {
        evaluation: EvaluationRunRecord;
        meta: {
          durationMs: number;
          adapterSizeBytes: number;
          trainableParams: number;
          estimatedMemoryGb: number;
        };
      };
      right: {
        evaluation: EvaluationRunRecord;
        meta: {
          durationMs: number;
          adapterSizeBytes: number;
          trainableParams: number;
          estimatedMemoryGb: number;
        };
      };
      sampleDiffs: Array<{
        index: number;
        prompt: string;
        groundTruth: string;
        leftPrediction: string;
        leftCorrect: boolean;
        leftConfidence: number;
        rightPrediction: string;
        rightCorrect: boolean;
        rightConfidence: number;
      }>;
    }>('/api/comparison/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ leftModelRef, rightModelRef, datasetId }),
    }),

  getInferenceModels: () =>
    apiFetch<{
      models: Array<{
        ref: string;
        label: string;
        method: string;
        baseModelId: string;
        datasetName?: string;
        status?: string;
        parameterCount?: string;
        provider?: string;
        aiTool?: string;
        modelTier?: string;
        trainableLocally?: boolean;
      }>;
    }>('/api/inference/models'),
  getConversations: () =>
    apiFetch<
      Array<{
        id: string;
        modelRef: string;
        modelLabel: string;
        systemPrompt: string;
        messages: PlaygroundMessage[];
        updatedAt: string;
      }>
    >('/api/inference/conversations'),
  saveConversation: (payload: {
    id?: string;
    modelRef: string;
    modelLabel: string;
    systemPrompt: string;
    messages: unknown[];
  }) =>
    apiFetch<{ id: string; saved: boolean }>('/api/inference/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  deleteConversation: (id: string) =>
    apiFetch<{ deleted: boolean }>(`/api/inference/conversations/${id}`, { method: 'DELETE' }),
  runPlaygroundTestReport: (payload: {
    modelRef: string;
    datasetId?: string;
    suiteMode?: PlaygroundTestReport['suiteMode'];
  }) =>
    apiFetch<PlaygroundTestReport>('/api/inference/test-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  getPlaygroundTestReports: () =>
    apiFetch<PlaygroundTestReport[]>('/api/inference/test-reports'),
  deletePlaygroundTestReport: (id: string) =>
    apiFetch<{ deleted: boolean }>(`/api/inference/test-reports/${id}`, { method: 'DELETE' }),

  getSettings: () =>
    apiFetch<{
      workspace: { id: string; name: string; description: string };
      users: Array<{ id: string; name: string; email: string; role: string }>;
      apiKeys: Array<{
        id: string;
        name: string;
        prefix: string;
        role: string;
        created_at: string;
      }>;
      auditLogs: Array<{
        id: number;
        actor: string;
        action: string;
        resource_type: string;
        resource_id: string;
        detail: string;
        created_at: string;
      }>;
      hardware: HardwareTelemetry;
      geminiConfigured: boolean;
    }>('/api/settings'),
  updateWorkspace: (name: string, description: string) =>
    apiFetch<{ updated: boolean }>('/api/settings/workspace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    }),
  createApiKey: (name: string, role: string) =>
    apiFetch<{ id: string; name: string; prefix: string; role: string; secretToken: string }>(
      '/api/settings/api-keys',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, role }),
      }
    ),
  revokeApiKey: (id: string) =>
    apiFetch<{ deleted: boolean }>(`/api/settings/api-keys/${id}`, { method: 'DELETE' }),
};
