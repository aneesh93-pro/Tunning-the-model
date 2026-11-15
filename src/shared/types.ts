export type DatasetFormat = 'instruction' | 'conversational' | 'raw_text';

export interface DatasetColumnStats {
  name: string;
  inferredType: 'string' | 'number' | 'boolean' | 'array' | 'object' | 'null';
  missingCount: number;
  uniqueCount: number;
  avgLength: number;
  sampleValues: string[];
}

export interface DatasetValidationReport {
  valid: boolean;
  format: DatasetFormat;
  rowCount: number;
  columnCount: number;
  missingValuesTotal: number;
  duplicateCount: number;
  leakageDetected: boolean;
  overlapCount: number;
  avgTokenEstimate: number;
  maxTokenEstimate: number;
  warnings: string[];
  errors: string[];
  columns: DatasetColumnStats[];
}

export interface DatasetSplits {
  trainCount: number;
  valCount: number;
  testCount: number;
  trainRatio: number;
  valRatio: number;
  testRatio: number;
  leakageFree: boolean;
}

export interface DatasetRecord {
  id: string;
  name: string;
  fileName: string;
  fileType: 'csv' | 'json' | 'jsonl' | 'txt';
  format: DatasetFormat;
  status: 'uploaded' | 'validated' | 'training_ready';
  rowCount: number;
  columnCount: number;
  sizeBytes: number;
  missingCount: number;
  duplicateCount: number;
  validationReport: DatasetValidationReport | null;
  splits: DatasetSplits | null;
  columnMapping: {
    instruction?: string;
    input?: string;
    output?: string;
    messages?: string;
  } | null;
  createdAt: string;
  updatedAt: string;
}

export type AiToolCategory =
  | 'Open SLM'
  | 'ChatGPT'
  | 'Claude'
  | 'Gemini'
  | 'DeepSeek'
  | 'Llama'
  | 'Mistral';

export interface ModelSpec {
  id: string;
  name: string;
  repoId: string;
  provider?: string;
  aiTool?: AiToolCategory;
  modelTier?: 'Edge SLM' | 'Frontier API' | 'Reasoning' | 'Open Weights';
  trainableLocally?: boolean;
  architecture: 'qwen2' | 'llama' | 'phi3' | 'gpt4' | 'claude3' | 'gemini' | 'deepseek' | 'mistral';
  parameterCount: string;
  parameterBillions: number;
  contextLength: number;
  defaultPrecision: 'bf16' | 'fp16' | 'nf4';
  license: string;
  source:
    | 'Hugging Face'
    | 'Local Cache'
    | 'OpenAI (ChatGPT)'
    | 'Anthropic (Claude)'
    | 'Google (Gemini)'
    | 'DeepSeek'
    | 'Meta Open Weights'
    | 'Mistral AI';
  description: string;
  supportedMethods: ('lora' | 'qlora' | 'full')[];
  targetModules: string[];
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  estimatedFp16MemoryGb: number;
  estimatedQloraMemoryGb: number;
  chatTemplate: 'chatml' | 'phi3';
}

export type TrainingMethod = 'lora' | 'qlora' | 'full';

export interface TrainingConfig {
  experimentName: string;
  description: string;
  datasetId: string;
  modelId: string;
  method: TrainingMethod;
  // LoRA parameters
  loraR: number;
  loraAlpha: number;
  loraDropout: number;
  targetModules: string[];
  bias: 'none' | 'all' | 'lora_only';
  // QLoRA parameters
  quantizationBits: 4 | 8 | 16;
  quantType: 'nf4' | 'fp4';
  doubleQuant: boolean;
  computeDtype: 'bfloat16' | 'float16' | 'float32';
  gradientCheckpointing: boolean;
  // Optimization parameters
  learningRate: number;
  epochs: number;
  batchSize: number;
  gradientAccumulation: number;
  maxSeqLength: number;
  weightDecay: number;
  warmupRatio: number;
  optimizer: 'adamw' | 'sgd';
  scheduler: 'cosine' | 'linear' | 'constant';
  earlyStoppingPatience: number;
  seed: number;
}

export type ExperimentStatus =
  | 'queued'
  | 'preparing'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface TrainingMetricPoint {
  step: number;
  epoch: number;
  trainLoss: number;
  valLoss: number | null;
  learningRate: number;
  gradNorm: number;
  memoryMb: number;
  tokensPerSec: number;
  timestamp: string;
}

export interface ExperimentRecord {
  id: string;
  name: string;
  description: string;
  datasetId: string;
  datasetName: string;
  modelId: string;
  modelName: string;
  method: TrainingMethod;
  status: ExperimentStatus;
  config: TrainingConfig;
  currentEpoch: number;
  currentStep: number;
  totalSteps: number;
  trainLoss: number | null;
  valLoss: number | null;
  trainableParams: number;
  totalParams: number;
  trainablePercent: number;
  quantizationRmse: number | null;
  adapterSizeBytes: number | null;
  artifactDir: string;
  errorMessage: string | null;
  durationMs: number | null;
  startedAt: string | null;
  completedAt: string | null;
  createdAt: string;
}

export interface CheckpointRecord {
  id: string;
  experimentId: string;
  step: number;
  epoch: number;
  trainLoss: number;
  valLoss: number | null;
  filePath: string;
  sizeBytes: number;
  createdAt: string;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface EvaluationSample {
  index: number;
  prompt: string;
  groundTruth: string;
  prediction: string;
  correct: boolean;
  invalidOutput: boolean;
  confidence: number;
  loss: number;
  latencyMs: number;
}

export interface EvaluationRunRecord {
  id: string;
  experimentId: string | null;
  registeredModelId: string | null;
  modelName: string;
  baseModelId: string;
  method: TrainingMethod | 'base';
  datasetId: string;
  datasetName: string;
  taskType: 'classification' | 'generation';
  testSampleCount: number;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  macroF1: number | null;
  weightedF1: number | null;
  exactMatch: number | null;
  tokenF1: number | null;
  perplexity: number;
  avgLoss: number;
  invalidOutputCount: number;
  avgLatencyMs: number;
  confusionMatrix: ConfusionMatrixData | null;
  samples: EvaluationSample[];
  createdAt: string;
}

export interface RegisteredModelRecord {
  id: string;
  name: string;
  version: string;
  baseModelId: string;
  baseModelName: string;
  method: TrainingMethod;
  experimentId: string;
  datasetId: string;
  datasetName: string;
  adapterPath: string;
  adapterSizeBytes: number;
  trainableParams: number;
  status: 'ready' | 'deployed' | 'archived';
  latestAccuracy: number | null;
  latestMacroF1: number | null;
  latestPerplexity: number | null;
  trainingConfig: TrainingConfig;
  createdAt: string;
}

export interface StructuredInferenceOutput {
  booleanValue: boolean;
  choice: string;
  score: number;
  choicesConsidered: string[];
  grounded: boolean;
  hallucinationFlag: boolean;
  explanation: string;
}

export interface PlaygroundMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  latencyMs?: number;
  tokensGenerated?: number;
  predictedClass?: string;
  confidence?: number;
  structured?: StructuredInferenceOutput;
  timestamp: string;
}

export interface PlaygroundTestCaseResult {
  index: number;
  category: 'dataset_eval' | 'anti_hallucination';
  prompt: string;
  expectedChoice: string;
  expectedBoolean: boolean;
  booleanValue: boolean;
  choice: string;
  score: number;
  passed: boolean;
  hallucinationFlag: boolean;
  latencyMs: number;
  explanation: string;
}

export interface PlaygroundTestReport {
  id: string;
  modelRef: string;
  modelLabel: string;
  method: string;
  datasetId: string;
  datasetName: string;
  suiteMode: 'dataset_and_probes' | 'dataset_only' | 'anti_hallucination';
  totalCases: number;
  passedCount: number;
  passRate: number;
  hallucinationCount: number;
  hallucinationRate: number;
  avgScore: number;
  avgLatencyMs: number;
  verdict: 'pass' | 'review' | 'fail';
  cases: PlaygroundTestCaseResult[];
  createdAt: string;
}

export interface HardwareTelemetry {
  platform: string;
  cpuModel: string;
  cpuCores: number;
  totalMemoryGb: number;
  freeMemoryGb: number;
  processMemoryMb: number;
  gpuAvailable: boolean;
  gpuName: string | null;
  gpuMemoryTotalGb: number | null;
  gpuMemoryUsedGb: number | null;
  gpuUtilizationPct: number | null;
  gpuTemperatureC: number | null;
  executionBackend: 'cpu-ts-lora-engine' | 'cuda-gpu';
  note: string;
}
