import type { ModelSpec, TrainingMethod } from '../shared/types';

export const PRETRAINED_MODELS: ModelSpec[] = [
  {
    id: 'smollm2-360m-instruct',
    name: 'SmolLM2-360M-Instruct',
    repoId: 'HuggingFaceTB/SmolLM2-360M-Instruct',
    architecture: 'llama',
    parameterCount: '362M',
    parameterBillions: 0.362,
    contextLength: 8192,
    defaultPrecision: 'bf16',
    license: 'Apache 2.0',
    source: 'Hugging Face',
    description:
      'Compact instruction-tuned Llama-family model optimized for fast edge fine-tuning, intent classification, and lightweight structured generation.',
    supportedMethods: ['lora', 'qlora', 'full'],
    targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    hiddenSize: 960,
    numLayers: 32,
    vocabSize: 49152,
    estimatedFp16MemoryGb: 1.8,
    estimatedQloraMemoryGb: 0.7,
    chatTemplate: 'chatml',
  },
  {
    id: 'qwen2.5-0.5b-instruct',
    name: 'Qwen2.5-0.5B-Instruct',
    repoId: 'Qwen/Qwen2.5-0.5B-Instruct',
    architecture: 'qwen2',
    parameterCount: '494M',
    parameterBillions: 0.494,
    contextLength: 32768,
    defaultPrecision: 'bf16',
    license: 'Apache 2.0',
    source: 'Hugging Face',
    description:
      'High-efficiency Qwen2.5 small language model with strong multilingual tokenization, structured JSON compliance, and instruction following.',
    supportedMethods: ['lora', 'qlora', 'full'],
    targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    hiddenSize: 896,
    numLayers: 24,
    vocabSize: 151936,
    estimatedFp16MemoryGb: 2.4,
    estimatedQloraMemoryGb: 0.95,
    chatTemplate: 'chatml',
  },
  {
    id: 'qwen2.5-1.5b-instruct',
    name: 'Qwen2.5-1.5B-Instruct',
    repoId: 'Qwen/Qwen2.5-1.5B-Instruct',
    architecture: 'qwen2',
    parameterCount: '1.54B',
    parameterBillions: 1.54,
    contextLength: 32768,
    defaultPrecision: 'bf16',
    license: 'Apache 2.0',
    source: 'Hugging Face',
    description:
      'Balanced 1.5B parameter Transformer offering strong domain reasoning, multi-turn chat fidelity, and code/SQL synthesis with low QLoRA memory overhead.',
    supportedMethods: ['lora', 'qlora'],
    targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    hiddenSize: 1536,
    numLayers: 28,
    vocabSize: 151936,
    estimatedFp16MemoryGb: 5.6,
    estimatedQloraMemoryGb: 2.1,
    chatTemplate: 'chatml',
  },
  {
    id: 'phi-3.5-mini-instruct',
    name: 'Phi-3.5-mini-instruct',
    repoId: 'microsoft/Phi-3.5-mini-instruct',
    architecture: 'phi3',
    parameterCount: '3.82B',
    parameterBillions: 3.82,
    contextLength: 128000,
    defaultPrecision: 'nf4',
    license: 'MIT',
    source: 'Hugging Face',
    description:
      'Dense 3.8B reasoning SLM using fused qkv_proj and gate_up_proj projections. Requires QLoRA (4-bit NF4) on consumer GPUs under 16 GB VRAM.',
    supportedMethods: ['qlora', 'lora'],
    targetModules: ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'],
    hiddenSize: 3072,
    numLayers: 32,
    vocabSize: 32064,
    estimatedFp16MemoryGb: 13.8,
    estimatedQloraMemoryGb: 4.4,
    chatTemplate: 'phi3',
  },
];

export function getModelById(id: string): ModelSpec | undefined {
  return PRETRAINED_MODELS.find((m) => m.id === id || m.repoId === id);
}

export interface MemoryEstimate {
  weightsGb: number;
  adapterAndGradsGb: number;
  optimizerStatesGb: number;
  activationsGb: number;
  totalEstimatedGb: number;
  trainableParams: number;
  totalParams: number;
  trainablePct: number;
  compatibleWithLocalCpu: boolean;
  warnings: string[];
}

export function estimateTrainingMemory(
  model: ModelSpec,
  method: TrainingMethod,
  loraR: number,
  targetModules: string[],
  batchSize: number,
  maxSeqLength: number,
  gradientCheckpointing: boolean,
  availableMemoryGb: number
): MemoryEstimate {
  const totalParams = Math.round(model.parameterBillions * 1e9);
  const activeModules = targetModules.filter((m) => model.targetModules.includes(m));
  const moduleCount = Math.max(1, activeModules.length);

  // Trainable params formula for LoRA: 2 * r * d_model * num_layers * num_modules
  const trainableParams =
    method === 'full'
      ? totalParams
      : 2 * loraR * model.hiddenSize * model.numLayers * moduleCount;

  const bytesPerBaseParam = method === 'qlora' ? 0.52 : 2; // 4-bit NF4 + double quant overhead vs BF16/FP16
  const weightsGb = Number(((totalParams * bytesPerBaseParam) / 1e9).toFixed(2));
  const adapterAndGradsGb = Number(((trainableParams * 6) / 1e9).toFixed(3));
  const optimizerStatesGb = Number(((trainableParams * 8) / 1e9).toFixed(3));

  const seqFactor = (batchSize * maxSeqLength) / 2048;
  const rawActGb = 0.45 * seqFactor * (model.hiddenSize / 1024);
  const activationsGb = Number((gradientCheckpointing ? rawActGb * 0.4 : rawActGb).toFixed(2));

  const totalEstimatedGb = Number(
    (weightsGb + adapterAndGradsGb + optimizerStatesGb + activationsGb).toFixed(2)
  );
  const trainablePct = Number(((trainableParams / totalParams) * 100).toFixed(4));

  const warnings: string[] = [];
  if (!model.supportedMethods.includes(method)) {
    warnings.push(
      `${model.name} does not recommend ${method.toUpperCase()} without high-VRAM multi-GPU nodes.`
    );
  }
  const invalidModules = targetModules.filter((m) => !model.targetModules.includes(m));
  if (invalidModules.length > 0) {
    warnings.push(
      `Modules [${invalidModules.join(', ')}] do not exist on ${model.architecture} architecture.`
    );
  }
  if (totalEstimatedGb > availableMemoryGb) {
    warnings.push(
      `Full Hugging Face PyTorch weights require ~${totalEstimatedGb} GB RAM/VRAM, exceeding available ${availableMemoryGb.toFixed(1)} GB. Local CPU execution runs the compact LoRA neural engine.`
    );
  }

  return {
    weightsGb,
    adapterAndGradsGb,
    optimizerStatesGb,
    activationsGb,
    totalEstimatedGb,
    trainableParams,
    totalParams,
    trainablePct,
    compatibleWithLocalCpu: totalEstimatedGb <= availableMemoryGb,
    warnings,
  };
}
