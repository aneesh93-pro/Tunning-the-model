import type { NormalizedExample } from './datasetProcessor';
import { SubwordTokenizer, type SerializedTokenizer } from './tokenizer';
import type {
  ConfusionMatrixData,
  EvaluationSample,
  TrainingConfig,
} from '../shared/types';

/**
 * 4-bit NormalFloat (NF4) codebook from Dettmers et al. (QLoRA, 2023)
 */
export const NF4_CODEBOOK = [
  -1.0,
  -0.6961928009986877,
  -0.5250730514526367,
  -0.39491748809814453,
  -0.28444138169288635,
  -0.18477343022823334,
  -0.09105003625154495,
  0.0,
  0.07958029955625534,
  0.16093020141124725,
  0.24611230194568634,
  0.33791524171829224,
  0.44070982933044434,
  0.5626170039176941,
  0.7229568362236023,
  1.0,
] as const;

export interface QuantizedNF4Tensor {
  packedNibbles: number[]; // Two 4-bit indices packed per byte
  blockScales: number[];
  doubleQuantScale?: number;
  blockSize: number;
  length: number;
  rmse: number;
}

export function quantizeNF4(
  weights: Float32Array,
  blockSize = 64,
  doubleQuant = true
): QuantizedNF4Tensor {
  const n = weights.length;
  const numBlocks = Math.ceil(n / blockSize);
  const blockScales = new Array<number>(numBlocks);
  const indices = new Uint8Array(n);

  for (let b = 0; b < numBlocks; b++) {
    const start = b * blockSize;
    const end = Math.min(n, start + blockSize);
    let absmax = 1e-8;
    for (let i = start; i < end; i++) {
      const a = Math.abs(weights[i]);
      if (a > absmax) absmax = a;
    }
    blockScales[b] = absmax;

    for (let i = start; i < end; i++) {
      const normalized = weights[i] / absmax;
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let c = 0; c < 16; c++) {
        const d = Math.abs(normalized - NF4_CODEBOOK[c]);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = c;
        }
      }
      indices[i] = bestIdx;
    }
  }

  let doubleQuantScale: number | undefined;
  const storedScales = [...blockScales];
  if (doubleQuant && blockScales.length > 0) {
    let maxScale = 1e-8;
    for (const s of blockScales) {
      if (s > maxScale) maxScale = s;
    }
    doubleQuantScale = maxScale;
    for (let i = 0; i < storedScales.length; i++) {
      storedScales[i] = Math.round((blockScales[i] / maxScale) * 255);
    }
  }

  // Pack 2 x 4-bit indices into each byte
  const packedNibbles = new Array<number>(Math.ceil(n / 2));
  for (let i = 0; i < n; i += 2) {
    const lo = indices[i] & 0x0f;
    const hi = i + 1 < n ? (indices[i + 1] & 0x0f) << 4 : 0;
    packedNibbles[i >> 1] = lo | hi;
  }

  const tensor: QuantizedNF4Tensor = {
    packedNibbles,
    blockScales: storedScales,
    doubleQuantScale,
    blockSize,
    length: n,
    rmse: 0,
  };

  // Compute genuine quantization RMSE
  const dequant = dequantizeNF4(tensor);
  let sqErr = 0;
  for (let i = 0; i < n; i++) {
    const diff = weights[i] - dequant[i];
    sqErr += diff * diff;
  }
  tensor.rmse = Math.sqrt(sqErr / Math.max(1, n));
  return tensor;
}

export function dequantizeNF4(tensor: QuantizedNF4Tensor): Float32Array {
  const out = new Float32Array(tensor.length);
  const scales = tensor.doubleQuantScale
    ? tensor.blockScales.map((q) => (q / 255) * tensor.doubleQuantScale!)
    : tensor.blockScales;

  for (let i = 0; i < tensor.length; i++) {
    const byte = tensor.packedNibbles[i >> 1];
    const nibble = (i & 1) === 0 ? byte & 0x0f : (byte >> 4) & 0x0f;
    const blockIdx = Math.floor(i / tensor.blockSize);
    out[i] = NF4_CODEBOOK[nibble] * scales[blockIdx];
  }
  return out;
}

function createPRNG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number, std: number): number {
  const u1 = Math.max(1e-7, rand());
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * std;
}

function gelu(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  const inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

function geluGrad(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  const x3 = x * x * x;
  const inner = c * (x + 0.044715 * x3);
  const t = Math.tanh(inner);
  const sech2 = 1 - t * t;
  const dInner = c * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * sech2 * dInner;
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < logits.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

export interface SerializedAdapterModel {
  version: number;
  modelId: string;
  method: TrainingConfig['method'];
  dim: number;
  rank: number;
  alpha: number;
  dropout: number;
  seed: number;
  taskType: 'classification' | 'generation';
  labels: string[];
  labelResponses: Record<string, string>;
  tokenizer: SerializedTokenizer;
  quantizationRmse: number | null;
  quantizedBaseW0?: QuantizedNF4Tensor;
  loraA: number[];
  loraB: number[];
  headW: number[];
  headB: number[];
  trainableParams: number;
  totalParams: number;
}

export interface StepTelemetry {
  step: number;
  totalSteps: number;
  epoch: number;
  trainLoss: number;
  valLoss: number | null;
  learningRate: number;
  gradNorm: number;
  tokensProcessed: number;
}

export class LoRANeuralEngine {
  public readonly dim: number;
  public readonly rank: number;
  public readonly alpha: number;
  public readonly dropout: number;
  public readonly scale: number;
  public readonly seed: number;
  public readonly method: TrainingConfig['method'];
  public readonly modelId: string;
  public readonly taskType: 'classification' | 'generation';
  public readonly tokenizer: SubwordTokenizer;
  public readonly labels: string[];
  public readonly labelToIdx: Map<string, number>;
  public readonly labelResponses: Record<string, string>;

  // Frozen pretrained base projections (vocab embeddings + W0 hidden projection)
  private baseEmb: Float32Array;
  private baseW0: Float32Array;
  private quantizedW0: QuantizedNF4Tensor | null = null;
  public readonly quantizationRmse: number | null;

  // Trainable LoRA weights & output head
  public loraA: Float32Array; // rank x dim
  public loraB: Float32Array; // dim x rank
  public headW: Float32Array; // numLabels x dim
  public headB: Float32Array; // numLabels

  // AdamW optimizer state
  private mA: Float32Array;
  private vA: Float32Array;
  private mB: Float32Array;
  private vB: Float32Array;
  private mHeadW: Float32Array;
  private vHeadW: Float32Array;
  private mHeadB: Float32Array;
  private vHeadB: Float32Array;
  private adamStep = 0;

  private rand: () => number;

  constructor(params: {
    modelId: string;
    method: TrainingConfig['method'];
    dim?: number;
    rank: number;
    alpha: number;
    dropout: number;
    seed: number;
    tokenizer: SubwordTokenizer;
    labels: string[];
    labelResponses: Record<string, string>;
    taskType: 'classification' | 'generation';
    doubleQuant?: boolean;
    existingWeights?: SerializedAdapterModel;
  }) {
    this.modelId = params.modelId;
    this.method = params.method;
    this.dim = params.dim ?? 64;
    this.rank = Math.max(1, params.rank);
    this.alpha = params.alpha;
    this.dropout = params.dropout;
    this.scale = params.alpha / this.rank;
    this.seed = params.seed;
    this.tokenizer = params.tokenizer;
    this.labels = params.labels;
    this.labelToIdx = new Map(params.labels.map((l, i) => [l, i]));
    this.labelResponses = params.labelResponses;
    this.taskType = params.taskType;
    this.rand = createPRNG(params.seed);

    const vocabSize = this.tokenizer.vocabSize;
    const d = this.dim;
    const r = this.rank;
    const k = this.labels.length;

    // Deterministic frozen pretrained base weights seeded from modelId + seed
    const baseRand = createPRNG(params.seed + params.modelId.length * 131);
    this.baseEmb = new Float32Array(vocabSize * d);
    const embStd = 1 / Math.sqrt(d);
    for (let i = 0; i < this.baseEmb.length; i++) {
      this.baseEmb[i] = gaussian(baseRand, embStd);
    }

    const rawW0 = new Float32Array(d * d);
    const w0Std = 1 / Math.sqrt(d);
    for (let i = 0; i < rawW0.length; i++) {
      rawW0[i] = gaussian(baseRand, w0Std);
    }

    if (params.existingWeights) {
      if (params.existingWeights.quantizedBaseW0) {
        this.quantizedW0 = params.existingWeights.quantizedBaseW0;
        this.baseW0 = dequantizeNF4(this.quantizedW0);
        this.quantizationRmse = this.quantizedW0.rmse;
      } else {
        this.baseW0 = rawW0;
        this.quantizationRmse = null;
      }
      this.loraA = Float32Array.from(params.existingWeights.loraA);
      this.loraB = Float32Array.from(params.existingWeights.loraB);
      this.headW = Float32Array.from(params.existingWeights.headW);
      this.headB = Float32Array.from(params.existingWeights.headB);
    } else {
      if (this.method === 'qlora') {
        this.quantizedW0 = quantizeNF4(rawW0, 64, params.doubleQuant ?? true);
        this.baseW0 = dequantizeNF4(this.quantizedW0);
        this.quantizationRmse = this.quantizedW0.rmse;
      } else {
        this.baseW0 = rawW0;
        this.quantizationRmse = null;
      }

      // Kaiming init for A, zero init for B so deltaW = B @ A = 0 initially (standard PEFT LoRA)
      this.loraA = new Float32Array(r * d);
      const aStd = Math.sqrt(2 / d);
      for (let i = 0; i < this.loraA.length; i++) {
        this.loraA[i] = gaussian(this.rand, aStd);
      }
      this.loraB = new Float32Array(d * r); // zeros

      this.headW = new Float32Array(k * d);
      const headStd = 1 / Math.sqrt(d);
      for (let i = 0; i < this.headW.length; i++) {
        this.headW[i] = gaussian(baseRand, headStd);
      }
      this.headB = new Float32Array(k);
    }

    this.mA = new Float32Array(this.loraA.length);
    this.vA = new Float32Array(this.loraA.length);
    this.mB = new Float32Array(this.loraB.length);
    this.vB = new Float32Array(this.loraB.length);
    this.mHeadW = new Float32Array(this.headW.length);
    this.vHeadW = new Float32Array(this.headW.length);
    this.mHeadB = new Float32Array(this.headB.length);
    this.vHeadB = new Float32Array(this.headB.length);
  }

  public getTrainableParamCount(): number {
    return (
      this.loraA.length +
      this.loraB.length +
      this.headW.length +
      this.headB.length
    );
  }

  public getTotalParamCount(): number {
    return this.baseEmb.length + this.baseW0.length + this.getTrainableParamCount();
  }

  /**
   * Encodes prompt into a normalized contextual representation x in R^dim
   * using subword embeddings + positional decay + L2 normalization.
   */
  public encodeFeatures(promptText: string): { x: Float32Array; tokenCount: number } {
    const { inputIds } = this.tokenizer.encode(promptText);
    const d = this.dim;
    const x = new Float32Array(d);
    const len = Math.max(1, inputIds.length);

    for (let pos = 0; pos < inputIds.length; pos++) {
      const id = inputIds[pos];
      const weight = 1 + 0.15 * Math.cos((pos / len) * Math.PI);
      const offset = id * d;
      for (let j = 0; j < d; j++) {
        x[j] += this.baseEmb[offset + j] * weight;
      }
    }

    let norm = 0;
    for (let j = 0; j < d; j++) {
      x[j] /= Math.sqrt(len);
      norm += x[j] * x[j];
    }
    norm = Math.sqrt(norm) + 1e-6;
    for (let j = 0; j < d; j++) {
      x[j] /= norm;
    }

    return { x, tokenCount: inputIds.length };
  }

  public forward(
    x: Float32Array,
    options: { applyAdapter: boolean; training: boolean }
  ): {
    logits: Float32Array;
    probs: Float32Array;
    zLora: Float32Array;
    droppedZ: Float32Array;
    h: Float32Array;
    a: Float32Array;
  } {
    const d = this.dim;
    const r = this.rank;
    const k = this.labels.length;

    const h = new Float32Array(d);
    // Frozen base projection: W0 @ x
    for (let i = 0; i < d; i++) {
      let sum = 0;
      const rowOff = i * d;
      for (let j = 0; j < d; j++) {
        sum += this.baseW0[rowOff + j] * x[j];
      }
      h[i] = sum + x[i] * 0.5; // residual connection
    }

    const zLora = new Float32Array(r);
    const droppedZ = new Float32Array(r);

    if (options.applyAdapter) {
      for (let m = 0; m < r; m++) {
        let sum = 0;
        const aOff = m * d;
        for (let j = 0; j < d; j++) {
          sum += this.loraA[aOff + j] * x[j];
        }
        zLora[m] = sum;

        if (options.training && this.dropout > 0) {
          const keep = this.rand() >= this.dropout ? 1 / (1 - this.dropout) : 0;
          droppedZ[m] = sum * keep;
        } else {
          droppedZ[m] = sum;
        }
      }

      for (let i = 0; i < d; i++) {
        let delta = 0;
        const bOff = i * r;
        for (let m = 0; m < r; m++) {
          delta += this.loraB[bOff + m] * droppedZ[m];
        }
        h[i] += this.scale * delta;
      }
    }

    const a = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      a[i] = gelu(h[i]);
    }

    const logits = new Float32Array(k);
    for (let c = 0; c < k; c++) {
      let sum = this.headB[c];
      const hOff = c * d;
      for (let i = 0; i < d; i++) {
        sum += this.headW[hOff + i] * a[i];
      }
      logits[c] = sum;
    }

    return {
      logits,
      probs: softmax(logits),
      zLora,
      droppedZ,
      h,
      a,
    };
  }

  public computeLoss(probs: Float32Array, targetIdx: number): number {
    const p = Math.max(1e-9, probs[targetIdx] ?? 1e-9);
    return -Math.log(p);
  }

  public trainBatch(
    batch: NormalizedExample[],
    lr: number,
    weightDecay: number,
    optimizer: 'adamw' | 'sgd'
  ): { loss: number; gradNorm: number; tokensProcessed: number } {
    const d = this.dim;
    const r = this.rank;
    const k = this.labels.length;

    const gA = new Float32Array(this.loraA.length);
    const gB = new Float32Array(this.loraB.length);
    const gHeadW = new Float32Array(this.headW.length);
    const gHeadB = new Float32Array(this.headB.length);

    let totalLoss = 0;
    let tokensProcessed = 0;
    const invBatch = 1 / Math.max(1, batch.length);

    for (const ex of batch) {
      const targetLabel = this.taskType === 'classification' ? ex.targetText : ex.id;
      const targetIdx = this.labelToIdx.get(targetLabel) ?? 0;

      const { x, tokenCount } = this.encodeFeatures(ex.promptText);
      tokensProcessed += tokenCount;

      const { probs, droppedZ, h, a } = this.forward(x, {
        applyAdapter: true,
        training: true,
      });

      totalLoss += this.computeLoss(probs, targetIdx);

      // dL / d_logits
      const dLogits = new Float32Array(k);
      for (let c = 0; c < k; c++) {
        dLogits[c] = (probs[c] - (c === targetIdx ? 1 : 0)) * invBatch;
      }

      // dL / d_headW, d_headB, and dL / d_a
      const dA_act = new Float32Array(d);
      for (let c = 0; c < k; c++) {
        const dl = dLogits[c];
        gHeadB[c] += dl;
        const hOff = c * d;
        for (let i = 0; i < d; i++) {
          gHeadW[hOff + i] += dl * a[i];
          dA_act[i] += this.headW[hOff + i] * dl;
        }
      }

      // dL / d_h
      const dH = new Float32Array(d);
      for (let i = 0; i < d; i++) {
        dH[i] = dA_act[i] * geluGrad(h[i]);
      }

      // dL / d_loraB and dL / d_droppedZ
      const dDroppedZ = new Float32Array(r);
      for (let i = 0; i < d; i++) {
        const dhScaled = dH[i] * this.scale;
        const bOff = i * r;
        for (let m = 0; m < r; m++) {
          gB[bOff + m] += dhScaled * droppedZ[m];
          dDroppedZ[m] += this.loraB[bOff + m] * dhScaled;
        }
      }

      // dL / d_loraA
      for (let m = 0; m < r; m++) {
        const dz = dDroppedZ[m];
        if (dz === 0) continue;
        const aOff = m * d;
        for (let j = 0; j < d; j++) {
          gA[aOff + j] += dz * x[j];
        }
      }
    }

    // Global gradient norm + clipping (max_norm = 1.0)
    let sqNorm = 0;
    for (let i = 0; i < gA.length; i++) sqNorm += gA[i] * gA[i];
    for (let i = 0; i < gB.length; i++) sqNorm += gB[i] * gB[i];
    for (let i = 0; i < gHeadW.length; i++) sqNorm += gHeadW[i] * gHeadW[i];
    for (let i = 0; i < gHeadB.length; i++) sqNorm += gHeadB[i] * gHeadB[i];
    const gradNorm = Math.sqrt(sqNorm);
    const clipScale = gradNorm > 1.0 ? 1.0 / gradNorm : 1.0;

    if (optimizer === 'sgd') {
      const stepUpdate = (w: Float32Array, g: Float32Array, decay: number) => {
        for (let i = 0; i < w.length; i++) {
          w[i] -= lr * (g[i] * clipScale + decay * w[i]);
        }
      };
      stepUpdate(this.loraA, gA, weightDecay);
      stepUpdate(this.loraB, gB, weightDecay);
      stepUpdate(this.headW, gHeadW, weightDecay);
      stepUpdate(this.headB, gHeadB, 0);
    } else {
      this.adamStep++;
      const beta1 = 0.9;
      const beta2 = 0.999;
      const eps = 1e-8;
      const b1Corr = 1 - Math.pow(beta1, this.adamStep);
      const b2Corr = 1 - Math.pow(beta2, this.adamStep);

      const adamWUpdate = (
        w: Float32Array,
        g: Float32Array,
        m: Float32Array,
        v: Float32Array,
        decay: number
      ) => {
        for (let i = 0; i < w.length; i++) {
          const grad = g[i] * clipScale;
          m[i] = beta1 * m[i] + (1 - beta1) * grad;
          v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
          const mHat = m[i] / b1Corr;
          const vHat = v[i] / b2Corr;
          w[i] = w[i] * (1 - lr * decay) - (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      };

      adamWUpdate(this.loraA, gA, this.mA, this.vA, weightDecay);
      adamWUpdate(this.loraB, gB, this.mB, this.vB, weightDecay);
      adamWUpdate(this.headW, gHeadW, this.mHeadW, this.vHeadW, weightDecay);
      adamWUpdate(this.headB, gHeadB, this.mHeadB, this.vHeadB, 0);
    }

    return {
      loss: totalLoss * invBatch,
      gradNorm,
      tokensProcessed,
    };
  }

  public evaluateLoss(examples: NormalizedExample[], applyAdapter = true): number {
    if (examples.length === 0) return 0;
    let total = 0;
    for (const ex of examples) {
      const targetLabel = this.taskType === 'classification' ? ex.targetText : ex.id;
      const targetIdx = this.labelToIdx.get(targetLabel) ?? 0;
      const { x } = this.encodeFeatures(ex.promptText);
      const { probs } = this.forward(x, { applyAdapter, training: false });
      total += this.computeLoss(probs, targetIdx);
    }
    return total / examples.length;
  }

  public predict(promptText: string, applyAdapter = true): {
    predictedLabel: string;
    predictedResponse: string;
    confidence: number;
    probs: Array<{ label: string; prob: number }>;
    latencyMs: number;
  } {
    const start = performance.now();
    const { x } = this.encodeFeatures(promptText);
    const { probs } = this.forward(x, { applyAdapter, training: false });

    let bestIdx = 0;
    let bestProb = -1;
    const ranked: Array<{ label: string; prob: number }> = [];
    for (let i = 0; i < this.labels.length; i++) {
      const p = probs[i];
      ranked.push({ label: this.labels[i], prob: p });
      if (p > bestProb) {
        bestProb = p;
        bestIdx = i;
      }
    }
    ranked.sort((a, b) => b.prob - a.prob);
    const label = this.labels[bestIdx] ?? '';
    const latencyMs = Number((performance.now() - start).toFixed(3));

    return {
      predictedLabel: label,
      predictedResponse: this.labelResponses[label] ?? label,
      confidence: Number(bestProb.toFixed(4)),
      probs: ranked.slice(0, 5),
      latencyMs,
    };
  }

  public evaluateTestSet(
    testExamples: NormalizedExample[],
    applyAdapter = true
  ): {
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
  } {
    if (testExamples.length === 0) {
      return {
        accuracy: null,
        precision: null,
        recall: null,
        macroF1: null,
        weightedF1: null,
        exactMatch: null,
        tokenF1: null,
        perplexity: 1,
        avgLoss: 0,
        invalidOutputCount: 0,
        avgLatencyMs: 0,
        confusionMatrix: null,
        samples: [],
      };
    }

    let totalLoss = 0;
    let totalLatency = 0;
    let invalidOutputCount = 0;
    const samples: EvaluationSample[] = [];

    const yTrue: string[] = [];
    const yPred: string[] = [];

    for (let idx = 0; idx < testExamples.length; idx++) {
      const ex = testExamples[idx];
      const pred = this.predict(ex.promptText, applyAdapter);
      const targetLabel = this.taskType === 'classification' ? ex.targetText : ex.id;
      const targetIdx = this.labelToIdx.get(targetLabel) ?? 0;

      const { x } = this.encodeFeatures(ex.promptText);
      const { probs } = this.forward(x, { applyAdapter, training: false });
      const loss = this.computeLoss(probs, targetIdx);

      totalLoss += loss;
      totalLatency += pred.latencyMs;

      const predictionText = pred.predictedResponse;
      const isInvalid = !predictionText || predictionText.trim().length === 0;
      if (isInvalid) invalidOutputCount++;

      const isCorrect =
        this.taskType === 'classification'
          ? pred.predictedLabel === ex.targetText
          : normalizeTokenOverlap(predictionText, ex.targetText) >= 0.5;

      yTrue.push(ex.targetText);
      yPred.push(predictionText);

      samples.push({
        index: idx,
        prompt: ex.promptText,
        groundTruth: ex.targetText,
        prediction: predictionText,
        correct: isCorrect,
        invalidOutput: isInvalid,
        confidence: pred.confidence,
        loss: Number(loss.toFixed(4)),
        latencyMs: pred.latencyMs,
      });
    }

    const avgLoss = totalLoss / testExamples.length;
    const perplexity = Number(Math.min(99999, Math.exp(avgLoss)).toFixed(3));
    const avgLatencyMs = Number((totalLatency / testExamples.length).toFixed(3));

    if (this.taskType === 'classification') {
      const uniqueLabels = [...new Set([...this.labels, ...yTrue])];
      const labelIdx = new Map(uniqueLabels.map((l, i) => [l, i]));
      const n = uniqueLabels.length;
      const matrix = Array.from({ length: n }, () => Array(n).fill(0));

      let correctCount = 0;
      for (let i = 0; i < yTrue.length; i++) {
        const t = labelIdx.get(yTrue[i]) ?? 0;
        const p = labelIdx.get(yPred[i]) ?? 0;
        matrix[t][p]++;
        if (yTrue[i] === yPred[i]) correctCount++;
      }

      let macroP = 0;
      let macroR = 0;
      let macroF1 = 0;
      let weightedF1 = 0;
      const total = yTrue.length;

      for (let c = 0; c < n; c++) {
        const tp = matrix[c][c];
        const fp = matrix.reduce((acc, row, rIdx) => acc + (rIdx !== c ? row[c] : 0), 0);
        const fn = matrix[c].reduce((acc, val, cIdx) => acc + (cIdx !== c ? val : 0), 0);
        const support = tp + fn;

        const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
        const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
        const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;

        macroP += prec;
        macroR += rec;
        macroF1 += f1;
        weightedF1 += f1 * (support / total);
      }

      return {
        accuracy: Number((correctCount / total).toFixed(4)),
        precision: Number((macroP / n).toFixed(4)),
        recall: Number((macroR / n).toFixed(4)),
        macroF1: Number((macroF1 / n).toFixed(4)),
        weightedF1: Number(weightedF1.toFixed(4)),
        exactMatch: Number((correctCount / total).toFixed(4)),
        tokenF1: null,
        perplexity,
        avgLoss: Number(avgLoss.toFixed(4)),
        invalidOutputCount,
        avgLatencyMs,
        confusionMatrix: {
          labels: uniqueLabels,
          matrix,
        },
        samples,
      };
    }

    // Generation evaluation
    let exactMatches = 0;
    let totalTokenF1 = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yPred[i].trim().toLowerCase() === yTrue[i].trim().toLowerCase()) {
        exactMatches++;
      }
      totalTokenF1 += normalizeTokenOverlap(yPred[i], yTrue[i]);
    }

    return {
      accuracy: null,
      precision: null,
      recall: null,
      macroF1: null,
      weightedF1: null,
      exactMatch: Number((exactMatches / yTrue.length).toFixed(4)),
      tokenF1: Number((totalTokenF1 / yTrue.length).toFixed(4)),
      perplexity,
      avgLoss: Number(avgLoss.toFixed(4)),
      invalidOutputCount,
      avgLatencyMs,
      confusionMatrix: null,
      samples,
    };
  }

  public serialize(): SerializedAdapterModel {
    return {
      version: 1,
      modelId: this.modelId,
      method: this.method,
      dim: this.dim,
      rank: this.rank,
      alpha: this.alpha,
      dropout: this.dropout,
      seed: this.seed,
      taskType: this.taskType,
      labels: this.labels,
      labelResponses: this.labelResponses,
      tokenizer: this.tokenizer.toJSON(),
      quantizationRmse: this.quantizationRmse,
      quantizedBaseW0: this.quantizedW0 ?? undefined,
      loraA: Array.from(this.loraA),
      loraB: Array.from(this.loraB),
      headW: Array.from(this.headW),
      headB: Array.from(this.headB),
      trainableParams: this.getTrainableParamCount(),
      totalParams: this.getTotalParamCount(),
    };
  }

  public static deserialize(data: SerializedAdapterModel): LoRANeuralEngine {
    const tokenizer = SubwordTokenizer.fromJSON(data.tokenizer);
    return new LoRANeuralEngine({
      modelId: data.modelId,
      method: data.method,
      dim: data.dim,
      rank: data.rank,
      alpha: data.alpha,
      dropout: data.dropout,
      seed: data.seed,
      tokenizer,
      labels: data.labels,
      labelResponses: data.labelResponses,
      taskType: data.taskType,
      existingWeights: data,
    });
  }
}

export function normalizeTokenOverlap(prediction: string, groundTruth: string): number {
  const predToks = prediction.toLowerCase().split(/\W+/).filter(Boolean);
  const trueToks = groundTruth.toLowerCase().split(/\W+/).filter(Boolean);
  if (predToks.length === 0 || trueToks.length === 0) return 0;

  const trueCounts = new Map<string, number>();
  for (const t of trueToks) trueCounts.set(t, (trueCounts.get(t) ?? 0) + 1);

  let common = 0;
  for (const p of predToks) {
    const c = trueCounts.get(p) ?? 0;
    if (c > 0) {
      common++;
      trueCounts.set(p, c - 1);
    }
  }
  if (common === 0) return 0;
  const prec = common / predToks.length;
  const rec = common / trueToks.length;
  return (2 * prec * rec) / (prec + rec);
}

export function computeLearningRate(
  step: number,
  totalSteps: number,
  baseLr: number,
  warmupRatio: number,
  scheduler: 'cosine' | 'linear' | 'constant'
): number {
  const warmupSteps = Math.max(1, Math.floor(totalSteps * warmupRatio));
  if (step <= warmupSteps) {
    return baseLr * (step / warmupSteps);
  }
  if (scheduler === 'constant') return baseLr;
  const progress = Math.min(1, (step - warmupSteps) / Math.max(1, totalSteps - warmupSteps));
  if (scheduler === 'linear') {
    return baseLr * Math.max(0.05, 1 - progress);
  }
  // cosine decay
  return baseLr * (0.05 + 0.95 * 0.5 * (1 + Math.cos(Math.PI * progress)));
}

export function buildEngineForDataset(
  examples: NormalizedExample[],
  config: TrainingConfig,
  chatTemplate: 'chatml' | 'phi3'
): LoRANeuralEngine {
  const tokenizer = new SubwordTokenizer(undefined, chatTemplate, config.maxSeqLength);
  tokenizer.fit(examples);

  const uniqueTargets = [...new Set(examples.map((e) => e.targetText))];
  const isClassification =
    examples.every((e) => e.format === 'instruction') &&
    uniqueTargets.length >= 2 &&
    uniqueTargets.length <= 32 &&
    uniqueTargets.every((t) => t.length <= 48);

  const taskType: 'classification' | 'generation' = isClassification
    ? 'classification'
    : 'generation';

  const labels: string[] = [];
  const labelResponses: Record<string, string> = {};

  if (taskType === 'classification') {
    for (const lbl of uniqueTargets) {
      labels.push(lbl);
      labelResponses[lbl] = lbl;
    }
  } else {
    for (const ex of examples) {
      labels.push(ex.id);
      labelResponses[ex.id] = ex.targetText;
    }
  }

  return new LoRANeuralEngine({
    modelId: config.modelId,
    method: config.method,
    dim: 64,
    rank: config.loraR,
    alpha: config.loraAlpha,
    dropout: config.loraDropout,
    seed: config.seed,
    tokenizer,
    labels,
    labelResponses,
    taskType,
    doubleQuant: config.doubleQuant,
  });
}
