import crypto from 'node:crypto';
import fs from 'node:fs';
import { GoogleGenAI, Type } from '@google/genai';
import { db, logAudit } from './db.ts';
import { ensureDatasetBundlePrepared } from './datasetBundleService.ts';
import { loadTrainedEngineFromAdapter } from './trainingManager.ts';
import { getModelById, PRETRAINED_MODELS } from '../src/ml/modelCatalog.ts';
import type { ProcessedDatasetBundle } from '../src/ml/datasetProcessor.ts';
import {
  aggregatePlaygroundReport,
  buildPlaygroundTestSuite,
  clampScore,
  evaluateStructuredOutput,
  inferBooleanFromChoice,
  parseCandidateChoices,
  resolveCandidateChoices,
  scoreZeroShotPrompt,
  summarizePlaygroundBatchReports,
} from '../src/ml/structuredInference.ts';
import type {
  PlaygroundBatchTestResponse,
  PlaygroundTestCaseResult,
  PlaygroundTestReport,
  StructuredInferenceOutput,
} from '../src/shared/types.ts';

function getGeminiClient(): GoogleGenAI | null {
  const key = process.env.GEMINI_API_KEY;
  if (!key || key === 'MY_GEMINI_API_KEY') return null;
  return new GoogleGenAI({
    apiKey: key,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

export interface InferenceRequest {
  modelRef: string; // e.g. 'base:qwen2.5-0.5b-instruct' or 'reg:<registered_model_id>'
  prompt: string;
  datasetId?: string;
  systemPrompt?: string;
  choices?: string[] | string;
  history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  temperature?: number;
  topP?: number;
  maxNewTokens?: number;
}

interface ResolvedModelHandle {
  modelRef: string;
  modelLabel: string;
  method: string;
  isBaseOnly: boolean;
  baseModelId: string;
  datasetId?: string;
  engine?: ReturnType<typeof loadTrainedEngineFromAdapter>;
  modelSpec?: ReturnType<typeof getModelById>;
}

function resolveModelHandle(modelRef: string): ResolvedModelHandle {
  const isBaseOnly = modelRef.startsWith('base:');
  if (!isBaseOnly) {
    const regId = modelRef.replace(/^reg:/, '');
    const regRow = db
      .prepare('SELECT * FROM registered_models WHERE id = ?')
      .get(regId) as Record<string, unknown> | undefined;

    if (!regRow) {
      throw new Error(`Registered model "${regId}" not found.`);
    }

    const engine = loadTrainedEngineFromAdapter(String(regRow.adapter_path));
    return {
      modelRef,
      modelLabel: `${String(regRow.name)} (${String(regRow.method).toUpperCase()} · ${String(regRow.base_model_name)})`,
      method: String(regRow.method),
      isBaseOnly: false,
      baseModelId: String(regRow.base_model_id),
      datasetId: String(regRow.dataset_id),
      engine,
      modelSpec: getModelById(String(regRow.base_model_id)),
    };
  }

  const baseId = modelRef.replace(/^base:/, '');
  const spec = getModelById(baseId);
  if (!spec) {
    throw new Error(`Pretrained base model "${baseId}" not found.`);
  }
  return {
    modelRef,
    modelLabel: `${spec.name} (Pretrained Base)`,
    method: 'base',
    isBaseOnly: true,
    baseModelId: spec.id,
    modelSpec: spec,
  };
}

function loadDatasetBundle(datasetId?: string): {
  datasetId: string;
  datasetName: string;
  bundle: ProcessedDatasetBundle | null;
} {
  let targetId = datasetId;
  if (!targetId) {
    const row = db
      .prepare(`SELECT id FROM datasets ORDER BY updated_at DESC LIMIT 1`)
      .get() as { id: string } | undefined;
    targetId = row?.id;
  }
  if (!targetId) {
    return {
      datasetId: 'default',
      datasetName: 'Workspace Verification Corpus',
      bundle: null,
    };
  }
  try {
    const { datasetRow, bundle } = ensureDatasetBundlePrepared(targetId);
    return {
      datasetId: String(datasetRow.id),
      datasetName: String(datasetRow.name),
      bundle,
    };
  } catch {
    return {
      datasetId: targetId,
      datasetName: 'Workspace Verification Corpus',
      bundle: null,
    };
  }
}

function runLocalStructuredInference(params: {
  handle: ResolvedModelHandle;
  prompt: string;
  choices: string[];
  referenceExamples?: ProcessedDatasetBundle['train'];
}): {
  structured: StructuredInferenceOutput;
  predictedClass?: string;
  confidence: number;
  adapterDetail?: string;
} {
  const { handle, prompt, choices, referenceExamples } = params;

  if (!handle.isBaseOnly && handle.engine) {
    const pred = handle.engine.predict(prompt, true);
    const engine = handle.engine;

    // If evaluating classification labels covered by the trained adapter
    if (
      engine.taskType === 'classification' &&
      choices.some((c) => engine.labels.some((l) => l.toLowerCase() === c.toLowerCase()))
    ) {
      // Score each requested candidate choice using the adapter's real class probabilities
      const choiceScores = choices.map((c) => {
        const hit = pred.probs.find((p) => p.label.toLowerCase() === c.toLowerCase());
        return { choice: c, prob: hit ? hit.prob : 0 };
      });
      choiceScores.sort((a, b) => b.prob - a.prob);
      const top = choiceScores[0];

      // Check adversarial probe even when banking choices were not requested
      if (top.prob > 0) {
        const structured = evaluateStructuredOutput({
          rawChoice: pred.predictedLabel,
          rawScore: pred.confidence,
          choices,
          explanation: `LoRA adapter predicted "${pred.predictedLabel}" with softmax probability ${(pred.confidence * 100).toFixed(1)}%.`,
        });
        return {
          structured,
          predictedClass: pred.predictedLabel,
          confidence: pred.confidence,
          adapterDetail: pred.predictedResponse,
        };
      }
    }

    // For boolean / verification prompts or generation adapters: check grounded support + zero-shot calibration
    const zeroShot = scoreZeroShotPrompt({
      modelSpec: handle.modelSpec,
      prompt,
      choices,
      referenceExamples,
    });
    const combinedScore = clampScore(zeroShot.score * 0.65 + pred.confidence * 0.35);
    const structured = evaluateStructuredOutput({
      rawChoice: zeroShot.choice,
      rawBoolean: zeroShot.booleanValue,
      rawScore: combinedScore,
      choices,
      explanation: `${handle.modelLabel} grounded verification: ${zeroShot.explanation} Adapter top signal="${pred.predictedResponse.slice(0, 80)}".`,
    });
    return {
      structured,
      predictedClass: pred.predictedLabel,
      confidence: structured.score,
      adapterDetail: pred.predictedResponse,
    };
  }

  // Pretrained base model deterministic zero-shot structured inference (always works offline)
  const structured = scoreZeroShotPrompt({
    modelSpec: handle.modelSpec,
    prompt,
    choices,
    referenceExamples,
  });
  return {
    structured,
    predictedClass: structured.choice,
    confidence: structured.score,
  };
}

export async function* streamModelInference(req: InferenceRequest): AsyncGenerator<{
  type: 'meta' | 'token' | 'done' | 'error';
  text?: string;
  meta?: {
    predictedClass?: string;
    confidence?: number;
    latencyMs?: number;
    tokensGenerated?: number;
    engineMode?: string;
    structured?: StructuredInferenceOutput;
  };
  error?: string;
}> {
  const startWall = performance.now();
  let handle: ResolvedModelHandle;
  try {
    handle = resolveModelHandle(req.modelRef);
  } catch (err) {
    yield { type: 'error', error: (err as Error).message };
    return;
  }

  const { bundle } = loadDatasetBundle(req.datasetId ?? handle.datasetId);
  const datasetLabels: string[] =
    handle.engine?.taskType === 'classification'
      ? handle.engine.labels
      : bundle
        ? [...new Set(bundle.examples.map((e) => e.targetText))].filter((t) => t.length <= 48)
        : [];

  const customChoices = parseCandidateChoices(req.choices);
  const choices = resolveCandidateChoices({
    prompt: req.prompt,
    customChoices,
    datasetLabels,
  });

  const localResult = runLocalStructuredInference({
    handle,
    prompt: req.prompt,
    choices,
    referenceExamples: bundle?.train,
  });

  let finalStructured = localResult.structured;
  let engineMode = handle.isBaseOnly ? 'base-structured-local' : 'lora-adapter-local';

  yield {
    type: 'meta',
    meta: {
      predictedClass: localResult.predictedClass,
      confidence: localResult.confidence,
      engineMode,
      structured: finalStructured,
    },
  };

  const ai = getGeminiClient();
  if (ai) {
    try {
      const sysParts = [
        req.systemPrompt?.trim() ||
          'You are a grounded small language model evaluation engine inside MiniMind AI.',
        `Active model profile: ${handle.modelLabel}.`,
        `Allowed choices: [${choices.join(', ')}].`,
        `Local adapter/model signal: choice="${localResult.structured.choice}", score=${localResult.structured.score}.`,
        'Respond STRICTLY with JSON containing booleanValue (boolean), choice (must be exact match from Allowed choices), score (0.0 to 1.0), and explanation (concise grounded justification, no fabricated entities).',
      ];

      const historyText = (req.history ?? [])
        .slice(-4)
        .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
        .join('\n');
      const fullPrompt = historyText
        ? `${historyText}\nUser: ${req.prompt}`
        : req.prompt;

      const result = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: fullPrompt,
        config: {
          systemInstruction: sysParts.join('\n'),
          temperature: Math.min(0.4, req.temperature ?? 0.2),
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              booleanValue: { type: Type.BOOLEAN },
              choice: { type: Type.STRING },
              score: { type: Type.NUMBER },
              explanation: { type: Type.STRING },
            },
            required: ['booleanValue', 'choice', 'score', 'explanation'],
          },
        },
      });

      const parsed = JSON.parse(result.text ?? '{}') as {
        booleanValue?: boolean;
        choice?: string;
        score?: number;
        explanation?: string;
      };

      finalStructured = evaluateStructuredOutput({
        rawChoice: String(parsed.choice ?? ''),
        rawBoolean: parsed.booleanValue,
        rawScore: Number(parsed.score),
        choices,
        explanation: parsed.explanation,
      });
      engineMode = handle.isBaseOnly ? 'base-structured-gemini' : 'lora-adapter+gemini';
    } catch {
      // Seamless local fallback keeps every model functional offline or if Gemini errors
    }
  }

  const formattedReply = [
    `**Structured Verification Output**`,
    `- **boolean:** \`${String(finalStructured.booleanValue)}\``,
    `- **choice:** \`${finalStructured.choice}\``,
    `- **score:** \`${finalStructured.score.toFixed(4)}\``,
    `- **grounded:** \`${finalStructured.grounded ? 'verified' : 'flagged_hallucination'}\``,
    ``,
    finalStructured.explanation,
  ].join('\n');

  const chunks = formattedReply.split(/(\s+)/);
  for (const piece of chunks) {
    yield { type: 'token', text: piece };
    await new Promise((r) => setTimeout(r, 6));
  }

  const latencyMs = Number((performance.now() - startWall).toFixed(1));
  const tokensGenerated = Math.max(1, Math.ceil(formattedReply.length / 4));
  yield {
    type: 'done',
    meta: {
      predictedClass: finalStructured.choice,
      confidence: finalStructured.score,
      latencyMs,
      tokensGenerated,
      engineMode,
      structured: finalStructured,
    },
  };
}

export function runPlaygroundModelDiagnosticReport(params: {
  modelRef: string;
  datasetId?: string;
  suiteMode?: PlaygroundTestReport['suiteMode'];
}): PlaygroundTestReport {
  const handle = resolveModelHandle(params.modelRef);
  const { datasetId, datasetName, bundle } = loadDatasetBundle(params.datasetId ?? handle.datasetId);

  const testExamples = bundle?.test?.length
    ? bundle.test
    : bundle?.examples?.slice(-6) ?? [];

  const allLabels: string[] = bundle
    ? [...new Set(bundle.examples.map((e) => e.targetText))].filter((t) => t.length <= 48)
    : [];

  const taskType: 'classification' | 'generation' =
    handle.engine?.taskType ?? (allLabels.length >= 2 && allLabels.length <= 24 ? 'classification' : 'generation');

  const suiteMode = params.suiteMode ?? 'dataset_and_probes';
  const suiteSpecs = buildPlaygroundTestSuite({
    testExamples,
    allLabels,
    suiteMode,
    taskType,
  });

  const caseResults: PlaygroundTestCaseResult[] = suiteSpecs.map((spec, idx) => {
    const t0 = performance.now();
    const res = runLocalStructuredInference({
      handle,
      prompt: spec.prompt,
      choices: spec.choices,
      referenceExamples: bundle?.train,
    });
    const latencyMs = Number((performance.now() - t0).toFixed(2));
    const s = res.structured;
    const passed =
      !s.hallucinationFlag &&
      s.choice.toLowerCase() === spec.expectedChoice.toLowerCase() &&
      s.booleanValue === spec.expectedBoolean;

    return {
      index: idx,
      category: spec.category,
      prompt: spec.prompt,
      expectedChoice: spec.expectedChoice,
      expectedBoolean: spec.expectedBoolean,
      booleanValue: s.booleanValue,
      choice: s.choice,
      score: s.score,
      passed,
      hallucinationFlag: s.hallucinationFlag,
      latencyMs,
      explanation: s.explanation,
    };
  });

  const report = aggregatePlaygroundReport({
    id: `pgrep_${Date.now().toString(36)}_${crypto.randomUUID().slice(0, 6)}`,
    modelRef: handle.modelRef,
    modelLabel: handle.modelLabel,
    method: handle.method,
    datasetId,
    datasetName,
    suiteMode,
    cases: caseResults,
  });

  db.prepare(
    `INSERT INTO playground_reports (
      id, workspace_id, model_ref, model_label, method, dataset_id, dataset_name, suite_mode,
      total_cases, passed_count, pass_rate, hallucination_count, hallucination_rate,
      avg_score, avg_latency_ms, verdict, cases_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    report.id,
    'ws_default',
    report.modelRef,
    report.modelLabel,
    report.method,
    report.datasetId,
    report.datasetName,
    report.suiteMode,
    report.totalCases,
    report.passedCount,
    report.passRate,
    report.hallucinationCount,
    report.hallucinationRate,
    report.avgScore,
    report.avgLatencyMs,
    report.verdict,
    JSON.stringify(report.cases),
    report.createdAt
  );

  logAudit(
    'PLAYGROUND_MODEL_REPORT',
    'playground_report',
    report.id,
    `Tested ${report.modelLabel}: ${report.passedCount}/${report.totalCases} passed (${(report.passRate * 100).toFixed(1)}%), verdict=${report.verdict}`
  );

  return report;
}

export function listAllPlaygroundModelRefs(): string[] {
  const regRows = db
    .prepare('SELECT id FROM registered_models ORDER BY created_at DESC')
    .all() as Array<{ id: unknown }>;
  const registeredRefs = regRows.map((row) => `reg:${String(row.id)}`);
  const baseRefs = PRETRAINED_MODELS.map((m) => `base:${m.id}`);
  return [...registeredRefs, ...baseRefs];
}

export function runPlaygroundBatchDiagnosticReports(params: {
  modelRefs?: string[];
  datasetId?: string;
  suiteMode?: PlaygroundTestReport['suiteMode'];
}): PlaygroundBatchTestResponse {
  const requestedRefs =
    params.modelRefs && params.modelRefs.length > 0
      ? [...new Set(params.modelRefs.map((ref) => String(ref).trim()).filter(Boolean))]
      : listAllPlaygroundModelRefs();

  if (requestedRefs.length === 0) {
    throw new Error('No models available to test in Playground.');
  }

  const generatedReports: PlaygroundTestReport[] = requestedRefs.map((modelRef) =>
    runPlaygroundModelDiagnosticReport({
      modelRef,
      datasetId: params.datasetId,
      suiteMode: params.suiteMode,
    })
  );

  const { sortedReports, summary } = summarizePlaygroundBatchReports(generatedReports);

  logAudit(
    'PLAYGROUND_BATCH_TEST',
    'playground_report',
    'batch_all_models',
    `Tested all ${summary.totalModels} Playground models (${summary.passedModels} pass, ${summary.reviewModels} review, ${summary.failedModels} fail, mean ${(summary.avgPassRate * 100).toFixed(1)}%)`
  );

  return {
    reports: sortedReports,
    summary,
  };
}

export function listPlaygroundTestReports(): PlaygroundTestReport[] {
  const rows = db
    .prepare('SELECT * FROM playground_reports ORDER BY created_at DESC LIMIT 100')
    .all() as Record<string, unknown>[];
  return rows.map((r) => ({
    id: String(r.id),
    modelRef: String(r.model_ref),
    modelLabel: String(r.model_label),
    method: String(r.method),
    datasetId: String(r.dataset_id),
    datasetName: String(r.dataset_name),
    suiteMode: r.suite_mode as PlaygroundTestReport['suiteMode'],
    totalCases: Number(r.total_cases),
    passedCount: Number(r.passed_count),
    passRate: Number(r.pass_rate),
    hallucinationCount: Number(r.hallucination_count),
    hallucinationRate: Number(r.hallucination_rate),
    avgScore: Number(r.avg_score),
    avgLatencyMs: Number(r.avg_latency_ms),
    verdict: r.verdict as PlaygroundTestReport['verdict'],
    cases: JSON.parse(String(r.cases_json)) as PlaygroundTestCaseResult[],
    createdAt: String(r.created_at),
  }));
}

export { inferBooleanFromChoice };
