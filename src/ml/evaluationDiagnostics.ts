import type { EvaluationRunRecord, EvaluationSample } from '../shared/types';
import type { NavSection } from '../views/OverviewView';

export type DiagnosticSeverity = 'critical' | 'warning' | 'info';

export interface EvaluationDiagnosticItem {
  id: string;
  severity: DiagnosticSeverity;
  title: string;
  explanation: string;
  rectification: string;
  recommendedAction?: {
    label: string;
    targetSection: NavSection;
  };
}

export interface ConfusionPair {
  actual: string;
  predicted: string;
  count: number;
}

export interface EvaluationDiagnosisReport {
  overallHealth: 'healthy' | 'warning' | 'critical';
  summary: string;
  errorRate: number;
  mismatchCount: number;
  topConfusions: ConfusionPair[];
  hardestSamples: EvaluationSample[];
  diagnostics: EvaluationDiagnosticItem[];
}

/**
 * Analyzes an evaluation run's real error patterns (misclassifications, confusion pairs,
 * perplexity/loss spikes, invalid outputs, precision-recall imbalance, or small test sets)
 * and provides concrete explanations and ways to rectify each issue.
 */
export function analyzeEvaluationRun(evalRun: EvaluationRunRecord): EvaluationDiagnosisReport {
  const diagnostics: EvaluationDiagnosticItem[] = [];
  const total = Math.max(1, evalRun.testSampleCount);
  const mismatchCount = evalRun.samples.filter((s) => !s.correct).length;
  const errorRate = Number((mismatchCount / total).toFixed(4));

  // Extract top confusion pairs
  const topConfusions: ConfusionPair[] = [];
  if (evalRun.confusionMatrix) {
    const { labels, matrix } = evalRun.confusionMatrix;
    for (let r = 0; r < labels.length; r++) {
      for (let c = 0; c < labels.length; c++) {
        if (r !== c && matrix[r]?.[c] > 0) {
          topConfusions.push({
            actual: labels[r],
            predicted: labels[c],
            count: matrix[r][c],
          });
        }
      }
    }
    topConfusions.sort((a, b) => b.count - a.count);
  }

  const hardestSamples = [...evalRun.samples]
    .filter((s) => !s.correct || s.invalidOutput || s.loss > evalRun.avgLoss)
    .sort((a, b) => b.loss - a.loss)
    .slice(0, 5);

  const primaryScore =
    evalRun.taskType === 'classification'
      ? (evalRun.accuracy ?? 0)
      : (evalRun.tokenF1 ?? evalRun.exactMatch ?? 0);

  // 1. Frozen base model vs fine-tuned adapter check
  if (evalRun.method === 'base' && primaryScore < 0.85) {
    diagnostics.push({
      id: 'frozen-base-gap',
      severity: primaryScore < 0.65 ? 'critical' : 'warning',
      title: `Zero-Shot Domain Gap on "${evalRun.modelName}"`,
      explanation: `Base model "${evalRun.modelName}" was evaluated without domain adapter weights, scoring ${(primaryScore * 100).toFixed(1)}% with mean loss ${evalRun.avgLoss.toFixed(3)}. Zero-shot priors often confuse specialized workspace classes.`,
      rectification:
        'Open Fine-Tuning Studio, pick LoRA (r=8 or r=16) or 4-bit QLoRA with target modules ["q_proj","v_proj","o_proj"], and train for 10–16 epochs on this dataset.',
      recommendedAction: {
        label: 'Fine-Tune in Studio',
        targetSection: 'studio',
      },
    });
  }

  // 2. Top confusion pair diagnosis
  if (topConfusions.length > 0) {
    const top = topConfusions[0];
    const totalMisclassified = topConfusions.reduce((s, p) => s + p.count, 0);
    diagnostics.push({
      id: 'confusion-pair',
      severity: top.count >= 2 || totalMisclassified >= total * 0.25 ? 'warning' : 'info',
      title: `Recurring Confusion: "${top.actual}" predicted as "${top.predicted}"`,
      explanation: `${top.count} test sample(s) labeled "${top.actual}" were misclassified as "${top.predicted}" (contributing to ${totalMisclassified} total class errors). Overlapping keywords or under-represented examples usually drive this boundary error.`,
      rectification: `Add 5–10 contrasting examples in Dataset Manager that highlight the lexical difference between "${top.actual}" and "${top.predicted}", re-run validation & split, then re-train your adapter.`,
      recommendedAction: {
        label: 'Inspect & Enrich Dataset',
        targetSection: 'datasets',
      },
    });
  }

  // 3. Invalid or empty outputs
  if (evalRun.invalidOutputCount > 0) {
    diagnostics.push({
      id: 'invalid-outputs',
      severity: 'critical',
      title: `${evalRun.invalidOutputCount} Invalid or Empty Model Output(s)`,
      explanation: `The model returned empty or malformed responses on ${evalRun.invalidOutputCount} of ${total} held-out samples, which breaks downstream JSON or choice parsing.`,
      rectification:
        'Verify dataset rows do not contain blank targets in Dataset Manager, enforce allowed candidate choices in Model Playground, and lower sampling temperature to 0.1–0.2.',
      recommendedAction: {
        label: 'Verify in Playground',
        targetSection: 'playground',
      },
    });
  }

  // 4. High perplexity / calibration uncertainty
  if (evalRun.perplexity > 3.5 || evalRun.avgLoss > 1.25) {
    diagnostics.push({
      id: 'high-perplexity',
      severity: evalRun.perplexity > 8 ? 'critical' : 'warning',
      title: `High Perplexity (${evalRun.perplexity}) / Uncalibrated Logits`,
      explanation: `Average cross-entropy loss is ${evalRun.avgLoss.toFixed(3)} (PPL ${evalRun.perplexity}), showing low probability margin on the true label—typically caused by under-training, too high dropout, or noisy labels.`,
      rectification:
        'Increase LoRA rank (r=16, alpha=32), switch scheduler to cosine decay with 10% warmup, lower weight decay to 0.005, and extend training epochs until validation loss plateaus.',
      recommendedAction: {
        label: 'Tune Hyperparameters',
        targetSection: 'studio',
      },
    });
  }

  // 5. Precision vs Recall gap
  if (
    evalRun.taskType === 'classification' &&
    evalRun.precision !== null &&
    evalRun.recall !== null &&
    Math.abs(evalRun.precision - evalRun.recall) > 0.12
  ) {
    const gap = (Math.abs(evalRun.precision - evalRun.recall) * 100).toFixed(1);
    const skew = evalRun.precision > evalRun.recall ? 'high false negatives' : 'high false positives';
    diagnostics.push({
      id: 'precision-recall-skew',
      severity: 'warning',
      title: `Precision / Recall Imbalance (${gap}% gap · ${skew})`,
      explanation: `Macro Precision (${(evalRun.precision * 100).toFixed(1)}%) and Recall (${(evalRun.recall * 100).toFixed(1)}%) diverge across classes, indicating class imbalance in the training split.`,
      rectification:
        'Balance class row counts in Dataset Manager before re-running stratified Train/Val/Test splitting so minority intents receive equal gradient updates.',
      recommendedAction: {
        label: 'Rebalance Dataset Splits',
        targetSection: 'datasets',
      },
    });
  }

  // 6. Small held-out test set notice
  if (evalRun.testSampleCount < 6) {
    diagnostics.push({
      id: 'small-test-split',
      severity: 'info',
      title: `Small Held-Out Test Split (${evalRun.testSampleCount} samples)`,
      explanation: `With only ${evalRun.testSampleCount} held-out examples, each single prediction swings accuracy by ${(100 / evalRun.testSampleCount).toFixed(1)}%, making benchmarks noisy.`,
      rectification:
        'Expand the dataset to 24+ diverse examples or adjust the test split ratio to 0.20 in Dataset Manager for statistically stable evaluation.',
      recommendedAction: {
        label: 'Adjust Splits in Datasets',
        targetSection: 'datasets',
      },
    });
  }

  // 7. Healthy confirmation if zero issues flagged
  if (diagnostics.length === 0) {
    diagnostics.push({
      id: 'healthy-eval',
      severity: 'info',
      title: 'Strong Held-Out Generalization & Calibration',
      explanation: `${evalRun.modelName} achieved ${(primaryScore * 100).toFixed(1)}% with low cross-entropy loss (${evalRun.avgLoss.toFixed(3)}) and zero malformed responses.`,
      rectification:
        'Promote this model in Model Comparison against alternative architectures or export the adapter bundle from Model Registry.',
      recommendedAction: {
        label: 'Compare in Model Comparison',
        targetSection: 'comparison',
      },
    });
  }

  const hasCritical = diagnostics.some((d) => d.severity === 'critical');
  const hasWarning = diagnostics.some((d) => d.severity === 'warning');
  const overallHealth: EvaluationDiagnosisReport['overallHealth'] = hasCritical
    ? 'critical'
    : hasWarning
      ? 'warning'
      : 'healthy';

  const summary =
    mismatchCount === 0
      ? `All ${evalRun.testSampleCount} held-out samples passed with ${(primaryScore * 100).toFixed(1)}% score and perplexity ${evalRun.perplexity}.`
      : `${mismatchCount} of ${evalRun.testSampleCount} held-out sample(s) had errors (${(errorRate * 100).toFixed(1)}% error rate). Follow the rectification steps below to improve accuracy and lower perplexity.`;

  return {
    overallHealth,
    summary,
    errorRate,
    mismatchCount,
    topConfusions: topConfusions.slice(0, 5),
    hardestSamples,
    diagnostics,
  };
}
