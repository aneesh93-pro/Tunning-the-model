import type { OverviewResponse } from '../lib/api';

function wrapLines(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number
): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = '';

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (ctx.measureText(candidate).width <= maxWidth) {
      current = candidate;
    } else {
      if (current) lines.push(current);
      current = word;
    }
  }
  if (current) lines.push(current);
  return lines;
}

export function renderMiniMindBriefFrontDataUrl(stats?: OverviewResponse['stats']): string {
  const width = 680;
  const height = 860;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  // Archival studio paper base
  ctx.fillStyle = '#F7F5EF';
  ctx.fillRect(0, 0, width, height);

  // Subtle technical grid
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.045)';
  ctx.lineWidth = 1;
  for (let x = 32; x < width; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 32; y < height; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Outer editorial border
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.16)';
  ctx.lineWidth = 2;
  ctx.strokeRect(24, 24, width - 48, height - 48);

  // Top studio header bar
  ctx.fillStyle = '#0F172A';
  ctx.fillRect(24, 24, width - 48, 52);

  ctx.fillStyle = '#F8FAFC';
  ctx.font = '700 15px "JetBrains Mono", ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  ctx.fillText('MINIMIND AI // FIELD BRIEF', 44, 50);

  ctx.fillStyle = '#60A5FA';
  ctx.font = '600 13px "JetBrains Mono", ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.fillText('LORA & QLORA STUDIO', width - 44, 50);
  ctx.textAlign = 'left';

  // Hero Headline & Intro
  ctx.fillStyle = '#1D4ED8';
  ctx.font = '700 13px "Plus Jakarta Sans", system-ui, sans-serif';
  ctx.fillText('CUSTOM SMALL LANGUAGE MODEL WORKSPACE', 44, 108);

  ctx.fillStyle = '#0F172A';
  ctx.font = '800 32px "Plus Jakarta Sans", system-ui, sans-serif';
  ctx.fillText('MiniMind AI Platform', 44, 146);

  ctx.fillStyle = '#334155';
  ctx.font = '400 16px "DM Sans", system-ui, sans-serif';
  const leadLines = wrapLines(
    ctx,
    'An end-to-end generative AI workbench built to preprocess instruction datasets, train low-rank adapters, verify grounded outputs, and evaluate Small Language Models transparently.',
    width - 88
  );
  let cursorY = 186;
  for (const line of leadLines) {
    ctx.fillText(line, 44, cursorY);
    cursorY += 24;
  }

  // Divider
  cursorY += 12;
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.14)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(44, cursorY);
  ctx.lineTo(width - 44, cursorY);
  ctx.stroke();

  cursorY += 28;
  const sections = [
    {
      num: '01',
      title: 'Dataset Manager & Split Integrity',
      body: 'Upload CSV, JSON, JSONL, or TXT corpora, normalize schemas, deduplicate examples, and lock zero-leakage train / validation / test splits.',
      accent: '#2563EB',
    },
    {
      num: '02',
      title: 'LoRA & 4-Bit NF4 QLoRA Training',
      body: 'Customize rank (r), scaling alpha, dropout, and target projections across SmolLM2, Qwen2.5, and Phi-3.5 with live loss & perplexity charts.',
      accent: '#0284C7',
    },
    {
      num: '03',
      title: 'Grounded Playground & Anti-Hallucination',
      body: 'Constrain predictions into verified boolean, choice, and score formats to reject ungrounded probes (such as JEV AI) and export diagnostic reports.',
      accent: '#7C3AED',
    },
    {
      num: '04',
      title: 'Evaluation Center & Adapter Registry',
      body: 'Measure held-out accuracy and token F1 against frozen base models, inspect delta gains side-by-side, and download ready-to-deploy adapters.',
      accent: '#059669',
    },
  ];

  for (const item of sections) {
    // Index badge block
    ctx.fillStyle = item.accent;
    ctx.fillRect(44, cursorY - 12, 34, 24);
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '700 13px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(item.num, 61, cursorY);
    ctx.textAlign = 'left';

    ctx.fillStyle = '#0F172A';
    ctx.font = '700 18px "Plus Jakarta Sans", system-ui, sans-serif';
    ctx.fillText(item.title, 90, cursorY);

    cursorY += 24;
    ctx.fillStyle = '#475569';
    ctx.font = '400 14px "DM Sans", system-ui, sans-serif';
    const lines = wrapLines(ctx, item.body, width - 134);
    for (const line of lines) {
      ctx.fillText(line, 90, cursorY);
      cursorY += 20;
    }
    cursorY += 20;
  }

  // Footer telemetry strip (real counts)
  const footerY = height - 92;
  ctx.fillStyle = '#EAE6DC';
  ctx.fillRect(44, footerY, width - 88, 46);
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.14)';
  ctx.lineWidth = 1;
  ctx.strokeRect(44, footerY, width - 88, 46);

  const expCount = stats?.totalExperiments ?? 0;
  const dsCount = stats?.totalDatasets ?? 0;
  const mdlCount = stats?.totalPretrainedModels ?? 4;

  ctx.fillStyle = '#0F172A';
  ctx.font = '600 13px "JetBrains Mono", ui-monospace, monospace';
  ctx.fillText(
    `DATASETS: ${dsCount}  |  BASE SLMS: ${mdlCount}  |  RUNS: ${expCount}`,
    62,
    footerY + 23
  );

  ctx.fillStyle = '#2563EB';
  ctx.textAlign = 'right';
  ctx.fillText('HOLD TO CRUMPLE', width - 62, footerY + 23);
  ctx.textAlign = 'left';

  return canvas.toDataURL('image/png');
}

export function renderMiniMindBriefBackDataUrl(): string {
  const width = 680;
  const height = 860;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  // Dark slate technical blueprint reverse side
  ctx.fillStyle = '#0F172A';
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = 'rgba(59, 130, 246, 0.16)';
  ctx.lineWidth = 1;
  for (let x = 32; x < width; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 32; y < height; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(96, 165, 250, 0.45)';
  ctx.lineWidth = 2;
  ctx.strokeRect(28, 28, width - 56, height - 56);

  ctx.fillStyle = '#60A5FA';
  ctx.font = '700 13px "JetBrains Mono", ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  ctx.fillText('REVERSE SPEC // PARAMETER-EFFICIENT ARCHITECTURE', 52, 74);

  ctx.fillStyle = '#F8FAFC';
  ctx.font = '800 30px "Plus Jakarta Sans", system-ui, sans-serif';
  ctx.fillText('LoRA & NF4 Mechanics', 52, 118);

  // Equation box
  ctx.fillStyle = '#1E293B';
  ctx.fillRect(52, 160, width - 104, 112);
  ctx.strokeStyle = '#3B82F6';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(52, 160, width - 104, 112);

  ctx.fillStyle = '#93C5FD';
  ctx.font = '600 13px "JetBrains Mono", ui-monospace, monospace';
  ctx.fillText('LOW-RANK UPDATE EQUATION', 74, 188);

  ctx.fillStyle = '#FFFFFF';
  ctx.font = '700 26px "JetBrains Mono", ui-monospace, monospace';
  ctx.fillText('h = W₀x + (α / r) · B(Ax)', 74, 232);

  const notes = [
    {
      label: 'FROZEN BASE WEIGHTS (W₀)',
      desc: 'Preserves general language priors while cutting optimizer memory by >85% on consumer hardware.',
    },
    {
      label: '4-BIT NORMALFLOAT (NF4)',
      desc: 'Symmetric block-wise quantization compresses base matrices to 4 bits without collapsing perplexity.',
    },
    {
      label: 'VERIFIED STRUCTURED OUTPUT',
      desc: 'Inference gates every response through boolean, allowed-choice, and confidence score validation.',
    },
  ];

  let y = 320;
  for (const n of notes) {
    ctx.fillStyle = '#38BDF8';
    ctx.font = '700 14px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillText(n.label, 52, y);
    y += 26;
    ctx.fillStyle = '#CBD5E1';
    ctx.font = '400 15px "DM Sans", system-ui, sans-serif';
    for (const line of wrapLines(ctx, n.desc, width - 104)) {
      ctx.fillText(line, 52, y);
      y += 22;
    }
    y += 26;
  }

  ctx.fillStyle = '#64748B';
  ctx.font = '600 13px "JetBrains Mono", ui-monospace, monospace';
  ctx.fillText('MINIMIND AI · CUSTOM SLM FINE-TUNING STUDIO', 52, height - 62);

  return canvas.toDataURL('image/png');
}
