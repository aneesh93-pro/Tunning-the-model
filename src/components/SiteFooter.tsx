import { useState } from 'react';
import {
  BookOpen,
  Check,
  Compass,
  Copy,
  Layers,
  ListChecks,
  Mail,
  Target,
} from 'lucide-react';
import type { NavSection } from '../views/OverviewView';

interface SiteFooterProps {
  onNavigate: (section: NavSection) => void;
}

const FOOTER_MAIL = 'mayurianeesh@gmail.com';

const QUICK_LINKS: Array<{ id: NavSection; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'datasets', label: 'Dataset Manager' },
  { id: 'models', label: 'Model Hub' },
  { id: 'studio', label: 'Fine-Tuning Studio' },
  { id: 'playground', label: 'Model Playground' },
  { id: 'evaluation', label: 'Evaluation Center' },
];

export function SiteFooter({ onNavigate }: SiteFooterProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(FOOTER_MAIL);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="w-full max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 pb-8 pt-4 space-y-6">
      {/* Above Footer: Website Information, Uses, How to Use, and Main Aim */}
      <section
        aria-label="About MiniMind AI, Uses, How to Use, and Main Aim"
        className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/95 dark:bg-[#11141C]/95 backdrop-blur-sm p-5 sm:p-6 space-y-5"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-200 dark:border-white/[0.06]">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 shrink-0">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                About MiniMind AI — Guide, Uses &amp; Mission
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Everything you need to know about what this platform does, how to use it, and its core objective
              </p>
            </div>
          </div>
          <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
            End-to-End LLM &amp; SLM Engineering Studio
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {/* 1. About the Website */}
          <article className="rounded-lg border border-slate-200/80 dark:border-white/[0.07] bg-slate-50/80 dark:bg-[#090B10]/70 p-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
              <Compass className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
              <h3>About the Website</h3>
            </div>
            <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
              <strong className="font-medium text-slate-900 dark:text-slate-100">MiniMind AI</strong> is a full-stack LLM &amp; Small Language Model Fine-Tuning Studio. It unifies dataset cleaning, multi-model testing across ChatGPT, Claude, Gemini, DeepSeek, and open-weight SLMs, CPU-native LoRA / 4-bit NF4 QLoRA training, and adapter export in one workspace.
            </p>
          </article>

          {/* 2. Uses of the Website */}
          <article className="rounded-lg border border-slate-200/80 dark:border-white/[0.07] bg-slate-50/80 dark:bg-[#090B10]/70 p-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
              <Layers className="w-4 h-4 text-cyan-600 dark:text-cyan-400 shrink-0" />
              <h3>Use of This Website</h3>
            </div>
            <ul className="space-y-1.5 text-xs leading-relaxed text-slate-600 dark:text-slate-300 list-disc pl-4">
              <li>Clean, deduplicate, and export Train / Val / Test dataset splits back as JSONL, CSV, JSON, or ZIP.</li>
              <li>Fine-tune compact SLMs (SmolLM2, Qwen2.5, Phi-3.5) with LoRA or QLoRA adapters.</li>
              <li>Benchmark and test ChatGPT, Claude, Gemini, DeepSeek, and custom adapters side by side.</li>
              <li>Audit hallucination rates with structured <span className="font-mono text-[11px]">boolean / choice / score</span> test reports.</li>
            </ul>
          </article>

          {/* 3. How to Use It */}
          <article className="rounded-lg border border-slate-200/80 dark:border-white/[0.07] bg-slate-50/80 dark:bg-[#090B10]/70 p-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
              <ListChecks className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
              <h3>How to Use It</h3>
            </div>
            <ol className="space-y-1.5 text-xs leading-relaxed text-slate-600 dark:text-slate-300 list-decimal pl-4">
              <li>
                Open{' '}
                <button
                  type="button"
                  onClick={() => onNavigate('datasets')}
                  className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Dataset Manager
                </button>{' '}
                to upload, validate, split, and export processed data.
              </li>
              <li>
                Browse{' '}
                <button
                  type="button"
                  onClick={() => onNavigate('models')}
                  className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Model Hub
                </button>{' '}
                and pick a model or test it directly in Playground.
              </li>
              <li>
                Train an adapter in{' '}
                <button
                  type="button"
                  onClick={() => onNavigate('studio')}
                  className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Fine-Tuning Studio
                </button>{' '}
                and watch live loss curves.
              </li>
              <li>
                Verify results in{' '}
                <button
                  type="button"
                  onClick={() => onNavigate('evaluation')}
                  className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Evaluation Center
                </button>{' '}
                &amp; download PEFT weights.
              </li>
            </ol>
          </article>

          {/* 4. Main Aim of the Website */}
          <article className="rounded-lg border border-slate-200/80 dark:border-white/[0.07] bg-slate-50/80 dark:bg-[#090B10]/70 p-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
              <Target className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
              <h3>Main Aim of the Website</h3>
            </div>
            <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
              The main aim of MiniMind AI is to make custom language model fine-tuning, dataset engineering, and AI model evaluation <strong className="font-medium text-slate-900 dark:text-slate-100">transparent, leakage-free, and reproducible</strong>—giving engineers, students, and researchers verifiable metrics instead of black-box guesswork.
            </p>
          </article>
        </div>
      </section>

      {/* Page Footer */}
      <footer className="rounded-xl border border-slate-200 dark:border-white/[0.08] bg-white/95 dark:bg-[#11141C]/95 backdrop-blur-sm px-5 py-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
              <span>MiniMind AI</span>
              <span className="text-slate-400">·</span>
              <span className="font-normal text-slate-600 dark:text-slate-300">
                Custom LLM Fine-Tuning Studio
              </span>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Engineered by <span className="font-medium text-slate-700 dark:text-slate-200">KANCHARLA ANEESH</span> (AI Engineer) · Leakage-Free Dataset Splits, LoRA/QLoRA &amp; Multi-Model Benchmarking
            </p>
          </div>

          <nav aria-label="Footer quick navigation" className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
            {QUICK_LINKS.map((link) => (
              <button
                key={link.id}
                type="button"
                onClick={() => onNavigate(link.id)}
                className="text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                {link.label}
              </button>
            ))}
          </nav>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <a
              href={`mailto:${FOOTER_MAIL}?subject=MiniMind%20AI%20Studio`}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-[#090B10] text-xs font-mono text-slate-800 dark:text-slate-200 hover:border-blue-500/50 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              <Mail className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400 shrink-0" />
              <span>{FOOTER_MAIL}</span>
            </a>
            <button
              type="button"
              onClick={handleCopy}
              aria-label={`Copy footer email ${FOOTER_MAIL}`}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-white/[0.1] bg-slate-50 dark:bg-white/[0.04] text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-white/[0.08] transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-500" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
