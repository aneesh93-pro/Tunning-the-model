import { useEffect, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { LiquidButton } from '@/components/ui/liquid-glass-button';
import { BlurText } from './reactbits/BlurText';
import { ShinyText } from './reactbits/ShinyText';

interface WelcomeIntroProps {
  theme: 'dark' | 'light';
  onComplete?: () => void;
}

function shouldSkipIntroOnLoad(): boolean {
  if (typeof window === 'undefined') return true;
  const params = new URLSearchParams(window.location.search);
  if (params.get('autostart') === '1') return true;
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return true;
  return false;
}

export function WelcomeIntro({ theme, onComplete }: WelcomeIntroProps) {
  const [visible, setVisible] = useState<boolean>(() => !shouldSkipIntroOnLoad());
  const isDark = theme === 'dark';

  const dismiss = () => {
    setVisible(false);
    onComplete?.();
  };

  useEffect(() => {
    if (!visible) return undefined;

    const timer = window.setTimeout(() => {
      dismiss();
    }, 2800);

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') {
        dismiss();
      }
    };
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [visible]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          role="dialog"
          aria-modal="true"
          aria-label="MiniMind AI Welcome Intro"
          data-testid="welcome-intro"
          initial={{ y: 0, opacity: 1 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{
            y: '-100%',
            transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] },
          }}
          className={`fixed inset-0 z-[100] flex flex-col items-center justify-center overflow-hidden px-6 ${
            isDark ? 'bg-[#07090F] text-slate-50' : 'bg-[#F8FAFC] text-slate-900'
          }`}
        >
          {/* Ambient radial glow backdrop (compositor transform + opacity only) */}
          <motion.div
            aria-hidden="true"
            initial={{ opacity: 0, scale: 0.72 }}
            animate={{ opacity: 1, scale: 1.05 }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
            className="pointer-events-none absolute inset-0 flex items-center justify-center"
          >
            <div
              className={`h-[440px] w-[440px] sm:h-[600px] sm:w-[600px] rounded-full blur-3xl ${
                isDark
                  ? 'bg-[radial-gradient(circle,rgba(59,130,246,0.28)_0%,rgba(6,182,212,0.14)_42%,transparent_72%)]'
                  : 'bg-[radial-gradient(circle,rgba(37,99,235,0.18)_0%,rgba(14,165,233,0.10)_44%,transparent_72%)]'
              }`}
            />
          </motion.div>

          {/* Animated geometric neural constellation SVG paths */}
          <svg
            aria-hidden="true"
            viewBox="0 0 600 320"
            className="pointer-events-none absolute w-[520px] sm:w-[700px] max-w-[92vw] opacity-40"
          >
            <motion.path
              d="M60 160 C160 60, 240 260, 300 160 C360 60, 440 260, 540 160"
              fill="none"
              stroke={isDark ? '#3B82F6' : '#2563EB'}
              strokeWidth="1.25"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.7 }}
              transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1] }}
            />
            <motion.path
              d="M60 160 C160 260, 240 60, 300 160 C360 260, 440 60, 540 160"
              fill="none"
              stroke={isDark ? '#06B6D4' : '#0284C7'}
              strokeWidth="1.25"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.6 }}
              transition={{ duration: 1.5, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
            />
          </svg>

          {/* Animated orbital frame & center emblem */}
          <div className="relative z-10 flex flex-col items-center text-center max-w-xl">
            <div className="relative mb-6 flex items-center justify-center">
              {/* Outer rotating hairline ring */}
              <motion.div
                aria-hidden="true"
                initial={{ opacity: 0, scale: 0.5, rotate: -45 }}
                animate={{ opacity: 1, scale: 1, rotate: 90 }}
                transition={{ duration: 1.8, ease: [0.16, 1, 0.3, 1] }}
                className={`absolute h-28 w-28 sm:h-32 sm:w-32 rounded-full border ${
                  isDark ? 'border-blue-400/35' : 'border-blue-600/25'
                }`}
              />
              {/* Inner counter-rotating dashed arc */}
              <motion.div
                aria-hidden="true"
                initial={{ opacity: 0, scale: 0.7, rotate: 30 }}
                animate={{ opacity: 0.9, scale: 1, rotate: -60 }}
                transition={{ duration: 2.0, ease: [0.16, 1, 0.3, 1] }}
                className={`absolute h-22 w-22 sm:h-24 sm:w-24 rounded-full border border-dashed ${
                  isDark ? 'border-cyan-400/45' : 'border-cyan-600/35'
                }`}
              />
              {/* Emblem icon tile */}
              <motion.div
                initial={{ opacity: 0, scale: 0.65, y: 14 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ type: 'spring', stiffness: 180, damping: 18 }}
                className={`relative flex h-16 w-16 sm:h-18 sm:w-18 items-center justify-center rounded-2xl border shadow-xl ${
                  isDark
                    ? 'border-white/[0.12] bg-[#11141C]'
                    : 'border-slate-200 bg-white'
                }`}
              >
                <img
                  src="/minimind-favicon.svg"
                  alt="MiniMind AI"
                  className="h-10 w-10 sm:h-11 sm:w-11 object-contain"
                />
              </motion.div>
            </div>

            {/* Hero title entrance */}
            <h1 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-balance">
              <BlurText
                text="MiniMind AI"
                animateBy="letters"
                effect="zoom"
                direction="bottom"
                delay={42}
                stepDuration={0.34}
                blur={8}
                distance={24}
                trigger="load"
                color={isDark ? '#F8FAFC' : '#0F172A'}
              />
            </h1>

            {/* Subtitle glide */}
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35, duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
              className="mt-2.5 text-sm sm:text-base font-medium"
            >
              <ShinyText
                text="Custom LLM & SLM Fine-Tuning Studio"
                trigger="loop"
                speed={2.4}
                delay={0.2}
                color={isDark ? '#60A5FA' : '#2563EB'}
                shineColor={isDark ? '#BAE6FD' : '#60A5FA'}
              />
            </motion.p>

            {/* Clean editorial capability strip (zero pills, typographic separators) */}
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.55, duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
              className={`mt-4 flex flex-wrap items-center justify-center gap-x-2.5 gap-y-1 text-xs ${
                isDark ? 'text-slate-400' : 'text-slate-600'
              }`}
            >
              <span>Dataset Splits &amp; Export</span>
              <span aria-hidden="true">·</span>
              <span>ChatGPT, Claude, Gemini &amp; DeepSeek</span>
              <span aria-hidden="true">·</span>
              <span>LoRA &amp; QLoRA Training</span>
            </motion.div>

            {/* Smooth scaleX progress bar (compositor-only) */}
            <div
              aria-hidden="true"
              className={`mt-7 h-0.5 w-48 sm:w-56 overflow-hidden rounded-full ${
                isDark ? 'bg-white/[0.08]' : 'bg-slate-200'
              }`}
            >
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 2.6, ease: [0.22, 1, 0.36, 1] }}
                className="h-full w-full origin-left bg-gradient-to-r from-blue-500 via-cyan-400 to-blue-600"
              />
            </div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.3 }}
              className="mt-6"
            >
              <LiquidButton
                type="button"
                variant="default"
                size="sm"
                onClick={dismiss}
              >
                <span>Enter Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </LiquidButton>
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
