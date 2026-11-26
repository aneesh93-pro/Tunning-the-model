'use client';

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

export const buttonVariants = cva(
  'inline-flex items-center cursor-pointer justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive:
          'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        cool: 'dark:inset-shadow-2xs dark:inset-shadow-white/10 bg-linear-to-t border border-b-2 border-zinc-950/40 from-primary to-primary/85 shadow-md shadow-primary/20 ring-1 ring-inset ring-white/25 transition-[filter] duration-200 hover:brightness-110 active:brightness-90 dark:border-x-0 text-primary-foreground dark:border-t-0 dark:border-primary/50 dark:ring-white/5',
        outline:
          'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
        secondary:
          'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-9 px-4 py-2',
        sm: 'h-8 rounded-md px-3 text-xs',
        lg: 'h-10 rounded-md px-8',
        icon: 'h-9 w-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export const liquidbuttonVariants = cva(
  'inline-flex items-center justify-center cursor-pointer gap-2 whitespace-nowrap rounded-xl text-sm font-medium transition-[color,background-color,box-shadow,transform] duration-200 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*="size-"])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:ring-ring/50 focus-visible:ring-[3px] backdrop-blur-xl',
  {
    variants: {
      variant: {
        default:
          'border border-white/95 bg-white/85 hover:bg-white/95 text-slate-900 shadow-[0_6px_22px_rgba(15,23,42,0.10),0_1px_4px_rgba(0,0,0,0.06)] dark:border-white/90 dark:bg-white/[0.84] dark:hover:bg-white/[0.96] dark:text-slate-950 dark:shadow-[0_6px_24px_rgba(255,255,255,0.18),0_2px_8px_rgba(0,0,0,0.35)] hover:scale-[1.03] active:scale-[0.98]',
        white:
          'border border-white/95 bg-white/85 hover:bg-white/95 text-slate-900 shadow-[0_6px_22px_rgba(15,23,42,0.10),0_1px_4px_rgba(0,0,0,0.06)] dark:border-white/90 dark:bg-white/[0.84] dark:hover:bg-white/[0.96] dark:text-slate-950 dark:shadow-[0_6px_24px_rgba(255,255,255,0.18),0_2px_8px_rgba(0,0,0,0.35)] hover:scale-[1.03] active:scale-[0.98]',
        outline:
          'border border-slate-300/85 bg-white/55 hover:bg-white/80 text-slate-900 shadow-[0_4px_16px_rgba(15,23,42,0.06)] dark:border-white/45 dark:bg-white/[0.16] dark:hover:bg-white/[0.28] dark:text-white dark:shadow-[0_4px_18px_rgba(0,0,0,0.28)] hover:scale-[1.02] active:scale-[0.98]',
        destructive:
          'border border-red-400/60 bg-red-500/85 text-white hover:bg-red-500 focus-visible:ring-destructive/20',
        secondary:
          'border border-white/70 bg-white/65 hover:bg-white/85 text-slate-800 dark:border-white/30 dark:bg-white/[0.22] dark:hover:bg-white/[0.34] dark:text-white',
        ghost:
          'bg-white/25 hover:bg-white/50 text-slate-800 dark:bg-white/[0.08] dark:hover:bg-white/[0.18] dark:text-white',
        link: 'text-slate-900 dark:text-white underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-9 px-4 py-2 text-xs',
        sm: 'h-8 text-xs gap-1.5 px-3',
        lg: 'h-10 rounded-xl px-5 text-xs',
        xl: 'h-11 rounded-xl px-7 text-sm',
        xxl: 'h-14 rounded-xl px-10 text-sm',
        icon: 'size-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export function GlassFilter({ filterId = 'container-glass' }: { filterId?: string }) {
  return (
    <svg className="hidden" aria-hidden="true">
      <defs>
        <filter
          id={filterId}
          x="0%"
          y="0%"
          width="100%"
          height="100%"
          colorInterpolationFilters="sRGB"
        >
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.05 0.05"
            numOctaves="1"
            seed="1"
            result="turbulence"
          />
          <feGaussianBlur in="turbulence" stdDeviation="2" result="blurredNoise" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="blurredNoise"
            scale="28"
            xChannelSelector="R"
            yChannelSelector="B"
            result="displaced"
          />
          <feGaussianBlur in="displaced" stdDeviation="2" result="finalBlur" />
          <feComposite in="finalBlur" in2="finalBlur" operator="over" />
        </filter>
      </defs>
    </svg>
  );
}

export function LiquidButton({
  className,
  variant,
  size,
  asChild = false,
  children,
  ...props
}: React.ComponentProps<'button'> &
  VariantProps<typeof liquidbuttonVariants> & {
    asChild?: boolean;
  }) {
  const Comp = asChild ? Slot : 'button';
  const rawId = React.useId();
  const filterId = `liquid-glass-${rawId.replace(/[^a-zA-Z0-9_-]/g, '')}`;

  return (
    <Comp
      data-slot="button"
      className={cn(
        'relative overflow-hidden',
        liquidbuttonVariants({ variant, size, className })
      )}
      {...props}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-0 rounded-[inherit]
          bg-gradient-to-b from-white/70 via-white/15 to-transparent
          shadow-[inset_0_1px_1px_rgba(255,255,255,0.95),inset_0_-2px_6px_rgba(148,163,184,0.22),inset_2px_2px_4px_rgba(255,255,255,0.45),inset_-2px_-2px_4px_rgba(255,255,255,0.25)]
          dark:from-white/50 dark:via-white/10 dark:to-white/5
          dark:shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),inset_0_-2px_8px_rgba(255,255,255,0.18),inset_2px_2px_4px_rgba(255,255,255,0.35)]
          transition-all"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 isolate -z-10 overflow-hidden rounded-[inherit]"
        style={{ backdropFilter: `url("#${filterId}") blur(12px)` }}
      />

      <span className="pointer-events-none relative z-10 inline-flex items-center justify-center gap-1.5 font-semibold tracking-[-0.01em]">
        {children}
      </span>
      <GlassFilter filterId={filterId} />
    </Comp>
  );
}

type ColorVariant =
  | 'default'
  | 'primary'
  | 'success'
  | 'error'
  | 'gold'
  | 'bronze';

interface MetalButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ColorVariant;
}

const colorVariants: Record<
  ColorVariant,
  {
    outer: string;
    inner: string;
    button: string;
    textColor: string;
    textShadow: string;
  }
> = {
  default: {
    outer: 'bg-gradient-to-b from-[#0F172A] to-[#94A3B8]',
    inner: 'bg-gradient-to-b from-[#E2E8F0] via-[#334155] to-[#CBD5E1]',
    button: 'bg-gradient-to-b from-[#475569] to-[#1E293B]',
    textColor: 'text-white',
    textShadow: '[text-shadow:_0_-1px_0_rgb(15_23_42_/_90%)]',
  },
  primary: {
    outer: 'bg-gradient-to-b from-[#1E3A8A] to-[#93C5FD]',
    inner: 'bg-gradient-to-b from-[#DBEAFE] via-[#1D4ED8] to-[#93C5FD]',
    button: 'bg-gradient-to-b from-[#2563EB] to-[#1E40AF]',
    textColor: 'text-white',
    textShadow: '[text-shadow:_0_-1px_0_rgb(30_58_138_/_95%)]',
  },
  success: {
    outer: 'bg-gradient-to-b from-[#064E3B] to-[#6EE7B7]',
    inner: 'bg-gradient-to-b from-[#D1FAE5] via-[#065F46] to-[#A7F3D0]',
    button: 'bg-gradient-to-b from-[#059669] to-[#047857]',
    textColor: 'text-white',
    textShadow: '[text-shadow:_0_-1px_0_rgb(6_78_59_/_95%)]',
  },
  error: {
    outer: 'bg-gradient-to-b from-[#5A0000] to-[#FCA5A5]',
    inner: 'bg-gradient-to-b from-[#FEE2E2] via-[#7F1D1D] to-[#FECACA]',
    button: 'bg-gradient-to-b from-[#DC2626] to-[#991B1B]',
    textColor: 'text-white',
    textShadow: '[text-shadow:_0_-1px_0_rgb(127_29_29_/_95%)]',
  },
  gold: {
    outer: 'bg-gradient-to-b from-[#78350F] to-[#FDE68A]',
    inner: 'bg-gradient-to-b from-[#FEF3C7] via-[#92400E] to-[#FDE68A]',
    button: 'bg-gradient-to-b from-[#D97706] to-[#92400E]',
    textColor: 'text-white',
    textShadow: '[text-shadow:_0_-1px_0_rgb(120_53_15_/_95%)]',
  },
  bronze: {
    outer: 'bg-gradient-to-b from-[#7C2D12] to-[#FDBA74]',
    inner: 'bg-gradient-to-b from-[#FFEDD5] via-[#7C2D12] to-[#FED7AA]',
    button: 'bg-gradient-to-b from-[#C2410C] to-[#7C2D12]',
    textColor: 'text-white',
    textShadow: '[text-shadow:_0_-1px_0_rgb(124_45_18_/_95%)]',
  },
};

export function metalButtonVariants(
  variant: ColorVariant = 'default',
  isPressed = false,
  isHovered = false,
  isTouchDevice = false
) {
  const colors = colorVariants[variant];
  const transitionStyle = 'all 250ms cubic-bezier(0.1, 0.4, 0.2, 1)';

  return {
    wrapperStyle: {
      ...(!isTouchDevice && isHovered && {
        transform: 'translateY(-1px)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.14)',
      }),
      ...(isPressed && {
        transform: 'translateY(1px) scale(0.985)',
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.15)',
      }),
      transition: transitionStyle,
    },
    outerStyle: {
      transition: transitionStyle,
    },
    innerStyle: {
      transition: transitionStyle,
      filter:
        isHovered && !isPressed && !isTouchDevice ? 'brightness(1.06)' : 'none',
    },
    buttonStyle: {
      transition: transitionStyle,
    },
    outerClass: colors.outer,
    innerClass: colors.inner,
    buttonClass: colors.button,
    textColor: colors.textColor,
    textShadow: colors.textShadow,
  };
}

function ShineEffect({ isPressed }: { isPressed: boolean }) {
  return (
    <div
      aria-hidden="true"
      className={cn(
        'pointer-events-none absolute inset-0 z-20 overflow-hidden transition-opacity duration-300',
        isPressed ? 'opacity-25' : 'opacity-0'
      )}
    >
      <div className="absolute inset-0 rounded-md bg-gradient-to-r from-transparent via-white to-transparent" />
    </div>
  );
}

export const MetalButton = React.forwardRef<HTMLButtonElement, MetalButtonProps>(
  ({ children, className, variant = 'default', ...props }, ref) => {
    const [isPressed, setIsPressed] = React.useState(false);
    const [isHovered, setIsHovered] = React.useState(false);
    const [isTouchDevice, setIsTouchDevice] = React.useState(false);

    React.useEffect(() => {
      setIsTouchDevice('ontouchstart' in window || navigator.maxTouchPoints > 0);
    }, []);

    const styles = metalButtonVariants(variant, isPressed, isHovered, isTouchDevice);

    return (
      <div
        className={cn(
          'relative inline-flex transform-gpu rounded-lg p-[1px] shadow-xs',
          styles.outerClass
        )}
        style={styles.wrapperStyle}
      >
        <div
          aria-hidden="true"
          className={cn(
            'pointer-events-none absolute inset-[1px] transform-gpu rounded-[7px]',
            styles.innerClass
          )}
          style={styles.innerStyle}
        />
        <button
          ref={ref}
          {...props}
          className={cn(
            'relative z-10 m-[1px] inline-flex h-9 cursor-pointer items-center justify-center gap-1.5 overflow-hidden rounded-[6px] px-4 py-2 text-xs font-semibold leading-none transform-gpu outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap',
            styles.buttonClass,
            styles.textColor,
            styles.textShadow,
            className
          )}
          style={styles.buttonStyle}
          onMouseDown={() => setIsPressed(true)}
          onMouseUp={() => setIsPressed(false)}
          onMouseLeave={() => {
            setIsPressed(false);
            setIsHovered(false);
          }}
          onMouseEnter={() => {
            if (!isTouchDevice) setIsHovered(true);
          }}
          onTouchStart={() => setIsPressed(true)}
          onTouchEnd={() => setIsPressed(false)}
          onTouchCancel={() => setIsPressed(false)}
        >
          <ShineEffect isPressed={isPressed} />
          {children}
          {isHovered && !isPressed && !isTouchDevice && (
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 rounded-[6px] bg-gradient-to-t from-transparent to-white/15"
            />
          )}
        </button>
      </div>
    );
  }
);
MetalButton.displayName = 'MetalButton';
