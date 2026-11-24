import type { OrbState } from 'thinking-orbs';
import type { ExperimentRecord } from '../shared/types';

/** What the AI model / runtime is doing. Screens speak in these states. */
export const AI_ORB_STATES = [
  'idle',
  'listening',
  'thinking',
  'reasoning',
  'tool-call',
  'tool-result',
  'streaming',
  'waiting-approval',
  'error',
  'done',
] as const;

export type AiOrbState = (typeof AI_ORB_STATES)[number];

/** Semantic tint token for the orb canvas. */
export type AiOrbTone = 'muted' | 'primary' | 'attention' | 'critical';

export interface OrbVisual {
  orb: OrbState;
  tone: AiOrbTone;
  /** Multiplier on the library's tuned speed. */
  speed: number;
  /** Freeze frame instead of animating continuously. */
  paused: boolean;
  /** Resting / settled state rather than active computation. */
  resting: boolean;
}

export const ORB_VISUAL_TABLE: Record<AiOrbState, OrbVisual> = {
  idle: {
    orb: 'breathing',
    tone: 'muted',
    speed: 0.45,
    paused: false,
    resting: true,
  },
  listening: {
    orb: 'listening',
    tone: 'primary',
    speed: 1.0,
    paused: false,
    resting: false,
  },
  thinking: {
    orb: 'working',
    tone: 'primary',
    speed: 1.05,
    paused: false,
    resting: false,
  },
  reasoning: {
    orb: 'solving',
    tone: 'primary',
    speed: 1.15,
    paused: false,
    resting: false,
  },
  'tool-call': {
    orb: 'connecting',
    tone: 'primary',
    speed: 1.1,
    paused: false,
    resting: false,
  },
  'tool-result': {
    orb: 'searching',
    tone: 'primary',
    speed: 0.95,
    paused: false,
    resting: false,
  },
  streaming: {
    orb: 'composing',
    tone: 'primary',
    speed: 1.25,
    paused: false,
    resting: false,
  },
  'waiting-approval': {
    orb: 'weaving',
    tone: 'attention',
    speed: 0.72,
    paused: false,
    resting: false,
  },
  error: {
    orb: 'shaping',
    tone: 'critical',
    speed: 0.75,
    paused: false,
    resting: false,
  },
  done: {
    orb: 'breathing',
    tone: 'primary',
    speed: 0.55,
    paused: false,
    resting: true,
  },
};

export const AI_ORB_LABELS: Record<AiOrbState, string> = {
  idle: 'Ready',
  listening: 'Listening',
  thinking: 'Thinking',
  reasoning: 'Reasoning',
  'tool-call': 'Running suite',
  'tool-result': 'Processing result',
  streaming: 'Generating',
  'waiting-approval': 'Paused',
  error: 'Error',
  done: 'Complete',
};

export interface AgentActivity {
  error?: boolean;
  waitingApproval?: boolean;
  streaming?: boolean;
  toolCalling?: boolean;
  toolResult?: boolean;
  reasoning?: boolean;
  thinking?: boolean;
  listening?: boolean;
  done?: boolean;
}

/**
 * Resolve highest-priority runtime activity flags into a single semantic AiOrbState.
 * Priority: error > waiting-approval > tool-call > streaming > reasoning > thinking > tool-result > listening > done > idle
 */
export function orbStateFor(activity: AgentActivity): AiOrbState {
  if (activity.error) return 'error';
  if (activity.waitingApproval) return 'waiting-approval';
  if (activity.toolCalling) return 'tool-call';
  if (activity.streaming) return 'streaming';
  if (activity.reasoning) return 'reasoning';
  if (activity.thinking) return 'thinking';
  if (activity.toolResult) return 'tool-result';
  if (activity.listening) return 'listening';
  if (activity.done) return 'done';
  return 'idle';
}

export function mapExperimentStatusToOrbState(
  status?: ExperimentRecord['status'] | null
): AiOrbState {
  switch (status) {
    case 'preparing':
      return 'thinking';
    case 'running':
      return 'reasoning';
    case 'paused':
      return 'waiting-approval';
    case 'completed':
      return 'done';
    case 'failed':
    case 'cancelled':
      return 'error';
    default:
      return 'idle';
  }
}

export function mapPlaygroundToOrbState(params: {
  error?: string | null;
  lastMessageIsError?: boolean;
  testingModel?: boolean;
  testingAllModels?: boolean;
  streaming?: boolean;
  hasDraftInput?: boolean;
  hasMessages?: boolean;
}): AiOrbState {
  return orbStateFor({
    error: Boolean(params.error || params.lastMessageIsError),
    toolCalling: Boolean(params.testingModel || params.testingAllModels),
    streaming: Boolean(params.streaming),
    listening: Boolean(params.hasDraftInput),
    done: Boolean(params.hasMessages && !params.streaming && !params.hasDraftInput),
  });
}

export function mapJobRunnerToOrbState(params: {
  error?: string | null;
  running?: boolean;
  hasResult?: boolean;
}): AiOrbState {
  return orbStateFor({
    error: Boolean(params.error),
    toolCalling: Boolean(params.running),
    done: Boolean(params.hasResult && !params.running),
  });
}
