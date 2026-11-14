import type { NormalizedExample } from './datasetProcessor';

export const SPECIAL_TOKENS = [
  '<|pad|>',
  '<|unk|>',
  '<|bos|>',
  '<|eos|>',
  '<|im_start|>',
  '<|im_end|>',
] as const;

export interface SerializedTokenizer {
  vocab: Record<string, number>;
  chatTemplate: 'chatml' | 'phi3';
  maxSeqLength: number;
}

/**
 * Deterministic subword + word tokenizer with chat template formatting.
 */
export class SubwordTokenizer {
  public readonly tokenToId: Map<string, number>;
  public readonly idToToken: string[];
  public readonly chatTemplate: 'chatml' | 'phi3';
  public readonly maxSeqLength: number;

  constructor(
    vocab?: Record<string, number>,
    chatTemplate: 'chatml' | 'phi3' = 'chatml',
    maxSeqLength = 256
  ) {
    this.tokenToId = new Map<string, number>();
    this.idToToken = [];
    this.chatTemplate = chatTemplate;
    this.maxSeqLength = maxSeqLength;

    if (vocab) {
      const entries = Object.entries(vocab).sort((a, b) => a[1] - b[1]);
      for (const [tok, id] of entries) {
        this.tokenToId.set(tok, id);
        this.idToToken[id] = tok;
      }
    } else {
      for (const tok of SPECIAL_TOKENS) {
        this.addToken(tok);
      }
    }
  }

  private addToken(tok: string): number {
    const existing = this.tokenToId.get(tok);
    if (existing !== undefined) return existing;
    const id = this.idToToken.length;
    this.tokenToId.set(tok, id);
    this.idToToken.push(tok);
    return id;
  }

  /**
   * Splits text into normalized word + bigram/subword pieces.
   */
  public static preTokenize(text: string): string[] {
    const cleaned = text
      .toLowerCase()
      .replace(/([.,!?;:()[\]{}"'`/\\-])/g, ' $1 ')
      .trim();
    if (!cleaned) return [];
    const words = cleaned.split(/\s+/);
    const pieces: string[] = [];
    for (const w of words) {
      pieces.push(w);
      if (w.length >= 5) {
        pieces.push(`##${w.slice(0, 3)}`);
        pieces.push(`##${w.slice(-3)}`);
      }
    }
    return pieces;
  }

  public fit(examples: NormalizedExample[], maxVocabSize = 2048): void {
    const freq = new Map<string, number>();
    for (const ex of examples) {
      const text = `${ex.promptText} ${ex.targetText}`;
      for (const piece of SubwordTokenizer.preTokenize(text)) {
        freq.set(piece, (freq.get(piece) ?? 0) + 1);
      }
    }

    const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1]);
    for (const [tok] of sorted) {
      if (this.idToToken.length >= maxVocabSize) break;
      this.addToken(tok);
    }
  }

  public formatPrompt(example: NormalizedExample): string {
    if (this.chatTemplate === 'phi3') {
      return `<|user|>\n${example.promptText}<|end|>\n<|assistant|>\n`;
    }
    return `<|im_start|>user\n${example.promptText}<|im_end|>\n<|im_start|>assistant\n`;
  }

  public encode(text: string, maxLength = this.maxSeqLength): { inputIds: number[]; attentionMask: number[] } {
    const unkId = this.tokenToId.get('<|unk|>') ?? 1;
    const bosId = this.tokenToId.get('<|bos|>') ?? 2;
    const eosId = this.tokenToId.get('<|eos|>') ?? 3;

    const pieces = SubwordTokenizer.preTokenize(text);
    const ids = [bosId];
    for (const p of pieces) {
      if (ids.length >= maxLength - 1) break;
      ids.push(this.tokenToId.get(p) ?? unkId);
    }
    ids.push(eosId);

    const attentionMask = ids.map(() => 1);
    return { inputIds: ids, attentionMask };
  }

  public get vocabSize(): number {
    return this.idToToken.length;
  }

  public toJSON(): SerializedTokenizer {
    const vocab: Record<string, number> = {};
    for (const [k, v] of this.tokenToId.entries()) {
      vocab[k] = v;
    }
    return {
      vocab,
      chatTemplate: this.chatTemplate,
      maxSeqLength: this.maxSeqLength,
    };
  }

  public static fromJSON(data: SerializedTokenizer): SubwordTokenizer {
    return new SubwordTokenizer(data.vocab, data.chatTemplate, data.maxSeqLength);
  }
}
