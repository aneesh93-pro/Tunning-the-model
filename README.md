# MiniMind AI — Custom LLM Fine-Tuning Studio

Full-stack Small Language Model (SLM) fine-tuning platform supporting dataset ingestion and leakage-free splitting, architecture-aware LoRA and 4-bit NF4 QLoRA training, live SSE metric streaming, held-out test evaluation, side-by-side model comparison, streaming inference playground, and reproducible adapter registry.

## Architecture

- **Frontend**: React 19 + TypeScript + Tailwind CSS + Recharts + ReactBits (`SpotlightCard`, `CountUp`, `BlurText`, `DotGrid`)
- **Live Server & Local Training Engine**: Express + `node:sqlite` (`data/minimind.db`) + CPU-native LoRA / 4-bit NF4 QLoRA backpropagation engine (`src/ml/loraEngine.ts`) + `@google/genai` streaming inference (`server/inferenceService.ts`)
- **GPU Cloud Worker Stack (`/backend`)**: FastAPI + SQLAlchemy + PostgreSQL + PyTorch + Hugging Face Transformers + PEFT + TRL + BitsAndBytes + Docker Compose (`docker-compose.yml`)

## Local Development

```bash
npm install
npm run dev
```

## Unit Tests

```bash
npx --yes tsx --test src/ml/datasetProcessor.test.ts src/ml/loraEngine.test.ts
```

## Build & Production

```bash
npm run build
npm run start
```
