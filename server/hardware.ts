import os from 'node:os';
import { execSync } from 'node:child_process';
import type { HardwareTelemetry } from '../src/shared/types.ts';

let cachedGpuCheck: {
  checkedAt: number;
  gpuAvailable: boolean;
  gpuName: string | null;
  gpuMemoryTotalGb: number | null;
  gpuMemoryUsedGb: number | null;
  gpuUtilizationPct: number | null;
  gpuTemperatureC: number | null;
} | null = null;

export function getHardwareTelemetry(): HardwareTelemetry {
  const cpus = os.cpus();
  const totalMemoryGb = Number((os.totalmem() / 1024 ** 3).toFixed(2));
  const freeMemoryGb = Number((os.freemem() / 1024 ** 3).toFixed(2));
  const processMemoryMb = Number((process.memoryUsage().rss / 1024 ** 2).toFixed(1));

  const now = Date.now();
  if (!cachedGpuCheck || now - cachedGpuCheck.checkedAt > 5000) {
    let gpuAvailable = false;
    let gpuName: string | null = null;
    let gpuMemoryTotalGb: number | null = null;
    let gpuMemoryUsedGb: number | null = null;
    let gpuUtilizationPct: number | null = null;
    let gpuTemperatureC: number | null = null;

    try {
      const out = execSync(
        'nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits',
        { encoding: 'utf-8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 1000 }
      ).trim();
      if (out) {
        const [name, memTotal, memUsed, util, temp] = out.split(',').map((s) => s.trim());
        gpuAvailable = true;
        gpuName = name || 'NVIDIA GPU';
        gpuMemoryTotalGb = Number((Number(memTotal) / 1024).toFixed(2));
        gpuMemoryUsedGb = Number((Number(memUsed) / 1024).toFixed(2));
        gpuUtilizationPct = Number(util);
        gpuTemperatureC = Number(temp);
      }
    } catch {
      gpuAvailable = false;
    }

    cachedGpuCheck = {
      checkedAt: now,
      gpuAvailable,
      gpuName,
      gpuMemoryTotalGb,
      gpuMemoryUsedGb,
      gpuUtilizationPct,
      gpuTemperatureC,
    };
  }

  return {
    platform: `${os.type()} ${os.arch()}`,
    cpuModel: cpus[0]?.model || 'x86_64 CPU',
    cpuCores: cpus.length,
    totalMemoryGb,
    freeMemoryGb,
    processMemoryMb,
    gpuAvailable: cachedGpuCheck.gpuAvailable,
    gpuName: cachedGpuCheck.gpuName,
    gpuMemoryTotalGb: cachedGpuCheck.gpuMemoryTotalGb,
    gpuMemoryUsedGb: cachedGpuCheck.gpuMemoryUsedGb,
    gpuUtilizationPct: cachedGpuCheck.gpuUtilizationPct,
    gpuTemperatureC: cachedGpuCheck.gpuTemperatureC,
    executionBackend: cachedGpuCheck.gpuAvailable ? 'cuda-gpu' : 'cpu-ts-lora-engine',
    note: cachedGpuCheck.gpuAvailable
      ? 'CUDA GPU telemetry active.'
      : 'GPU hardware unavailable in this container. Executing real CPU-native LoRA/QLoRA NF4 engine; export train_peft.py or /backend for CUDA GPU nodes.',
  };
}
