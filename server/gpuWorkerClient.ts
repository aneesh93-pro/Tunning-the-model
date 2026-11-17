export interface GpuWorkerHealth {
  configured: boolean;
  reachable: boolean;
  status: string;
  cuda: boolean;
  device: string | null;
  error?: string;
}

let cachedWorkerHealth: { checkedAt: number; value: GpuWorkerHealth } | null = null;

export async function getGpuWorkerHealth(): Promise<GpuWorkerHealth> {
  const baseUrl = process.env.GPU_WORKER_URL?.trim();
  if (!baseUrl) {
    return {
      configured: false,
      reachable: false,
      status: 'unconfigured',
      cuda: false,
      device: null,
    };
  }

  const now = Date.now();
  if (cachedWorkerHealth && now - cachedWorkerHealth.checkedAt < 10_000) {
    return cachedWorkerHealth.value;
  }

  try {
    const target = new URL('health', baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`);
    const res = await fetch(target, {
      method: 'GET',
      signal: AbortSignal.timeout(1500),
    });
    if (!res.ok) {
      const value: GpuWorkerHealth = {
        configured: true,
        reachable: false,
        status: `http_${res.status}`,
        cuda: false,
        device: null,
        error: `GPU worker returned HTTP ${res.status}`,
      };
      cachedWorkerHealth = { checkedAt: now, value };
      return value;
    }

    const body = (await res.json()) as { status?: string; cuda?: boolean; device?: string };
    const value: GpuWorkerHealth = {
      configured: true,
      reachable: true,
      status: body.status ?? 'ok',
      cuda: Boolean(body.cuda),
      device: body.device ?? null,
    };
    cachedWorkerHealth = { checkedAt: now, value };
    return value;
  } catch (err) {
    const value: GpuWorkerHealth = {
      configured: true,
      reachable: false,
      status: 'unreachable',
      cuda: false,
      device: null,
      error: (err as Error).message,
    };
    cachedWorkerHealth = { checkedAt: now, value };
    return value;
  }
}
