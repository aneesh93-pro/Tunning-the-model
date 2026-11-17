import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { getGpuWorkerHealth } from './gpuWorkerClient.ts';

test('getGpuWorkerHealth returns unconfigured when GPU_WORKER_URL is unset', async () => {
  const prev = process.env.GPU_WORKER_URL;
  delete process.env.GPU_WORKER_URL;
  try {
    const status = await getGpuWorkerHealth();
    assert.equal(status.configured, false);
    assert.equal(status.reachable, false);
    assert.equal(status.status, 'unconfigured');
  } finally {
    if (prev !== undefined) process.env.GPU_WORKER_URL = prev;
  }
});

test('getGpuWorkerHealth queries GPU_WORKER_URL/health from service binding', async () => {
  const server = http.createServer((req, res) => {
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', cuda: true, device: 'NVIDIA L4' }));
      return;
    }
    res.writeHead(404);
    res.end();
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address();
  assert.ok(addr && typeof addr === 'object');

  const prev = process.env.GPU_WORKER_URL;
  process.env.GPU_WORKER_URL = `http://127.0.0.1:${addr.port}`;
  try {
    const status = await getGpuWorkerHealth();
    assert.equal(status.configured, true);
    assert.equal(status.reachable, true);
    assert.equal(status.status, 'ok');
    assert.equal(status.cuda, true);
    assert.equal(status.device, 'NVIDIA L4');
  } finally {
    if (prev !== undefined) process.env.GPU_WORKER_URL = prev;
    else delete process.env.GPU_WORKER_URL;
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
