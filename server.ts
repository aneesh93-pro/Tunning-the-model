import 'dotenv/config';
import express from 'express';
import http from 'node:http';
import path from 'node:path';
import { createServer as createViteServer } from 'vite';
import { createApiApp } from './server/index.ts';

function setStaticHeaders(res: express.Response, filePath: string) {
  if (filePath.endsWith('.glb')) {
    res.setHeader('Content-Type', 'model/gltf-binary');
  } else if (filePath.endsWith('.gltf')) {
    res.setHeader('Content-Type', 'model/gltf+json');
  }
}

async function startServer() {
  const app = createApiApp();
  const httpServer = http.createServer(app);
  const PORT = Number(process.env.PORT || 3000);

  if (process.env.NODE_ENV !== 'production') {
    const publicDir = path.resolve(process.cwd(), 'public');
    app.use(express.static(publicDir, { setHeaders: setStaticHeaders }));
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR === 'true' ? false : { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir, { setHeaders: setStaticHeaders }));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`MiniMind AI server listening on http://localhost:${PORT}`);
  });
}

startServer();
