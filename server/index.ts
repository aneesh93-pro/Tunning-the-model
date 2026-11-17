import 'dotenv/config';
import express from 'express';
import { createApiRouter } from './routes.ts';

export function createApiApp() {
  const app = express();
  app.use(express.json({ limit: '12mb' }));
  app.use('/api', createApiRouter());
  return app;
}

const app = createApiApp();

export default app;
