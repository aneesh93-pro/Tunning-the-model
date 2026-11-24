import type { DatasetRecord } from '../shared/types';

export function pickPreferredDatasetId(
  datasets: Pick<DatasetRecord, 'id' | 'status'>[],
  preferredId?: string | null
): string {
  if (datasets.length === 0) return '';
  if (preferredId && datasets.some((d) => d.id === preferredId)) {
    return preferredId;
  }
  const ready = datasets.find((d) => d.status === 'training_ready');
  if (ready) return ready.id;
  return datasets[0].id;
}
