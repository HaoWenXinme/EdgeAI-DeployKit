import { Metric } from './ui';
import { countPass } from '@/lib/format';
import type { ArtifactItem, HealthResponse, MatrixRow, ModelItem } from '@/lib/types';

export function KpiGrid({ models, matrix, artifacts, health }: { models: ModelItem[]; matrix: MatrixRow[]; artifacts: ArtifactItem[]; health: HealthResponse }) {
  const packages = artifacts.filter((item) => item.kind === 'package').length;
  const reports = artifacts.filter((item) => item.kind === 'report').length;
  const requiredChecks = health.checks.filter((item) => item.required !== false);
  const scoreChecks = requiredChecks.length ? requiredChecks : health.checks;
  const available = scoreChecks.filter((item) => item.available).length;

  return (
    <section className="mt-6 grid grid-cols-4 gap-4">
      <Metric label="Models" value={String(models.length)} hint="zoo / examples / inputs / outputs" tone="cyan" />
      <Metric label="Board passed" value={`${countPass(matrix, 'board_run')}/${matrix.length || 0}`} hint="Board Run PASS from matrix.json" tone="green" />
      <Metric label="Packages" value={String(packages)} hint="outputs/packages artifacts" tone="violet" />
      <Metric label="Reports" value={String(reports)} hint={`${available}/${scoreChecks.length || 0} core runtime ready`} tone="amber" />
    </section>
  );
}
