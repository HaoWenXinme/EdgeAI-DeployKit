import type { ArtifactItem, DashboardData, HealthResponse, Job, MatrixRow, ModelItem } from './types';

export const fallbackHealth: HealthResponse = {
  project_root: '/root/edge-ai-deploy-kit',
  outputs_dir: 'outputs',
  reports_dir: 'reports',
  checks: [
    { name: 'Python runtime', command: 'python', available: false, detail: 'waiting for backend', required: true, category: 'core' },
    { name: 'edgeai module', command: 'python -m edgeai.cli', available: false, detail: 'waiting for backend', required: true, category: 'core' },
    { name: 'Node.js', command: 'node', available: false, required: true, category: 'core' },
    { name: 'cmake', command: 'cmake', available: false, required: false, category: 'native-build' },
    { name: 'gcc', command: 'gcc', available: false, required: false, category: 'native-build' },
    { name: 'qemu-system-aarch64', command: 'qemu-system-aarch64', available: false, required: false, category: 'board' },
    { name: 'atc', command: 'atc', available: false, required: false, category: 'board' },
  ],
};

export const fallbackModels: ModelItem[] = [
  { name: 'mnist', path: 'models/zoo/mnist/model.onnx', type: 'mnist', size_mb: 0.28, source: 'zoo' },
  { name: 'mobilenetv2', path: 'models/zoo/mobilenetv2/model_opset11.onnx', type: 'mobilenetv2', size_mb: 13.6, source: 'zoo' },
  { name: 'resnet18', path: 'models/zoo/resnet18/model.onnx', type: 'resnet18', size_mb: 44.7, source: 'zoo' },
  { name: 'yolov5n_opset11', path: 'models/zoo/yolov5n_opset11/model.onnx', type: 'yolov5n', size_mb: 7.2, source: 'zoo' },
];

export const fallbackMatrix: MatrixRow[] = [
  { model: 'mnist', model_type: 'mnist', benchmark: 'PASS', package: 'PASS', board_run: 'PASS', om_convert: 'PASS', board_latency_ms: 1.8, predict_label: 'digit' },
  { model: 'mobilenetv2', model_type: 'classification', benchmark: 'PASS', package: 'PASS', board_run: 'NOT_RUN', om_convert: 'NOT_RUN', board_latency_ms: null, predict_label: 'cat' },
  { model: 'yolov5n_opset11', model_type: 'detection', benchmark: 'PASS', package: 'PASS', board_run: 'FAIL', om_convert: 'PASS', board_latency_ms: 31.2, predict_label: 'object' },
];

export const fallbackArtifacts: ArtifactItem[] = [
  { name: 'matrix.json', path: 'outputs/model_matrix/matrix.json', kind: 'matrix', size_mb: 0.01, modified_at: '-' },
  { name: 'edgeai_report.html', path: 'reports/edgeai_report.html', kind: 'report', size_mb: 0.12, modified_at: '-' },
];

export const fallbackJobs: Job[] = [
  { id: 'demo', action: 'check', status: 'queued', command: ['edgeai', 'check', '--model', 'models/zoo/mnist/model.onnx'], created_at: '-' },
];

export const fallbackDashboard: DashboardData = {
  health: fallbackHealth,
  models: fallbackModels,
  matrix: fallbackMatrix,
  artifacts: fallbackArtifacts,
  jobs: fallbackJobs,
};
