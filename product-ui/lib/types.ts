export type Tone = 'neutral' | 'green' | 'cyan' | 'amber' | 'violet' | 'red';

export type HealthItem = {
  name: string;
  command?: string;
  available: boolean;
  detail?: string;
  required?: boolean;
  category?: string;
};

export type HealthResponse = {
  project_root: string;
  outputs_dir: string;
  reports_dir: string;
  checks: HealthItem[];
};

export type ModelItem = {
  name: string;
  path: string;
  type: string;
  size_mb: number;
  source: string;
  modified_at?: string;
};

export type MatrixRow = {
  model?: string;
  model_type?: string;
  onnx_check?: string;
  benchmark?: string;
  package?: string;
  board_sync?: string;
  om_convert?: string;
  board_run?: string;
  avg_latency_ms?: number | string | null;
  p50_ms?: number | string | null;
  p95_ms?: number | string | null;
  board_latency_ms?: number | string | null;
  predict?: string | number | null;
  predict_label?: string | null;
  top1?: string | number | null;
  top1_label?: string | null;
  detection_count?: number | null;
  detections?: InferDetection[] | null;
  annotated_image?: string | null;
};

export type ArtifactItem = {
  name: string;
  path: string;
  kind: 'report' | 'package' | 'benchmark' | 'matrix' | 'other';
  size_mb: number;
  modified_at: string;
};

export type JobStatus = 'queued' | 'running' | 'success' | 'failed' | 'timeout' | 'cancelled';

export type Job = {
  id: string;
  action: string;
  status: JobStatus;
  command: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  code?: number | null;
  log_path?: string | null;
  error?: string | null;
};

export type JobCreateRequest = {
  action: string;
  params?: Record<string, string | number | boolean | null | undefined>;
};

export type InferDetection = {
  class_id?: number | string | null;
  label_en?: string | null;
  label_zh?: string | null;
  confidence?: number | string | null;
  bbox?: Array<number | string> | null;
  raw?: unknown;
};

export type InferPrediction = {
  class_id?: number | string | null;
  label_en?: string | null;
  label_zh?: string | null;
  confidence?: number | string | null;
  top5?: Array<Record<string, unknown>>;
};

export type InferResult = {
  model: string;
  model_type?: string | null;
  result_type: 'classification' | 'detection' | 'unknown';
  status?: string | null;
  package_dir?: string | null;
  source_input_path?: string | null;
  input_image?: string | null;
  result_image?: string | null;
  prediction?: InferPrediction | null;
  detections?: InferDetection[];
  detection_count?: number | string | null;
  latency_ms?: number | string | null;
  device?: string | null;
  runtime?: string | null;
  updated_at?: string | null;
  artifacts?: Record<string, string | null>;
  raw?: unknown;
};

export type DashboardData = {
  health: HealthResponse;
  models: ModelItem[];
  matrix: MatrixRow[];
  artifacts: ArtifactItem[];
  jobs: Job[];
};
