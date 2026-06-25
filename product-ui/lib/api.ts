import { fallbackArtifacts, fallbackDashboard, fallbackHealth, fallbackJobs, fallbackMatrix, fallbackModels } from './fallback-data';
import type { ArtifactItem, DashboardData, HealthResponse, InferResult, Job, JobCreateRequest, MatrixRow, ModelItem } from './types';

function defaultApiBase() {
  if (typeof window !== 'undefined') {
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:8000`;
  }
  return 'http://127.0.0.1:8000';
}

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || defaultApiBase()).replace(/\/$/, '');

async function apiGet<T>(path: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(await res.text());
    return (await res.json()) as T;
  } catch {
    return fallback;
  }
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as T;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return apiGet('/api/health', fallbackHealth);
}

export async function fetchModels(): Promise<ModelItem[]> {
  return apiGet('/api/models', fallbackModels);
}

export async function fetchMatrix(): Promise<MatrixRow[]> {
  return apiGet('/api/matrix', fallbackMatrix);
}

export async function fetchArtifacts(): Promise<ArtifactItem[]> {
  return apiGet('/api/artifacts', fallbackArtifacts);
}

export async function fetchJobs(): Promise<Job[]> {
  return apiGet('/api/jobs', fallbackJobs);
}

export async function fetchDashboard(): Promise<DashboardData> {
  return apiGet('/api/dashboard', fallbackDashboard);
}

export async function createJob(payload: JobCreateRequest): Promise<Job> {
  return apiPost<Job>('/api/jobs', payload);
}

export async function fetchJobLogs(jobId: string): Promise<string> {
  try {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/logs`, { cache: 'no-store' });
    if (!res.ok) return '';
    return await res.text();
  } catch {
    return '';
  }
}


export type UploadResult = {
  path: string;
  name: string;
};

export async function uploadModel(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/api/uploads/model`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return (await res.json()) as UploadResult;
}


export async function fetchInferResults(): Promise<InferResult[]> {
  return apiGet('/api/infer-results', []);
}

export async function fetchInferResult(modelName: string): Promise<InferResult | null> {
  return apiGet(`/api/infer-result/${encodeURIComponent(modelName)}`, null);
}

export function fileUrl(path?: string | null, version?: string | number | null): string {
  if (!path) return '';
  const cleaned = path.replace(/^\/+/, '');
  const encoded = cleaned.split('/').map((part) => encodeURIComponent(part)).join('/');
  const suffix = version ? `?v=${encodeURIComponent(String(version))}` : '';
  return `${API_BASE}/api/files/${encoded}${suffix}`;
}

export async function uploadInputImage(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/api/uploads/image`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return (await res.json()) as UploadResult;
}

// ---- Local DeepSeek / Ollama assistant ----
export type AssistantMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type AssistantChatResponse = {
  model: string;
  content: string;
  usage?: unknown;
};

export async function chatWithAssistant(
  messages: AssistantMessage[],
): Promise<AssistantChatResponse> {
  const res = await fetch(`${API_BASE}/api/assistant/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ messages }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Assistant request failed: ${res.status}`);
  }

  return (await res.json()) as AssistantChatResponse;
}

export type LocalReportItem = {
  model_name: string;
  report_path: string;
  pdf_path?: string | null;
  has_pdf?: boolean;
  pdf_size_bytes?: number | null;
  pdf_modified_time?: number | null;
  source?: string;
  size_bytes?: number;
  modified_time?: number;
  modified_at?: string;
};

export async function fetchLocalReports(): Promise<LocalReportItem[]> {
  return apiGet<LocalReportItem[]>('/api/local-reports', []);
}

export async function fetchLocalReportContent(modelName: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/local-reports/${encodeURIComponent(modelName)}`, {
    cache: 'no-store',
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return await res.text();
}

export async function uploadInput(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/uploads/input`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return (await res.json()) as UploadResult;
}

export type ConvertRequirementsResponse = {
  ready?: boolean;
  framework?: string;
  detected_source_kind?: string;
  missing_params?: string[];
  suggested_params?: Record<string, unknown>;
  questions?: Array<{
    name: string;
    label?: string;
    help?: string;
    default?: string | number | boolean | null;
    options?: string[];
  }>;
  install_commands?: string[];
  warnings?: string[];
};

export async function fetchConvertRequirements(payload: Record<string, unknown>): Promise<ConvertRequirementsResponse> {
  return apiPost<ConvertRequirementsResponse>('/api/convert/requirements', payload);
}



// EDGEAI_LOCAL_INFERENCE_FLOW_V1
export type LocalInferenceFlowRequest = {
  package_name: string;
  input?: string;
  input_path?: string;
  prompt?: string;
  max_tokens?: number;
  temperature?: number;
  force_report?: boolean;
  force_analyze?: boolean;
  force_task?: boolean;
};

export type LocalInferenceFlowResponse = {
  ok?: boolean;
  message?: string;
  package_name?: string;
  package_dir?: string;
  input?: string | null;
  stages?: Array<{
    stage?: string;
    command?: string;
    code?: number;
    ok?: boolean;
    skipped?: boolean;
    output?: string;
  }>;
  artifacts?: Record<string, boolean | string | null | undefined>;
  task_result?: unknown;
};

export async function runLocalInferenceFlow(payload: LocalInferenceFlowRequest): Promise<LocalInferenceFlowResponse> {
  return apiPost<LocalInferenceFlowResponse>('/api/local-inference-flow', payload);
}
