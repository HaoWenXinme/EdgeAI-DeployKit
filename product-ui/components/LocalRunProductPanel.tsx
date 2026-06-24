"use client";

import React from "react";
import { FileJson, FileText, Play, Radar, RefreshCcw, UploadCloud } from "lucide-react";
import { createJob, uploadInput, uploadModel } from "@/lib/api";
import { compactPath } from "@/lib/format";
import type { ModelItem } from "@/lib/types";
import { Badge, SectionTitle, Surface } from "./ui";

const { useEffect, useMemo, useState } = React;

const ACTIVE_SESSION_KEY = "edgeai.activeLocalSession.v1";
const OLD_SELECTION_KEYS = [
  "edgeai.localRunProductPanel.v1",
  "edgeai.localRunProductPanel.v2",
  "edgeai.localRunProductPanel.v3",
];

type ActiveLocalSession = {
  modelPath: string;
  packageName: string;
  testInput: string;
  source: "selected" | "uploaded" | "manual";
  status: "editing" | "running" | "done";
  updatedAt: number;
};

function cleanName(value: string | undefined) {
  const raw = (value || "user_model").replace(/\.(onnx|pt|pth|ckpt|h5|hdf5|keras|pb|tflite|pkl|joblib|sav|bst|xgb|lgb|gguf|zip)$/i, "");
  const cleaned = raw.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
  return cleaned || "user_model";
}

function basenameWithoutExt(path?: string) {
  if (!path) return "user_model";
  const clean = path.replace(/\\/g, "/");
  const parts = clean.split("/").filter(Boolean);
  const file = parts[parts.length - 1] || "user_model";
  const parent = parts[parts.length - 2] || "user_model";
  const stem = file.replace(/\.(onnx|pt|pth|ckpt|h5|hdf5|keras|pb|tflite|pkl|joblib|sav|bst|xgb|lgb|gguf|zip)$/i, "");

  // Zoo models are usually stored as .../<model_name>/model.onnx.
  // Using the parent avoids wrong package names such as model_local.
  if (stem === "model" || stem.startsWith("model_")) return cleanName(parent);
  return cleanName(stem);
}

function defaultInputForPath(path?: string) {
  const text = (path || "").toLowerCase();
  if (text.includes("mnist")) return "photo/1.png";
  return "photo/cat.png";
}

function selectedModelPath(model?: ModelItem) {
  return ((model as (ModelItem & { path?: string }) | undefined)?.path || "").trim();
}

function selectedModelName(model?: ModelItem) {
  return ((model as (ModelItem & { name?: string }) | undefined)?.name || "").trim();
}

function sessionFromSelected(selectedPath: string, selectedName: string): ActiveLocalSession {
  const source = selectedPath || selectedName || "models/zoo/mnist/model.onnx";
  return {
    modelPath: source,
    packageName: `${basenameWithoutExt(source)}_local`,
    testInput: defaultInputForPath(source),
    source: "selected",
    status: "editing",
    updatedAt: Date.now(),
  };
}

function readActiveSession(): ActiveLocalSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(ACTIVE_SESSION_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw) as Partial<ActiveLocalSession>;
    if (!data || typeof data !== "object") return null;
    if (!data.modelPath || !data.packageName) return null;
    return {
      modelPath: String(data.modelPath || ""),
      packageName: cleanName(String(data.packageName || "user_model")),
      testInput: String(data.testInput || ""),
      source: data.source === "uploaded" || data.source === "manual" ? data.source : "selected",
      status: data.status === "running" || data.status === "done" ? data.status : "editing",
      updatedAt: Number(data.updatedAt || Date.now()),
    };
  } catch {
    return null;
  }
}

function writeActiveSession(session: ActiveLocalSession) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(ACTIVE_SESSION_KEY, JSON.stringify({ ...session, updatedAt: Date.now() }));
  } catch {
    // ignore localStorage write errors
  }
}

function clearOldSelectionCaches() {
  if (typeof window === "undefined") return;
  try {
    OLD_SELECTION_KEYS.forEach((key) => window.localStorage.removeItem(key));
  } catch {
    // ignore
  }
}

export function LocalRunProductPanel({
  selectedModel,
  onRefresh,
}: {
  selectedModel?: ModelItem;
  onRefresh: () => void | Promise<void>;
}) {
  const selectedPath = selectedModelPath(selectedModel);
  const selectedName = selectedModelName(selectedModel);
  const initialSession = sessionFromSelected(selectedPath, selectedName);

  const [hydrated, setHydrated] = useState(false as boolean);
  const [session, setSession] = useState(initialSession as ActiveLocalSession);
  const [busyAction, setBusyAction] = useState(null as string | null);
  const [message, setMessage] = useState("" as string);
  const [modelFile, setModelFile] = useState(null as File | null);
  const [inputFile, setInputFile] = useState(null as File | null);
  const [uploadBusy, setUploadBusy] = useState(null as string | null);

  // Active local inference is a workflow session, not a temporary form cache.
  // Once the user uploads/edits model and input, switching pages must not reset them.
  useEffect(() => {
    clearOldSelectionCaches();
    const saved = readActiveSession();
    if (saved) {
      setSession(saved);
      setMessage(`已恢复本次本地推理流程：${saved.packageName}`);
    }
    setHydrated(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    writeActiveSession(session);
  }, [hydrated, session]);

  const safePackageName = useMemo(() => cleanName(session.packageName), [session.packageName]);
  const packageDir = `outputs/packages/${safePackageName}`;

  function updateSession(patch: Partial<ActiveLocalSession>) {
    setSession((prev: any) => ({
      ...prev,
      ...patch,
      packageName: cleanName(patch.packageName ?? prev.packageName),
      updatedAt: Date.now(),
    }));
  }

  function startNewLocalSession() {
    const next = sessionFromSelected(selectedPath, selectedName);
    setSession(next);
    setModelFile(null);
    setInputFile(null);
    setMessage("已开始新的本地推理流程。请上传/填写模型和测试输入，然后从 01 开始执行。");
    writeActiveSession(next);
  }

  function useSelectedModelAsSession() {
    const next = sessionFromSelected(selectedPath, selectedName);
    setSession(next);
    setMessage(`已把左侧当前选中模型设为本次本地推理目标：${next.modelPath}`);
    writeActiveSession(next);
  }

  function markSessionDone() {
    updateSession({ status: "done" });
    setMessage(`本次本地推理流程已完成：${safePackageName}。Reports 页面会默认查看这个 package 的报告。`);
  }

  async function uploadSelectedModel() {
    if (!modelFile) {
      window.alert("请先选择要上传的 ONNX 模型文件");
      return;
    }
    setUploadBusy("model");
    setMessage(`正在上传模型：${modelFile.name} ...`);
    try {
      const result = (await uploadModel(modelFile)) as { path?: string; name?: string };
      const nextPath = String(result.path || "");
      const nextName = `${basenameWithoutExt(result.path || result.name)}_local`;
      updateSession({
        modelPath: nextPath,
        packageName: nextName,
        testInput: session.testInput || defaultInputForPath(result.path || result.name),
        source: "uploaded",
        status: "editing",
      });
      setMessage(`模型上传成功：${nextPath}。本次本地推理流程已锁定 package：${nextName}`);
      await onRefresh();
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setMessage(`模型上传失败：${detail}`);
      window.alert(`模型上传失败：\n\n${detail}`);
    } finally {
      setUploadBusy(null);
    }
  }

  async function uploadSelectedInput() {
    if (!inputFile) {
      window.alert("请先选择测试输入文件，例如 jpg、png、npy 或 csv");
      return;
    }
    setUploadBusy("input");
    setMessage(`正在上传测试输入：${inputFile.name} ...`);
    try {
      const result = (await uploadInput(inputFile)) as { path?: string; name?: string };
      const nextInput = String(result.path || result.name || "");
      updateSession({ testInput: nextInput, source: session.source === "selected" ? "manual" : session.source, status: "editing" });
      setMessage(`测试输入上传成功：${nextInput}。后续 03 Prepare Input 将使用这个输入。`);
      await onRefresh();
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setMessage(`测试输入上传失败：${detail}`);
      window.alert(`测试输入上传失败：\n\n${detail}`);
    } finally {
      setUploadBusy(null);
    }
  }

  async function launch(action: string, params: Record<string, string | boolean>) {
    if (!session.modelPath.trim() && action === "local-model-setup") {
      window.alert("请先选择、上传或填写 ONNX 模型路径");
      return;
    }
    if (!session.testInput.trim() && action === "prepare-input") {
      window.alert("请先选择、上传或填写测试输入路径");
      return;
    }

    updateSession({ status: "running" });
    setBusyAction(action);
    setMessage(`正在提交 ${action} 后端任务，本次流程 package=${safePackageName} ...`);

    try {
      const job = (await createJob({ action, params })) as { id?: string; job_id?: string };
      const id = job.id || job.job_id || "已创建";
      setMessage(`已创建 ${action} 任务：${id}。本次流程仍锁定 ${safePackageName}，可到 Runtime Console 查看日志。`);
      if (action === "local-report") markSessionDone();
      await onRefresh();
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      updateSession({ status: "editing" });
      setMessage(`${action} 操作失败：${detail}`);
      window.alert(`Local ONNX Runtime 操作失败：\n\n${detail}`);
    } finally {
      setBusyAction(null);
    }
  }

  const steps = [
    {
      key: "local-model-setup",
      title: "Setup Model",
      desc: "把本次流程的 ONNX 模型整理为本地推理 package。",
      icon: UploadCloud,
      params: {
        name: safePackageName,
        source_model: session.modelPath.trim(),
        framework: "auto",
        overwrite: true,
      },
    },
    {
      key: "analyze",
      title: "Analyze ONNX",
      desc: "读取输入输出、opset、dtype、shape 和算子统计。",
      icon: Radar,
      params: { package: packageDir },
    },
    {
      key: "prepare-input",
      title: "Prepare Input",
      desc: "根据 preprocess.json 把本次流程的测试输入转换为 input.npy。",
      icon: FileJson,
      params: { package: packageDir, input: session.testInput.trim() },
    },
    {
      key: "local-run",
      title: "Local Run",
      desc: "使用 ONNX Runtime CPUExecutionProvider 完成本地推理。",
      icon: Play,
      params: { package: packageDir },
    },
    {
      key: "local-report",
      title: "Local Report",
      desc: "按本次流程 package 生成本地推理 Markdown / PDF 报告。",
      icon: FileText,
      params: { package: packageDir },
    },
  ] as const;

  return (
    <Surface className="p-5">
      <SectionTitle
        label="LOCAL ONNX RUNTIME"
        title="本地推理流程"
        description="这是一次独立的本地推理流程：用户选择/上传的模型和输入会一直作为本次流程目标，直到生成报告或开始新的本地推理。"
        right={<Badge tone={session.status === "done" ? "green" : "cyan"}>{session.status === "done" ? "Report ready" : "Active session"}</Badge>}
      />

      <div className="mb-4 rounded-2xl border border-cyan/20 bg-cyan/10 p-4 text-xs leading-6 text-cyan">
        <div className="font-semibold text-ink">当前本地推理流程</div>
        <div>Package：<span className="font-mono">{safePackageName}</span></div>
        <div>模型：<span className="font-mono">{compactPath(session.modelPath, 92)}</span></div>
        <div>输入：<span className="font-mono">{compactPath(session.testInput || "未设置", 92)}</span></div>
      </div>

      <div className="mb-4 grid gap-3 lg:grid-cols-2">
        <div className="rounded-2xl border border-line bg-black/20 p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-muted">Upload model</div>
          <input
            type="file"
            accept=".onnx,.pt,.pth,.ckpt,.h5,.hdf5,.keras,.pb,.tflite,.pkl,.joblib,.sav,.bst,.xgb,.lgb,.gguf,.txt,.json,.zip"
            onChange={(event) => setModelFile(event.target.files?.[0] || null)}
            className="mb-3 block w-full rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-muted file:mr-3 file:rounded-lg file:border-0 file:bg-cyan/20 file:px-3 file:py-1 file:text-cyan"
          />
          <button
            type="button"
            onClick={uploadSelectedModel}
            disabled={uploadBusy === "model" || !modelFile}
            className="rounded-xl border border-cyan/30 bg-cyan/10 px-4 py-2 text-xs font-semibold text-cyan transition hover:border-cyan/60 disabled:opacity-50"
          >
            {uploadBusy === "model" ? "上传中..." : "上传并设为本次模型"}
          </button>
        </div>

        <div className="rounded-2xl border border-line bg-black/20 p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-muted">Upload input</div>
          <input
            type="file"
            accept=".jpg,.jpeg,.png,.bmp,.npy,.csv"
            onChange={(event) => setInputFile(event.target.files?.[0] || null)}
            className="mb-3 block w-full rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-muted file:mr-3 file:rounded-lg file:border-0 file:bg-cyan/20 file:px-3 file:py-1 file:text-cyan"
          />
          <button
            type="button"
            onClick={uploadSelectedInput}
            disabled={uploadBusy === "input" || !inputFile}
            className="rounded-xl border border-cyan/30 bg-cyan/10 px-4 py-2 text-xs font-semibold text-cyan transition hover:border-cyan/60 disabled:opacity-50"
          >
            {uploadBusy === "input" ? "上传中..." : "上传并设为本次输入"}
          </button>
        </div>
      </div>

      <div className="mb-4 grid gap-3 lg:grid-cols-3">
        <label className="block rounded-2xl border border-line bg-black/20 p-3">
          <span className="label">ONNX model path</span>
          <input
            value={session.modelPath}
            onChange={(event) => {
              const nextPath = event.target.value;
              updateSession({
                modelPath: nextPath,
                packageName: `${basenameWithoutExt(nextPath)}_local`,
                source: "manual",
                status: "editing",
              });
            }}
            className="mt-2 w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink outline-none transition focus:border-cyan/50"
          />
        </label>

        <label className="block rounded-2xl border border-line bg-black/20 p-3">
          <span className="label">Package name</span>
          <input
            value={session.packageName}
            onChange={(event) => updateSession({ packageName: event.target.value, source: "manual", status: "editing" })}
            className="mt-2 w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink outline-none transition focus:border-cyan/50"
          />
        </label>

        <label className="block rounded-2xl border border-line bg-black/20 p-3">
          <span className="label">Test input path</span>
          <input
            value={session.testInput}
            onChange={(event) => updateSession({ testInput: event.target.value, source: "manual", status: "editing" })}
            className="mt-2 w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink outline-none transition focus:border-cyan/50"
          />
        </label>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        <button type="button" onClick={startNewLocalSession} className="rounded-xl border border-line bg-white/[0.04] px-3 py-2 text-xs font-semibold text-ink transition hover:border-cyan/30">
          开始新的本地推理
        </button>
        <button type="button" onClick={useSelectedModelAsSession} className="rounded-xl border border-line bg-white/[0.04] px-3 py-2 text-xs font-semibold text-ink transition hover:border-cyan/30">
          使用左侧选中模型
        </button>
        <button type="button" onClick={markSessionDone} className="rounded-xl border border-line bg-white/[0.04] px-3 py-2 text-xs font-semibold text-muted transition hover:border-cyan/30">
          标记本次流程结束
        </button>
      </div>

      {message ? <div className="mb-4 rounded-xl border border-line bg-black/20 p-3 text-sm text-muted">{message}</div> : null}

      <div className="grid gap-3 lg:grid-cols-5">
        {steps.map((step, index) => {
          const Icon = step.icon;
          const active = busyAction === step.key;
          return (
            <button
              type="button"
              key={step.key}
              onClick={() => launch(step.key, step.params)}
              disabled={Boolean(busyAction)}
              className="group rounded-2xl border border-line bg-black/20 p-4 text-left transition hover:-translate-y-0.5 hover:border-cyan/40 hover:bg-cyan/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="rounded-xl border border-line bg-black/30 px-2 py-1 text-[10px] font-semibold text-muted">0{index + 1}</span>
                <Icon className="h-4 w-4 text-cyan" />
              </div>
              <div className="text-sm font-bold text-ink">{active ? "Running..." : step.title}</div>
              <p className="mt-2 text-xs leading-5 text-muted">{step.desc}</p>
            </button>
          );
        })}
      </div>

      <div className="mt-4 flex items-center gap-2 text-xs text-muted">
        <RefreshCcw className="h-3.5 w-3.5" />
        Runtime / Reports 页面会读取当前本地推理流程的 package：<span className="font-mono text-ink">{safePackageName}</span>
      </div>
    </Surface>
  );
}
