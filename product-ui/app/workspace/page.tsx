"use client";

import { useEffect, useMemo, useState } from "react";
import { AppShell, type WorkspacePanel } from "@/components/AppShell";
import { AssistantPanel } from "@/components/AssistantPanel";
import { BenchmarkPanel } from "@/components/BenchmarkPanel";
import { BenchmarkProductPanel } from "@/components/BenchmarkProductPanel";
import { BenchmarkRolePanel } from "@/components/BenchmarkRolePanel";
import { BoardInferencePanel } from "@/components/BoardInferencePanel";
import { BoardProductPanel } from "@/components/BoardProductPanel";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { CompactReportAssets } from "@/components/CompactReportAssets";
import { ConvertModelPanel } from "@/components/ConvertModelPanel";
import { DeployModeDialog, DeployModeSwitch, type DeployMode } from "@/components/DeployModeDialog";
import { FeatureLaunchGrid } from "@/components/FeatureLaunchGrid";
import { InferResultPanel } from "@/components/InferResultPanel";
import { LocalInferencePanel } from "@/components/LocalInferencePanel";
import { LocalLlmChatPanel } from "@/components/LocalLlmChatPanel";
import { ModelRegistry } from "@/components/ModelRegistry";
import { ModelProductPanel } from "@/components/ModelProductPanel";
import { ProductOverviewV1 } from "@/components/ProductOverviewV1";
import { ReportPreviewGrid } from "@/components/ReportPreviewGrid";
import { RuntimeChecksPanel } from "@/components/RuntimeChecksPanel";
import { RuntimeConsolePreview } from "@/components/RuntimeConsolePreview";
import { TaskGuidancePanel } from "@/components/TaskGuidancePanel";
import { TaskResultPanel } from "@/components/TaskResultPanel";
import { TopBar } from "@/components/TopBar";
import { WorkQueue } from "@/components/WorkQueue";
import { createJob, fetchDashboard, fetchInferResults, uploadInputImage, uploadModel } from "@/lib/api";
import { fallbackDashboard } from "@/lib/fallback-data";
import type { DashboardData, InferResult, ModelItem } from "@/lib/types";

const DEPLOY_MODE_KEY = "edgeai.deployMode.v1";
const ACTIVE_RUN_KEY = "edgeai.currentRun";
const ACTIVE_LOCAL_KEYS = [
  "edgeai.activeLocalSession",
  "edgeai.activeLocalSession.v1",
  "edgeai.activeLocalRunSession",
  "edgeai.localRun.session",
];

const panelMeta: Record<WorkspacePanel, { kicker: string; title: string; desc: string }> = {
  overview: {
    kicker: "Workspace",
    title: "Workspace Overview",
    desc: "本地模型部署是主流程；香橙派部署保留为可选流程。",
  },
  models: {
    kicker: "Model Registry",
    title: "Model Registry",
    desc: "管理模型，查看路径、来源和模型资产状态。",
  },
  pipeline: {
    kicker: "Product Pipeline",
    title: "Model Deployment Pipeline",
    desc: "先上传/转换模型，再选择本地推理或香橙派推理。页面只显示当前选择的流程。",
  },
  benchmark: {
    kicker: "Performance Lab",
    title: "Benchmark Lab",
    desc: "作为本地模型跑通后的性能测试实验室，不再作为主部署入口。",
  },
  board: {
    kicker: "OrangePi AIPro",
    title: "Board Session",
    desc: "香橙派部署保留为可选高级流程。",
  },
  "infer-result": {
    kicker: "Inference Result",
    title: "Infer Result",
    desc: "根据当前部署方式显示本地推理结果或香橙派推理结果。",
  },
  reports: {
    kicker: "Reports",
    title: "Reports",
    desc: "紧凑预览当前报告，并提供浏览器预览和下载入口。",
  },
  runtime: {
    kicker: "Runtime",
    title: "Runtime Console",
    desc: "默认显示 Runtime output 和 Work queue & logs；项目助手和能力表默认折叠。",
  },
};

function safeJson(value: string | null) {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function cleanName(value: unknown) {
  return String(value || "")
    .trim()
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .pop()
    ?.replace(/\.onnx$/i, "")
    .replace(/[^a-zA-Z0-9_.-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "";
}

function extractPackageName(value: unknown): string {
  if (!value) return "";
  if (typeof value === "string") return cleanName(value);
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = extractPackageName(item);
      if (found) return found;
    }
    return "";
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const direct = obj.packageName || obj.package_name || obj.package || obj.model_name || obj.modelName || obj.name;
    const directClean = cleanName(direct);
    if (directClean) return directClean;
    const pathLike = obj.packageDir || obj.package_dir || obj.outputDir || obj.output_dir || obj.modelPath || obj.model_path;
    const pathClean = cleanName(pathLike);
    if (pathClean) return pathClean;
  }
  return "";
}

function readDeployMode(): DeployMode | null {
  if (typeof window === "undefined") return null;
  const value = window.localStorage.getItem(DEPLOY_MODE_KEY);
  return value === "local" || value === "board" ? value : null;
}

function readActiveLocalPackage() {
  if (typeof window === "undefined") return "";
  const preferred = safeJson(window.localStorage.getItem(ACTIVE_RUN_KEY));
  const fromPreferred = extractPackageName(preferred);
  if (fromPreferred) return fromPreferred;

  for (const key of ACTIVE_LOCAL_KEYS) {
    const found = extractPackageName(safeJson(window.localStorage.getItem(key)) || window.localStorage.getItem(key));
    if (found) return found;
  }
  return "";
}

function readActiveLocalSessionText() {
  if (typeof window === "undefined") return "";
  const values: string[] = [];
  for (const key of ACTIVE_LOCAL_KEYS) {
    const raw = window.localStorage.getItem(key);
    if (raw) values.push(raw);
  }
  const current = window.localStorage.getItem(ACTIVE_RUN_KEY);
  if (current) values.push(current);
  return values.join(" ").toLowerCase();
}

function writeDeployMode(mode: DeployMode) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(DEPLOY_MODE_KEY, mode);
}

function modelSearchText(model: ModelItem) {
  const value = model as ModelItem & { type?: string; source?: string; name?: string; path?: string };
  return [value.name, value.path, value.type, value.source].filter(Boolean).join(" ").toLowerCase();
}

function displayPackageName(model?: ModelItem) {
  if (!model) return "model";
  const parts = model.path.split("/").filter(Boolean);
  const file = parts.at(-1) || model.name || "model.onnx";
  const parent = parts.at(-2);
  const rawName = String(model.name || "").trim();

  if (parent && (rawName === "model" || rawName === "model.onnx" || file === "model.onnx")) return parent;
  if (rawName && rawName !== "model" && rawName !== "model.onnx") return rawName.replace(/\.onnx$/i, "");
  return parent || file.replace(/\.onnx$/i, "") || "model";
}

function edgeaiModelType(model?: ModelItem) {
  const name = displayPackageName(model).toLowerCase();
  const path = String(model?.path || "").toLowerCase();
  const type = String(model?.type || "").toLowerCase();
  const value = `${name} ${path} ${type}`;
  if (value.includes("yolov5n") || value.includes("yolov5") || value.includes("yolo")) return "yolov5n";
  if (value.includes("resnet18")) return "resnet18";
  if (value.includes("mobilenetv2")) return "mobilenetv2";
  if (value.includes("mnist")) return "mnist";
  if (type && type !== "auto" && type !== "onnx" && type !== "model") return type;
  return name;
}

function isYoloModel(model?: ModelItem) {
  return edgeaiModelType(model) === "yolov5n";
}

export default function WorkspacePage() {
  const [data, setData] = useState<DashboardData>(fallbackDashboard);
  const [loading, setLoading] = useState(false);
  const [activePanel, setActivePanel] = useState<WorkspacePanel>("overview");
  const [searchQuery, setSearchQuery] = useState("");
  const [inferResults, setInferResults] = useState<InferResult[]>([]);
  const [selectedInputPath, setSelectedInputPath] = useState<string | undefined>(undefined);
  const [selectedInputVersion, setSelectedInputVersion] = useState<number>(0);
  const [boardHost, setBoardHost] = useState("192.168.0.36");
  const [selectedPath, setSelectedPath] = useState<string | undefined>(fallbackDashboard.models[0]?.path);
  const [deployMode, setDeployModeState] = useState<DeployMode | null>(null);
  const [deployDialogOpen, setDeployDialogOpen] = useState(false);
  const [activeLocalPackage, setActiveLocalPackage] = useState("");
  const [pipelineSessionReady, setPipelineSessionReady] = useState(false);

  const selectedModel = useMemo(
    () => data.models.find((model) => model.path === selectedPath) || data.models[0],
    [data.models, selectedPath],
  );

  const currentPackageName = pipelineSessionReady && activeLocalPackage ? activeLocalPackage : displayPackageName(selectedModel);

  async function refresh() {
    setLoading(true);
    try {
      const [next, nextInferResults] = await Promise.all([fetchDashboard(), fetchInferResults()]);
      setData(next);
      setInferResults(nextInferResults);
      if (!selectedPath && next.models[0]) setSelectedPath(next.models[0].path);
      if (pipelineSessionReady) {
        const latestLocal = readActiveLocalPackage();
        if (latestLocal) setActiveLocalPackage(latestLocal);
      }
    } finally {
      setLoading(false);
    }
  }

  function setDeployMode(mode: DeployMode) {
    setDeployModeState(mode);
    writeDeployMode(mode);
    setDeployDialogOpen(false);
  }

  useEffect(() => {
    // Do not restore deploy mode / active package on first open.
    // Pipeline must start with only Upload / Convert until the user uploads or explicitly activates a model in this session.
    setDeployModeState(null);
    setActiveLocalPackage("");
    setPipelineSessionReady(false);

    function onConvertSubmitted(event: Event) {
      const detail = (event as CustomEvent<{ packageName?: string }>).detail;
      const pkg = cleanName(detail?.packageName) || readActiveLocalPackage();
      if (pkg) setActiveLocalPackage(pkg);
      setPipelineSessionReady(true);
      setDeployModeState(null);
      setActivePanel("pipeline");
      setDeployDialogOpen(true);
    }

    window.addEventListener("edgeai:convert-submitted", onConvertSubmitted as EventListener);
    window.addEventListener("edgeai:convert-success", onConvertSubmitted as EventListener);
    window.addEventListener("edgeai:activate-local-session", onConvertSubmitted as EventListener);
    return () => {
      window.removeEventListener("edgeai:convert-submitted", onConvertSubmitted as EventListener);
      window.removeEventListener("edgeai:convert-success", onConvertSubmitted as EventListener);
      window.removeEventListener("edgeai:activate-local-session", onConvertSubmitted as EventListener);
    };
  }, []);

  useEffect(() => {
    let scrolling = false;
    let scrollTimer: number | undefined;

    function handleScroll() {
      scrolling = true;
      if (scrollTimer) window.clearTimeout(scrollTimer);
      scrollTimer = window.setTimeout(() => {
        scrolling = false;
      }, 500);
    }

    async function tick() {
      if (document.visibilityState !== "visible") return;
      if (scrolling) return;
      await refresh();
    }

    refresh();
    const scroller = document.querySelector(".main-stage");
    if (scroller) scroller.addEventListener("scroll", handleScroll, { passive: true });
    else window.addEventListener("scroll", handleScroll, { passive: true });

    const timer = window.setInterval(tick, 60000);
    return () => {
      window.clearInterval(timer);
      if (scroller) scroller.removeEventListener("scroll", handleScroll);
      else window.removeEventListener("scroll", handleScroll);
      if (scrollTimer) window.clearTimeout(scrollTimer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activePanel !== "runtime") return;
    const hasActiveJob = data.jobs.some((job) => job.status === "running" || job.status === "queued");
    if (!hasActiveJob) return;
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [activePanel, data.jobs]);

  function selectModel(model: ModelItem) {
    setSelectedPath(model.path);
  }

  function searchModel(query: string) {
    const q = query.trim().toLowerCase();
    if (!q) return;
    const hit = data.models.find((model) => modelSearchText(model).includes(q));
    if (!hit) {
      window.alert(`没有找到匹配模型：${query}`);
      return;
    }
    setSelectedPath(hit.path);
    setSearchQuery(query);
    setActivePanel("models");
  }

  async function importOnnx(file: File) {
    setLoading(true);
    try {
      const result = await uploadModel(file);
      const next = await fetchDashboard();
      setData(next);
      setSelectedPath(result.path);
      setSearchQuery(result.name);
      setActivePanel("models");
    } finally {
      setLoading(false);
    }
  }

  async function importInputImage(file: File) {
    setLoading(true);
    try {
      const result = await uploadInputImage(file);
      setSelectedInputPath(result.path);
      setSelectedInputVersion(Date.now());
      setActivePanel("infer-result");
      await refresh();
      window.alert(`测试输入上传成功：${result.path}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      window.alert(`测试输入上传失败：${message}`);
    } finally {
      setLoading(false);
    }
  }

  function selectedModelType() {
    return edgeaiModelType(selectedModel);
  }

  function selectedPackageName() {
    return currentPackageName || displayPackageName(selectedModel);
  }

  function selectedPackageIsLlm() {
    const value = `${selectedPackageName()} ${selectedModel?.name || ""} ${selectedModel?.path || ""} ${readActiveLocalSessionText()}`.toLowerCase();
    return value.includes("gguf") || value.includes("llm") || value.includes("chat") || value.includes("tinyllama") || value.includes("qwen") || value.includes("deepseek");
  }

  function packageDir() {
    return `outputs/packages/${selectedPackageName()}`;
  }

  function modelInt8Output() {
    return `outputs/${selectedPackageName()}_int8.onnx`;
  }

  type PipelineAction =
    | "check"
    | "quantize"
    | "benchmark"
    | "package"
    | "board-sync"
    | "board-run"
    | "board-deploy"
    | "matrix"
    | "report"
    | "html"
    | "pdf";

  async function runPipelineAction(action: PipelineAction) {
    if (!["matrix", "report", "html", "pdf"].includes(action) && !selectedModel) {
      window.alert("请先选择一个模型");
      return;
    }
    if (["board-sync", "board-run", "board-deploy"].includes(action) && !boardHost.trim()) {
      window.alert("请先填写 OrangePi AIPro 的 host，例如 192.168.0.36");
      setActivePanel("infer-result");
      return;
    }
    if (["benchmark", "package", "board-sync", "board-run", "board-deploy"].includes(action) && !selectedInputPath) {
      window.alert("请先上传测试输入，避免使用旧输入生成部署包。当前建议在本地推理流程中完成输入上传。");
      setActivePanel("infer-result");
      return;
    }

    const model = selectedModel?.path;
    const type = selectedModelType();
    const packageName = selectedPackageName();
    const input = selectedInputPath;

    const paramsByAction: Record<PipelineAction, Record<string, string | number | boolean | undefined>> = {
      check: { model },
      quantize: { model, output: modelInt8Output() },
      benchmark: { model, type, input, output: `outputs/benchmark/${packageName}.json` },
      package: { model, type, input, output: packageDir() },
      "board-sync": { host: boardHost.trim(), user: "HwHiAiUser", package: packageDir(), model_name: packageName },
      "board-run": { host: boardHost.trim(), user: "HwHiAiUser", port: 7891, package: packageDir(), output: packageDir(), model_name: packageName, force_convert: false, wait: isYoloModel(selectedModel) ? 8 : 3 },
      "board-deploy": { model, host: boardHost.trim(), user: "HwHiAiUser", port: 7891, type, input, package_output: packageDir(), force_convert: false, update_matrix: true, wait: isYoloModel(selectedModel) ? 8 : 3 },
      matrix: {},
      report: { report_model: packageName },
      html: { report_model: packageName },
      pdf: {},
    };

    setLoading(true);
    try {
      await createJob({ action, params: paramsByAction[action] });
      setActivePanel("runtime");
      await refresh();
    } finally {
      setLoading(false);
    }
  }

  async function runCheck() {
    if (!selectedModel) {
      window.alert("请先选择一个模型");
      return;
    }
    setLoading(true);
    try {
      await createJob({ action: "check", params: { model: selectedModel.path } });
      setActivePanel("runtime");
      await refresh();
    } finally {
      setLoading(false);
    }
  }

  const meta = panelMeta[activePanel];

  return (
    <AppShell health={data.health} activePanel={activePanel} onPanelChange={setActivePanel}>
      <TopBar
        loading={loading}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onSearchSubmit={searchModel}
        onImportModel={importOnnx}
        onRunCheck={runCheck}
      />

      <DeployModeDialog
        open={deployDialogOpen}
        packageName={selectedPackageName()}
        onClose={() => setDeployDialogOpen(false)}
        onSelect={setDeployMode}
      />

      <div className="workspace-panel-shell">
        <section className="workspace-panel-head">
          <div>
            <div className="workspace-panel-kicker">{meta.kicker}</div>
            <h1>{meta.title}</h1>
            {meta.desc ? <p>{meta.desc}</p> : null}
          </div>
          <div className="workspace-panel-current">
            <span>Current package</span>
            <strong>{selectedPackageName()}</strong>
          </div>
        </section>

        {activePanel === "overview" && (
          <div className="workspace-panel-stack">
            <ProductOverviewV1
              selectedPackage={selectedPackageName()}
              deployMode={deployMode}
              onSelectPanel={setActivePanel}
              onSelectDeployMode={setDeployMode}
            />
            <FeatureLaunchGrid selectedModel={selectedModel} onSelectPanel={setActivePanel} />
          </div>
        )}

        {activePanel === "models" && (
          <div className="workspace-product-stack">
            <ModelProductPanel models={data.models} />
            <div className="workspace-panel-card workspace-panel-card-large">
              <ModelRegistry
                models={data.models}
                selectedPath={selectedModel?.path}
                onSelect={selectModel}
                onOpenRuntime={() => setActivePanel("runtime")}
                onOpenBenchmark={() => setActivePanel("benchmark")}
                onRefresh={refresh}
              />
            </div>
          </div>
        )}

        {activePanel === "pipeline" && (
          <div className="workspace-product-stack">
            <ConvertModelPanel onRefresh={refresh} />

            {!pipelineSessionReady ? (
              <section className="rounded-[28px] border border-white/10 bg-slate-950/35 p-5">
                <div className="product-kicker">Waiting for model</div>
                <h2 className="mt-1 text-xl font-black text-white">请先上传或转换模型</h2>
                <p className="mt-2 text-sm leading-7 text-slate-300">
                  按你的流程要求，Pipeline 初始只显示模型上传 / 转换。完成上传并提交转换后，才会弹出“本地推理 / 香橙派推理”选择，并展示对应后续流程。
                </p>
              </section>
            ) : deployMode ? (
              <DeployModeSwitch mode={deployMode} onSelect={setDeployMode} />
            ) : (
              <section className="rounded-[28px] border border-white/10 bg-slate-950/45 p-5">
                <div className="product-kicker">Choose deploy route</div>
                <h2 className="mt-1 text-xl font-black text-white">选择接下来的部署方式</h2>
                <p className="mt-2 text-sm leading-7 text-slate-300">
                  模型已进入当前流程：{selectedPackageName()}。请选择本地推理或香橙派推理；选择后页面只显示对应流程。
                </p>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button type="button" onClick={() => setDeployMode("local")} className="rounded-xl border border-cyan-300/25 bg-cyan-300/10 px-4 py-2 text-xs font-bold text-cyan-100">本地推理</button>
                  <button type="button" onClick={() => setDeployMode("board")} className="rounded-xl border border-pink-200/25 bg-pink-200/10 px-4 py-2 text-xs font-bold text-pink-100">香橙派推理</button>
                </div>
              </section>
            )}

            {pipelineSessionReady && deployMode === "local" ? (
              <>
                <TaskGuidancePanel packageName={selectedPackageName()} />
                {selectedPackageIsLlm() ? (
                  <LocalLlmChatPanel
                    packageName={selectedPackageName()}
                    onRefresh={refresh}
                    onOpenReports={() => setActivePanel("reports")}
                  />
                ) : (
                  <LocalInferencePanel
                    packageName={selectedPackageName()}
                    selectedInputPath={selectedInputPath}
                    onUploadInput={importInputImage}
                    onRefresh={refresh}
                    onOpenRuntime={() => setActivePanel("runtime")}
                    onOpenReports={() => setActivePanel("reports")}
                  />
                )}
                <TaskResultPanel packageName={selectedPackageName()} />
              </>
            ) : null}

            {pipelineSessionReady && deployMode === "board" ? (
              <BoardInferencePanel
                selectedModel={selectedModel}
                boardHost={boardHost}
                onBoardHostChange={setBoardHost}
                onRunAction={runPipelineAction}
                onOpenRuntime={() => setActivePanel("runtime")}
                onOpenReports={() => setActivePanel("reports")}
              />
            ) : null}
          </div>
        )}

        {activePanel === "benchmark" && (
          <div className="workspace-product-stack benchmark-page-stack">
            <BenchmarkRolePanel />
            <BenchmarkProductPanel matrix={data.matrix} />
            <div className="benchmark-main-chart-card">
              <BenchmarkPanel matrix={data.matrix} />
            </div>
          </div>
        )}

        {activePanel === "board" && (
          <div className="workspace-product-stack">
            <BoardProductPanel health={data.health} jobs={data.jobs} />
          </div>
        )}

        {activePanel === "infer-result" && (
          <div className="workspace-product-stack">
            {!pipelineSessionReady ? (
              <section className="rounded-[28px] border border-white/10 bg-slate-950/35 p-5">
                <div className="product-kicker">Inference locked</div>
                <h2 className="mt-1 text-xl font-black text-white">请先在 Pipeline 上传 / 转换模型</h2>
                <p className="mt-2 text-sm leading-7 text-slate-300">
                  推理结果页不会在没有当前模型流程时提前展示任务向导。完成模型上传并选择部署方式后，这里才显示本地或香橙派推理结果。
                </p>
                <button type="button" onClick={() => setActivePanel("pipeline")} className="mt-4 rounded-xl border border-cyan-300/25 bg-cyan-300/10 px-4 py-2 text-xs font-bold text-cyan-100">前往 Pipeline</button>
              </section>
            ) : deployMode === "board" ? (
              <InferResultPanel
                results={inferResults}
                selectedModel={selectedModel}
                selectedInputPath={selectedInputPath}
                selectedInputVersion={selectedInputVersion}
                boardHost={boardHost}
                onBoardHostChange={setBoardHost}
                onUploadInput={importInputImage}
                onRunAction={runPipelineAction}
                onOpenRuntime={() => setActivePanel("runtime")}
                onOpenReports={() => setActivePanel("reports")}
              />
            ) : deployMode === "local" ? (
              <>
                <TaskGuidancePanel packageName={selectedPackageName()} compact />
                {selectedPackageIsLlm() ? (
                  <LocalLlmChatPanel
                    packageName={selectedPackageName()}
                    onRefresh={refresh}
                    onOpenReports={() => setActivePanel("reports")}
                  />
                ) : (
                  <LocalInferencePanel
                    packageName={selectedPackageName()}
                    selectedInputPath={selectedInputPath}
                    onUploadInput={importInputImage}
                    onRefresh={refresh}
                    onOpenRuntime={() => setActivePanel("runtime")}
                    onOpenReports={() => setActivePanel("reports")}
                  />
                )}
                <TaskResultPanel packageName={selectedPackageName()} />
              </>
            ) : (
              <section className="rounded-[28px] border border-white/10 bg-slate-950/45 p-5">
                <div className="product-kicker">Choose deploy route</div>
                <h2 className="mt-1 text-xl font-black text-white">请先选择部署方式</h2>
                <p className="mt-2 text-sm leading-7 text-slate-300">模型已进入当前流程，但尚未选择本地推理或香橙派推理。</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button type="button" onClick={() => setDeployMode("local")} className="rounded-xl border border-cyan-300/25 bg-cyan-300/10 px-4 py-2 text-xs font-bold text-cyan-100">本地推理</button>
                  <button type="button" onClick={() => setDeployMode("board")} className="rounded-xl border border-pink-200/25 bg-pink-200/10 px-4 py-2 text-xs font-bold text-pink-100">香橙派推理</button>
                </div>
              </section>
            )}
          </div>
        )}

        {activePanel === "reports" && (
          <div className="workspace-product-stack">
            <ReportPreviewGrid artifacts={data.artifacts} />
            <CompactReportAssets artifacts={data.artifacts} packageName={selectedPackageName()} />
          </div>
        )}

        {activePanel === "runtime" && (
          <div className="workspace-product-stack">
            <RuntimeConsolePreview health={data.health} jobs={data.jobs} />
            <WorkQueue jobs={data.jobs} />
            <CollapsibleSection title="项目助手" kicker="Assistant" defaultOpen={false}>
              <AssistantPanel selectedModel={selectedModel} />
            </CollapsibleSection>
            <CollapsibleSection title="Runtime capability" kicker="Capabilities" defaultOpen={false}>
              <RuntimeChecksPanel health={data.health} />
            </CollapsibleSection>
          </div>
        )}
      </div>
    </AppShell>
  );
}
