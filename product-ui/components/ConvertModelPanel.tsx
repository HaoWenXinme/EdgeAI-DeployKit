"use client";

import React from "react";
import { createJob, fetchConvertRequirements, uploadModel } from "@/lib/api";

type ConvertModelPanelProps = {
  onRefresh?: () => void | Promise<void>;
};

type Framework = "auto" | "onnx" | "pytorch" | "torchscript" | "tensorflow" | "sklearn" | "xgboost" | "lightgbm" | "llm";

type ParamValue = string | number | boolean | null | undefined;
type ParamMap = Record<string, ParamValue>;

type Question = {
  name: string;
  label?: string;
  help?: string;
  default?: string | number | boolean | null;
  options?: string[];
};

type Requirements = {
  ready?: boolean;
  framework?: string;
  detected_source_kind?: string;
  missing_params?: string[];
  suggested_params?: Record<string, unknown>;
  questions?: Question[];
  install_commands?: string[];
  warnings?: string[];
};

const ACTIVE_RUN_KEY = "edgeai.currentRun";
const ACTIVE_LOCAL_KEYS = [
  "edgeai.activeLocalSession",
  "edgeai.activeLocalSession.v1",
  "edgeai.activeLocalRunSession",
  "edgeai.localRun.session",
];

const FRAMEWORK_OPTIONS: Array<{ value: Framework; label: string }> = [
  { value: "auto", label: "Auto detect" },
  { value: "onnx", label: "ONNX 导入" },
  { value: "pytorch", label: "PyTorch / TorchScript / state_dict" },
  { value: "tensorflow", label: "TensorFlow / Keras" },
  { value: "sklearn", label: "Scikit-Learn" },
  { value: "xgboost", label: "XGBoost" },
  { value: "lightgbm", label: "LightGBM" },
  { value: "llm", label: "LLM / GGUF" },
];

function cleanName(value: unknown) {
  const raw = String(value || "user_model")
    .trim()
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .pop() || "user_model";

  const withoutExt = raw.replace(/\.(onnx|pt|pth|ckpt|h5|hdf5|keras|pb|tflite|pkl|joblib|sav|bst|xgb|lgb|gguf|txt|json|zip)$/i, "");
  const cleaned = withoutExt.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^_+|_+$/g, "");
  return cleaned || "user_model";
}

function normalizePackageName(value: unknown) {
  const cleaned = cleanName(value);
  const collapsed = cleaned.replace(/(?:_local)+$/g, "_local");
  return collapsed.endsWith("_local") ? collapsed : `${collapsed}_local`;
}

function defaultPackageName(pathOrName: string) {
  return normalizePackageName(pathOrName);
}

function inferFramework(pathOrName: string): Framework {
  const lower = pathOrName.toLowerCase();
  if (lower.endsWith(".onnx")) return "onnx";
  if (lower.endsWith(".pt") || lower.endsWith(".pth") || lower.endsWith(".ckpt")) return "pytorch";
  if (lower.endsWith(".h5") || lower.endsWith(".hdf5") || lower.endsWith(".keras") || lower.endsWith(".pb") || lower.endsWith(".tflite") || lower.includes("saved_model")) return "tensorflow";
  if (lower.endsWith(".pkl") || lower.endsWith(".joblib") || lower.endsWith(".sav")) return "sklearn";
  if (lower.endsWith(".bst") || lower.endsWith(".xgb") || lower.endsWith(".json")) return "xgboost";
  if (lower.endsWith(".lgb") || lower.endsWith(".txt")) return "lightgbm";
  if (lower.endsWith(".gguf") || lower.includes("llm") || lower.includes("chat")) return "llm";
  return "auto";
}

function writeLocalSession(packageName: string, sourceModel: string) {
  if (typeof window === "undefined") return;
  const now = Date.now();
  const session = {
    mode: "local",
    status: "converted",
    sourceModel,
    modelPath: `outputs/packages/${packageName}/model.onnx`,
    packageName,
    updatedAt: now,
  };

  try {
    for (const key of ACTIVE_LOCAL_KEYS) {
      window.localStorage.setItem(key, JSON.stringify(session));
    }
    window.localStorage.setItem(ACTIVE_RUN_KEY, JSON.stringify({ mode: "local", packageName, updatedAt: now }));
  } catch {
    // localStorage may be blocked; ignore.
  }
}

function questionDefault(q: Question, suggestions: Record<string, unknown>) {
  const fromSuggested = suggestions[q.name];
  if (fromSuggested !== undefined && fromSuggested !== null && String(fromSuggested) !== "") return String(fromSuggested);
  if (q.default !== undefined && q.default !== null) return String(q.default);
  return "";
}

function repairDuplicatedLocalPackageSession() {
  // EDGEAI_PACKAGE_NORMALIZE_FIX: collapse stale xxx_local_local localStorage sessions.
  if (typeof window === "undefined") return;
  try {
    const keys = [...ACTIVE_LOCAL_KEYS, ACTIVE_RUN_KEY];
    for (const key of keys) {
      const raw = window.localStorage.getItem(key);
      if (!raw || !raw.includes("_local_local")) continue;
      const data = JSON.parse(raw);
      if (typeof data.packageName === "string") {
        data.packageName = normalizePackageName(data.packageName);
      }
      if (typeof data.modelPath === "string") {
        data.modelPath = data.modelPath.replace(/_local_local/g, "_local");
      }
      data.updatedAt = Date.now();
      window.localStorage.setItem(key, JSON.stringify(data));
    }
  } catch {
    // ignore malformed localStorage.
  }
}

export function ConvertModelPanel({ onRefresh }: ConvertModelPanelProps) {
  const [framework, setFramework] = React.useState("auto" as Framework);
  const [sourceModel, setSourceModel] = React.useState("");
  const [packageName, setPackageName] = React.useState("");
  const [opset, setOpset] = React.useState(11);
  const [inputShape, setInputShape] = React.useState("");
  const [inputName, setInputName] = React.useState("input");
  const [outputName, setOutputName] = React.useState("output");
  const [arch, setArch] = React.useState("");
  const [featureCount, setFeatureCount] = React.useState("");
  const [dynamicBatch, setDynamicBatch] = React.useState(true);
  const [requirements, setRequirements] = React.useState(null as Requirements | null);
  const [modalValues, setModalValues] = React.useState({} as Record<string, string>);
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    repairDuplicatedLocalPackageSession();
  }, []);

  const nextPackage = React.useMemo(() => packageName || defaultPackageName(sourceModel), [packageName, sourceModel]);

  function buildParams(extra?: ParamMap): ParamMap {
    const params: ParamMap = {
      framework,
      source_model: sourceModel.trim(),
      package_name: normalizePackageName(nextPackage),
      opset,
      input_name: inputName || "input",
      output_name: outputName || "output",
      overwrite: true,
      dynamic_batch: dynamicBatch,
      ...(extra || {}),
    };
    if (inputShape.trim()) params.input_shape = inputShape.trim();
    if (arch.trim()) params.arch = arch.trim();
    if (featureCount.trim()) params.feature_count = Number(featureCount.trim());
    return params;
  }

  function applyModalValues(values: Record<string, string>) {
    if (values.input_shape) setInputShape(values.input_shape);
    if (values.input_name) setInputName(values.input_name);
    if (values.output_name) setOutputName(values.output_name);
    if (values.arch) setArch(values.arch);
    if (values.feature_count) setFeatureCount(values.feature_count);
  }

  function needsModal(req: { missing_params?: string[]; install_commands?: string[] }) {
    return Boolean((req.missing_params && req.missing_params.length > 0) || (req.install_commands && req.install_commands.length > 0));
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    setBusy(true);
    setError("");
    setMessage(`正在上传 ${file.name} ...`);
    try {
      const result = await uploadModel(file);
      const guessed = inferFramework(result.name || result.path);
      setSourceModel(result.path);
      setFramework(guessed);
      setPackageName(defaultPackageName(result.name || result.path));
      setMessage(`上传完成：${result.path}。点击“检测并转换”，系统会自动检查缺少哪些参数。`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function probeRequirements() {
    const src = sourceModel.trim();
    if (!src) {
      setError("请先上传模型文件，或填写服务器上的模型路径。");
      return null;
    }
    setBusy(true);
    setError("");
    setMessage("正在检测模型类型和转换参数 ...");
    try {
      const req = await fetchConvertRequirements(buildParams());
      setRequirements(req);
      const suggestions = req.suggested_params || {};
      if (typeof suggestions.framework === "string") setFramework(suggestions.framework as Framework);
      const initialValues: Record<string, string> = {};
      for (const q of req.questions || []) {
        initialValues[q.name] = questionDefault(q, suggestions);
      }
      setModalValues(initialValues);
      if (needsModal(req)) {
        if (req.install_commands && req.install_commands.length > 0) {
          setMessage("检测到缺少转换依赖，请按弹窗中的命令安装后再转换。");
        } else {
          setMessage("检测到参数不足，请在弹窗中补足后继续转换。");
        }
      } else {
        setMessage("参数检测通过，可以开始转换。");
      }
      return req;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function submitConvert(extra?: ParamMap) {
    const src = sourceModel.trim();
    const pkg = normalizePackageName(nextPackage);
    if (!src) {
      setError("请先上传模型文件，或填写服务器上的模型路径。");
      return;
    }
    setBusy(true);
    setError("");
    setMessage("正在提交本地模型自动初始化任务 ...");
    try {
      const job = await createJob({
        action: "local-model-setup",
        params: buildParams(extra),
      });
      writeLocalSession(pkg, src);
      setPackageName(pkg);
      setRequirements(null);
      setMessage(`已提交转换任务：${job.id}。请选择接下来的部署方式：本地推理或香橙派推理。`);
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("edgeai:convert-submitted", { detail: { packageName: pkg, sourceModel: src, jobId: job.id } }));
      }
      await onRefresh?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function detectAndConvert() {
    const req = await probeRequirements();
    if (!req) return;
    if (needsModal(req)) return;
    await submitConvert();
  }

  async function confirmModalAndConvert() {
    applyModalValues(modalValues);
    const extra: ParamMap = {};
    if (modalValues.input_shape) extra.input_shape = modalValues.input_shape;
    if (modalValues.input_name) extra.input_name = modalValues.input_name;
    if (modalValues.output_name) extra.output_name = modalValues.output_name;
    if (modalValues.arch) extra.arch = modalValues.arch;
    if (modalValues.feature_count) extra.feature_count = Number(modalValues.feature_count);
    await submitConvert(extra);
  }

  function activateCurrent() {
    const pkg = normalizePackageName(nextPackage);
    if (!pkg) return;
    writeLocalSession(pkg, sourceModel || `outputs/packages/${pkg}/model.onnx`);
    setMessage(`已设置当前本地推理流程：${pkg}`);
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("edgeai:activate-local-session", { detail: { packageName: pkg, sourceModel: sourceModel || `outputs/packages/${pkg}/model.onnx` } }));
    }
  }

  const showModal = Boolean(requirements && needsModal(requirements));

  return (
    <section className="workspace-panel-card workspace-panel-card-large p-5">
      <div className="report-preview-head">
        <div>
          <div className="product-kicker">Convert Model</div>
          <h2>01 Upload / Convert Model</h2>
          <p>上传任意常见框架模型，系统先检测缺少的转换参数；参数不足时弹窗补齐，然后自动转换为 ONNX package。</p>
        </div>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-2">
        <label className="rounded-2xl border border-line bg-black/20 p-3 text-xs text-muted">
          <span className="mb-2 block font-semibold text-ink">上传模型文件</span>
          <input
            type="file"
            accept=".onnx,.pt,.pth,.ckpt,.h5,.hdf5,.keras,.pb,.tflite,.pkl,.joblib,.sav,.bst,.xgb,.lgb,.gguf,.txt,.json,.zip"
            onChange={(event) => handleUpload(event.target.files?.[0] || null)}
            disabled={busy}
            className="block w-full text-xs"
          />
          <span className="mt-2 block">支持 ONNX、PyTorch、TensorFlow/Keras、Scikit-Learn、XGBoost、LightGBM。</span>
        </label>

        <label className="rounded-2xl border border-line bg-black/20 p-3 text-xs text-muted">
          <span className="mb-2 block font-semibold text-ink">服务器模型路径</span>
          <input
            value={sourceModel}
            onChange={(event) => {
              const value = event.target.value;
              setSourceModel(value);
              const guessed = inferFramework(value);
              setFramework(guessed);
              if (!packageName) setPackageName(defaultPackageName(value));
            }}
            placeholder="inputs/models/shufflenetv2.pth 或 models/zoo/mobilenetv2/model.onnx"
            className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink"
          />
        </label>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-4">
        <label className="text-xs text-muted">
          <span className="mb-1 block text-ink">Framework</span>
          <select
            value={framework}
            onChange={(event) => setFramework(event.target.value as Framework)}
            className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-ink"
          >
            {FRAMEWORK_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
        </label>

        <label className="text-xs text-muted">
          <span className="mb-1 block text-ink">Package name</span>
          <input
            value={nextPackage}
            onChange={(event) => setPackageName(cleanName(event.target.value))}
            className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink"
          />
        </label>

        <label className="text-xs text-muted">
          <span className="mb-1 block text-ink">ONNX opset</span>
          <input
            type="number"
            value={opset}
            onChange={(event) => setOpset(Number(event.target.value || 11))}
            className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-ink"
          />
        </label>

        <label className="flex items-end gap-2 pb-2 text-xs text-muted">
          <input type="checkbox" checked={dynamicBatch} onChange={(event) => setDynamicBatch(event.target.checked)} />
          <span>动态 batch</span>
        </label>
      </div>

      <details className="mb-4 rounded-2xl border border-line bg-black/10 p-3 text-xs text-muted">
        <summary className="cursor-pointer font-semibold text-ink">高级参数，可留空，由检测弹窗提示补齐</summary>
        <div className="mt-3 grid gap-3 md:grid-cols-5">
          <label>
            <span className="mb-1 block text-ink">input_shape</span>
            <input value={inputShape} onChange={(event) => setInputShape(event.target.value)} placeholder="1,3,224,224" className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink" />
          </label>
          <label>
            <span className="mb-1 block text-ink">input_name</span>
            <input value={inputName} onChange={(event) => setInputName(event.target.value)} className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink" />
          </label>
          <label>
            <span className="mb-1 block text-ink">output_name</span>
            <input value={outputName} onChange={(event) => setOutputName(event.target.value)} className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink" />
          </label>
          <label>
            <span className="mb-1 block text-ink">arch</span>
            <input value={arch} onChange={(event) => setArch(event.target.value)} placeholder="torchvision:shufflenet_v2_x1_0" className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink" />
          </label>
          <label>
            <span className="mb-1 block text-ink">feature_count</span>
            <input value={featureCount} onChange={(event) => setFeatureCount(event.target.value)} placeholder="4" className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink" />
          </label>
        </div>
      </details>

      <div className="mb-4 flex flex-wrap gap-2">
        <button type="button" onClick={detectAndConvert} disabled={busy} className="rounded-xl border border-cyan/30 bg-cyan/10 px-3 py-2 text-xs font-semibold text-cyan">
          {busy ? "处理中..." : "检测并转换为 ONNX Package"}
        </button>
        <button type="button" onClick={probeRequirements} disabled={busy} className="rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-ink">
          只检测参数
        </button>
        <button type="button" onClick={activateCurrent} className="rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-ink">
          设置为当前本地流程
        </button>
      </div>

      <div className="rounded-2xl border border-line bg-black/20 p-3 text-xs leading-6 text-muted">
        <div>当前 package：<span className="font-mono text-ink">{nextPackage || "<package>"}</span></div>
        <div>ONNX 输出：<span className="font-mono text-ink">outputs/packages/{nextPackage || "<package>"}/model.onnx</span></div>
        <div>自动步骤：Convert → Analyze → Task Init；随后由用户上传测试输入 → Local Run → Report</div>
      </div>

      {message ? <div className="mt-3 rounded-xl border border-cyan/20 bg-cyan/10 p-3 text-xs text-cyan">{message}</div> : null}
      {error ? <div className="mt-3 rounded-xl border border-red-400/30 bg-red-500/10 p-3 text-xs text-red-100">{error}</div> : null}

      {showModal ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm">
          <div className="max-h-[86vh] w-full max-w-2xl overflow-auto rounded-3xl border border-line bg-[#0b1020] p-5 shadow-2xl">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <div className="product-kicker">Convert Wizard</div>
                <h3 className="text-xl font-semibold text-ink">需要补充转换参数</h3>
                <p className="mt-1 text-xs leading-6 text-muted">检测结果：{requirements?.detected_source_kind || requirements?.framework || "unknown"}</p>
              </div>
              <button type="button" onClick={() => setRequirements(null)} className="rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-ink">关闭</button>
            </div>

            {requirements?.install_commands?.length ? (
              <div className="mb-4 rounded-2xl border border-amber-400/30 bg-amber-500/10 p-3 text-xs leading-6 text-amber-100">
                <div className="font-semibold">缺少转换依赖，请先在服务器执行：</div>
                <pre className="mt-2 overflow-auto rounded-xl bg-black/30 p-3 font-mono text-[11px]">{requirements.install_commands.join("\n")}</pre>
              </div>
            ) : null}

            {(requirements?.questions || []).map((q: Question) => (
              <label key={q.name} className="mb-3 block rounded-2xl border border-line bg-black/20 p-3 text-xs text-muted">
                <span className="mb-1 block font-semibold text-ink">{q.label || q.name}</span>
                {q.help ? <span className="mb-2 block leading-5">{q.help}</span> : null}
                <input
                  list={`${q.name}-options`}
                  value={modalValues[q.name] || ""}
                  onChange={(event) => setModalValues({ ...modalValues, [q.name]: event.target.value })}
                  className="w-full rounded-xl border border-line bg-black/30 px-3 py-2 font-mono text-xs text-ink"
                />
                {q.options?.length ? (
                  <datalist id={`${q.name}-options`}>
                    {q.options.map((option: string) => <option key={option} value={option} />)}
                  </datalist>
                ) : null}
              </label>
            ))}

            {requirements?.warnings?.length ? (
              <div className="mb-4 rounded-2xl border border-line bg-black/20 p-3 text-xs leading-6 text-muted">
                {requirements.warnings.map((item: string) => <div key={item}>- {item}</div>)}
              </div>
            ) : null}

            <div className="flex flex-wrap justify-end gap-2">
              <button type="button" onClick={() => setRequirements(null)} className="rounded-xl border border-line bg-black/30 px-3 py-2 text-xs text-ink">取消</button>
              <button type="button" onClick={confirmModalAndConvert} disabled={busy || Boolean(requirements?.install_commands?.length)} className="rounded-xl border border-cyan/30 bg-cyan/10 px-3 py-2 text-xs font-semibold text-cyan">
                补齐参数并开始转换
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
