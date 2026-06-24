"use client";

import { useEffect, useState } from "react";
import type { ModelItem } from "@/lib/types";

const FILE_INPUT_ID = "workspace-onnx-upload-input";
const MODEL_EXTENSIONS = [
  ".onnx", ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".keras", ".pb", ".tflite",
  ".pkl", ".joblib", ".sav", ".bst", ".xgb", ".lgb", ".gguf", ".txt", ".json", ".zip",
];

export function TopBar({
  selectedModel,
  loading = false,
  searchQuery = "",
  onSearchChange,
  onSearchSubmit,
  onRefresh,
  onImportModel,
  onRunCheck,
}: {
  selectedModel?: ModelItem;
  loading?: boolean;
  searchQuery?: string;
  onSearchChange?: (value: string) => void;
  onSearchSubmit?: (value: string) => void;
  onRefresh?: () => void | Promise<void>;
  onImportModel?: (file: File) => void | Promise<void>;
  onRunCheck?: () => void | Promise<void>;
}) {
  const [draft, setDraft] = useState(searchQuery);
  const [busy, setBusy] = useState<"refresh" | "import" | "check" | null>(null);

  useEffect(() => {
    setDraft(searchQuery);
  }, [searchQuery]);

  const disabled = loading || busy !== null;

  async function runAction(
    name: "refresh" | "import" | "check",
    action?: () => void | Promise<void>,
  ) {
    if (!action || disabled) return;

    setBusy(name);

    try {
      await action();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      window.alert(message || "Action failed");
    } finally {
      setBusy(null);
    }
  }

  function submitSearch(event: any) {
    event.preventDefault();

    const value = draft.trim();
    if (!value) return;

    onSearchSubmit?.(value);
  }

  function changeSearch(value: string) {
    setDraft(value);
    onSearchChange?.(value);
  }

  function openFilePicker() {
    const input = document.getElementById(FILE_INPUT_ID) as HTMLInputElement | null;
    input?.click();
  }

  async function handleFileChange(event: any) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];

    input.value = "";

    if (!file) return;

    const lower = file.name.toLowerCase();
    if (!MODEL_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
      window.alert("请选择 .onnx 模型文件");
      return;
    }

    await runAction("import", () => onImportModel?.(file));
  }

  return (
    <header className="topbar workspace-topbar">
      <div className="workspace-topbar-inner">
        <div className="workspace-topbar-brand">
          <div className="workspace-topbar-kicker">EdgeAI Control</div>
          <h1 className="workspace-topbar-title">Deployment Workbench</h1>
          <p className="workspace-topbar-model">
            {selectedModel?.path || "No model selected"}
          </p>
        </div>

        <form className="workspace-search" onSubmit={submitSearch}>
          <span className="workspace-search-icon">⌕</span>
          <input
            value={draft}
            onChange={(event: any) => changeSearch(event.target.value)}
            placeholder="Search model or path"
            aria-label="Search model"
          />
        </form>

        <div className="workspace-topbar-actions">
          <button
            type="button"
            className="workspace-topbar-button"
            disabled={disabled}
            onClick={() => runAction("refresh", onRefresh)}
          >
            {busy === "refresh" ? "Refreshing..." : "Refresh"}
          </button>

          <button
            type="button"
            className="workspace-topbar-button"
            disabled={disabled}
            onClick={openFilePicker}
          >
            {busy === "import" ? "Uploading..." : "Import Model"}
          </button>

          <input
            id={FILE_INPUT_ID}
            type="file"
            accept={MODEL_EXTENSIONS.join(",")}
            style={{ display: "none" }}
            onChange={handleFileChange}
          />

          <button
            type="button"
            className="workspace-topbar-button workspace-topbar-button-primary"
            disabled={disabled || !selectedModel}
            onClick={() => runAction("check", onRunCheck)}
          >
            {busy === "check" ? "Running..." : "Run Check"}
          </button>
        </div>
      </div>
    </header>
  );
}
