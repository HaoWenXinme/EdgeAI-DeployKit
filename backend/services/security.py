from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from fastapi import HTTPException

from backend.services.paths import INPUTS_DIR, OUTPUTS_DIR, PROJECT_ROOT, REPORTS_DIR

ALLOWED_PATH_ROOTS = [PROJECT_ROOT, INPUTS_DIR, OUTPUTS_DIR, REPORTS_DIR]

PATH_PARAM_NAMES = {
    "model",
    "input",
    "json",
    "output",
    "output_dir",
    "package",
    "package_dir",
    "package_output",
    "qemu_dir",
    "initramfs",
    "kernel",
    "toolchain_dir",
    "onnxruntime_root",
    "matrix",
    "markdown_output",
    "html_output",
    "md_path",
    "pdf_path",
    "dockerfile",
    "source_model",
    "label_map",
}

FLAG_MAP = {
    "report_model": "--model",
    "model": "--model",
    "all_models": "--all-models",
    "input": "--input",
    "json": "--json",
    "output": "--output",
    "output_dir": "--output-dir",
    "package": "--package",
    "package_dir": "--package",
    "package_output": "--package-output",
    "type": "--type",
    "model_type": "--type",
    "model_name": "--model-name",
    "host": "--host",
    "user": "--user",
    "port": "--port",
    "remote_dir": "--remote-dir",
    "remote_root": "--remote-root",
    "min_free_gb": "--min-free-gb",
    "wait": "--wait",
    "force_convert": "--force-convert",
    "update_matrix": "--matrix",
    "repeat": "--repeat",
    "soc_version": "--soc-version",
    "input_format": "--input-format",
    "input_shape": "--input-shape",
    "atc_args": "--atc-args",
    "timeout": "--timeout",
    "dummy": "--dummy",
    "markdown_output": "--markdown-output",
    "html_output": "--html-output",
    "matrix": "--matrix",
    "qemu_dir": "--qemu-dir",
    "initramfs": "--initramfs",
    "kernel": "--kernel",
    "toolchain_dir": "--toolchain-dir",
    "memory": "--memory",
    "onnxruntime_root": "--onnxruntime-root",
    "tag": "--tag",
    "dockerfile": "--dockerfile",
    "name": "--name",
    "source_model": "--source-model",
    "framework": "--framework",
    "overwrite": "--overwrite",
    "package_name": "--package",
    "opset": "--opset",
    "input_name": "--input-name",
    "output_name": "--output-name",
    "torchscript": "--torchscript",
    "arch": "--arch",
    "feature_count": "--feature-count",
    "dynamic_batch": "--dynamic-batch",
    "interactive": "--interactive",
    "task_type": "--task-type",
    "label_map": "--label-map",
    "label_language": "--label-language",
    "prompt": "--prompt",
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
}

EDGEAI_MODULE_COMMAND = [sys.executable, "-m", "edgeai.cli"]
SCRIPT_COMMAND = [sys.executable]

COMMANDS: dict[str, list[str]] = {
    "model-info": EDGEAI_MODULE_COMMAND + ["model-info"],
    "check": EDGEAI_MODULE_COMMAND + ["check"],
    "quantize": EDGEAI_MODULE_COMMAND + ["quantize"],
    "benchmark": EDGEAI_MODULE_COMMAND + ["benchmark"],
    "package": EDGEAI_MODULE_COMMAND + ["package"],
    "prepare-input": EDGEAI_MODULE_COMMAND + ["prepare-input"],
    "om-convert": EDGEAI_MODULE_COMMAND + ["om-convert"],
    "board-sync": EDGEAI_MODULE_COMMAND + ["board-sync"],
    "board-run": EDGEAI_MODULE_COMMAND + ["board-run"],
    "board-deploy": EDGEAI_MODULE_COMMAND + ["board-deploy"],
    "deploy-qemu": EDGEAI_MODULE_COMMAND + ["deploy-qemu"],
    "matrix": EDGEAI_MODULE_COMMAND + ["matrix"],
    "report": EDGEAI_MODULE_COMMAND + ["report"],
    "html": EDGEAI_MODULE_COMMAND + ["html"],
    "pdf": EDGEAI_MODULE_COMMAND + ["pdf"],
    "matrix-report": EDGEAI_MODULE_COMMAND + ["matrix-report"],
    "pc-aipro-report": EDGEAI_MODULE_COMMAND + ["pc-aipro-report"],
    "docker-build": EDGEAI_MODULE_COMMAND + ["docker-build"],
    "docker-run-qemu": EDGEAI_MODULE_COMMAND + ["docker-run-qemu"],
    "init-package": EDGEAI_MODULE_COMMAND + ["init-package"],
    "analyze": EDGEAI_MODULE_COMMAND + ["analyze"],
    "local-run": EDGEAI_MODULE_COMMAND + ["local-run"],
    "local-report": EDGEAI_MODULE_COMMAND + ["report"],
    "convert": EDGEAI_MODULE_COMMAND + ["convert"],
    "local-model-setup": SCRIPT_COMMAND + ["scripts/edgeai_local_model_setup.py"],
}


def safe_filename(filename: str) -> str:
    name = Path(filename).name.replace(" ", "_")
    allowed = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", "."}:
            allowed.append(ch)
    cleaned = "".join(allowed).strip("._")
    if not cleaned:
        raise HTTPException(status_code=400, detail="invalid filename")
    return cleaned


def safe_package_name(name: str) -> str:
    cleaned = safe_filename((name or "").replace("\\", "/").split("/")[-1])
    return cleaned


def resolve_project_path(value: str) -> str:
    raw = Path(value).expanduser()
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve()
    if not any(resolved == root or root in resolved.parents for root in ALLOWED_PATH_ROOTS):
        raise HTTPException(status_code=400, detail=f"path is outside allowed roots: {value}")
    return str(resolved)


def build_command(action: str, params: dict[str, str | int | float | bool | None]) -> list[str]:
    if action not in COMMANDS:
        raise HTTPException(status_code=400, detail=f"unsupported action: {action}")
    cmd = list(COMMANDS[action])
    for key, value in params.items():
        if value is None or value is False or value == "":
            continue
        if key not in FLAG_MAP:
            raise HTTPException(status_code=400, detail=f"unsupported parameter: {key}")
        flag = FLAG_MAP[key]
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        text_value = str(value)
        if key in PATH_PARAM_NAMES and not (action in {"report", "html"} and key == "model"):
            text_value = resolve_project_path(text_value)
        cmd.extend([flag, text_value])
    return cmd
