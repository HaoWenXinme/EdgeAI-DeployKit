#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# EDGEAI_MODEL_INTELLIGENCE_NOTE: convert_model now uses registry-driven architecture matching.
from edgeai.convert_model import convert_model
from edgeai.model_signature import analyze_package
from edgeai.task_system import create_or_update_model_task


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert/import a model, analyze it, and create model_task.json in one local setup job."
    )
    ap.add_argument("--source-model", required=True)
    ap.add_argument("--framework", default="auto")
    ap.add_argument("--package", dest="package_name", required=True)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--input-shape", default=None)
    ap.add_argument("--input-name", default="input")
    ap.add_argument("--output-name", default="output")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--feature-count", type=int, default=None)
    ap.add_argument("--torchscript", action="store_true")
    ap.add_argument("--dynamic-batch", action="store_true")
    ap.add_argument("--static-batch", dest="dynamic_batch", action="store_false")
    ap.set_defaults(dynamic_batch=True)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--task-type", default="auto")
    ap.add_argument("--label-map", default=None)
    ap.add_argument("--label-language", default="zh")
    args = ap.parse_args()

    try:
        convert = convert_model(
            source_model=args.source_model,
            framework=args.framework,
            package_name=args.package_name,
            opset=args.opset,
            input_shape=args.input_shape,
            input_name=args.input_name,
            output_name=args.output_name,
            torchscript=args.torchscript,
            arch=args.arch,
            feature_count=args.feature_count,
            dynamic_batch=args.dynamic_batch,
            interactive=args.interactive,
            overwrite=args.overwrite,
        )
        if not convert.get("ok"):
            print(json.dumps(convert, ensure_ascii=False, indent=2))
            return 1

        package_dir = Path(convert.get("package_dir") or (ROOT / "outputs" / "packages" / args.package_name))
        if not package_dir.is_absolute():
            package_dir = ROOT / package_dir
        resolved_framework = str(convert.get("framework") or args.framework or "auto").lower()
        is_llm = resolved_framework == "llm" or (package_dir / "model.gguf").exists() or (package_dir / "llm_runtime.json").exists()
        model_path = package_dir / ("model.gguf" if is_llm and (package_dir / "model.gguf").exists() else "model.onnx")
        if not is_llm and not model_path.exists():
            result = {
                "ok": False,
                "stage": "convert",
                "message": "convert reported ok but model.onnx was not generated",
                "package_dir": str(package_dir),
                "missing": str(model_path),
                "convert": convert,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        analyze = None if is_llm else analyze_package(package_dir)
        task = create_or_update_model_task(
            package_dir,
            task_type="llm_chat" if is_llm and args.task_type in (None, "", "auto") else args.task_type,
            label_map=args.label_map,
            label_language=args.label_language,
        )

        result: dict[str, Any] = {
            "ok": True,
            "action": "local-model-setup",
            "message": "model converted, analyzed, and task config generated" if not is_llm else "LLM package created and task config generated",
            "package_name": package_dir.name,
            "package_dir": str(package_dir),
            "output_model": str(model_path),
            "output_onnx": str(package_dir / "model.onnx") if (package_dir / "model.onnx").exists() else None,
            "output_llm": str(package_dir / "model.gguf") if (package_dir / "model.gguf").exists() else None,
            "model_signature": str(package_dir / "model_signature.json"),
            "operator_report": str(package_dir / "operator_report.json"),
            "model_task": str(package_dir / "model_task.json"),
            "task_type": task.get("task_type"),
            "inferred_task_type": task.get("inferred_task_type"),
            "input_prompt": task.get("input_prompt"),
            "convert": convert,
            "analyze": analyze,
            "task": task,
            "next_steps": [
                f"edgeai local-run --package {_rel(package_dir)} --prompt <your_prompt>" if is_llm else f"edgeai prepare-input --package {_rel(package_dir)} --input <your_input_image_or_npy>",
                f"edgeai report --package {_rel(package_dir)}" if is_llm else f"edgeai local-run --package {_rel(package_dir)}",
                "Open the WebUI chat view for LLM interaction." if is_llm else f"edgeai report --package {_rel(package_dir)}",
            ],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        result = {
            "ok": False,
            "action": "local-model-setup",
            "stage": "exception",
            "message": "local model setup failed",
            "error": f"{type(exc).__name__}: {exc}",
            "package_name": args.package_name,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
