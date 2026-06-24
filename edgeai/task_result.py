from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_package_dir(package: str | Path) -> Path:
    p = Path(package)
    if p.exists():
        return p.resolve()
    root_pkg = PROJECT_ROOT / "outputs" / "packages" / str(package)
    return root_pkg.resolve()


def _load_labels(label_map: Optional[str]) -> List[str]:
    candidates: List[Path] = []
    if label_map:
        lm = Path(label_map)
        candidates.append(lm if lm.is_absolute() else PROJECT_ROOT / lm)
    candidates.append(PROJECT_ROOT / "models" / "labels" / "imagenet_classes.txt")
    candidates.append(PROJECT_ROOT / "imagenet_classes.txt")
    for p in candidates:
        if p.exists() and p.is_file():
            return [line.strip() for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    return []


def _load_zh_overrides() -> Dict[str, str]:
    path = PROJECT_ROOT / "models" / "labels" / "imagenet_zh_overrides.json"
    data = _read_json(path, {}) or {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()} if isinstance(data, dict) else {}


def _localize_label(label: Any, label_language: str, overrides: Dict[str, str]) -> tuple[Any, Any]:
    if not label or not str(label_language).lower().startswith("zh"):
        return label, None
    text = str(label)
    localized = overrides.get(text)
    if not localized:
        return label, None
    return localized, text


def _normalize_topk(topk: Any, labels: List[str], label_language: str = "en") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(topk, list):
        return rows
    zh_overrides = _load_zh_overrides()
    for i, item in enumerate(topk):
        if not isinstance(item, dict):
            continue
        idx = item.get("index", item.get("class_id", item.get("id")))
        label = item.get("label")
        try:
            idx_int = int(idx)
        except Exception:
            idx_int = None
        if (not label) and idx_int is not None and 0 <= idx_int < len(labels):
            label = labels[idx_int]
        label, label_en = _localize_label(label, label_language, zh_overrides)
        score = item.get("score", item.get("confidence", item.get("prob")))
        row = {
            "rank": int(item.get("rank", i + 1)),
            "index": idx_int if idx_int is not None else idx,
            "label": label,
            "score": score,
        }
        if label_en:
            row["label_en"] = label_en
        rows.append(row)
    return rows


def _summary_for(task_type: str, topk: List[Dict[str, Any]], local_result: Dict[str, Any]) -> Dict[str, Any]:
    best = topk[0] if topk else {}
    if task_type == "digit_classification":
        digit = best.get("label") or best.get("index")
        if isinstance(digit, str) and digit.strip().isdigit():
            digit = int(digit.strip())
        return {
            "title": "数字识别结果",
            "primary": f"预测数字：{digit}" if digit is not None else "未能解析预测数字",
            "confidence": best.get("score"),
        }
    if task_type == "image_classification":
        label = best.get("label") or best.get("index")
        return {
            "title": "图像分类结果",
            "primary": f"Top1：{label}" if label is not None else "未能解析 Top1 类别",
            "confidence": best.get("score"),
        }
    if task_type == "object_detection":
        dets = local_result.get("detections") or local_result.get("boxes") or []
        count = len(dets) if isinstance(dets, list) else 0
        return {
            "title": "目标检测结果",
            "primary": f"检测到 {count} 个目标",
            "confidence": None,
        }
    if task_type == "segmentation":
        return {"title": "图像分割结果", "primary": "已生成分割输出，请查看可视化产物。", "confidence": None}
    if task_type == "llm_chat":
        response = local_result.get("response")
        if not response and isinstance(local_result.get("outputs"), list) and local_result["outputs"]:
            first = local_result["outputs"][0]
            response = first.get("text") if isinstance(first, dict) else None
        return {"title": "Local LLM Chat Result", "primary": str(response or "Chat output generated.")[:220], "confidence": None}
    if task_type in {"text_classification", "llm_chat"}:
        return {"title": "文本/对话推理结果", "primary": "已生成文本推理输出。", "confidence": None}
    return {"title": "通用推理结果", "primary": "已生成本地推理输出。", "confidence": best.get("score")}


def render_task_result(package: str | Path, force: bool = False) -> Dict[str, Any]:
    package_dir = _safe_package_dir(package)
    if not package_dir.exists():
        raise FileNotFoundError(f"package not found: {package_dir}")

    out_path = package_dir / "task_result.json"
    if out_path.exists() and not force:
        cached = _read_json(out_path, None)
        if isinstance(cached, dict):
            return cached

    task = _read_json(package_dir / "model_task.json", {}) or {}
    local = _read_json(package_dir / "local_result.json", {}) or {}

    task_type = str(task.get("task_type") or "unknown")
    task_title = str(task.get("task_title") or task_type)
    ui = task.get("ui") if isinstance(task.get("ui"), dict) else {}
    output_cfg = task.get("output") if isinstance(task.get("output"), dict) else {}
    label_map = output_cfg.get("label_map") if isinstance(output_cfg, dict) else None
    label_language = str(output_cfg.get("label_language") or "en")
    labels = _load_labels(label_map)
    topk = _normalize_topk(local.get("topk"), labels, label_language=label_language)

    artifacts = {
        "source_input": "source_input.png" if (package_dir / "source_input.png").exists() else None,
        "local_topk_result": "local_topk_result.png" if (package_dir / "local_topk_result.png").exists() else None,
        "report_md": "report.md" if (package_dir / "report.md").exists() else None,
        "report_pdf": "report.pdf" if (package_dir / "report.pdf").exists() else None,
        "raw_output": "local_output.npy" if (package_dir / "local_output.npy").exists() else None,
        "text_output": "local_output.txt" if (package_dir / "local_output.txt").exists() else None,
    }

    result: Dict[str, Any] = {
        "ok": True,
        "schema_version": "1.0",
        "package_name": package_dir.name,
        "package_dir": str(package_dir),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "task_type": task_type,
        "task_title": task_title,
        "result_view": ui.get("result_view") or output_cfg.get("type") or "generic",
        "input_prompt": ui.get("input_prompt") or (task.get("input") or {}).get("prompt"),
        "summary": _summary_for(task_type, topk, local),
        "topk": topk,
        "latency_ms": local.get("latency_ms"),
        "backend": local.get("backend"),
        "provider": local.get("provider"),
        "artifacts": artifacts,
        "raw": {
            "local_success": local.get("success"),
            "output_count": len(local.get("outputs") or []) if isinstance(local.get("outputs"), list) else None,
        },
    }
    if task_type == "llm_chat":
        input_obj = local.get("input") if isinstance(local.get("input"), dict) else {}
        result["conversation"] = {
            "prompt": input_obj.get("text"),
            "response": local.get("response"),
        }

    _write_json(out_path, result)
    return result
