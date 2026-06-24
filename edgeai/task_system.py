from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TASK_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "digit_classification": {"title": "数字识别", "input_type": "image", "input_prompt": "请上传一张只包含单个数字 0~9 的图片，例如 MNIST 手写数字。", "result_type": "digit_topk", "recommended_examples": ["examples/digits/7.png", "photo/digit.png"]},
    "image_classification": {"title": "图像分类", "input_type": "image", "input_prompt": "请上传一张实物图片，例如猫、狗、汽车、杯子等。系统会输出 TopK 分类结果和中文/英文标签。", "result_type": "classification_topk", "recommended_examples": ["photo/cat.png", "photo/dog.png"]},
    "object_detection": {"title": "目标检测", "input_type": "image", "input_prompt": "请上传一张包含目标物体的图片。系统会输出带检测框的结果图、类别和置信度。", "result_type": "detection_boxes", "recommended_examples": ["photo/street.jpg", "photo/cat.png"]},
    "segmentation": {"title": "图像分割", "input_type": "image", "input_prompt": "请上传一张需要分割的图片。系统会输出 mask 和叠加预览图。", "result_type": "segmentation_mask", "recommended_examples": ["photo/cat.png"]},
    "text_classification": {"title": "文本分类", "input_type": "text", "input_prompt": "请输入一段待分类文本。系统会输出分类标签和置信度。", "result_type": "text_label", "recommended_examples": []},
    "llm_chat": {"title": "本地大模型对话", "input_type": "chat", "input_prompt": "请输入对话内容。系统会使用本地大模型 runtime 生成回复。", "result_type": "chat_message", "recommended_examples": []},
    "unknown": {"title": "未知任务类型", "input_type": "unknown", "input_prompt": "系统暂时无法自动判断模型任务类型，请手动选择数字识别、图像分类、目标检测、图像分割、文本分类或大模型对话。", "result_type": "unknown", "recommended_examples": []},
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_shape(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _last_int(shape: List[Any]) -> Optional[int]:
    if not shape:
        return None
    try:
        return int(shape[-1])
    except Exception:
        return None


def _rank(shape: List[Any]) -> int:
    return len(shape or [])


def _name_contains(items: List[Dict[str, Any]], *tokens: str) -> bool:
    text = " ".join(str(x.get("name", "")).lower() for x in items if isinstance(x, dict))
    return any(t.lower() in text for t in tokens)


def resolve_package_dir(package: str | Path) -> Path:
    p = Path(package)
    s = str(p)
    if p.exists() or s.startswith("outputs/") or p.is_absolute():
        return p
    return Path("outputs") / "packages" / s


def project_root_from_package(package_dir: Path) -> Path:
    p = package_dir.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "edgeai").exists() and (parent / "outputs" / "packages").exists():
            return parent
    if len(p.parents) >= 3 and p.parents[1].name == "packages" and p.parents[2].name == "outputs":
        return p.parents[3]
    return Path.cwd()


def infer_task_type_from_signature(signature: Dict[str, Any], package_name: str = "") -> Tuple[str, List[str]]:
    inputs = signature.get("inputs") or []
    outputs = signature.get("outputs") or []
    name = package_name.lower()
    if "mnist" in name or "digit" in name:
        return "digit_classification", ["package name contains mnist/digit"]
    if "yolo" in name or "ssd" in name or "detect" in name:
        return "object_detection", ["package name contains yolo/ssd/detect"]
    if "llm" in name or "chat" in name or "gguf" in name or "deepseek" in name:
        return "llm_chat", ["package name contains llm/chat/gguf/deepseek"]
    if _name_contains(inputs, "input_ids", "attention_mask", "token", "tokens"):
        if _name_contains(outputs, "logits", "scores"):
            return "text_classification", ["token-like input names and logits/scores output"]
        return "llm_chat", ["token-like input names"]
    output_shapes = [_as_shape(o.get("shape")) for o in outputs if isinstance(o, dict)]
    output_names = " ".join(str(o.get("name", "")).lower() for o in outputs if isinstance(o, dict))
    if any(t in output_names for t in ["box", "boxes", "score", "scores", "label", "labels", "bbox"]):
        return "object_detection", ["output names contain boxes/scores/labels"]
    for shape in output_shapes:
        if _rank(shape) == 3:
            vals = []
            for v in shape:
                try: vals.append(int(v))
                except Exception: pass
            last = _last_int(shape)
            if (last is not None and last >= 5 and (last in {6, 7, 84, 85} or last > 20)) or any(v in {84, 85} for v in vals):
                return "object_detection", [f"rank-3 detection-like output shape {shape}"]
    for shape in output_shapes:
        if _rank(shape) == 4:
            try:
                h, w = int(shape[-2]), int(shape[-1])
                c = int(shape[1]) if len(shape) > 1 else -1
                if h >= 16 and w >= 16 and c >= 1:
                    return "segmentation", [f"rank-4 spatial output {shape}"]
            except Exception:
                pass
    for shape in output_shapes:
        if _rank(shape) == 2:
            classes = _last_int(shape)
            if classes == 10: return "digit_classification", ["rank-2 output with 10 classes"]
            if classes == 1000: return "image_classification", ["rank-2 output with 1000 classes"]
            if classes and classes > 1: return "image_classification", [f"rank-2 classification-like output with {classes} classes"]
        if _rank(shape) == 1:
            classes = _last_int(shape)
            if classes == 10: return "digit_classification", ["rank-1 output with 10 classes"]
            if classes == 1000 or (classes and classes > 1): return "image_classification", [f"rank-1 classification-like output with {classes} classes"]
    return "unknown", ["no confident task inference rule matched"]


def default_label_map(project_root: Path, task_type: str, class_count: Optional[int] = None) -> Optional[str]:
    if task_type == "digit_classification":
        p = project_root / "models" / "labels" / "digits_0_9.txt"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(str(i) for i in range(10)) + "\n", encoding="utf-8")
        return str(p.relative_to(project_root))
    if task_type == "image_classification":
        p = project_root / "models" / "labels" / "imagenet_classes.txt"
        if p.exists(): return str(p.relative_to(project_root))
    return None


def create_or_update_model_task(package: str | Path, task_type: str = "auto", label_map: Optional[str | Path] = None, label_language: str = "zh", save: bool = True) -> Dict[str, Any]:
    package_dir = resolve_package_dir(package)
    project_root = project_root_from_package(package_dir)
    package_name = package_dir.name
    signature_path = package_dir / "model_signature.json"
    operator_path = package_dir / "operator_report.json"
    task_path = package_dir / "model_task.json"
    signature = _read_json(signature_path)
    operator_report = _read_json(operator_path)
    llm_runtime = _read_json(package_dir / "llm_runtime.json")
    inferred, reasons = infer_task_type_from_signature(signature, package_name)
    if llm_runtime or (package_dir / "model.gguf").exists():
        inferred = "llm_chat"
        if "LLM package metadata found" not in reasons:
            reasons.append("LLM package metadata found")
    final_task = inferred if task_type in (None, "", "auto") else task_type
    if final_task not in TASK_GUIDANCE:
        reasons.append(f"unsupported task_type={final_task}; fallback to unknown")
        final_task = "unknown"
    inputs = signature.get("inputs") or []
    outputs = signature.get("outputs") or []
    model_input = inputs[0] if inputs and isinstance(inputs[0], dict) else {}
    model_output = outputs[0] if outputs and isinstance(outputs[0], dict) else {}
    input_shape = _as_shape(model_input.get("shape"))
    output_shape = _as_shape(model_output.get("shape"))
    class_count = _last_int(output_shape) if _rank(output_shape) in (1, 2) else None
    guidance = TASK_GUIDANCE.get(final_task, TASK_GUIDANCE["unknown"])
    label_map_value = str(label_map) if label_map else default_label_map(project_root, final_task, class_count)
    runtime_name = "onnxruntime"
    model_path_value = "model.onnx"
    if final_task == "llm_chat":
        runtime_name = str((llm_runtime or {}).get("runtime") or signature.get("runtime") or "llama.cpp")
        model_path_value = str((llm_runtime or {}).get("model_path") or ("model.gguf" if (package_dir / "model.gguf").exists() else "hf_model"))
    config: Dict[str, Any] = {
        "schema_version": "1.0",
        "package_name": package_name,
        "created_by": "edgeai task system",
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "runtime": runtime_name,
        "task_type": final_task,
        "task_title": guidance["title"],
        "inferred_task_type": inferred,
        "inference_reasons": reasons,
        "model_path": model_path_value,
        "input": {"type": guidance["input_type"], "prompt": guidance["input_prompt"], "name": model_input.get("name", "input"), "shape": input_shape, "dtype": model_input.get("dtype"), "layout": model_input.get("layout_guess") or ("NCHW" if len(input_shape) == 4 and input_shape[1] in (1, 3) else "unknown")},
        "output": {"type": guidance["result_type"], "name": model_output.get("name", "output"), "shape": output_shape, "dtype": model_output.get("dtype"), "class_count": class_count, "label_map": label_map_value, "label_language": label_language},
        "ui": {"input_prompt": guidance["input_prompt"], "recommended_examples": guidance.get("recommended_examples", []), "result_view": guidance["result_type"]},
        "artifacts": {"model_signature": "model_signature.json" if signature_path.exists() else None, "operator_report": "operator_report.json" if operator_path.exists() else None, "local_result": "local_result.json" if (package_dir / "local_result.json").exists() else None, "report_md": "report.md" if (package_dir / "report.md").exists() else None, "report_pdf": "report.pdf" if (package_dir / "report.pdf").exists() else None},
    }
    if operator_report: config["operator_summary"] = operator_report.get("operators") or operator_report.get("op_counts") or operator_report
    if save: _write_json(task_path, config)
    return {"ok": True, "package_dir": str(package_dir), "task_file": str(task_path), "task_type": final_task, "inferred_task_type": inferred, "reasons": reasons, "input_prompt": guidance["input_prompt"], "config": config}


def read_model_task(package: str | Path, auto_create: bool = False) -> Dict[str, Any]:
    package_dir = resolve_package_dir(package)
    task_path = package_dir / "model_task.json"
    if task_path.exists(): return {"ok": True, "package_dir": str(package_dir), "task_file": str(task_path), "config": _read_json(task_path)}
    if auto_create: return create_or_update_model_task(package_dir)
    return {"ok": False, "package_dir": str(package_dir), "task_file": str(task_path), "error": "model_task.json not found"}
