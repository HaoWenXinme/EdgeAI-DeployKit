from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


# EdgeAI hotfix: normalize package names produced by WebUI/CLI convert flow.
def normalize_package_name(name: str, *_, **__)-> str:
    """Normalize a package name without changing ordinary names.

    Rules:
    - If a path-like value is passed accidentally, keep the final component.
    - Collapse duplicated WebUI suffixes: xxx_local_local -> xxx_local.
    - Keep empty/None values as an empty string so callers can handle validation.
    """
    value = str(name or "").strip().replace("\\", "/")
    if "/" in value:
        value = value.rstrip("/").split("/")[-1]
    while value.endswith("_local_local"):
        value = value[:-6]
    return value



SUPPORTED_FRAMEWORKS = {
    "auto",
    "onnx",
    "pytorch",
    "torch",
    "torchscript",
    "tensorflow",
    "tf",
    "keras",
    "savedmodel",
    "pb",
    "tflite",
    "sklearn",
    "xgboost",
    "lightgbm",
    "llm",
    "gguf",
}

TORCHVISION_ARCH_CANDIDATES = [
    "torchvision:shufflenet_v2_x1_0",
    "torchvision:shufflenet_v2_x0_5",
    "torchvision:shufflenet_v2_x1_5",
    "torchvision:shufflenet_v2_x2_0",
    "torchvision:resnet18",
    "torchvision:resnet34",
    "torchvision:resnet50",
    "torchvision:mobilenet_v2",
    "torchvision:mobilenet_v3_small",
    "torchvision:mobilenet_v3_large",
    "torchvision:efficientnet_b0",
    "torchvision:vgg16",
]


@dataclass
class ConvertStatus:
    ok: bool
    framework: str
    source_model: str
    package_name: str
    package_dir: str
    output_onnx: str
    opset: int
    message: str
    warnings: list[str]
    next_steps: list[str]
    error: Optional[str] = None
    requires_input: bool = False
    missing_params: Optional[list[str]] = None
    suggested_params: Optional[dict[str, Any]] = None
    questions: Optional[list[dict[str, Any]]] = None
    install_commands: Optional[list[str]] = None


class ConversionNeedsInput(RuntimeError):
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        super().__init__(json.dumps(payload, indent=2, ensure_ascii=False))


def project_root() -> Path:
    return Path.cwd().resolve()


def sanitize_name(value: str | None, fallback: str = "user_model") -> str:
    import re

    raw = (value or fallback).strip().replace("\\", "/").split("/")[-1]
    raw = re.sub(r"\.(onnx|pt|pth|pkl|joblib|h5|hdf5|keras|pb|tflite|ckpt|sav|bst|xgb|lgb|gguf|txt|zip)$", "", raw, flags=re.I)
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return raw or fallback


def infer_framework(source: Path, requested: str | None) -> str:
    requested = (requested or "auto").lower()
    mapping = {
        "torch": "pytorch",
        "tf": "tensorflow",
        "savedmodel": "tensorflow",
        "keras": "tensorflow",
        "pb": "tensorflow",
        "tflite": "tensorflow",
        "gguf": "llm",
    }
    if requested != "auto":
        if requested not in SUPPORTED_FRAMEWORKS:
            raise ValueError(f"unsupported framework: {requested}; supported={sorted(SUPPORTED_FRAMEWORKS)}")
        return mapping.get(requested, requested)

    if source.is_dir():
        if (source / "saved_model.pb").exists():
            return "tensorflow"
        if (source / "config.json").exists() and (
            (source / "tokenizer.json").exists()
            or (source / "tokenizer.model").exists()
            or list(source.glob("*.safetensors"))
            or list(source.glob("*.gguf"))
        ):
            return "llm"
        return "tensorflow"

    suffix = source.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "pytorch"
    if suffix in {".h5", ".hdf5", ".keras", ".pb", ".tflite"}:
        return "tensorflow"
    if suffix in {".pkl", ".joblib", ".sav"}:
        return "sklearn"
    if suffix in {".bst", ".xgb"}:
        return "xgboost"
    if suffix in {".lgb"}:
        return "lightgbm"
    if suffix == ".json":
        return "xgboost"
    if suffix == ".txt":
        return "lightgbm"
    if suffix in {".gguf", ".safetensors"}:
        return "llm"
    return "onnx"


def parse_shape(shape: str | None) -> list[int] | None:
    if not shape:
        return None
    text = str(shape).strip().lower().replace("×", "x").replace(" ", "")
    if not text:
        return None
    if ";" in text:
        # First-input shortcut for multi-input notation: input1:1,3,224,224;input2:1,10
        text = text.split(";", 1)[0]
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.replace("x", ",")
    values: list[int] = []
    for item in text.split(","):
        if not item:
            continue
        if item in {"?", "-1", "none", "dynamic", "batch", "n"}:
            values.append(1)
        else:
            values.append(int(item))
    return values or None


def shape_to_text(shape: list[int] | None) -> str | None:
    return ",".join(str(x) for x in shape) if shape else None


def copy_source(source: Path, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        target = dest_dir / source.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return str(target)
    target = dest_dir / source.name
    shutil.copy2(source, target)
    return str(target)


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or project_root()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout or ""


def import_ok(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def _auto_install_conversion_deps(requirements: dict[str, Any], package_dir: Path, warnings: list[str]) -> bool:
    framework = str(requirements.get("framework") or "").lower()
    if not requirements.get("install_commands"):
        return False

    install_plan: list[list[str]] = []
    if framework in {"pytorch", "torchscript"}:
        install_plan.append(["--index-url", os.environ.get("PYTORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu"), "torch", "torchvision"])
    elif framework == "tensorflow":
        install_plan.append(["tensorflow-cpu", "tf2onnx", "h5py"])
    elif framework == "sklearn":
        install_plan.append(["joblib", "skl2onnx"])
    elif framework == "xgboost":
        install_plan.append(["onnxmltools", "xgboost"])
    elif framework == "lightgbm":
        install_plan.append(["onnxmltools", "lightgbm"])
    elif framework == "llm":
        install_plan.append(["llama-cpp-python"])
    else:
        return False

    log_path = package_dir / "dependency_install.log"
    with log_path.open("a", encoding="utf-8", errors="ignore") as log:
        for args in install_plan:
            cmd = [sys.executable, "-m", "pip", "install", *args]
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            code, text = run(cmd)
            log.write(text + "\n")
            log.flush()
            if code != 0:
                raise RuntimeError(
                    "automatic dependency installation failed; see "
                    f"{log_path}. Last output:\n{text[-4000:]}"
                )
    warnings.append(f"Auto-installed missing conversion dependencies for {framework}; log: {log_path}")
    return True


def _safe_torch_load(path: Path, device: Any) -> Any:
    import torch  # type: ignore

    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _extract_state_dict_from_checkpoint(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    for key in ("state_dict", "model_state_dict", "model", "net", "module", "ema", "weights"):
        value = obj.get(key)
        if hasattr(value, "state_dict"):
            try:
                return dict(value.state_dict())
            except Exception:
                pass
        if isinstance(value, dict):
            nested = _extract_state_dict_from_checkpoint(value)
            if nested:
                return nested
    tensor_like = 0
    for value in obj.values():
        if hasattr(value, "shape") or hasattr(value, "size"):
            tensor_like += 1
    if tensor_like >= max(1, min(3, len(obj))):
        return dict(obj)
    return None


def _strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.", "net.", "_orig_mod.")
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        next_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if next_key.startswith(prefix):
                    next_key = next_key[len(prefix):]
                    changed = True
        cleaned[next_key] = value
    return cleaned


def _infer_num_classes_from_state_dict(state_dict: dict[str, Any], default: int = 1000) -> int:
    priority = (
        "fc.weight",
        "classifier.weight",
        "classifier.1.weight",
        "classifier.3.weight",
        "head.weight",
        "heads.head.weight",
        "linear.weight",
    )
    for key in priority:
        value = state_dict.get(key)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2:
            try:
                return int(shape[0])
            except Exception:
                pass
    for key in reversed(list(state_dict.keys())):
        value = state_dict[key]
        shape = getattr(value, "shape", None)
        lower = str(key).lower()
        if shape is not None and len(shape) == 2 and any(t in lower for t in ("fc", "classifier", "head", "linear")):
            try:
                return int(shape[0])
            except Exception:
                pass
    return default


def _guess_torchvision_arch(source: Path, state_dict: dict[str, Any] | None = None) -> str | None:
    name = source.name.lower().replace("-", "_")
    keys = list((state_dict or {}).keys())
    joined_keys = "\n".join(keys[:80]).lower()

    if "shufflenet" in name or ("stage2" in joined_keys and "conv5" in joined_keys):
        return "torchvision:shufflenet_v2_x1_0"
    if "mobilenetv2" in name or "mobilenet_v2" in name or "features.18" in joined_keys:
        return "torchvision:mobilenet_v2"
    if "resnet18" in name:
        return "torchvision:resnet18"
    if "resnet34" in name:
        return "torchvision:resnet34"
    if "resnet50" in name:
        return "torchvision:resnet50"
    if "efficientnet_b0" in name or "efficientnetb0" in name:
        return "torchvision:efficientnet_b0"
    if "vgg16" in name:
        return "torchvision:vgg16"
    return None


def _guess_torchvision_arch(source: Path, state_dict: dict[str, Any] | None = None) -> str | None:
    """Registry-driven torchvision arch guesser.

    This replaces single-model heuristics with EdgeAI's model intelligence registry.
    It still falls back to filename matching if probing is not possible.
    """
    try:
        from .model_intelligence import guess_torchvision_arch
        return guess_torchvision_arch(source, state_dict)
    except Exception:
        name = source.name.lower().replace("-", "_")
        keys = list((state_dict or {}).keys())
        joined_keys = "\n".join(keys[:80]).lower()
        if "shufflenet" in name or ("stage2" in joined_keys and "conv5" in joined_keys):
            return "torchvision:shufflenet_v2_x0_5"
        if "mobilenetv2" in name or "mobilenet_v2" in name or "features.18" in joined_keys:
            return "torchvision:mobilenet_v2"
        if "mobilenet_v3_small" in name:
            return "torchvision:mobilenet_v3_small"
        if "mobilenet_v3_large" in name:
            return "torchvision:mobilenet_v3_large"
        if "resnet18" in name:
            return "torchvision:resnet18"
        if "resnet34" in name:
            return "torchvision:resnet34"
        if "resnet50" in name:
            return "torchvision:resnet50"
        if "efficientnet_b0" in name or "efficientnetb0" in name:
            return "torchvision:efficientnet_b0"
        if "vgg16" in name:
            return "torchvision:vgg16"
        return None


def _candidate_torchvision_arches(source: Path, state_dict: dict[str, Any] | None = None) -> list[str]:
    try:
        from .model_intelligence import get_torchvision_arch_candidates
        return get_torchvision_arch_candidates(source, state_dict)
    except Exception:
        first = _guess_torchvision_arch(source, state_dict)
        candidates = []
        if first:
            candidates.append(first)
        for arch in TORCHVISION_ARCH_CANDIDATES:
            if arch not in candidates:
                candidates.append(arch)
        return candidates


def _make_torchvision_model(arch: str, num_classes: int):
    normalized = (arch or "").strip()
    for prefix in ("torchvision:", "torchvision.models:", "tv:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    normalized = normalized.strip()
    if not normalized:
        raise RuntimeError("--arch is empty; example: --arch torchvision:shufflenet_v2_x1_0")

    try:
        import torchvision.models as models  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--arch torchvision:* requires torchvision. Install with: pip install torchvision") from exc

    factory = getattr(models, normalized, None)
    if factory is None:
        available = [n for n in dir(models) if not n.startswith("_") and callable(getattr(models, n))]
        close = [n for n in available if normalized.lower() in n.lower() or n.lower() in normalized.lower()][:12]
        hint = f" Similar candidates: {close}" if close else ""
        raise RuntimeError(f"unknown torchvision architecture: {normalized}.{hint}")

    errors: list[str] = []
    for kwargs in (
        {"weights": None, "num_classes": num_classes},
        {"pretrained": False, "num_classes": num_classes},
        {"num_classes": num_classes},
        {"weights": None},
        {"pretrained": False},
        {},
    ):
        try:
            return factory(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError(f"failed to instantiate torchvision model {normalized}: {' | '.join(errors[-3:])}")


def _build_model_from_state_dict(checkpoint: dict[str, Any], arch: str, device: Any):
    state_dict = _extract_state_dict_from_checkpoint(checkpoint)
    if not state_dict:
        raise RuntimeError("checkpoint dict does not contain a recognizable state_dict")
    state_dict = _strip_state_dict_prefixes(state_dict)
    num_classes = _infer_num_classes_from_state_dict(state_dict, default=1000)
    model = _make_torchvision_model(arch, num_classes=num_classes)
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"failed to load state_dict into --arch {arch}: {exc}") from exc
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    missing = list(getattr(load_result, "missing_keys", []) or [])
    if state_dict and len(unexpected) >= max(8, int(len(state_dict) * 0.8)):
        raise RuntimeError(
            f"state_dict does not match --arch {arch}; unexpected_keys={unexpected[:12]}, missing_keys={missing[:12]}"
        )
    model.to(device)
    model.eval()
    return model


def _question(name: str, label: str, help_text: str, default: Any = None, options: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "label": label, "help": help_text, "default": default, "options": options or []}


def inspect_conversion_requirements(
    source_model: str | Path,
    framework: str = "auto",
    opset: int = 11,
    input_shape: str | None = None,
    input_name: str = "input",
    output_name: str = "output",
    arch: str | None = None,
    feature_count: int | None = None,
    torchscript: bool = False,
    dynamic_batch: bool = True,
) -> dict[str, Any]:
    root = project_root()
    source = Path(source_model).expanduser()
    if not source.is_absolute():
        source = (root / source).resolve()

    missing: list[str] = []
    questions: list[dict[str, Any]] = []
    warnings: list[str] = []
    install_commands: list[str] = []
    suggested: dict[str, Any] = {}
    detected_source_kind = "unknown"

    if not source.exists():
        return {
            "ready": False,
            "framework": framework,
            "source_model": str(source),
            "detected_source_kind": detected_source_kind,
            "missing_params": ["source_model"],
            "suggested_params": {},
            "questions": [_question("source_model", "模型路径", "请填写服务器上存在的模型文件或 SavedModel 目录路径。")],
            "install_commands": [],
            "warnings": [f"source model not found: {source}"],
            "opset": opset,
        }

    resolved = infer_framework(source, framework)
    suggested["framework"] = resolved
    suggested["input_name"] = input_name or "input"
    suggested["output_name"] = output_name or "output"
    suggested["dynamic_batch"] = dynamic_batch

    if resolved == "onnx":
        detected_source_kind = "onnx_file"

    elif resolved in {"pytorch", "torchscript"}:
        if not import_ok("torch"):
            install_commands.append("python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        if not parse_shape(input_shape):
            missing.append("input_shape")
            suggested["input_shape"] = "1,3,224,224"
            questions.append(
                _question(
                    "input_shape",
                    "模型输入形状",
                    "PyTorch 导出 ONNX 需要 dummy input。图像分类常用 1,3,224,224；检测模型常用 1,3,640,640。",
                    "1,3,224,224",
                )
            )
        if source.suffix.lower() in {".pt", ".pth", ".ckpt"} and import_ok("torch"):
            try:
                import torch  # type: ignore

                device = torch.device("cpu")
                obj = None
                if torchscript or resolved == "torchscript":
                    try:
                        obj = torch.jit.load(str(source), map_location=device)
                        detected_source_kind = "torchscript"
                    except Exception:
                        obj = _safe_torch_load(source, device)
                else:
                    try:
                        obj = torch.jit.load(str(source), map_location=device)
                        detected_source_kind = "torchscript"
                    except Exception:
                        obj = _safe_torch_load(source, device)
                if isinstance(obj, dict):
                    state = _strip_state_dict_prefixes(_extract_state_dict_from_checkpoint(obj) or {})
                    detected_source_kind = "pytorch_state_dict_or_checkpoint"
                    guess = _guess_torchvision_arch(source, state)
                    if guess:
                        suggested["arch"] = guess
                    if not arch:
                        missing.append("arch")
                        questions.append(
                            _question(
                                "arch",
                                "PyTorch 网络结构",
                                "这个 .pth 是 state_dict/checkpoint 权重，必须指定网络结构。优先使用建议值；如不匹配再换 x0_5/x1_5/x2_0 等。",
                                guess or "torchvision:shufflenet_v2_x1_0",
                                TORCHVISION_ARCH_CANDIDATES,
                            )
                        )
                elif hasattr(obj, "forward"):
                    detected_source_kind = "pytorch_executable_model"
                else:
                    detected_source_kind = type(obj).__name__
            except Exception as exc:
                warnings.append(f"PyTorch probe failed: {type(exc).__name__}: {exc}")
        elif source.suffix.lower() in {".pt", ".pth", ".ckpt"} and not arch:
            guess = _guess_torchvision_arch(source, None)
            if guess:
                suggested["arch"] = guess

    elif resolved == "tensorflow":
        detected_source_kind = "tensorflow_savedmodel_or_keras"
        if not import_ok("tf2onnx"):
            install_commands.append("python -m pip install tf2onnx")
        suffix = source.suffix.lower()
        if suffix == ".pb":
            detected_source_kind = "tensorflow_graphdef_pb"
            if not input_name or input_name == "input":
                missing.append("input_name")
                questions.append(_question("input_name", "GraphDef 输入节点", "例如 input:0。GraphDef .pb 通常无法自动可靠识别输入节点。", "input:0"))
            if not output_name or output_name == "output":
                missing.append("output_name")
                questions.append(_question("output_name", "GraphDef 输出节点", "例如 output:0。GraphDef .pb 通常无法自动可靠识别输出节点。", "output:0"))

    elif resolved == "sklearn":
        detected_source_kind = "sklearn_pickle_or_joblib"
        if not import_ok("skl2onnx"):
            install_commands.append("python -m pip install skl2onnx joblib")
        if not feature_count and not parse_shape(input_shape):
            missing.append("feature_count")
            suggested["feature_count"] = 4
            questions.append(_question("feature_count", "特征数量", "传统机器学习模型需要输入特征数，例如 Iris 是 4。", 4))

    elif resolved in {"xgboost", "lightgbm"}:
        detected_source_kind = f"{resolved}_booster"
        if resolved == "xgboost" and not import_ok("onnxmltools"):
            install_commands.append("python -m pip install onnxmltools xgboost")
        if resolved == "lightgbm" and not import_ok("onnxmltools"):
            install_commands.append("python -m pip install onnxmltools lightgbm")
        if not feature_count and not parse_shape(input_shape):
            missing.append("feature_count")
            suggested["feature_count"] = 4
            questions.append(_question("feature_count", "特征数量", f"{resolved} 转 ONNX 需要输入特征数。", 4))

    elif resolved == "llm":
        has_gguf = source.suffix.lower() == ".gguf" or (source.is_dir() and bool(list(source.rglob("*.gguf"))))
        detected_source_kind = "llm_gguf" if has_gguf else "llm_directory"
        suggested["task_type"] = "llm_chat"
        suggested["runtime"] = "llama.cpp" if has_gguf else "external"
        if not has_gguf:
            warnings.append("HuggingFace-style LLM directories are packaged, but chat inference needs an external runtime adapter.")

    ready = not missing and not install_commands
    return {
        "ready": ready,
        "framework": resolved,
        "source_model": str(source),
        "detected_source_kind": detected_source_kind,
        "missing_params": list(dict.fromkeys(missing)),
        "suggested_params": suggested,
        "questions": questions,
        "install_commands": install_commands,
        "warnings": warnings,
        "opset": opset,
    }


def _prompt_for_missing(requirements: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    print("\n[EdgeAI Convert Wizard] 检测到转换参数不足，请按提示补全。直接回车使用默认值。\n", file=sys.stderr)
    for q in requirements.get("questions", []):
        name = q.get("name")
        if not name:
            continue
        default = q.get("default")
        help_text = q.get("help") or ""
        options = q.get("options") or []
        print(f"- {q.get('label', name)}", file=sys.stderr)
        if help_text:
            print(f"  {help_text}", file=sys.stderr)
        if options:
            print("  可选示例：" + ", ".join(str(x) for x in options[:8]), file=sys.stderr)
        prompt = f"  {name}"
        if default not in (None, ""):
            prompt += f" [{default}]"
        prompt += ": "
        value = input(prompt).strip()
        if not value and default not in (None, ""):
            value = str(default)
        if value:
            current[name] = value
        print("", file=sys.stderr)
    return current


def validate_onnx(path: Path, warnings: list[str]) -> None:
    try:
        import onnx  # type: ignore

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except ImportError:
        warnings.append("onnx package not installed; skipped onnx.checker validation")
    except Exception as exc:
        raise RuntimeError(f"ONNX validation failed: {exc}") from exc


def convert_onnx(source: Path, output_onnx: Path, warnings: list[str]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"ONNX source file not found: {source}")
    shutil.copy2(source, output_onnx)
    validate_onnx(output_onnx, warnings)


def convert_tensorflow(source: Path, output_onnx: Path, opset: int, input_name: str | None, output_name: str | None) -> None:
    try:
        import tf2onnx  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError("TensorFlow conversion requires tf2onnx. Install with: python -m pip install tf2onnx tensorflow") from exc

    cmd = [sys.executable, "-m", "tf2onnx.convert", "--output", str(output_onnx), "--opset", str(opset)]
    if source.is_dir():
        cmd.extend(["--saved-model", str(source)])
    elif source.suffix.lower() in {".h5", ".keras"}:
        cmd.extend(["--keras", str(source)])
    elif source.suffix.lower() == ".pb":
        cmd.extend(["--graphdef", str(source)])
        if input_name:
            cmd.extend(["--inputs", input_name])
        if output_name:
            cmd.extend(["--outputs", output_name])
    else:
        raise RuntimeError(f"unsupported TensorFlow source: {source}; use SavedModel dir, .h5, .keras or .pb")
    code, text = run(cmd)
    if code != 0 or not output_onnx.exists():
        raise RuntimeError("tf2onnx conversion failed:\n" + text[-6000:])


def convert_pytorch(
    source: Path,
    output_onnx: Path,
    opset: int,
    input_shape: str | None,
    input_name: str,
    output_name: str,
    torchscript: bool,
    arch: str | None = None,
    dynamic_batch: bool = True,
) -> None:
    shape = parse_shape(input_shape)
    if not shape:
        raise RuntimeError("PyTorch conversion requires --input-shape, e.g. --input-shape 1,3,224,224")
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch conversion requires torch. Install with: python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu") from exc

    device = torch.device("cpu")
    dummy = torch.zeros(*shape, dtype=torch.float32, device=device)
    load_errors: list[str] = []
    if torchscript:
        try:
            model = torch.jit.load(str(source), map_location=device)
        except Exception as exc:
            raise RuntimeError(f"failed to load TorchScript model: {exc}") from exc
    else:
        try:
            model = torch.jit.load(str(source), map_location=device)
        except Exception as exc:
            load_errors.append(f"torch.jit.load failed: {exc}")
            try:
                model = _safe_torch_load(source, device)
            except Exception as exc2:
                load_errors.append(f"torch.load failed: {exc2}")
                raise RuntimeError("unable to load PyTorch model. " + " | ".join(load_errors)) from exc2

    if isinstance(model, dict):
        checkpoint = model
        state_dict = _strip_state_dict_prefixes(_extract_state_dict_from_checkpoint(checkpoint) or {})
        candidates: list[str] = []
        if arch:
            candidates.append(arch)
        for candidate in _candidate_torchvision_arches(source, state_dict):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError(
                "loaded PyTorch file is a state_dict/checkpoint dict, not an executable model. "
                "Model intelligence could not infer an architecture. Please provide --arch or a model adapter."
            )
        errors: dict[str, str] = {}
        loaded_model = None
        selected_arch = None
        for candidate in candidates:
            try:
                loaded_model = _build_model_from_state_dict(checkpoint, arch=candidate, device=device)
                selected_arch = candidate
                break
            except Exception as exc:
                errors[candidate] = f"{type(exc).__name__}: {exc}"
        if loaded_model is None:
            tail = "\n".join(f"- {k}: {v[:700]}" for k, v in list(errors.items())[:8])
            raise RuntimeError(
                "failed to match PyTorch state_dict with known model registry candidates. "
                "Please choose the correct --arch or provide custom model code. Tried:\n" + tail
            )
        if selected_arch and selected_arch != arch:
            warnings.warn(f"Model intelligence selected {selected_arch} for this state_dict instead of {arch!r}.")
        model = loaded_model
    if hasattr(model, "eval"):
        model.eval()

    dyn_axes = None
    if dynamic_batch:
        dyn_axes = {input_name or "input": {0: "batch_size"}, output_name or "output": {0: "batch_size"}}

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_onnx),
            export_params=True,
            input_names=[input_name or "input"],
            output_names=[output_name or "output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dyn_axes,
            # PyTorch 2.x may default to the dynamo exporter.
            # dynamic_axes belongs to the legacy exporter; force legacy mode
            # so --dynamic-batch remains stable without requiring dynamic_shapes.
            dynamo=False,
        )


def _feature_count_from_args(feature_count: int | None, input_shape: str | None) -> int:
    if feature_count:
        return int(feature_count)
    shape = parse_shape(input_shape)
    if shape:
        return int(shape[-1])
    raise RuntimeError("traditional ML conversion requires --feature-count or --input-shape, e.g. --feature-count 4")


def convert_sklearn(source: Path, output_onnx: Path, input_shape: str | None, input_name: str, feature_count: int | None = None) -> None:
    features = _feature_count_from_args(feature_count, input_shape)
    try:
        import joblib  # type: ignore
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sklearn conversion requires joblib and skl2onnx. Install with: python -m pip install joblib skl2onnx") from exc

    try:
        model = joblib.load(str(source))
    except Exception:
        with source.open("rb") as f:
            model = pickle.load(f)
    onnx_model = convert_sklearn(model, initial_types=[(input_name or "input", FloatTensorType([None, features]))])
    output_onnx.write_bytes(onnx_model.SerializeToString())


def convert_booster(framework: str, source: Path, output_onnx: Path, input_shape: str | None, input_name: str, feature_count: int | None = None) -> None:
    features = _feature_count_from_args(feature_count, input_shape)
    try:
        from onnxmltools.convert.common.data_types import FloatTensorType  # type: ignore
    except ImportError as exc:
        raise RuntimeError("booster conversion requires onnxmltools. Install with: python -m pip install onnxmltools") from exc

    if framework == "xgboost":
        try:
            import xgboost as xgb  # type: ignore
            from onnxmltools import convert_xgboost  # type: ignore
        except ImportError as exc:
            raise RuntimeError("xgboost conversion requires xgboost and onnxmltools. Install with: python -m pip install xgboost onnxmltools") from exc
        model = xgb.Booster()
        model.load_model(str(source))
        onnx_model = convert_xgboost(model, initial_types=[(input_name or "input", FloatTensorType([None, features]))])
    elif framework == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
            from onnxmltools import convert_lightgbm  # type: ignore
        except ImportError as exc:
            raise RuntimeError("lightgbm conversion requires lightgbm and onnxmltools. Install with: python -m pip install lightgbm onnxmltools") from exc
        model = lgb.Booster(model_file=str(source))
        onnx_model = convert_lightgbm(model, initial_types=[(input_name or "input", FloatTensorType([None, features]))])
    else:
        raise RuntimeError(f"unsupported booster framework: {framework}")
    output_onnx.write_bytes(onnx_model.SerializeToString())


def convert_llm_package(source: Path, package_dir: Path, warnings: list[str]) -> Path:
    """Create a package for local LLM runtimes.

    GGUF can run with llama.cpp or llama-cpp-python. HuggingFace-style directories
    are recorded as deployable model sources, but need a runtime such as Ollama,
    transformers, or a later ONNX GenAI export path.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    source_model_dir = package_dir / "source_model"
    source_model_dir.mkdir(parents=True, exist_ok=True)

    model_path: Path
    if source.is_file() and source.suffix.lower() == ".gguf":
        model_path = package_dir / "model.gguf"
        shutil.copy2(source, model_path)
        runtime = "llama.cpp"
        runnable = True
    elif source.is_dir():
        ggufs = sorted(source.rglob("*.gguf"))
        if ggufs:
            model_path = package_dir / "model.gguf"
            shutil.copy2(ggufs[0], model_path)
            runtime = "llama.cpp"
            runnable = True
        else:
            target = package_dir / "hf_model"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            model_path = target
            runtime = "external"
            runnable = False
            warnings.append("HuggingFace directory package created; install/configure a local runtime before chat inference.")
    else:
        raise RuntimeError(f"unsupported LLM source: {source}; provide .gguf or a HuggingFace model directory")

    signature = {
        "model_path": str(model_path),
        "format": "gguf" if model_path.suffix.lower() == ".gguf" else "huggingface_directory",
        "inputs": [{"name": "prompt", "shape": ["text"], "dtype": "string", "dynamic": True, "layout_guess": "text"}],
        "outputs": [{"name": "response", "shape": ["text"], "dtype": "string", "dynamic": True, "layout_guess": "text"}],
        "runtime": runtime,
        "has_dynamic_shape": True,
    }
    operator_report = {
        "model_path": str(model_path),
        "node_count": None,
        "operator_count": {},
        "runtime": runtime,
        "note": "LLM packages are executed by a text-generation runtime, not ONNX graph operators.",
    }
    model_json = {
        "model_name": package_dir.name,
        "framework": "llm",
        "source_model": str(source),
        "copied_source": str(model_path),
        "runtime": runtime,
        "runnable": runnable,
        "model_path": model_path.name if model_path.is_file() else model_path.name,
        "stage": "convert",
    }
    llm_runtime = {
        "runtime": runtime,
        "model_path": model_path.name if model_path.is_file() else model_path.name,
        "provider_order": ["llama_cpp_python", "llama_cpp_cli", "ollama_external"],
        "default_max_tokens": 256,
        "default_temperature": 0.7,
        "runnable": runnable,
    }
    for name, data in {
        "model_signature.json": signature,
        "operator_report.json": operator_report,
        "model.json": model_json,
        "llm_runtime.json": llm_runtime,
    }.items():
        (package_dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return model_path


def write_compatibility_report(package_dir: Path, result: ConvertStatus) -> Path:
    path = package_dir / "compatibility_report.md"
    lines = [
        f"# EdgeAI-DeployKit 模型转换兼容性报告：{result.package_name}",
        "",
        f"- 转换是否成功：{'成功' if result.ok else '失败'}",
        f"- 是否需要补充参数：{'是' if result.requires_input else '否'}",
        f"- 来源框架：`{result.framework}`",
        f"- 原始模型：`{result.source_model}`",
        f"- 输出 ONNX：`{result.output_onnx}`",
        f"- opset：`{result.opset}`",
        "",
        "## 信息",
        "",
        result.message,
        "",
    ]
    if result.missing_params:
        lines += ["## 需要补充的参数", ""] + [f"- `{p}`" for p in result.missing_params] + [""]
    if result.suggested_params:
        lines += ["## 建议参数", "", "```json", json.dumps(result.suggested_params, indent=2, ensure_ascii=False), "```", ""]
    if result.install_commands:
        lines += ["## 需要安装的依赖", "", "```bash"] + result.install_commands + ["```", ""]
    if result.warnings:
        lines += ["## 警告", ""] + [f"- {w}" for w in result.warnings] + [""]
    if result.error:
        lines += ["## 错误", "", "```text", result.error, "```", ""]
    lines += ["## 后续操作建议", ""] + [f"- {s}" for s in result.next_steps] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def convert_model(
    source_model: str | Path,
    framework: str = "auto",
    package_name: str | None = None,
    output_dir: str | Path | None = None,
    opset: int = 11,
    input_shape: str | None = None,
    input_name: str = "input",
    output_name: str = "output",
    torchscript: bool = False,
    arch: str | None = None,
    feature_count: int | None = None,
    dynamic_batch: bool = True,
    interactive: bool | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = project_root()
    source = Path(source_model).expanduser()
    if not source.is_absolute():
        source = (root / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"source model not found: {source}")

    package = normalize_package_name(package_name, sanitize_name(source.name) + "_local")
    package_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "outputs" / "packages" / package
    output_onnx = package_dir / "model.onnx"
    if package_dir.exists() and overwrite:
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "source_model").mkdir(parents=True, exist_ok=True)

    resolved_framework = infer_framework(source, framework)
    warnings: list[str] = []
    source_copy = copy_source(source, package_dir / "source_model")

    params: dict[str, Any] = {
        "source_model": str(source),
        "framework": resolved_framework,
        "opset": opset,
        "input_shape": input_shape,
        "input_name": input_name,
        "output_name": output_name,
        "arch": arch,
        "feature_count": feature_count,
        "torchscript": torchscript,
        "dynamic_batch": dynamic_batch,
    }

    # Probe missing dependencies and parameters. If CLI is interactive, ask the user until ready.
    for _ in range(3):
        requirements = inspect_conversion_requirements(**params)
        (package_dir / "convert_requirements.json").write_text(json.dumps(requirements, indent=2, ensure_ascii=False), encoding="utf-8")
        if requirements.get("install_commands"):
            if _auto_install_conversion_deps(requirements, package_dir, warnings):
                continue
        if not requirements.get("missing_params"):
            break
        allow_prompt = sys.stdin.isatty() if interactive is None else bool(interactive)
        if allow_prompt:
            params = _prompt_for_missing(requirements, params)
            input_shape = params.get("input_shape") or input_shape
            input_name = params.get("input_name") or input_name
            output_name = params.get("output_name") or output_name
            arch = params.get("arch") or arch
            if params.get("feature_count") not in (None, ""):
                feature_count = int(params.get("feature_count"))
            continue
        payload = {
            "ok": False,
            "requires_input": True,
            "framework": requirements.get("framework", resolved_framework),
            "source_model": str(source),
            "package_name": package,
            "package_dir": str(package_dir),
            "output_onnx": str(output_onnx),
            "opset": opset,
            "message": "conversion parameters are incomplete; please provide missing_params and retry",
            "missing_params": requirements.get("missing_params", []),
            "suggested_params": requirements.get("suggested_params", {}),
            "questions": requirements.get("questions", []),
            "install_commands": requirements.get("install_commands", []),
            "source_copy": source_copy,
            "compatibility_report": str(package_dir / "compatibility_report.md"),
        }
        status = ConvertStatus(
            ok=False,
            framework=resolved_framework,
            source_model=str(source),
            package_name=package,
            package_dir=str(package_dir),
            output_onnx=str(output_onnx),
            opset=opset,
            message=payload["message"],
            warnings=requirements.get("warnings", []),
            next_steps=["补充 missing_params 后重新执行 edgeai convert。"],
            requires_input=True,
            missing_params=payload["missing_params"],
            suggested_params=payload["suggested_params"],
            questions=payload["questions"],
            install_commands=payload["install_commands"],
        )
        payload["compatibility_report"] = str(write_compatibility_report(package_dir, status))
        (package_dir / "convert_result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ConversionNeedsInput(payload)

    # Use prompted values.
    input_shape = params.get("input_shape") or input_shape
    input_name = params.get("input_name") or input_name
    output_name = params.get("output_name") or output_name
    arch = params.get("arch") or arch
    if params.get("feature_count") not in (None, ""):
        feature_count = int(params.get("feature_count"))

    try:
        if output_onnx.exists():
            output_onnx.unlink()
        if resolved_framework == "onnx":
            convert_onnx(source, output_onnx, warnings)
        elif resolved_framework == "tensorflow":
            convert_tensorflow(source, output_onnx, opset, input_name, output_name, input_shape=input_shape)
            validate_onnx(output_onnx, warnings)
        elif resolved_framework in {"pytorch", "torchscript"}:
            convert_pytorch(
                source,
                output_onnx,
                opset,
                input_shape,
                input_name,
                output_name,
                torchscript or resolved_framework == "torchscript",
                arch=arch,
                dynamic_batch=dynamic_batch,
            )
            validate_onnx(output_onnx, warnings)
        elif resolved_framework == "sklearn":
            convert_sklearn(source, output_onnx, input_shape, input_name, feature_count=feature_count)
            validate_onnx(output_onnx, warnings)
        elif resolved_framework in {"xgboost", "lightgbm"}:
            convert_booster(resolved_framework, source, output_onnx, input_shape, input_name, feature_count=feature_count)
            validate_onnx(output_onnx, warnings)
        elif resolved_framework == "llm":
            llm_model = convert_llm_package(source, package_dir, warnings)
            output_onnx = llm_model
        else:
            raise RuntimeError(f"framework not implemented: {resolved_framework}")

        status = ConvertStatus(
            ok=True,
            framework=resolved_framework,
            source_model=str(source),
            package_name=package,
            package_dir=str(package_dir),
            output_onnx=str(output_onnx),
            opset=opset,
            message="model converted/imported to package successfully" if resolved_framework == "llm" else "model converted/imported to ONNX package successfully",
            warnings=warnings,
            next_steps=[
                f"edgeai task-init --package {package_dir} --task-type llm_chat" if resolved_framework == "llm" else f"edgeai analyze --package {package_dir}",
                f"edgeai local-run --package {package_dir} --prompt <your_prompt>" if resolved_framework == "llm" else f"edgeai prepare-input --package {package_dir} --input <your_input_image_or_npy>",
                f"edgeai report --package {package_dir}" if resolved_framework == "llm" else f"edgeai local-run --package {package_dir}",
                f"edgeai report --package {package_dir}" if resolved_framework != "llm" else "Open the WebUI chat panel for package-local conversation.",
            ],
            suggested_params={
                "framework": resolved_framework,
                "input_shape": input_shape,
                "input_name": input_name,
                "output_name": output_name,
                "arch": arch,
                "feature_count": feature_count,
                "dynamic_batch": dynamic_batch,
            },
        )
    except Exception as exc:
        req = inspect_conversion_requirements(
            source_model=source,
            framework=resolved_framework,
            opset=opset,
            input_shape=input_shape,
            input_name=input_name,
            output_name=output_name,
            arch=arch,
            feature_count=feature_count,
            torchscript=torchscript,
            dynamic_batch=dynamic_batch,
        )
        status = ConvertStatus(
            ok=False,
            framework=resolved_framework,
            source_model=str(source),
            package_name=package,
            package_dir=str(package_dir),
            output_onnx=str(output_onnx),
            opset=opset,
            message="model conversion failed; see error and compatibility report",
            warnings=warnings + req.get("warnings", []),
            next_steps=[
                "根据 compatibility_report.md 检查依赖、input_shape、input_name/output_name、feature_count 或 arch。",
                "如果是 PyTorch state_dict，请提供 --arch，例如 torchvision:shufflenet_v2_x1_0。",
                "如果是自定义 PyTorch 模型，请先导出 TorchScript/full nn.Module，或提供模型定义适配器。",
            ],
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            requires_input=bool(req.get("missing_params")),
            missing_params=req.get("missing_params", []),
            suggested_params=req.get("suggested_params", {}),
            questions=req.get("questions", []),
            install_commands=req.get("install_commands", []),
        )

    result = asdict(status)
    result["source_copy"] = source_copy
    result["compatibility_report"] = str(write_compatibility_report(package_dir, status))
    (package_dir / "convert_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if not status.ok:
        raise RuntimeError(json.dumps(result, indent=2, ensure_ascii=False))
    return result
# === EdgeAI TensorFlow Universal Importer v1b override START ===
# === EdgeAI TensorFlow Universal Importer v1b override START ===
def convert_tensorflow(
    source,
    output_onnx,
    opset,
    input_name=None,
    output_name=None,
    input_shape=None,
    *_,
    **__,
):
    """Universal TensorFlow converter override.

    Supports modern .keras, Keras H5/HDF5 with a generic legacy fallback,
    SavedModel directories, frozen GraphDef .pb, and .tflite when tf2onnx
    supports the model operators. This is intentionally not tied to one model.
    """
    from pathlib import Path as _Path
    import json as _json
    from edgeai.tf_universal_importer import convert_tensorflow_universal

    source = _Path(source)
    output_onnx = _Path(output_onnx)
    info = convert_tensorflow_universal(
        source,
        output_onnx,
        opset=int(opset or 15),
        input_name=input_name,
        output_name=output_name,
        input_shape=input_shape,
    )
    try:
        (output_onnx.parent / "tensorflow_convert_info.json").write_text(
            _json.dumps(info, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass
# === EdgeAI TensorFlow Universal Importer v1b override END ===
# === EdgeAI TensorFlow Universal Importer v1b override END ===
