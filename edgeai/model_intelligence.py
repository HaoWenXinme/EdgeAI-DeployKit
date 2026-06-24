from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from .model_registry import load_model_registry
except Exception:  # pragma: no cover
    def load_model_registry() -> list[dict[str, Any]]:
        return []


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(x) for x in list(shape)]
    except Exception:
        return None


def _is_tensor_like(value: Any) -> bool:
    return _shape(value) is not None


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Return the most likely state_dict from a PyTorch checkpoint object."""
    if not isinstance(checkpoint, dict):
        return {}
    for key in ("state_dict", "model_state_dict", "model", "net", "network", "module"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and any(_is_tensor_like(v) for v in value.values()):
            return strip_state_dict_prefixes(value)
    if any(_is_tensor_like(v) for v in checkpoint.values()):
        return strip_state_dict_prefixes(checkpoint)
    return {}


def strip_state_dict_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.", "net.", "network.", "_orig_mod.")
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        k = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True
        out[k] = value
    return out


def _torch_load_state_dict(source: str | Path) -> tuple[str, dict[str, Any], str | None]:
    """Return (kind, state_dict, error). Does not require torchvision."""
    try:
        import torch  # type: ignore
    except Exception as exc:
        return "missing_torch", {}, f"torch import failed: {type(exc).__name__}: {exc}"
    path = Path(source)
    device = torch.device("cpu")
    try:
        obj = torch.jit.load(str(path), map_location=device)
        return "torchscript", {}, None
    except Exception:
        pass
    try:
        try:
            obj = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location=device)
    except Exception as exc:
        return "pytorch_load_failed", {}, f"torch.load failed: {type(exc).__name__}: {exc}"
    state = extract_state_dict(obj)
    if state:
        return "pytorch_state_dict_or_checkpoint", state, None
    if hasattr(obj, "forward"):
        return "pytorch_executable_model", {}, None
    return type(obj).__name__, {}, None


def state_dict_fingerprint(state_dict: dict[str, Any]) -> dict[str, Any]:
    keys = list(state_dict.keys())
    shapes = {k: _shape(v) for k, v in state_dict.items() if _shape(v) is not None}
    lower_keys = [k.lower() for k in keys]

    def find_out_channels(patterns: list[str]) -> int | None:
        for pat in patterns:
            rx = re.compile(pat)
            for k, sh in shapes.items():
                if sh and len(sh) >= 1 and rx.search(k):
                    return int(sh[0])
        return None

    def find_second_dim(patterns: list[str]) -> int | None:
        for pat in patterns:
            rx = re.compile(pat)
            for k, sh in shapes.items():
                if sh and len(sh) >= 2 and rx.search(k):
                    return int(sh[1])
        return None

    classifier_out = None
    for k in reversed(keys):
        sh = shapes.get(k)
        kl = k.lower()
        if sh and len(sh) == 2 and any(tok in kl for tok in ("classifier", "fc", "head", "linear")):
            classifier_out = int(sh[0])
            break

    fp = {
        "key_count": len(keys),
        "sample_keys": keys[:40],
        "classifier_out": classifier_out,
        "has_stage2": any("stage2" in k for k in lower_keys),
        "has_stage3": any("stage3" in k for k in lower_keys),
        "has_stage4": any("stage4" in k for k in lower_keys),
        "has_conv5": any("conv5" in k for k in lower_keys),
        "has_features": any(k.startswith("features.") for k in lower_keys),
        "has_layer4": any(k.startswith("layer4.") for k in lower_keys),
        "stage2_channels": find_out_channels([r"stage2\.0\.branch1\.2\.weight", r"stage2\.0\.branch2\.0\.weight"]),
        "stage3_channels": find_out_channels([r"stage3\.0\.branch1\.1\.weight", r"stage3\.0\.branch2\.0\.weight"]),
        "stage4_channels": find_out_channels([r"stage4\.0\.branch1\.1\.weight", r"stage4\.0\.branch2\.0\.weight"]),
        "conv5_in_channels": find_second_dim([r"conv5\.0\.weight", r"conv5\.weight"]),
    }
    return fp


def _match_name_score(entry: dict[str, Any], name: str) -> int:
    name_l = name.lower().replace("-", "_")
    score = 0
    if entry.get("family") and str(entry["family"]).lower() in name_l.replace("_", ""):
        score += 20
    for alias in entry.get("aliases", []) or []:
        alias_l = str(alias).lower().replace("-", "_")
        if alias_l and alias_l in name_l:
            score += 40
    model_id = str(entry.get("id", "")).lower().split(":")[-1]
    if model_id and model_id in name_l:
        score += 50
    return score


def _match_signature_score(entry: dict[str, Any], fp: dict[str, Any], keys: set[str]) -> tuple[int, list[str]]:
    sig = entry.get("state_dict_signature") or {}
    score = 0
    reasons: list[str] = []

    # Exact channel signatures, useful for families such as ShuffleNetV2.
    for field in ("stage2_channels", "stage3_channels", "stage4_channels", "conv5_in_channels"):
        expected = sig.get(field)
        actual = fp.get(field)
        if expected is not None and actual is not None:
            if int(expected) == int(actual):
                score += 35
                reasons.append(f"{field}={actual}")
            else:
                score -= 30

    for k in sig.get("required_keys", []) or []:
        if k in keys:
            score += 20
            reasons.append(f"key:{k}")
        else:
            score -= 5

    for prefix in sig.get("required_key_prefixes", []) or []:
        if any(k.startswith(prefix) for k in keys):
            score += 15
            reasons.append(f"prefix:{prefix}")
        else:
            score -= 3

    return score, reasons


def match_state_dict_to_registry(source: str | Path, state_dict: dict[str, Any]) -> dict[str, Any]:
    registry = [m for m in load_model_registry() if m.get("framework") == "pytorch"]
    fp = state_dict_fingerprint(state_dict)
    keys = set(state_dict.keys())
    name = Path(source).name
    scored: list[dict[str, Any]] = []
    for entry in registry:
        sig_score, reasons = _match_signature_score(entry, fp, keys)
        name_score = _match_name_score(entry, name)
        total = sig_score + name_score
        if total > 0:
            candidate = dict(entry)
            candidate["score"] = total
            candidate["match_reasons"] = reasons + (["filename"] if name_score else [])
            scored.append(candidate)
    scored.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
    best = scored[0] if scored else None
    confidence = "none"
    if best:
        score = int(best.get("score", 0))
        if score >= 95:
            confidence = "high"
        elif score >= 55:
            confidence = "medium"
        else:
            confidence = "low"
    return {
        "ok": True,
        "fingerprint": fp,
        "best_match": best,
        "candidates": scored[:12],
        "confidence": confidence,
    }


def get_torchvision_arch_candidates(source: str | Path, state_dict: dict[str, Any] | None = None) -> list[str]:
    """Return registry-ranked torchvision arch candidates for a PyTorch file/state_dict."""
    state = state_dict or {}
    if not state:
        kind, loaded, _ = _torch_load_state_dict(source)
        if kind == "pytorch_state_dict_or_checkpoint":
            state = loaded
    candidates: list[str] = []
    if state:
        result = match_state_dict_to_registry(source, state)
        for item in result.get("candidates", []) or []:
            model_id = item.get("id")
            loader = item.get("loader") or {}
            if model_id and loader.get("type") == "torchvision":
                candidates.append(str(model_id))
    # Filename fallback from registry aliases.
    name = Path(source).name.lower().replace("-", "_")
    for item in load_model_registry():
        if item.get("framework") != "pytorch":
            continue
        if item.get("loader", {}).get("type") != "torchvision":
            continue
        model_id = str(item.get("id") or "")
        if not model_id:
            continue
        if any(str(a).lower().replace("-", "_") in name for a in item.get("aliases", []) or []):
            candidates.append(model_id)
    # Stable broad fallbacks.
    fallbacks = [
        "torchvision:shufflenet_v2_x0_5",
        "torchvision:shufflenet_v2_x1_0",
        "torchvision:shufflenet_v2_x1_5",
        "torchvision:shufflenet_v2_x2_0",
        "torchvision:mobilenet_v3_small",
        "torchvision:mobilenet_v3_large",
        "torchvision:mobilenet_v2",
        "torchvision:resnet18",
        "torchvision:resnet34",
        "torchvision:resnet50",
        "torchvision:efficientnet_b0",
        "torchvision:vgg16",
    ]
    for arch in fallbacks:
        if arch not in candidates:
            candidates.append(arch)
    return candidates


def guess_torchvision_arch(source: str | Path, state_dict: dict[str, Any] | None = None) -> str | None:
    candidates = get_torchvision_arch_candidates(source, state_dict)
    return candidates[0] if candidates else None


def probe_model(source_model: str | Path, framework: str = "auto") -> dict[str, Any]:
    source = Path(source_model).expanduser()
    suffix = source.suffix.lower()
    result: dict[str, Any] = {
        "ok": source.exists(),
        "source_model": str(source),
        "framework_hint": framework,
        "suffix": suffix,
        "detected_kind": "missing" if not source.exists() else "unknown",
        "suggested_params": {},
        "questions": [],
        "warnings": [],
        "candidates": [],
    }
    if not source.exists():
        result["warnings"].append(f"source model not found: {source}")
        return result

    if suffix == ".onnx":
        result.update({"detected_kind": "onnx_file", "framework": "onnx"})
        result["suggested_params"].update({"framework": "onnx"})
        return result

    if suffix in {".pt", ".pth", ".ckpt"} or framework in {"pytorch", "torch", "torchscript"}:
        kind, state, err = _torch_load_state_dict(source)
        result.update({"detected_kind": kind, "framework": "pytorch"})
        result["suggested_params"].update({"framework": "pytorch", "input_shape": "1,3,224,224", "input_name": "input", "output_name": "output"})
        if err:
            result["warnings"].append(err)
        if state:
            matched = match_state_dict_to_registry(source, state)
            result["fingerprint"] = matched.get("fingerprint")
            result["candidates"] = matched.get("candidates", [])
            result["confidence"] = matched.get("confidence")
            best = matched.get("best_match")
            if best:
                result["suggested_params"]["arch"] = best.get("id")
                result["suggested_params"]["task_type"] = best.get("task_type")
                result["suggested_params"]["preprocess_profile"] = best.get("preprocess_profile")
        return result

    if source.is_dir() and (source / "config.json").exists() and (
        (source / "tokenizer.json").exists()
        or (source / "tokenizer.model").exists()
        or list(source.glob("*.safetensors"))
        or list(source.glob("*.gguf"))
    ):
        result.update({"detected_kind": "llm_directory", "framework": "llm"})
        result["suggested_params"].update({"framework": "llm", "task_type": "llm_chat"})
        return result

    if suffix in {".h5", ".hdf5", ".keras", ".pb", ".tflite"} or source.is_dir():
        result.update({"detected_kind": "tensorflow_or_keras", "framework": "tensorflow"})
        result["suggested_params"].update({"framework": "tensorflow"})
        return result

    if suffix in {".pkl", ".joblib"}:
        result.update({"detected_kind": "sklearn_or_booster_pickle", "framework": "sklearn"})
        result["suggested_params"].update({"framework": "sklearn", "feature_count": None})
        result["questions"].append({"name": "feature_count", "label": "特征数量", "help": "传统机器学习模型转 ONNX 需要输入特征数。"})
        return result

    if suffix == ".gguf":
        result.update({"detected_kind": "llm_gguf", "framework": "llm"})
        result["suggested_params"].update({"framework": "llm", "task_type": "llm_chat"})
        return result

    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Probe a model and suggest conversion parameters.")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--framework", default="auto")
    args = parser.parse_args()
    print(json.dumps(probe_model(args.source_model, args.framework), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
