from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .model_signature import analyze_package
from .package_layout import load_json, package_paths, save_json


def _preview(arr: np.ndarray, limit: int = 12) -> List[float]:
    flat = arr.reshape(-1)
    return [float(x) for x in flat[: min(limit, flat.size)]]


def _load_label_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {int(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        return {i: str(v) for i, v in enumerate(data)}
    return {}


def _topk(arr: np.ndarray, labels: Dict[int, str], k: int = 5) -> Optional[List[Dict[str, Any]]]:
    x = np.asarray(arr)
    if x.ndim == 2 and x.shape[0] == 1 and x.shape[1] > 1:
        scores = x[0].astype(np.float64)
    elif x.ndim == 1 and x.shape[0] > 1:
        scores = x.astype(np.float64)
    else:
        return None
    # 如果不像概率，也仍然按分数排序，不强制 softmax。
    idx = np.argsort(scores)[::-1][:k]
    return [
        {
            "index": int(i),
            "score": float(scores[i]),
            "label": labels.get(int(i)),
        }
        for i in idx
    ]


def run_local_package(package_dir: Path, repeat: int = 1, warmup: int = 1, prompt: str | None = None) -> Dict[str, Any]:
    from .llm_runner import is_llm_package, run_llm_package

    if is_llm_package(package_dir):
        return run_llm_package(package_dir, prompt=prompt)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("需要安装 onnxruntime: pip install onnxruntime") from exc

    paths = package_paths(Path(package_dir))
    if not paths.model_onnx.exists():
        raise FileNotFoundError(f"model.onnx not found: {paths.model_onnx}")
    if not paths.input_npy.exists():
        raise FileNotFoundError(f"input.npy not found: {paths.input_npy}; run edgeai prepare-input first")
    if not paths.model_signature_json.exists():
        analyze_package(paths.root)

    signature = load_json(paths.model_signature_json)
    inputs = signature.get("inputs") or []
    if not inputs:
        raise ValueError("model has no runtime inputs")
    if len(inputs) > 1:
        raise ValueError("local-run foundation patch currently supports single-input ONNX models only")

    input_name = inputs[0]["name"]
    input_arr = np.load(paths.input_npy)

    sess = ort.InferenceSession(str(paths.model_onnx), providers=["CPUExecutionProvider"])

    for _ in range(max(0, int(warmup))):
        sess.run(None, {input_name: input_arr})

    repeat = max(1, int(repeat))
    started = time.perf_counter()
    outputs = None
    for _ in range(repeat):
        outputs = sess.run(None, {input_name: input_arr})
    elapsed = (time.perf_counter() - started) * 1000.0 / repeat
    assert outputs is not None

    output_names = [x.name for x in sess.get_outputs()]
    output_infos = []
    labels = _load_label_map(paths.label_map_json)
    first_topk = None

    for idx, out in enumerate(outputs):
        out_arr = np.asarray(out)
        out_name = output_names[idx] if idx < len(output_names) else f"output_{idx}"
        if len(outputs) == 1:
            np.save(paths.local_output_npy, out_arr)
            output_path = paths.local_output_npy
        else:
            output_path = paths.root / f"local_output_{idx}.npy"
            np.save(output_path, out_arr)
        item = {
            "name": out_name,
            "shape": list(out_arr.shape),
            "dtype": str(out_arr.dtype),
            "path": str(output_path),
            "preview": _preview(out_arr),
        }
        tk = _topk(out_arr, labels)
        if tk is not None and first_topk is None:
            first_topk = tk
        output_infos.append(item)

    result = {
        "success": True,
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "package_dir": str(paths.root),
        "model": str(paths.model_onnx),
        "input": {
            "name": input_name,
            "path": str(paths.input_npy),
            "shape": list(input_arr.shape),
            "dtype": str(input_arr.dtype),
        },
        "repeat": repeat,
        "latency_ms": round(float(elapsed), 4),
        "outputs": output_infos,
    }
    if first_topk is not None:
        result["topk"] = first_topk

    save_json(paths.local_result_json, result)
    return result
