from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from backend.schemas import ModelItem
from backend.services.paths import INPUTS_DIR, PROJECT_ROOT, rel
from backend.services.security import safe_filename


def infer_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".gguf") or "llm" in lower or "chat" in lower:
        return "llm_chat"
    if lower.endswith((".h5", ".hdf5", ".keras", ".pb", ".tflite")) or "saved_model" in lower:
        return "tensorflow"
    if lower.endswith((".pt", ".pth", ".ckpt")):
        return "pytorch"
    if lower.endswith((".pkl", ".joblib", ".sav")):
        return "sklearn"
    if lower.endswith((".bst", ".xgb", ".lgb")):
        return "booster"
    if "mnist" in lower:
        return "mnist"
    if "mobilenet" in lower:
        return "mobilenetv2"
    if "resnet" in lower:
        return "resnet18"
    if "yolo" in lower:
        return "yolov5n"
    return "auto"


def source_for(path: Path) -> str:
    text = rel(path)
    if text.startswith("models/"):
        return "zoo"
    if text.startswith("examples/"):
        return "example"
    if text.startswith("inputs/"):
        return "upload"
    if text.startswith("outputs/"):
        return "output"
    return "custom"


def list_models() -> list[ModelItem]:
    suffixes = (
        ".onnx", ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".keras", ".pb", ".tflite",
        ".pkl", ".joblib", ".sav", ".bst", ".xgb", ".lgb", ".gguf",
    )
    patterns = ["models/**/*", "examples/**/*", "inputs/models/**/*", "outputs/packages/**/*"]
    seen: set[str] = set()
    items: list[ModelItem] = []
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_dir():
                if not ((path / "saved_model.pb").exists() or (path / "config.json").exists()):
                    continue
            elif not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            rp = rel(path)
            if rp in seen:
                continue
            seen.add(rp)
            stat = path.stat()
            items.append(ModelItem(
                name=path.stem,
                path=rp,
                type=infer_type(rp),
                size_mb=round(stat.st_size / 1024 / 1024, 4),
                source=source_for(path),
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            ))
    return sorted(items, key=lambda item: (item.source, item.name))


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe zip entry: {info.filename}")
            out = (target_dir / name).resolve()
            if not (out == root or root in out.parents):
                raise ValueError(f"unsafe zip entry: {info.filename}")
        zf.extractall(target_dir)
    children = [p for p in target_dir.iterdir() if p.name != zip_path.name]
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return target_dir


def save_upload(kind: str, file: UploadFile) -> str:
    subdir = {"model": "models", "image": "images", "input": "inputs", "json": "json"}[kind]
    folder = INPUTS_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    name = safe_filename(file.filename or "upload.bin")
    target = folder / name
    with target.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            handle.write(chunk)
    if kind == "model" and target.suffix.lower() == ".zip":
        extract_dir = folder / f"{target.stem}_dir"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extracted = _safe_extract_zip(target, extract_dir)
        return rel(extracted)
    return rel(target)
