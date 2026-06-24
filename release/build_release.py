#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = ROOT / "release_dist"

TARGETS = {
    "windows-x86_64": {"archive": "zip", "launcher": "bat"},
    "macos-x86_64": {"archive": "tar.gz", "launcher": "sh"},
    "macos-arm64": {"archive": "tar.gz", "launcher": "sh"},
    "linux-x86_64": {"archive": "tar.gz", "launcher": "sh"},
    "linux-arm64": {"archive": "tar.gz", "launcher": "sh"},
}

INCLUDE_DIRS = [
    "backend",
    "board",
    "configs",
    "docs",
    "edgeai",
    "examples",
    "models",
    "product-ui",
    "scripts",
    "templates",
    "webui",
]

INCLUDE_FILES = [
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",
    "README_PRODUCT_UI.md",
    "RELEASE_NOTES.md",
    "PRODUCT_UI_OPERATION_GUIDE.md",
    "docker-compose.product.yml",
    "start-windows.bat",
    "stop-windows.bat",
]

EXCLUDED_NAMES = {
    ".env",
    ".env.local",
    ".git",
    ".idea",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".streamlit",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "outputs",
    "release_dist",
    "reports",
    "third_party",
}

EXCLUDED_SUFFIXES = {
    ".onnx",
    ".om",
    ".pt",
    ".pth",
    ".h5",
    ".hdf5",
    ".keras",
    ".tflite",
    ".npy",
    ".npz",
    ".tgz",
    ".tar",
    ".gz",
    ".zip",
    ".7z",
    ".rar",
    ".so",
    ".dll",
    ".dylib",
    ".log",
    ".pid",
    ".tmp",
}


def current_target() -> str:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x86_64"
    if sys_name.startswith("windows"):
        return "windows-x86_64"
    if sys_name == "darwin":
        return f"macos-{arch}"
    return f"linux-{arch}"


def should_ignore(path: Path) -> bool:
    part_names = [part.lower() for part in path.parts]
    if any(".bak" in part or "backup" in part or part.endswith(("~", ".orig", ".rej", ".tsbuildinfo")) for part in part_names):
        return True
    if path.name in EXCLUDED_NAMES:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    parts = set(path.parts)
    return bool(parts & EXCLUDED_NAMES)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if should_ignore(path) or should_ignore(rel):
            if path.is_dir():
                continue
            continue
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def write_launchers(stage: Path, target: str) -> None:
    if TARGETS[target]["launcher"] == "bat":
        (stage / "start-backend.bat").write_text(
            "\r\n".join(
                [
                    "@echo off",
                    "setlocal",
                    "cd /d %~dp0",
                    "if exist .venv\\Scripts\\activate.bat call .venv\\Scripts\\activate.bat",
                    "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (stage / "start-ui.bat").write_text(
            "\r\n".join(
                [
                    "@echo off",
                    "setlocal",
                    "cd /d %~dp0\\product-ui",
                    "set NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000",
                    "pnpm install --frozen-lockfile",
                    "pnpm dev --hostname 127.0.0.1 --port 3000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    for name, body in {
        "start-backend.sh": [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            'if [ -f ".venv/bin/activate" ]; then source ".venv/bin/activate"; fi',
            'PY="${PYTHON_BIN:-python3}"',
            'if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi',
            'API_HOST="${EDGEAI_API_HOST:-127.0.0.1}"',
            'API_PORT="${EDGEAI_API_PORT:-8000}"',
            '"$PY" -m uvicorn backend.main:app --host "$API_HOST" --port "$API_PORT"',
            "",
        ],
        "start-ui.sh": [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")/product-ui"',
            'API_PORT="${EDGEAI_API_PORT:-8000}"',
            'UI_HOST="${EDGEAI_UI_HOST:-127.0.0.1}"',
            'UI_PORT="${EDGEAI_UI_PORT:-3000}"',
            'export NEXT_PUBLIC_API_BASE="${NEXT_PUBLIC_API_BASE:-http://127.0.0.1:${API_PORT}}"',
            'if command -v corepack >/dev/null 2>&1; then',
            "  corepack pnpm install --frozen-lockfile",
            '  corepack pnpm dev --hostname "$UI_HOST" --port "$UI_PORT"',
            'elif command -v pnpm >/dev/null 2>&1; then',
            "  pnpm install --frozen-lockfile",
            '  pnpm dev --hostname "$UI_HOST" --port "$UI_PORT"',
            "else",
            '  echo "[ERROR] pnpm/Corepack not found. Install Node.js 20+ and rerun scripts/install-linux.sh." >&2',
            "  exit 1",
            "fi",
            "",
        ],
    }.items():
        path = stage / name
        path.write_text("\n".join(body), encoding="utf-8")
        os.chmod(path, 0o755)


def write_release_readme(stage: Path, target: str) -> None:
    (stage / "README_RELEASE.md").write_text(
        f"""# EdgeAI-DeployKit Release Package

Target: `{target}`

This package contains the local WebUI, FastAPI backend, `edgeai` CLI source,
task-aware local inference flow, and report generation code.

## Quick Start

### Windows

Prerequisites:

- Windows 10/11 x86_64
- Python 3.10-3.12 on PATH
- Node.js 20+ with Corepack, or a global `pnpm`
- Network access to Python/npm package indexes on first start

Double-click:

```text
start-windows.bat
```

The first run creates `.venv`, installs Python dependencies, installs WebUI
dependencies, starts the FastAPI backend and Next.js WebUI, then opens:

```text
http://127.0.0.1:3000/workspace
```

Stop both services:

```text
stop-windows.bat
```

### Linux

```bash
scripts/install-linux.sh
./start-linux.sh
```

For PyTorch `.pt` / `.pth` conversion support:

```bash
scripts/install-linux.sh --with-pytorch
```

For TensorFlow/Keras conversion support:

```bash
scripts/install-linux.sh --with-tensorflow
```

For both PyTorch and TensorFlow/Keras conversion support:

```bash
scripts/install-linux.sh --with-pytorch --with-tensorflow
```

### Manual Backend

```bash
python -m pip install -e .[pdf]
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Windows users can run `start-backend.bat` and `start-ui.bat`. macOS/Linux
users can run `./start-linux.sh`, or run `./start-backend.sh` and `./start-ui.sh`
in separate terminals.

## Notes

- Generated packages, reports, rootfs images, ONNX Runtime SDKs, `node_modules`,
  and model weights are intentionally excluded from this archive.
- Put user models under `inputs/models/` or upload them from the WebUI.
- The local path is: Convert -> Analyze -> Task Init -> Prepare Input ->
  Local Run -> Task Render -> Report.
""",
        encoding="utf-8",
    )
    if target.startswith("windows"):
        (stage / "WINDOWS_QUICK_START_先看我.txt").write_text(
            """EdgeAI-DeployKit Windows 快速说明

这个 zip 是轻量启动包，不内置完整 Python/Node 环境。
第一次启动会自动创建 .venv 并安装依赖，但电脑上必须先有：

1. Windows 10/11 x86_64
2. Python 3.10/3.11/3.12，安装时勾选 Add python.exe to PATH
3. Node.js 20 LTS 或更高版本，安装后自带 corepack
4. 能访问 pip/npm 下载源的网络

使用方法：

1. 解压整个文件夹，不要只双击压缩包里的文件
2. 双击 start-windows.bat
3. 第一次启动会下载并安装依赖，时间可能较长
4. 浏览器打开 http://127.0.0.1:3000/workspace
5. 结束时双击 stop-windows.bat

如果提示没有环境：

- 没有 Python：安装 Python 3.10-3.12 后重新运行 start-windows.bat
- 没有 Node.js/corepack：安装 Node.js 20 LTS 后重新运行 start-windows.bat
- Runtime 页面显示 PyTorch/TensorFlow/LLM/Board 为 Optional：这不是启动失败，只表示这些模型类型或板端流程的可选依赖尚未安装

当前轻量包适合有基础运行环境的电脑。完全离线、真正解压即用的版本需要单独发布 portable/full 包，体积会大很多。
""",
            encoding="utf-8",
        )


def stage_target(target: str, clean: bool) -> Path:
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}; choose one of {', '.join(TARGETS)}")
    stage = DIST_ROOT / f"EdgeAI-DeployKit-{target}"
    if clean and stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    for dirname in INCLUDE_DIRS:
        copy_tree(ROOT / dirname, stage / dirname)
    for filename in INCLUDE_FILES:
        src = ROOT / filename
        if src.exists():
            shutil.copy2(src, stage / filename)

    (stage / "inputs" / "models").mkdir(parents=True, exist_ok=True)
    (stage / "inputs" / "images").mkdir(parents=True, exist_ok=True)
    (stage / "outputs" / "packages").mkdir(parents=True, exist_ok=True)
    (stage / "reports").mkdir(parents=True, exist_ok=True)
    if TARGETS[target]["launcher"] == "sh" and (stage / "scripts" / "install-linux.sh").exists():
        install_path = stage / "install-linux.sh"
        shutil.copy2(stage / "scripts" / "install-linux.sh", install_path)
        os.chmod(install_path, 0o755)
    if TARGETS[target]["launcher"] == "sh" and (stage / "scripts" / "start-linux.sh").exists():
        start_path = stage / "start-linux.sh"
        shutil.copy2(stage / "scripts" / "start-linux.sh", start_path)
        os.chmod(start_path, 0o755)
    if TARGETS[target]["launcher"] == "sh":
        for script in stage.rglob("*.sh"):
            try:
                os.chmod(script, 0o755)
            except OSError:
                pass
    write_launchers(stage, target)
    write_release_readme(stage, target)
    return stage


def make_archive(stage: Path, target: str) -> Path:
    archive_kind = TARGETS[target]["archive"]
    if archive_kind == "zip":
        out = DIST_ROOT / f"{stage.name}.zip"
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in stage.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(stage.parent))
        return out

    out = DIST_ROOT / f"{stage.name}.tar.gz"
    if out.exists():
        out.unlink()
    def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
        if info.isfile() and (info.name.endswith(".sh") or info.name.endswith("/install-linux.sh")):
            info.mode = 0o755
        elif info.isdir():
            info.mode = 0o755
        return info

    with tarfile.open(out, "w:gz") as tf:
        tf.add(stage, arcname=stage.name, filter=tar_filter)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GitHub Release-style EdgeAI-DeployKit archives.")
    parser.add_argument("--target", choices=["current", "all", *TARGETS.keys()], default="current")
    parser.add_argument("--no-clean", action="store_true", help="Keep existing staged directories before copying.")
    parser.add_argument("--no-archive", action="store_true", help="Only stage files, do not create zip/tar.gz archives.")
    args = parser.parse_args()

    targets = list(TARGETS) if args.target == "all" else [current_target() if args.target == "current" else args.target]
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    for target in targets:
        stage = stage_target(target, clean=not args.no_clean)
        if args.no_archive:
            print(f"staged {target}: {stage}")
        else:
            archive = make_archive(stage, target)
            print(f"built {target}: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
