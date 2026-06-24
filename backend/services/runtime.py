from __future__ import annotations

import shutil
import sys
import importlib.util
from pathlib import Path

from backend.schemas import HealthItem, HealthResponse
from backend.services.paths import OUTPUTS_DIR, PROJECT_ROOT, REPORTS_DIR


def tool_check(name: str, command: str, *, required: bool = True, category: str = "core") -> HealthItem:
    found = shutil.which(command)
    return HealthItem(name=name, command=command, available=found is not None, detail=found, required=required, category=category)


def module_check(name: str, module: str, *, required: bool = True, category: str = "core") -> HealthItem:
    found = importlib.util.find_spec(module) is not None
    return HealthItem(name=name, command=f"python -c import {module}", available=found, detail=module, required=required, category=category)


def runtime_health() -> HealthResponse:
    has_corepack = shutil.which("corepack") is not None
    has_pnpm = shutil.which("pnpm") is not None
    checks = [
        HealthItem(name="Python runtime", command=sys.executable, available=Path(sys.executable).exists(), detail=sys.version.split()[0]),
        HealthItem(name="edgeai module", command=f"{sys.executable} -m edgeai.cli", available=(PROJECT_ROOT / "edgeai" / "cli.py").exists(), detail=str(PROJECT_ROOT / "edgeai" / "cli.py")),
        module_check("ONNX Runtime", "onnxruntime"),
        module_check("FastAPI", "fastapi"),
        HealthItem(
            name="WebUI package manager",
            command="pnpm or corepack",
            available=has_pnpm or has_corepack,
            detail=shutil.which("pnpm") or shutil.which("corepack"),
            required=True,
            category="core",
        ),
        tool_check("Node.js", "node", required=True, category="core"),
        tool_check("edgeai console script", "edgeai", required=False, category="cli"),
        tool_check("Python launcher", "python", required=False, category="cli"),
        tool_check("Python 3 launcher", "python3", required=False, category="cli"),
        module_check("PyTorch conversion", "torch", required=False, category="model-adapter"),
        module_check("TorchVision conversion", "torchvision", required=False, category="model-adapter"),
        module_check("TensorFlow conversion", "tensorflow", required=False, category="model-adapter"),
        module_check("tf2onnx conversion", "tf2onnx", required=False, category="model-adapter"),
        module_check("Traditional ML conversion", "skl2onnx", required=False, category="model-adapter"),
        module_check("LLM GGUF runtime", "llama_cpp", required=False, category="model-adapter"),
        tool_check("cmake", "cmake", required=False, category="native-build"),
        tool_check("make", "make", required=False, category="native-build"),
        tool_check("gcc", "gcc", required=False, category="native-build"),
        tool_check("g++", "g++", required=False, category="native-build"),
        tool_check("qemu-system-aarch64", "qemu-system-aarch64", required=False, category="board"),
        tool_check("atc", "atc", required=False, category="board"),
        tool_check("docker", "docker", required=False, category="board"),
    ]
    sdk = Path("/opt/openeuler-aarch64/environment-setup-aarch64-openeuler-linux")
    checks.append(HealthItem(name="openEuler aarch64 SDK", command=None, available=sdk.exists(), detail=str(sdk), required=False, category="board"))
    return HealthResponse(project_root=str(PROJECT_ROOT), outputs_dir=str(OUTPUTS_DIR), reports_dir=str(REPORTS_DIR), checks=checks)
