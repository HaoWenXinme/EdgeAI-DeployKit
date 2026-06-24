from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .local_runner import run_local_package
from .model_signature import analyze_package
from .package_layout import init_package
from .preprocess import prepare_package_input


def _print_json(data):
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


def register_local_run_commands(app: typer.Typer) -> None:
    @app.command("init-package")
    def init_package_cmd(
        name: str = typer.Option(..., "--name", help="Package/model name"),
        source_model: Path = typer.Option(..., "--source-model", help="User uploaded model path"),
        framework: str = typer.Option("onnx", "--framework", help="onnx/tensorflow/pytorch/paddle/sklearn/xgboost/lightgbm"),
        output_root: Path = typer.Option(Path("outputs/packages"), "--output-root", help="Package root directory"),
        overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite", help="Overwrite existing package"),
    ):
        """Create a standard user-model package directory."""
        _print_json(init_package(name, source_model, framework, output_root, overwrite))

    @app.command("analyze")
    def analyze_cmd(
        package: Path = typer.Option(..., "--package", help="Package directory containing model.onnx"),
    ):
        """Analyze package model.onnx and write model_signature.json/operator_report.json."""
        _print_json(analyze_package(package))

    @app.command("prepare-input")
    def prepare_input_foundation_cmd(
        package: Path = typer.Option(..., "--package", help="Package directory"),
        input_file: Path = typer.Option(..., "--input", "-i", help="Image file or .npy tensor"),
        preprocess: Optional[Path] = typer.Option(None, "--preprocess", help="Optional preprocess.json override"),
        force_analyze: bool = typer.Option(False, "--force-analyze/--no-force-analyze", help="Regenerate model_signature.json first"),
    ):
        """Prepare user input according to preprocess.json and create input.npy."""
        _print_json(prepare_package_input(package, input_file, preprocess, force_analyze))

    @app.command("local-run")
    def local_run_cmd(
        package: Path = typer.Option(..., "--package", help="Package directory"),
        repeat: int = typer.Option(1, "--repeat", "-r", help="Average latency over N runs"),
        warmup: int = typer.Option(1, "--warmup", help="Warmup runs before measurement"),
        prompt: Optional[str] = typer.Option(None, "--prompt", help="Prompt text for llm_chat packages"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max generated tokens for llm_chat packages"),
        temperature: Optional[float] = typer.Option(None, "--temperature", help="Sampling temperature for llm_chat packages"),
    ):
        """Run package model.onnx locally with ONNX Runtime CPUExecutionProvider."""
        if max_tokens is not None or temperature is not None:
            from .llm_runner import is_llm_package, run_llm_package
            if is_llm_package(package):
                _print_json(run_llm_package(package, prompt=prompt, max_tokens=max_tokens, temperature=temperature))
                return
        _print_json(run_local_package(package, repeat=repeat, warmup=warmup, prompt=prompt))

# ---------------------------------------------------------------------------
# EdgeAI Local Task System commands
# ---------------------------------------------------------------------------
# disabled broken top-level task-init decorator: @app.command("task-init")
def task_init_cmd(
    package: Path = typer.Option(..., "--package", help="Package name or outputs/packages/<name> path"),
    task_type: str = typer.Option("auto", "--task-type", help="auto/digit_classification/image_classification/object_detection/segmentation/text_classification/llm_chat"),
    label_map: Optional[Path] = typer.Option(None, "--label-map", help="Optional label map file"),
    label_language: str = typer.Option("zh", "--label-language", help="Label language, default zh"),
):
    from .task_system import create_or_update_model_task
    import json as _json
    result = create_or_update_model_task(package, task_type=task_type, label_map=label_map, label_language=label_language)
    print(_json.dumps(result, ensure_ascii=False, indent=2))

# disabled broken top-level task command: @app.command("task-info")
def task_info_cmd(
    package: Path = typer.Option(..., "--package", help="Package name or outputs/packages/<name> path"),
    auto_create: bool = typer.Option(False, "--auto-create/--no-auto-create", help="Create model_task.json when missing"),
):
    from .task_system import read_model_task
    import json as _json
    result = read_model_task(package, auto_create=auto_create)
    print(_json.dumps(result, ensure_ascii=False, indent=2))
