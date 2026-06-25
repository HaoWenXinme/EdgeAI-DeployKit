# EdgeAI-DeployKit

EdgeAI-DeployKit 是一个面向本地 AI 模型部署的跨平台工具包。用户可以通过
WebUI 上传自己训练好的模型，工具会自动识别模型类型、转换或导入为本地推理
package，按任务类型提示上传测试输入，并输出可视化结果与 Markdown/PDF 报告。

It is designed for local deployment workflows: upload a model, convert/import it
into an ONNX or GGUF package, detect the task, run local inference, render the
result, and export a report.

## Downloads

Latest preview release:

[v0.2.0-local-preview](https://github.com/HaoWenXinme/EdgeAI-DeployKit/releases/tag/v0.2.0-local-preview)

| Platform | Package | Notes |
| --- | --- | --- |
| Windows x86_64 | [EdgeAI-DeployKit-windows-x86_64.zip](https://github.com/HaoWenXinme/EdgeAI-DeployKit/releases/download/v0.2.0-local-preview/EdgeAI-DeployKit-windows-x86_64.zip) | Double-click `start-windows.bat`. If Python is missing, run `install-runtime-windows.bat`. |
| macOS Apple Silicon | [EdgeAI-DeployKit-macos-arm64.tar.gz](https://github.com/HaoWenXinme/EdgeAI-DeployKit/releases/download/v0.2.0-local-preview/EdgeAI-DeployKit-macos-arm64.tar.gz) | For M1/M2/M3/M4 Macs. |
| macOS Intel | [EdgeAI-DeployKit-macos-x86_64.tar.gz](https://github.com/HaoWenXinme/EdgeAI-DeployKit/releases/download/v0.2.0-local-preview/EdgeAI-DeployKit-macos-x86_64.tar.gz) | For Intel Macs. |
| Linux x86_64 | [EdgeAI-DeployKit-linux-x86_64.tar.gz](https://github.com/HaoWenXinme/EdgeAI-DeployKit/releases/download/v0.2.0-local-preview/EdgeAI-DeployKit-linux-x86_64.tar.gz) | Validated first on openEuler/Linux local inference flow. |
| Linux arm64 | [EdgeAI-DeployKit-linux-arm64.tar.gz](https://github.com/HaoWenXinme/EdgeAI-DeployKit/releases/download/v0.2.0-local-preview/EdgeAI-DeployKit-linux-arm64.tar.gz) | For ARM64 Linux VM/device validation. |

## What It Does

- Upload user-trained models from the WebUI.
- Detect model framework and task type.
- Convert/import ONNX, PyTorch, TensorFlow/Keras, traditional ML, and GGUF LLM packages where dependencies are available.
- Run local inference with ONNX Runtime CPU for ONNX packages.
- Render task-aware results: TopK classification, digit recognition, object detection previews, and LLM chat-style outputs.
- Export package-local `report.md`, `report.html`, and `report.pdf`.
- Package releases for Windows, macOS, Linux x86_64, and Linux arm64.

## Current Status

- Linux local inference flow: validated on openEuler 24.03 in VMware.
- Windows x86_64: one-click lightweight starter with runtime health checks and PDF preview.
- macOS x86_64 / arm64: release packages, install script, start/stop/status script, and WebUI launchers are available.
- Linux arm64: package is generated; runtime validation should be done on an ARM64 Linux VM or device.

Validated model paths:

- ONNX import -> local package -> local inference -> task result -> Markdown/PDF report.
- PyTorch `.pth` / state_dict -> ONNX package with model intelligence parameter suggestions.
- TensorFlow/Keras H5 -> fallback load -> ONNX package.

## Supported Tasks

EdgeAI-DeployKit uses model signature analysis to pick a task-oriented result
view:

- Image classification: upload an image, show TopK labels and scores.
- Digit classification: upload a digit image, show the predicted digit.
- Object detection / YOLO-like models: upload an image, show a result image with boxes when the model output is supported.
- Large language model style workflows: planned as a chat interaction path.

## Quick Start On Linux

Download the release package for your Linux architecture:

- `EdgeAI-DeployKit-linux-x86_64.tar.gz`
- `EdgeAI-DeployKit-linux-arm64.tar.gz`

Extract and install:

```bash
tar -xzf EdgeAI-DeployKit-linux-x86_64.tar.gz
cd EdgeAI-DeployKit-linux-x86_64
./install-linux.sh
```

If you need PyTorch `.pt` / `.pth` conversion support:

```bash
./install-linux.sh --with-pytorch
```

If you need TensorFlow/Keras `.h5`, `.keras`, or SavedModel conversion support:

```bash
./install-linux.sh --with-tensorflow
```

For both:

```bash
./install-linux.sh --with-pytorch --with-tensorflow
```

Start the WebUI and backend:

```bash
./start-linux.sh
```

For a VM where the host machine needs to open the WebUI:

```bash
./start-linux.sh --lan
```

Then open the printed `/workspace` URL.

Stop services:

```bash
./start-linux.sh stop
```

Check services:

```bash
./start-linux.sh status
```

## Quick Start On Windows

Download:

- `EdgeAI-DeployKit-windows-x86_64.zip`

Extract the zip, then double-click:

```text
start-windows.bat
```

The first run creates a local `.venv`, installs Python dependencies, installs
WebUI dependencies, starts the backend and WebUI, and opens:

```text
http://127.0.0.1:3000/workspace
```

To stop the local services, double-click:

```text
stop-windows.bat
```

Optional conversion dependencies can be installed from PowerShell:

```powershell
.\start-windows.bat -WithPytorch
.\start-windows.bat -WithTensorflow
```

Windows prerequisites: Python 3.9-3.13. Python 3.10-3.12 is recommended for
best AI package compatibility. Node.js is prepared automatically when missing
if the machine can download the portable Node runtime.

## Quick Start On macOS

Download the package for your Mac:

- `EdgeAI-DeployKit-macos-arm64.tar.gz` for Apple Silicon Macs.
- `EdgeAI-DeployKit-macos-x86_64.tar.gz` for Intel Macs.

Install prerequisites with Homebrew:

```bash
brew install python@3.12 node
corepack enable
```

Extract, install, and start:

```bash
tar -xzf EdgeAI-DeployKit-macos-arm64.tar.gz
cd EdgeAI-DeployKit-macos-arm64
./install-macos.sh
./start-macos.sh
```

Stop services:

```bash
./start-macos.sh stop
```

## WebUI Flow

1. Upload a model file or enter a server-side model path.
2. Click the convert/detect action.
3. Fill missing conversion parameters if the wizard asks for them.
4. Upload a test input, such as an image.
5. Run local inference.
6. View the task-aware result and download the report.

Generated package artifacts are written under:

```text
outputs/packages/<package_name>/
```

Important files include:

- `model.onnx`
- `model_signature.json`
- `model_task.json`
- `input.npy`
- `local_result.json`
- `task_result.json`
- `local_topk_result.png`
- `report.md`
- `report.pdf`

## Release Packages

Build release-style archives from the project root:

```bash
python release/build_release.py --target linux-x86_64
python release/build_release.py --target linux-arm64
```

Build all platform-named packages:

```bash
python release/build_release.py --target all
```

Artifacts are written to `release_dist/`.

The release builder intentionally excludes generated outputs, reports,
`node_modules`, `.next`, virtual environments, model weights, ONNX exports, and
backup files.

## Documentation

- Linux release guide: `docs/LINUX_LOCAL_RELEASE.md`
- Windows release guide: `docs/WINDOWS_LOCAL_RELEASE.md`
- macOS release guide: `docs/MACOS_LOCAL_RELEASE.md`
- Release packaging notes: `docs/RELEASE_PACKAGING.md`
- Local task system notes: `docs/LOCAL_TASK_SYSTEM_GUIDE.md`
- Conversion wizard notes: `docs/CONVERT_SMART_WIZARD_GUIDE.md`
- TensorFlow importer notes: `docs/TENSORFLOW_UNIVERSAL_IMPORTER.md`
- Model adapter matrix: `docs/MODEL_ADAPTER_MATRIX.md`

## Development Notes

The backend runs allowlisted jobs through the current Python runtime and uses:

```bash
python -m edgeai.cli
```

This keeps the release package self-contained around its `.venv` and avoids
depending on a globally installed `edgeai` command.
