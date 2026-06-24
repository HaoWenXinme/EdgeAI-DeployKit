# EdgeAI-DeployKit

EdgeAI-DeployKit is a local AI model deployment toolkit with a WebUI, a FastAPI
backend, ONNX Runtime local inference, task-aware result rendering, and
Markdown/PDF report export.

The goal is simple: a user uploads a trained model, the toolkit converts or
imports it into a local ONNX package, detects the task type, asks for the right
test input, runs local inference, and shows a useful result.

## Current Status

The Linux local inference path is the primary validated path. Windows packaging
is now available as a portable zip with a one-click starter.

- Linux x86_64 release package: validated on openEuler 24.03 in VMware.
- Linux arm64 release package: package is generated; runtime validation should
  be done on an arm64 Linux VM or device.
- Windows x86_64 release package: zip package with `start-windows.bat` and
  `stop-windows.bat`; runtime validation is in progress.
- macOS packages: release naming and launchers are scaffolded, but full runtime
  validation is still pending.

Validated model paths:

- ONNX import -> local package -> local inference -> task result -> Markdown/PDF report.
- PyTorch `.pth` / state_dict -> ONNX package on the development Linux env.
- TensorFlow/Keras H5 -> fallback load -> ONNX package on the development Linux env.

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

Windows prerequisites: Python 3.9-3.13 and Node.js 20+ with Corepack. Python
3.10-3.12 is recommended for best AI package compatibility.

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
