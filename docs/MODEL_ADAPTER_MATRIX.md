# EdgeAI-DeployKit Model Adapter Matrix

This document defines the model formats accepted by the WebUI/CLI upload and
local deployment flow.

## Upload Formats

| Source type | Extensions / layout | Package output | Runtime |
|---|---|---|---|
| ONNX | `.onnx` | `model.onnx` | ONNX Runtime CPU |
| PyTorch executable / TorchScript | `.pt`, `.pth`, `.ckpt` | `model.onnx` | ONNX Runtime CPU |
| PyTorch state_dict | `.pt`, `.pth`, `.ckpt` plus known `torchvision:*` arch | `model.onnx` | ONNX Runtime CPU |
| TensorFlow / Keras | SavedModel directory, `.h5`, `.hdf5`, `.keras`, `.pb`, `.tflite` | `model.onnx` | ONNX Runtime CPU |
| Scikit-Learn | `.pkl`, `.joblib`, `.sav` | `model.onnx` | ONNX Runtime CPU |
| XGBoost | `.bst`, `.xgb`, `.json` | `model.onnx` | ONNX Runtime CPU |
| LightGBM | `.lgb`, `.txt` | `model.onnx` | ONNX Runtime CPU |
| LLM GGUF | `.gguf` | `model.gguf` | llama.cpp / llama-cpp-python |
| Directory models | `.zip` containing SavedModel or HuggingFace/GGUF layout | extracted directory | selected by detector |

## Dependency Switches

Base install supports ONNX import and ONNX Runtime inference.

Linux:

```bash
scripts/install-linux.sh --with-pytorch --with-tensorflow --with-ml --with-llm
```

Windows:

```powershell
start-windows.bat -WithPytorch -WithTensorflow -WithML -WithLLM
```

Use only the switches needed for your model type.

## Flow

The unified setup flow is:

```text
upload/provide model -> local-model-setup -> model_task.json -> local-inference-flow
```

For ONNX-compatible models, `local-model-setup` converts/imports to
`outputs/packages/<name>/model.onnx`, analyzes the graph, and creates task
guidance for digit classification, image classification, detection,
segmentation, text classification, or unknown outputs.

For GGUF LLM models, `local-model-setup` creates an LLM package with
`model.gguf`, `llm_runtime.json`, pseudo signature metadata, and an `llm_chat`
task. Chat inference requires either `llama-cpp-python` or a system `llama-cli`
binary.

