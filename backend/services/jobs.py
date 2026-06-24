from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from backend.schemas import JobCreateRequest, JobRecord
from backend.services.paths import JOBS_DIR, PROJECT_ROOT, ensure_runtime_dirs, rel
from backend.services.security import build_command


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def meta_path(job_id: str) -> Path:
    return job_dir(job_id) / "meta.json"


def log_path(job_id: str) -> Path:
    return job_dir(job_id) / "stdout.log"


def write_job(job: JobRecord) -> None:
    job_dir(job.id).mkdir(parents=True, exist_ok=True)
    meta_path(job.id).write_text(job.model_dump_json(indent=2), encoding="utf-8")


def read_job(path: Path) -> JobRecord | None:
    try:
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_jobs() -> list[JobRecord]:
    ensure_runtime_dirs()
    jobs = [job for path in JOBS_DIR.glob("*/meta.json") if (job := read_job(path)) is not None]
    return sorted(jobs, key=lambda item: item.created_at, reverse=True)


def get_job(job_id: str) -> JobRecord | None:
    return read_job(meta_path(job_id)) if meta_path(job_id).exists() else None


def read_job_log(job_id: str) -> str:
    path = log_path(job_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")




# EDGEAI_JOB_ARTIFACT_VALIDATION: validate real output files before a job is marked success.
def _json_objects_from_text(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    out: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except Exception:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i += max(end, 1)
    return out


def _last_json_dict_from_log(path: Path) -> dict | None:
    try:
        objs = _json_objects_from_text(path.read_text(encoding="utf-8", errors="ignore"))
        return objs[-1] if objs else None
    except Exception:
        return None


def _command_option(command: list[str], *names: str) -> str | None:
    for idx, item in enumerate(command):
        for name in names:
            if item == name and idx + 1 < len(command):
                return command[idx + 1]
            prefix = name + "="
            if item.startswith(prefix):
                return item[len(prefix):]
    return None


def _package_dir_from_job(job: JobRecord, parsed: dict | None = None) -> Path | None:
    parsed = parsed or {}
    package_dir = parsed.get("package_dir")
    if package_dir:
        p = Path(str(package_dir))
        return p if p.is_absolute() else PROJECT_ROOT / p

    raw = _command_option(job.command, "--package", "--package-dir")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and (p.name == "outputs" or "outputs/packages" in str(p)):
        # Explicit package path.
        return p
    if raw.startswith("outputs/packages/"):
        return PROJECT_ROOT / raw
    # UI may pass either a package name or an accidental absolute path as package.
    name = p.name
    return PROJECT_ROOT / "outputs" / "packages" / name


def _contains_action(job: JobRecord, needle: str) -> bool:
    action = (job.action or "").lower()
    command = " ".join(job.command or []).lower()
    return needle in action or f" {needle}" in command or command.endswith(f" {needle}")


def _missing_files(base: Path, names: list[str]) -> list[str]:
    return [name for name in names if not (base / name).exists()]


def _validate_job_artifacts(job: JobRecord, path: Path) -> str | None:
    parsed = _last_json_dict_from_log(path)

    if parsed and parsed.get("ok") is False:
        err = parsed.get("error") or parsed.get("message") or "unknown conversion error"
        return f"command returned ok:false: {err}"

    pkg = _package_dir_from_job(job, parsed)

    # Local model setup must finish conversion, analyze, and task config generation.
    if _contains_action(job, "local-model-setup"):
        if pkg is None:
            return None
        is_llm = (pkg / "model.gguf").exists() or (pkg / "llm_runtime.json").exists()
        required = ["model_signature.json", "operator_report.json", "model_task.json"]
        required.insert(0, "model.gguf" if is_llm else "model.onnx")
        missing = _missing_files(pkg, required)
        if missing:
            return f"local-model-setup finished but missing artifact(s) in {pkg}: {', '.join(missing)}"
        return None

    # Convert must generate model.onnx.
    if _contains_action(job, "convert"):
        if pkg is None:
            return None
        if not (pkg / "model.onnx").exists() and not (pkg / "model.gguf").exists() and not (pkg / "llm_runtime.json").exists():
            return f"conversion finished but no runnable model artifact was generated in: {pkg}"
        return None

    # Later pipeline steps should not be green unless their expected artifact exists.
    checks: list[tuple[str, list[str]]] = [
        ("analyze", ["model_signature.json", "operator_report.json"]),
        ("prepare-input", ["input.npy", "preprocess.json"]),
        ("local-run", ["local_result.json"]),
        ("report", ["report.md", "report.pdf"]),
    ]
    for needle, files in checks:
        if _contains_action(job, needle):
            if pkg is None:
                return None
            missing = _missing_files(pkg, files)
            if missing:
                return f"{needle} finished but missing artifact(s) in {pkg}: {', '.join(missing)}"
            return None
    return None


def run_job(job: JobRecord) -> None:
    job.status = "running"
    job.started_at = now()
    write_job(job)
    lp = log_path(job.id)
    try:
        with lp.open("w", encoding="utf-8", errors="ignore") as log:
            log.write("$ " + " ".join(job.command) + "\n\n")
            log.flush()
            env = os.environ.copy()
            venv_bin = PROJECT_ROOT / ".venv" / "bin"
            if venv_bin.exists():
                env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
                env.setdefault("VIRTUAL_ENV", str(PROJECT_ROOT / ".venv"))
            env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.Popen(
                job.command,
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            code = proc.wait(timeout=1)
        job.code = code
        if code == 0:
            artifact_error = _validate_job_artifacts(job, lp)
            if artifact_error:
                job.status = "failed"
                job.code = 1
                job.error = artifact_error
                with lp.open("a", encoding="utf-8", errors="ignore") as log:
                    log.write(f"\n[ARTIFACT-VALIDATION-ERROR] {artifact_error}\n")
            else:
                job.status = "success"
        else:
            job.status = "failed"
    except subprocess.TimeoutExpired:
        job.status = "timeout"
        job.error = "command timed out"
    except FileNotFoundError as exc:
        job.status = "failed"
        job.code = 127
        job.error = str(exc)
        with lp.open("a", encoding="utf-8", errors="ignore") as log:
            log.write(f"\n[ERROR] {exc}\n")
    except Exception as exc:  # defensive guard for UI visibility
        job.status = "failed"
        job.error = str(exc)
        with lp.open("a", encoding="utf-8", errors="ignore") as log:
            log.write(f"\n[ERROR] {exc}\n")
    finally:
        job.finished_at = now()
        write_job(job)


def create_job(payload: JobCreateRequest) -> JobRecord:
    ensure_runtime_dirs()
    command = build_command(payload.action, payload.params)
    job_id = uuid.uuid4().hex[:12]
    job = JobRecord(
        id=job_id,
        action=payload.action,
        status="queued",
        command=command,
        created_at=now(),
        log_path=rel(log_path(job_id)),
    )
    write_job(job)
    thread = threading.Thread(target=run_job, args=(job,), daemon=True)
    thread.start()
    return job
