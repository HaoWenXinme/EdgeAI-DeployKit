from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from backend.schemas import DashboardResponse, JobCreateRequest, JobRecord
from backend.services.artifacts import list_artifacts, read_matrix
from backend.services.jobs import create_job, get_job, list_jobs, read_job_log
from backend.services.models import list_models, save_upload
from backend.services.infer_results import get_infer_result, list_infer_results, resolve_public_file
from backend.services.runtime import runtime_health

router = APIRouter()


@router.get("/health")
def health():
    return runtime_health()


@router.get("/models")
def models():
    return list_models()


@router.get("/matrix")
def matrix():
    return read_matrix()


@router.get("/artifacts")
def artifacts():
    return list_artifacts()


@router.get("/jobs", response_model=list[JobRecord])
def jobs():
    return list_jobs()


@router.post("/jobs", response_model=JobRecord)
def launch_job(payload: JobCreateRequest):
    return create_job(payload)


@router.get("/jobs/{job_id}", response_model=JobRecord)
def job_detail(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
def job_logs(job_id: str):
    return read_job_log(job_id)


@router.post("/uploads/{kind}")
def upload(kind: str, file: UploadFile = File(...)):
    if kind not in {"model", "image", "json", "input"}:
        raise HTTPException(status_code=400, detail="kind must be one of model/image/json/input")
    path = save_upload(kind, file)
    return {"path": path, "name": Path(path).name}


@router.get("/infer-results")
def infer_results():
    return list_infer_results()


@router.get("/infer-result/{model_name}")
def infer_result(model_name: str):
    return get_infer_result(model_name)


# -----------------------------------------------------------------------------
# Force-refresh package-local report endpoints
# Source of truth:
#   Markdown: outputs/packages/<package>/report.md
#   PDF:      outputs/packages/<package>/report.pdf
# These endpoints never fall back to reports/edgeai_report.pdf.
# -----------------------------------------------------------------------------

def _edgeai_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_local_report_name(model_name: str) -> str:
    if not model_name or "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail="invalid package/model name")
    return model_name


def _package_report_paths(model_name: str):
    project_root = _edgeai_project_root()
    package_dir = project_root / "outputs" / "packages" / model_name
    return project_root, package_dir, package_dir / "report.md", package_dir / "report.pdf"


def _rel_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except Exception:
        return str(path)


def _generate_package_pdf_from_md(model_name: str, force: bool = False) -> Path:
    project_root, package_dir, md_path, pdf_path = _package_report_paths(model_name)
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"package report Markdown not found: outputs/packages/{model_name}/report.md")

    must_generate = force or (not pdf_path.exists()) or pdf_path.stat().st_size <= 0 or (pdf_path.stat().st_mtime < md_path.stat().st_mtime)
    if must_generate:
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            pass
        try:
            from edgeai.report import markdown_to_pdf
            candidate = Path(markdown_to_pdf(md_path=md_path, pdf_path=pdf_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to generate package PDF: {exc}")

        # If the converter returned another path, copy it back to package/report.pdf.
        try:
            if candidate.exists() and candidate.resolve() != pdf_path.resolve():
                import shutil
                shutil.copy2(candidate, pdf_path)
        except Exception:
            pass

    if not pdf_path.exists() or pdf_path.stat().st_size <= 0:
        raise HTTPException(status_code=500, detail="PDF was not created. Install weasyprint or pandoc, then run edgeai report again.")
    return pdf_path


@router.get("/local-reports")
def local_reports():
    """List package-local reports. Source of truth: outputs/packages/<package>/report.md."""
    from datetime import datetime

    project_root = _edgeai_project_root()
    packages = project_root / "outputs" / "packages"
    items = []

    if packages.exists():
        for md_path in sorted(packages.glob("*/report.md")):
            model_name = md_path.parent.name
            try:
                md_stat = md_path.stat()
            except Exception:
                continue

            pdf_path = md_path.parent / "report.pdf"
            pdf_exists = pdf_path.exists() and pdf_path.is_file() and pdf_path.stat().st_size > 0
            pdf_stat = pdf_path.stat() if pdf_exists else None

            items.append({
                "model_name": model_name,
                "report_path": _rel_to_project(md_path, project_root),
                "pdf_path": _rel_to_project(pdf_path, project_root) if pdf_exists else None,
                "has_pdf": bool(pdf_exists),
                "source": "package",
                "size_bytes": md_stat.st_size,
                "pdf_size_bytes": pdf_stat.st_size if pdf_stat else None,
                "modified_time": md_stat.st_mtime,
                "pdf_modified_time": pdf_stat.st_mtime if pdf_stat else None,
                "modified_at": datetime.fromtimestamp(md_stat.st_mtime).isoformat(timespec="seconds"),
            })

    return sorted(items, key=lambda item: item["modified_time"], reverse=True)


@router.get("/local-reports/{model_name}", response_class=PlainTextResponse)
def local_report_content(model_name: str, source: str = "package", t: str = ""):
    """Read package Markdown report from outputs/packages/<package>/report.md only."""
    model_name = _safe_local_report_name(model_name)
    _project_root, _package_dir, md_path, _pdf_path = _package_report_paths(model_name)
    if md_path.exists() and md_path.is_file():
        return md_path.read_text(encoding="utf-8", errors="replace")
    raise HTTPException(status_code=404, detail=f"package report not found: outputs/packages/{model_name}/report.md")


@router.get("/local-reports/{model_name}/pdf")
def local_report_pdf(model_name: str, source: str = "package", refresh: int = 0, t: str = ""):
    """Return package-local PDF. refresh=1 deletes/regenerates PDF from current Markdown."""
    from fastapi.responses import FileResponse

    model_name = _safe_local_report_name(model_name)
    pdf_path = _generate_package_pdf_from_md(model_name, force=bool(refresh))
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Disposition": f'inline; filename="{model_name}_local_report.pdf"',
        "X-EdgeAI-Local-Report-Package": model_name,
        "X-EdgeAI-Local-Report-Path": str(pdf_path),
        "X-EdgeAI-Local-Report-Refresh": str(int(bool(refresh))),
    }
    return FileResponse(path=str(pdf_path), media_type="application/pdf", headers=headers)


# ---------------------------------------------------------------------
# Current Run Report API
# ---------------------------------------------------------------------
# mode=local:  read outputs/packages/<package_name>/report.md/pdf
# mode=board:  read reports/edgeai_report.md/pdf
# This avoids mixing local-run reports with old global board reports.


def _edgeai_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_package_name(name: str) -> str:
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "")).strip("_")
    if not cleaned:
        raise HTTPException(status_code=400, detail="package_name is required for local report")
    if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise HTTPException(status_code=400, detail="invalid package_name")
    return cleaned


def _latest_local_report_package(project_root: Path) -> str | None:
    packages = project_root / "outputs" / "packages"
    if not packages.exists():
        return None
    reports = [p for p in packages.glob("*/report.md") if p.is_file()]
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime).parent.name


def _current_run_paths(mode: str, package_name: str | None):
    project_root = _edgeai_project_root()
    mode = (mode or "auto").lower()

    if mode == "local" or (mode == "auto" and package_name):
        package = _safe_package_name(package_name or _latest_local_report_package(project_root) or "")
        base = project_root / "outputs" / "packages" / package
        return "local", package, base / "report.md", base / "report.pdf"

    if mode == "auto":
        latest = _latest_local_report_package(project_root)
        if latest:
            base = project_root / "outputs" / "packages" / latest
            return "local", latest, base / "report.md", base / "report.pdf"

    # Board/global report fallback is only used in board mode.
    return "board", "board", project_root / "reports" / "edgeai_report.md", project_root / "reports" / "edgeai_report.pdf"


@router.get("/current-run-report", response_class=PlainTextResponse)
def current_run_report_content(mode: str = "auto", package_name: str | None = None):
    resolved_mode, name, md_path, _pdf_path = _current_run_paths(mode, package_name)
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"current {resolved_mode} report markdown not found: {md_path}")
    return md_path.read_text(encoding="utf-8", errors="replace")


@router.get("/current-run-report/pdf")
def current_run_report_pdf(mode: str = "auto", package_name: str | None = None, t: str | None = None):
    resolved_mode, name, md_path, pdf_path = _current_run_paths(mode, package_name)

    if resolved_mode == "local":
        # Local report PDF must come from outputs/packages/<package>/report.pdf.
        # Do not fall back to reports/edgeai_report.pdf.
        if not pdf_path.exists() and md_path.exists():
            # Generate PDF from the package markdown through edgeai report.
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "edgeai.cli", "report", "--package", str(md_path.parent)],
                cwd=str(_edgeai_project_root()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if not pdf_path.exists() or not pdf_path.is_file():
            raise HTTPException(status_code=404, detail=f"local package pdf not found: {pdf_path}")
        filename = f"{name}_current_run_report.pdf"
    else:
        if not pdf_path.exists() or not pdf_path.is_file():
            raise HTTPException(status_code=404, detail=f"board report pdf not found: {pdf_path}")
        filename = "board_current_run_report.pdf"

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-EdgeAI-Current-Run-Mode": resolved_mode,
        "X-EdgeAI-Current-Run-Name": name,
    }
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        headers=headers,
    )


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard():
    return DashboardResponse(
        health=runtime_health(),
        models=list_models(),
        matrix=read_matrix(),
        artifacts=list_artifacts(),
        jobs=list_jobs(),
    )


# ---- WebUI public artifact/PDF file serving ----
def _edgeai_resolve_public_file(path: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    raw = Path(path)
    candidate = raw if raw.is_absolute() else project_root / raw
    resolved = candidate.resolve()

    allowed_roots = [
        project_root / "outputs",
        project_root / "reports",
        project_root / "inputs",
        project_root / "photo",
    ]

    if not any(resolved == root.resolve() or root.resolve() in resolved.parents for root in allowed_roots):
        raise HTTPException(status_code=400, detail="file path is outside public artifact roots")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return resolved


def _edgeai_public_file_response(path: str, download: bool = False):
    try:
        resolver = resolve_public_file  # type: ignore[name-defined]
    except NameError:
        resolver = _edgeai_resolve_public_file

    resolved = Path(resolver(path))
    if download:
        return FileResponse(resolved, filename=resolved.name)
    return FileResponse(resolved)


@router.get("/files/{path:path}")
def public_file(path: str, download: bool = False):
    return _edgeai_public_file_response(path, download)


@router.head("/files/{path:path}")
def public_file_head(path: str):
    return _edgeai_public_file_response(path, False)

# ---- EdgeAI package-local report API v2 --------------------------------------
# New package-scoped local report endpoints.
# Source of truth:
#   Markdown: outputs/packages/<package>/report.md
#   PDF:      regenerated from the same Markdown on every PDF request.
# This block never reads reports/edgeai_report.pdf, avoiding stale YOLO report cache.


def _edgeai_pkg_report_root_v2():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def _edgeai_pkg_report_safe_name_v2(model_name: str) -> str:
    from fastapi import HTTPException
    if not model_name or "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail="invalid package name")
    return model_name


def _edgeai_pkg_report_paths_v2(model_name: str):
    project_root = _edgeai_pkg_report_root_v2()
    package_dir = project_root / "outputs" / "packages" / model_name
    md_path = package_dir / "report.md"
    pdf_path = package_dir / "report.pdf"
    return project_root, package_dir, md_path, pdf_path


def _edgeai_pkg_report_rel_v2(path, root) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)


def _edgeai_pkg_report_escape_v2(value) -> str:
    import html
    return html.escape(str(value), quote=True)


def _edgeai_pkg_report_resolve_img_v2(src: str, md_path):
    """Resolve Markdown image paths relative to package report.md or project root."""
    from pathlib import Path
    src = src.strip()
    if src.startswith("file://") or src.startswith("http://") or src.startswith("https://"):
        return src
    project_root = _edgeai_pkg_report_root_v2()
    base = md_path.parent.resolve()
    candidates = []
    candidates.append((base / src).resolve())
    candidates.append((project_root / src).resolve())
    # Some reports use package-local filenames without ./ prefix.
    candidates.append((base / Path(src).name).resolve())
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.as_uri()
        except Exception:
            pass
    return None


def _edgeai_pkg_report_inline_v2(text: str) -> str:
    import re
    text = _edgeai_pkg_report_escape_v2(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def _edgeai_pkg_report_markdown_to_html_v2(md_path) -> str:
    """Small Markdown-to-HTML converter for package reports.

    Supports headings, blockquotes, tables, lists, fenced code blocks, inline code,
    and Markdown images. It is intentionally dependency-light so the endpoint is stable.
    """
    import re
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out = []
    in_table = False
    in_ul = False
    in_code = False

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def inline(s: str) -> str:
        return _edgeai_pkg_report_inline_v2(s)

    for raw in lines:
        stripped = raw.strip()

        if stripped.startswith("```"):
            close_table(); close_ul()
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            continue

        if in_code:
            out.append(_edgeai_pkg_report_escape_v2(raw) + "\n")
            continue

        if not stripped:
            close_table(); close_ul()
            out.append("<br>")
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            close_table(); close_ul()
            alt = image_match.group(1)
            src = image_match.group(2)
            uri = _edgeai_pkg_report_resolve_img_v2(src, md_path)
            if uri:
                out.append(f'<figure><img src="{_edgeai_pkg_report_escape_v2(uri)}" alt="{_edgeai_pkg_report_escape_v2(alt)}"><figcaption>{_edgeai_pkg_report_escape_v2(alt)}</figcaption></figure>')
            else:
                out.append(f'<p class="warn">Image not found: {_edgeai_pkg_report_escape_v2(src)}</p>')
            continue

        if stripped.startswith("# "):
            close_table(); close_ul(); out.append(f"<h1>{inline(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            close_table(); close_ul(); out.append(f"<h2>{inline(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            close_table(); close_ul(); out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith(">"):
            close_table(); close_ul(); out.append(f"<blockquote>{inline(stripped[1:].strip())}</blockquote>")
        elif stripped.startswith("- "):
            close_table()
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
        elif stripped.startswith("|") and stripped.endswith("|"):
            close_ul()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            if not in_table:
                out.append("<table><tbody>"); in_table = True
            out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
        elif stripped == "---":
            close_table(); close_ul(); out.append("<hr>")
        else:
            close_table(); close_ul(); out.append(f"<p>{inline(stripped)}</p>")

    close_table(); close_ul()
    if in_code:
        out.append("</code></pre>")

    body = "\n".join(out)
    css = """
@page { size: A4; margin: 18mm; }
body { font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "DejaVu Sans", sans-serif; color:#111827; line-height:1.55; font-size:11px; }
h1 { font-size:24px; margin:0 0 16px; color:#0f172a; border-bottom:2px solid #0f172a; padding-bottom:8px; }
h2 { font-size:17px; margin-top:24px; color:#111827; border-bottom:1px solid #e5e7eb; padding-bottom:4px; }
h3 { font-size:13px; margin-top:16px; color:#1f2937; }
blockquote { margin:8px 0; padding:6px 10px; background:#f8fafc; border-left:4px solid #38bdf8; color:#334155; }
table { width:100%; border-collapse:collapse; margin:8px 0 12px; table-layout:fixed; }
td, th { border:1px solid #d1d5db; padding:5px 7px; word-break:break-all; vertical-align:top; }
tr:first-child td { font-weight:700; background:#f3f4f6; }
code { background:#f1f5f9; padding:1px 4px; border-radius:3px; }
pre { background:#0f172a; color:#e2e8f0; padding:10px; border-radius:6px; white-space:pre-wrap; }
figure { margin:12px 0; page-break-inside:avoid; }
img { display:block; max-width:100%; max-height:360px; object-fit:contain; border:1px solid #e5e7eb; margin:0 auto; }
figcaption { text-align:center; color:#64748b; font-size:10px; margin-top:4px; }
.warn { color:#b45309; background:#fffbeb; padding:6px 8px; }
"""
    return "<!doctype html><html><head><meta charset=\"utf-8\"><style>" + css + "</style></head><body>" + body + "</body></html>"


def _edgeai_pkg_report_generate_pdf_v2(model_name: str):
    from fastapi import HTTPException
    project_root, package_dir, md_path, pdf_path = _edgeai_pkg_report_paths_v2(model_name)
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"package Markdown report not found: outputs/packages/{model_name}/report.md")

    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception:
        pass

    html_text = _edgeai_pkg_report_markdown_to_html_v2(md_path)
    try:
        from weasyprint import HTML as WeasyHTML
        WeasyHTML(string=html_text, base_url=str(md_path.parent.resolve())).write_pdf(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to generate package PDF from {md_path}: {exc}")

    if not pdf_path.exists() or pdf_path.stat().st_size <= 0:
        raise HTTPException(status_code=500, detail=f"PDF was not created: {pdf_path}")

    try:
        import shutil
        copy_path = project_root / "reports" / f"{model_name}_local_report.pdf"
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, copy_path)
    except Exception:
        pass

    return pdf_path


@router.get("/package-local-reports")
def edgeai_package_local_reports_v2(t: str = ""):
    from datetime import datetime
    project_root = _edgeai_pkg_report_root_v2()
    packages_dir = project_root / "outputs" / "packages"
    items = []
    if packages_dir.exists():
        for md_path in sorted(packages_dir.glob("*/report.md")):
            model_name = md_path.parent.name
            try:
                md_stat = md_path.stat()
                pdf_path = md_path.parent / "report.pdf"
                pdf_exists = pdf_path.exists() and pdf_path.is_file() and pdf_path.stat().st_size > 0
                pdf_stat = pdf_path.stat() if pdf_exists else None
                items.append({
                    "model_name": model_name,
                    "report_path": _edgeai_pkg_report_rel_v2(md_path, project_root),
                    "pdf_path": _edgeai_pkg_report_rel_v2(pdf_path, project_root) if pdf_exists else None,
                    "has_pdf": bool(pdf_exists),
                    "source": "package-local-v2",
                    "size_bytes": md_stat.st_size,
                    "pdf_size_bytes": pdf_stat.st_size if pdf_stat else None,
                    "modified_time": md_stat.st_mtime,
                    "pdf_modified_time": pdf_stat.st_mtime if pdf_stat else None,
                    "modified_at": datetime.fromtimestamp(md_stat.st_mtime).isoformat(timespec="seconds"),
                })
            except Exception:
                continue
    return sorted(items, key=lambda x: x.get("modified_time") or 0, reverse=True)


@router.get("/package-local-reports/{model_name}")
def edgeai_package_local_report_markdown_v2(model_name: str, t: str = ""):
    from fastapi import HTTPException
    from fastapi.responses import PlainTextResponse
    model_name = _edgeai_pkg_report_safe_name_v2(model_name)
    _root, _pkg, md_path, _pdf = _edgeai_pkg_report_paths_v2(model_name)
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"package Markdown report not found: outputs/packages/{model_name}/report.md")
    return PlainTextResponse(md_path.read_text(encoding="utf-8", errors="replace"), media_type="text/markdown; charset=utf-8")


@router.get("/package-local-reports/{model_name}/pdf")
def edgeai_package_local_report_pdf_v2(model_name: str, t: str = ""):
    from fastapi.responses import FileResponse
    model_name = _edgeai_pkg_report_safe_name_v2(model_name)
    pdf_path = _edgeai_pkg_report_generate_pdf_v2(model_name)
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Disposition": f'inline; filename="{model_name}_package_report.pdf"',
        "X-EdgeAI-Package-Report-API": "v2",
        "X-EdgeAI-Package-Report-Name": model_name,
        "X-EdgeAI-Package-Report-Path": str(pdf_path),
    }
    return FileResponse(str(pdf_path), media_type="application/pdf", headers=headers)

# ---- EdgeAI current-run report API v2 read-only PDF
# ---- EdgeAI current-run report API v2 read-only PDF --------------------------
# Source of truth:
#   local: outputs/packages/<package_name>/report.md + report.pdf
#   board: reports/edgeai_report.md + edgeai_report.pdf
# Markdown is read directly. PDF is served read-only by default. PDF generation
# is explicit through /pdf/build, so opening Reports does not create many files.


def _edgeai_current_report_root_v2():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def _edgeai_current_report_safe_package_v2(name: str | None) -> str:
    from fastapi import HTTPException
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "")).strip("_")
    if not cleaned:
        raise HTTPException(status_code=400, detail="package_name is required for local current-run report")
    if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise HTTPException(status_code=400, detail="invalid package_name")
    return cleaned


def _edgeai_current_latest_local_package_v2(project_root):
    packages = project_root / "outputs" / "packages"
    if not packages.exists():
        return None
    reports = [p for p in packages.glob("*/report.md") if p.is_file()]
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime).parent.name


def _edgeai_current_paths_v2(mode: str = "auto", package_name: str | None = None):
    from fastapi import HTTPException
    project_root = _edgeai_current_report_root_v2()
    mode = (mode or "auto").lower()

    if mode == "local" or (mode == "auto" and package_name):
        package = _edgeai_current_report_safe_package_v2(package_name or _edgeai_current_latest_local_package_v2(project_root))
        base = project_root / "outputs" / "packages" / package
        return "local", package, base / "report.md", base / "report.pdf"

    if mode == "auto":
        latest = _edgeai_current_latest_local_package_v2(project_root)
        if latest:
            base = project_root / "outputs" / "packages" / latest
            return "local", latest, base / "report.md", base / "report.pdf"

    if mode in {"board", "aipro", "orange", "orange-pi"}:
        return "board", "board", project_root / "reports" / "edgeai_report.md", project_root / "reports" / "edgeai_report.pdf"

    raise HTTPException(status_code=400, detail=f"unsupported current-run report mode: {mode}")


def _edgeai_current_html_escape_v2(value) -> str:
    import html
    return html.escape(str(value), quote=True)


def _edgeai_current_inline_md_v2(text: str) -> str:
    import re
    text = _edgeai_current_html_escape_v2(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def _edgeai_current_resolve_image_v2(src: str, md_path):
    from pathlib import Path
    src = src.strip()
    if src.startswith("file://") or src.startswith("http://") or src.startswith("https://"):
        return src
    project_root = _edgeai_current_report_root_v2()
    base = md_path.parent.resolve()
    candidates = [
        (base / src).resolve(),
        (project_root / src).resolve(),
        (base / Path(src).name).resolve(),
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.as_uri()
        except Exception:
            pass
    return None


def _edgeai_current_markdown_to_html_v2(md_path) -> str:
    import re
    lines = md_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    in_table = False
    in_ul = False
    in_code = False

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("```"):
            close_table(); close_ul()
            if not in_code:
                out.append("<pre><code>"); in_code = True
            else:
                out.append("</code></pre>"); in_code = False
            continue
        if in_code:
            out.append(_edgeai_current_html_escape_v2(raw) + "\n")
            continue
        if not stripped:
            close_table(); close_ul(); out.append("<br>"); continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            close_table(); close_ul()
            alt = image_match.group(1)
            src = image_match.group(2)
            uri = _edgeai_current_resolve_image_v2(src, md_path)
            if uri:
                out.append(f'<figure><img src="{_edgeai_current_html_escape_v2(uri)}" alt="{_edgeai_current_html_escape_v2(alt)}"><figcaption>{_edgeai_current_html_escape_v2(alt)}</figcaption></figure>')
            else:
                out.append(f'<p class="warn">Image not found: {_edgeai_current_html_escape_v2(src)}</p>')
            continue

        if stripped.startswith("# "):
            close_table(); close_ul(); out.append(f"<h1>{_edgeai_current_inline_md_v2(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            close_table(); close_ul(); out.append(f"<h2>{_edgeai_current_inline_md_v2(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            close_table(); close_ul(); out.append(f"<h3>{_edgeai_current_inline_md_v2(stripped[4:])}</h3>")
        elif stripped.startswith(">"):
            close_table(); close_ul(); out.append(f"<blockquote>{_edgeai_current_inline_md_v2(stripped[1:].strip())}</blockquote>")
        elif stripped.startswith("- "):
            close_table()
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{_edgeai_current_inline_md_v2(stripped[2:])}</li>")
        elif stripped.startswith("|") and stripped.endswith("|"):
            close_ul()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            if not in_table:
                out.append("<table><tbody>"); in_table = True
            out.append("<tr>" + "".join(f"<td>{_edgeai_current_inline_md_v2(c)}</td>" for c in cells) + "</tr>")
        elif stripped == "---":
            close_table(); close_ul(); out.append("<hr>")
        else:
            close_table(); close_ul(); out.append(f"<p>{_edgeai_current_inline_md_v2(stripped)}</p>")

    close_table(); close_ul()
    if in_code:
        out.append("</code></pre>")

    body = "\n".join(out)
    css = """
@page { size: A4; margin: 18mm; }
body { font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "DejaVu Sans", sans-serif; color:#111827; line-height:1.58; font-size:11px; }
h1 { font-size:24px; margin:0 0 16px; color:#0f172a; border-bottom:2px solid #0f172a; padding-bottom:8px; }
h2 { font-size:17px; margin-top:24px; color:#111827; border-bottom:1px solid #e5e7eb; padding-bottom:4px; }
h3 { font-size:13px; margin-top:16px; color:#1f2937; }
blockquote { margin:8px 0; padding:6px 10px; background:#f8fafc; border-left:4px solid #38bdf8; color:#334155; }
table { width:100%; border-collapse:collapse; margin:8px 0 12px; table-layout:fixed; }
td, th { border:1px solid #d1d5db; padding:5px 7px; word-break:break-all; vertical-align:top; }
tr:first-child td { font-weight:700; background:#f3f4f6; }
code { background:#f1f5f9; padding:1px 4px; border-radius:3px; }
pre { background:#0f172a; color:#e2e8f0; padding:10px; border-radius:6px; white-space:pre-wrap; }
figure { margin:12px 0; page-break-inside:avoid; }
img { display:block; max-width:100%; max-height:360px; object-fit:contain; border:1px solid #e5e7eb; margin:0 auto; }
figcaption { text-align:center; color:#64748b; font-size:10px; margin-top:4px; }
.warn { color:#b45309; background:#fffbeb; padding:6px 8px; }
"""
    return "<!doctype html><html><head><meta charset=\"utf-8\"><style>" + css + "</style></head><body>" + body + "</body></html>"


def _edgeai_current_generate_pdf_v2(md_path, pdf_path, force: bool = False):
    from fastapi import HTTPException
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"markdown report not found: {md_path}")
    if pdf_path.exists() and pdf_path.is_file() and pdf_path.stat().st_size > 0 and not force:
        try:
            if pdf_path.stat().st_mtime >= md_path.stat().st_mtime:
                return False
        except Exception:
            return False

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pdf_path.with_suffix(pdf_path.suffix + ".tmp")
    html_text = _edgeai_current_markdown_to_html_v2(md_path)
    try:
        from weasyprint import HTML as WeasyHTML
        WeasyHTML(string=html_text, base_url=str(md_path.parent.resolve())).write_pdf(str(tmp_path))
        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            raise RuntimeError("temporary PDF is empty")
        tmp_path.replace(pdf_path)
        return True
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"failed to generate PDF from {md_path}: {exc}")


@router.get("/current-run-report-v2")
def current_run_report_v2(mode: str = "auto", package_name: str | None = None, t: str = ""):
    from fastapi import HTTPException
    from fastapi.responses import PlainTextResponse
    resolved_mode, name, md_path, _pdf_path = _edgeai_current_paths_v2(mode, package_name)
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"current {resolved_mode} markdown report not found: {md_path}")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "X-EdgeAI-Current-Run-API": "v2-readonly",
        "X-EdgeAI-Current-Run-Mode": resolved_mode,
        "X-EdgeAI-Current-Run-Name": name,
    }
    return PlainTextResponse(md_path.read_text(encoding="utf-8", errors="replace"), media_type="text/markdown; charset=utf-8", headers=headers)


@router.get("/current-run-report-v2/pdf")
def current_run_report_pdf_v2(mode: str = "auto", package_name: str | None = None, t: str = ""):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    resolved_mode, name, _md_path, pdf_path = _edgeai_current_paths_v2(mode, package_name)
    if not pdf_path.exists() or not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        raise HTTPException(status_code=404, detail=f"current {resolved_mode} PDF report not found; click generate PDF first: {pdf_path}")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "Content-Disposition": f'inline; filename="{name}_current_run_report.pdf"',
        "X-EdgeAI-Current-Run-API": "v2-readonly",
        "X-EdgeAI-Current-Run-Mode": resolved_mode,
        "X-EdgeAI-Current-Run-Name": name,
        "X-EdgeAI-Current-Run-Path": str(pdf_path),
    }
    return FileResponse(str(pdf_path), media_type="application/pdf", headers=headers)


@router.post("/current-run-report-v2/pdf/build")
def current_run_report_pdf_build_v2(mode: str = "auto", package_name: str | None = None, force: str = "0"):
    from fastapi import HTTPException
    resolved_mode, name, md_path, pdf_path = _edgeai_current_paths_v2(mode, package_name)
    if resolved_mode == "board":
        if not pdf_path.exists() or not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
            raise HTTPException(status_code=404, detail=f"board PDF report not found: {pdf_path}")
        generated = False
    else:
        generated = _edgeai_current_generate_pdf_v2(md_path, pdf_path, force=(force == "1"))
    return {
        "ok": True,
        "mode": resolved_mode,
        "name": name,
        "generated": bool(generated),
        "markdown_path": str(md_path),
        "pdf_path": str(pdf_path),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else None,
    }

# === EdgeAI Current Run Report V3 exact-file endpoints ===
# Added to avoid older duplicate /current-run-report/pdf routes and stale global reports.
# These endpoints never generate/copy reports automatically. They only read the exact
# current run files:
#   local: outputs/packages/<package_name>/report.md + report.pdf
#   board: reports/edgeai_report.md + edgeai_report.pdf

def _edgeai_v3_project_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def _edgeai_v3_latest_local_package(project_root):
    packages_dir = project_root / "outputs" / "packages"
    if not packages_dir.exists():
        return None
    candidates = []
    for md in packages_dir.glob("*/report.md"):
        try:
            candidates.append((md.stat().st_mtime, md.parent.name))
        except OSError:
            pass
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _edgeai_v3_resolve_current_report(mode="auto", package_name=None):
    from fastapi import HTTPException
    project_root = _edgeai_v3_project_root()
    requested_mode = (mode or "auto").strip().lower()
    package = (package_name or "").strip()

    if requested_mode == "local" or package:
        if not package:
            package = _edgeai_v3_latest_local_package(project_root)
        if not package:
            raise HTTPException(status_code=404, detail="No local package report found")
        base = project_root / "outputs" / "packages" / package
        return "local", package, base / "report.md", base / "report.pdf"

    if requested_mode == "board":
        return "board", "board", project_root / "reports" / "edgeai_report.md", project_root / "reports" / "edgeai_report.pdf"

    latest = _edgeai_v3_latest_local_package(project_root)
    if latest:
        base = project_root / "outputs" / "packages" / latest
        return "local", latest, base / "report.md", base / "report.pdf"
    return "board", "board", project_root / "reports" / "edgeai_report.md", project_root / "reports" / "edgeai_report.pdf"


@router.get("/current-run-report-v3")
def edgeai_current_run_report_v3(mode: str = "auto", package_name: str | None = None, t: str = ""):
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    run_mode, name, md_path, pdf_path = _edgeai_v3_resolve_current_report(mode, package_name)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"current run markdown report not found: {md_path}")
    markdown = md_path.read_text(encoding="utf-8", errors="replace")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-EdgeAI-Current-Run-API": "v3",
        "X-EdgeAI-Report-Mode": run_mode,
        "X-EdgeAI-Report-Name": name,
        "X-EdgeAI-Report-MD-Path": str(md_path),
        "X-EdgeAI-Report-PDF-Path": str(pdf_path),
        "X-EdgeAI-Report-Force": str(int(force_rebuild)),
    }
    return JSONResponse({
        "ok": True,
        "api": "current-run-report-v3",
        "mode": run_mode,
        "package_name": name,
        "markdown": markdown,
        "report_md": str(md_path),
        "report_pdf": str(pdf_path),
        "pdf_exists": pdf_path.exists(),
    }, headers=headers)


@router.get("/current-run-report-v3/pdf")
def edgeai_current_run_report_pdf_v3(mode: str = "auto", package_name: str | None = None, t: str = ""):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    run_mode, name, md_path, pdf_path = _edgeai_v3_resolve_current_report(mode, package_name)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"current run markdown report not found: {md_path}")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"current run pdf report not found: {pdf_path}; run Local Report / Board Report first")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-EdgeAI-Current-Run-API": "v3",
        "X-EdgeAI-Report-Mode": run_mode,
        "X-EdgeAI-Report-Name": name,
        "X-EdgeAI-Report-MD-Path": str(md_path),
        "X-EdgeAI-Report-PDF-Path": str(pdf_path),
        "X-EdgeAI-Report-Force": "1" if str(force).lower() in {"1", "true", "yes", "force"} else "0",
        "X-EdgeAI-PDF-Renderer": "weasyprint-markdown-html-v4",
    }
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"{name}_current_run_report.pdf",
        headers=headers,
    )
# === End EdgeAI Current Run Report V3 exact-file endpoints ===

# -----------------------------------------------------------------------------
# EdgeAI Current Run Report PDF v4
# - local: outputs/packages/<package_name>/report.md + report.pdf
# - board: reports/edgeai_report.md + edgeai_report.pdf
# - PDF is rebuilt only when missing or older than markdown.
# - No local report copy is written into reports/*_local_report.*.
# -----------------------------------------------------------------------------
def _edgeai_current_run_v4_project_root():
    from pathlib import Path as _Path
    # backend/api/routes.py -> backend/api -> backend -> project root
    return _Path(__file__).resolve().parents[2]


def _edgeai_current_run_v4_latest_package(project_root):
    packages_dir = project_root / "outputs" / "packages"
    candidates = []
    if packages_dir.exists():
        for child in packages_dir.iterdir():
            md = child / "report.md"
            if child.is_dir() and md.exists():
                candidates.append((md.stat().st_mtime, child.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _edgeai_current_run_v4_resolve(mode="auto", package_name=None):
    from fastapi import HTTPException as _HTTPException
    project_root = _edgeai_current_run_v4_project_root()
    mode = (mode or "auto").strip().lower()
    package_name = (package_name or "").strip()

    if mode in {"local", "auto"}:
        pkg = package_name or _edgeai_current_run_v4_latest_package(project_root)
        if pkg:
            base = project_root / "outputs" / "packages" / pkg
            md = base / "report.md"
            pdf = base / "report.pdf"
            if md.exists() or pdf.exists():
                return "local", pkg, md, pdf
            if mode == "local":
                raise _HTTPException(status_code=404, detail=f"local package report not found: {pkg}")

    md = project_root / "reports" / "edgeai_report.md"
    pdf = project_root / "reports" / "edgeai_report.pdf"
    if mode == "board" or mode == "auto":
        return "board", "board", md, pdf

    raise _HTTPException(status_code=400, detail=f"unsupported current run mode: {mode}")


def _edgeai_current_run_v4_markdown_to_pdf(md_path, pdf_path):
    """Render current-run Markdown to PDF.

    Do not use the old raw-<pre> fallback. Prefer the package-local Markdown
    renderer already defined in this file because it supports headings, tables,
    blockquotes, fenced code, inline code and package-relative images such as
    source_input.png / local_topk_result.png.
    """
    from pathlib import Path as _Path
    from fastapi import HTTPException as _HTTPException

    md_path = _Path(md_path)
    pdf_path = _Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if not md_path.exists() or not md_path.is_file():
        raise _HTTPException(status_code=404, detail=f"markdown report not found: {md_path}")

    try:
        from weasyprint import HTML as _HTML
    except Exception as exc:
        raise RuntimeError(f"weasyprint is required to render Markdown PDF: {exc}")

    # Use the strongest renderer currently present in routes.py.
    try:
        html_text = _edgeai_pkg_report_markdown_to_html_v2(md_path)  # type: ignore[name-defined]
    except Exception:
        try:
            html_text = _edgeai_current_markdown_to_html_v2(md_path)  # type: ignore[name-defined]
        except Exception as exc:
            raise RuntimeError(f"failed to render Markdown to HTML: {exc}")

    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception:
        pass

    _HTML(string=html_text, base_url=str(md_path.parent.resolve())).write_pdf(str(pdf_path))

    if not pdf_path.exists() or pdf_path.stat().st_size <= 0:
        raise RuntimeError(f"PDF was not created: {pdf_path}")
    return str(pdf_path)


@router.get("/current-run-report-v4/pdf")
def edgeai_current_run_report_pdf_v4(mode: str = "auto", package_name: str | None = None, force: str = "0", refresh: str = "0", t: str = ""):
    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import FileResponse as _FileResponse

    run_mode, name, md_path, pdf_path = _edgeai_current_run_v4_resolve(mode, package_name)

    if not md_path.exists() and not pdf_path.exists():
        raise _HTTPException(status_code=404, detail=f"report not found: {md_path}")

    # Rebuild when force=1, or when PDF is missing/older than Markdown.
    # This is important for WebUI: report.md may be correct while report.pdf is stale.
    force_rebuild = str(force).lower() in {"1", "true", "yes", "y"} or str(refresh).lower() in {"1", "true", "yes", "y"}
    pdf_missing_or_empty = (not pdf_path.exists()) or (pdf_path.stat().st_size <= 0 if pdf_path.exists() else True)
    pdf_older_than_md = md_path.exists() and pdf_path.exists() and (pdf_path.stat().st_mtime < md_path.stat().st_mtime)
    if md_path.exists() and (force_rebuild or pdf_missing_or_empty or pdf_older_than_md):
        try:
            _edgeai_current_run_v4_markdown_to_pdf(md_path, pdf_path)
        except Exception as exc:
            raise _HTTPException(status_code=500, detail=str(exc))

    if not pdf_path.exists():
        raise _HTTPException(status_code=404, detail=f"report pdf not found: {pdf_path}")

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-EdgeAI-Current-Run-API": "v4",
        "X-EdgeAI-Report-Mode": run_mode,
        "X-EdgeAI-Report-Name": name,
        "X-EdgeAI-Report-MD-Path": str(md_path),
        "X-EdgeAI-Report-PDF-Path": str(pdf_path),
    }
    return _FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"{name}_current_run_report.pdf",
        headers=headers,
    )

# -----------------------------------------------------------------------------
# Smart Convert Wizard endpoint
# -----------------------------------------------------------------------------
@router.post("/convert/requirements")
def convert_requirements(payload: dict):
    """Return missing/suggested parameters before launching edgeai convert.

    Used by WebUI to show a small modal when conversion params are incomplete.
    """
    try:
        from edgeai.convert_model import inspect_conversion_requirements

        source_model = payload.get("source_model") or payload.get("model") or ""
        if not source_model:
            raise HTTPException(status_code=400, detail="source_model is required")
        return inspect_conversion_requirements(
            source_model=source_model,
            framework=str(payload.get("framework") or "auto"),
            opset=int(payload.get("opset") or 11),
            input_shape=payload.get("input_shape"),
            input_name=str(payload.get("input_name") or "input"),
            output_name=str(payload.get("output_name") or "output"),
            arch=payload.get("arch"),
            feature_count=payload.get("feature_count"),
            torchscript=bool(payload.get("torchscript") or False),
            dynamic_batch=bool(payload.get("dynamic_batch", True)),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

# ---------------------------------------------------------------------------
# EdgeAI Local Task System API
# ---------------------------------------------------------------------------
@router.get("/api/local-task")
def api_local_task(package_name: str, auto_create: bool = False):
    from pathlib import Path as _Path
    from backend.services.security import safe_package_name as _safe_package_name
    from edgeai.task_system import read_model_task
    pkg = _safe_package_name(package_name)
    return read_model_task(_Path("outputs") / "packages" / pkg, auto_create=auto_create)

@router.post("/api/local-task/refresh")
def api_local_task_refresh(payload: dict):
    from pathlib import Path as _Path
    from backend.services.security import safe_package_name as _safe_package_name
    from edgeai.task_system import create_or_update_model_task
    package_name = _safe_package_name(str(payload.get("package_name") or payload.get("package") or ""))
    task_type = str(payload.get("task_type") or "auto")
    label_map = payload.get("label_map")
    label_language = str(payload.get("label_language") or "zh")
    return create_or_update_model_task(_Path("outputs") / "packages" / package_name, task_type=task_type, label_map=label_map, label_language=label_language)

# -----------------------------------------------------------------------------
# Local Task System API
# -----------------------------------------------------------------------------
@router.get("/local-task-config")
def local_task_config(package_name: str, auto_create: bool = False):
    """Read outputs/packages/<package>/model_task.json for the WebUI task guidance panel."""
    import json
    import re
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(package_name or "").strip()).strip("_")
    if not clean:
        raise HTTPException(status_code=400, detail="package_name is required")

    package_dir = project_root / "outputs" / "packages" / clean
    task_file = package_dir / "model_task.json"

    if not task_file.exists():
        if auto_create:
            signature_file = package_dir / "model_signature.json"
            if signature_file.exists():
                try:
                    from edgeai.task_system import create_or_update_model_task
                    created = create_or_update_model_task(package_dir, task_type="auto")
                    data = created.get("config") if isinstance(created, dict) else None
                    if isinstance(data, dict):
                        data.setdefault("ok", True)
                        data.setdefault("package_name", clean)
                        data.setdefault("package_dir", str(package_dir.relative_to(project_root)))
                        data.setdefault("task_file", str(task_file.relative_to(project_root)))
                        data.setdefault("auto_created", True)
                        return data
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"自动生成任务配置失败：{exc}") from exc
            raise HTTPException(
                status_code=404,
                detail=f"尚未生成任务配置：{clean}/model_task.json。请等待本地模型初始化任务完成，或先运行 Analyze 后自动生成。",
            )
        raise HTTPException(
            status_code=404,
            detail=f"未找到任务配置：{clean}/model_task.json。请先运行 edgeai task-init --package outputs/packages/{clean} --task-type auto",
        )

    try:
        data = json.loads(task_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取任务配置失败：{exc}") from exc

    if isinstance(data, dict):
        data.setdefault("ok", True)
        data.setdefault("package_name", clean)
        data.setdefault("package_dir", str(package_dir.relative_to(project_root)))
        data.setdefault("task_file", str(task_file.relative_to(project_root)))
    return data



# EDGEAI_LOCAL_TASK_RESULT_ENDPOINT
@router.get("/local-task-result")
def local_task_result(package_name: str, force: bool = False):
    """Return task-aware inference result for the current local package."""
    from pathlib import Path
    from fastapi import HTTPException
    from edgeai.task_result import render_task_result

    root = Path(__file__).resolve().parents[2]
    safe_name = package_name.strip().replace("\\", "/").split("/")[-1]
    if not safe_name:
        raise HTTPException(status_code=400, detail="package_name is required")
    package_dir = root / "outputs" / "packages" / safe_name
    if not package_dir.exists():
        raise HTTPException(status_code=404, detail=f"未找到本地软件包：{safe_name}")
    try:
        return render_task_result(package_dir, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成任务结果失败：{exc}")


# EDGEAI_LOCAL_INFERENCE_FLOW_V1
@router.post("/local-inference-flow")
def local_inference_flow(payload: dict):
    """Run package-local inference loop in one request.

    Steps:
      optional Analyze / Task Init if missing
      Prepare Input -> Local Run -> Task Render -> Report
    """
    import json
    import os
    import re
    import subprocess
    import sys
    from pathlib import Path

    project_root = _edgeai_project_root()

    def clean_package_name(value: object) -> str:
        raw = str(value or "").strip()
        raw = raw.replace("\\", "/").split("/")[-1]
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
        if not cleaned:
            raise HTTPException(status_code=400, detail="package_name is required")
        return cleaned

    def resolve_project_file(value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="input is required")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = project_root / p
        resolved = p.resolve()
        allowed_roots = [project_root, project_root / "inputs", project_root / "outputs", project_root / "photo", project_root / "models"]
        if not any(resolved == root.resolve() or root.resolve() in resolved.parents for root in allowed_roots if root.exists() or root == project_root):
            raise HTTPException(status_code=400, detail=f"input path is outside project: {raw}")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"input file not found: {raw}")
        return str(resolved)

    def edgeai_cmd(*args: str) -> list[str]:
        return [sys.executable, "-m", "edgeai.cli", *args]

    def run_stage(stage: str, command: list[str], timeout: int = 900) -> dict:
        env = os.environ.copy()
        venv_bin = project_root / ".venv" / "bin"
        if venv_bin.exists():
            env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
            env.setdefault("VIRTUAL_ENV", str(project_root / ".venv"))
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = proc.stdout or ""
        return {
            "stage": stage,
            "command": " ".join(command),
            "code": proc.returncode,
            "ok": proc.returncode == 0,
            "output": output[-12000:],
        }

    def run_python_stage(stage: str, func, *args, **kwargs) -> dict:
        try:
            result = func(*args, **kwargs)
            return {
                "stage": stage,
                "command": f"python:{getattr(func, '__module__', '')}.{getattr(func, '__name__', stage)}",
                "code": 0,
                "ok": True,
                "output": json.dumps(result, ensure_ascii=False, indent=2, default=str)[-12000:],
            }
        except Exception as exc:
            return {
                "stage": stage,
                "command": f"python:{getattr(func, '__module__', '')}.{getattr(func, '__name__', stage)}",
                "code": 1,
                "ok": False,
                "output": f"{type(exc).__name__}: {exc}",
            }

    package_name = clean_package_name(payload.get("package_name") or payload.get("package"))
    package_dir = project_root / "outputs" / "packages" / package_name
    if not package_dir.exists():
        raise HTTPException(status_code=404, detail=f"package not found: outputs/packages/{package_name}")
    from edgeai.llm_runner import is_llm_package
    is_llm = is_llm_package(package_dir)
    if not is_llm and not (package_dir / "model.onnx").exists():
        raise HTTPException(status_code=404, detail=f"model.onnx not found in package: outputs/packages/{package_name}")

    input_value = payload.get("input") or payload.get("input_path")
    llm_prompt = str(payload.get("prompt") or input_value or "").strip() if is_llm else None
    input_path = None if is_llm else (resolve_project_file(input_value) if input_value else None)
    stages: list[dict] = []

    def append_or_fail(stage: str, command: list[str]):
        result = run_stage(stage, command)
        stages.append(result)
        if not result.get("ok"):
            return False
        return True

    def append_python_or_fail(stage: str, func, *args, **kwargs):
        result = run_python_stage(stage, func, *args, **kwargs)
        stages.append(result)
        return bool(result.get("ok"))

    # Keep package self-healing: if user only converted model, generate analysis/task config automatically.
    if is_llm:
        stages.append({"stage": "analyze", "ok": True, "skipped": True, "output": "LLM package uses runtime metadata instead of ONNX graph analysis"})
    elif payload.get("force_analyze") or not (package_dir / "model_signature.json").exists() or not (package_dir / "operator_report.json").exists():
        if not append_or_fail("analyze", edgeai_cmd("analyze", "--package", str(package_dir))):
            return {"ok": False, "message": "Analyze failed", "package_name": package_name, "package_dir": str(package_dir), "input": input_path, "stages": stages}
    else:
        stages.append({"stage": "analyze", "ok": True, "skipped": True, "output": "model_signature.json/operator_report.json already exist"})

    if payload.get("force_task") or not (package_dir / "model_task.json").exists():
        if not append_or_fail("task-init", edgeai_cmd("task-init", "--package", str(package_dir), "--task-type", "auto")):
            return {"ok": False, "message": "Task init failed", "package_name": package_name, "package_dir": str(package_dir), "input": input_path, "stages": stages}
    else:
        stages.append({"stage": "task-init", "ok": True, "skipped": True, "output": "model_task.json already exists"})

    if is_llm:
        if not llm_prompt:
            return {"ok": False, "message": "No prompt provided for LLM package", "package_name": package_name, "package_dir": str(package_dir), "input": None, "stages": stages}
        (package_dir / "input.txt").write_text(llm_prompt + "\n", encoding="utf-8")
        stages.append({"stage": "prepare-input", "ok": True, "skipped": False, "output": "prompt saved to input.txt"})
    elif input_path:
        from edgeai.preprocess import prepare_package_input

        if not append_python_or_fail("prepare-input", prepare_package_input, package_dir, Path(input_path)):
            return {"ok": False, "message": "Prepare input failed", "package_name": package_name, "package_dir": str(package_dir), "input": input_path, "stages": stages}
    elif not (package_dir / "input.npy").exists():
        return {"ok": False, "message": "No input provided and input.npy does not exist", "package_name": package_name, "package_dir": str(package_dir), "input": None, "stages": stages}
    else:
        stages.append({"stage": "prepare-input", "ok": True, "skipped": True, "output": "input.npy already exists"})

    from edgeai.local_runner import run_local_package

    if not append_python_or_fail("local-run", run_local_package, package_dir, prompt=llm_prompt):
        return {"ok": False, "message": "Local run failed", "package_name": package_name, "package_dir": str(package_dir), "input": input_path, "stages": stages}

    # Task render is optional in older installs, but this project should have it. Fail clearly if it breaks.
    from edgeai.task_result import render_task_result

    if not append_python_or_fail("task-render", render_task_result, package_dir, force=True):
        return {"ok": False, "message": "Task render failed", "package_name": package_name, "package_dir": str(package_dir), "input": input_path, "stages": stages}

    from edgeai.local_report import generate_local_package_report

    if not append_python_or_fail("report", generate_local_package_report, package_dir):
        return {"ok": False, "message": "Report generation failed", "package_name": package_name, "package_dir": str(package_dir), "input": input_path, "stages": stages}

    artifacts = {
        "model.onnx": (package_dir / "model.onnx").exists(),
        "model.gguf": (package_dir / "model.gguf").exists(),
        "model_signature.json": (package_dir / "model_signature.json").exists(),
        "model_task.json": (package_dir / "model_task.json").exists(),
        "input.npy": (package_dir / "input.npy").exists(),
        "input.txt": (package_dir / "input.txt").exists(),
        "preprocess.json": (package_dir / "preprocess.json").exists(),
        "local_output.npy": (package_dir / "local_output.npy").exists(),
        "local_output.txt": (package_dir / "local_output.txt").exists(),
        "local_result.json": (package_dir / "local_result.json").exists(),
        "task_result.json": (package_dir / "task_result.json").exists(),
        "report.md": (package_dir / "report.md").exists(),
        "report.pdf": (package_dir / "report.pdf").exists(),
    }

    task_result = None
    task_result_path = package_dir / "task_result.json"
    if task_result_path.exists():
        try:
            task_result = json.loads(task_result_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            task_result = None

    required_artifacts = ["input.txt", "local_output.txt", "local_result.json", "task_result.json", "report.md", "report.pdf"] if is_llm else ["input.npy", "preprocess.json", "local_output.npy", "local_result.json", "task_result.json", "report.md", "report.pdf"]
    missing_artifacts = [name for name in required_artifacts if not artifacts.get(name)]
    if missing_artifacts:
        return {
            "ok": False,
            "message": f"local inference flow finished but missing artifact(s): {', '.join(missing_artifacts)}",
            "package_name": package_name,
            "package_dir": str(package_dir),
            "input": input_path,
            "stages": stages,
            "artifacts": artifacts,
            "task_result": task_result,
        }

    return {
        "ok": True,
        "message": "local inference flow completed: prepare-input -> local-run -> task-render -> report",
        "package_name": package_name,
        "package_dir": str(package_dir),
        "input": input_path,
        "stages": stages,
        "artifacts": artifacts,
        "task_result": task_result,
    }
