from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthItem(BaseModel):
    name: str
    command: str | None = None
    available: bool
    detail: str | None = None
    required: bool = True
    category: str = "core"


class HealthResponse(BaseModel):
    project_root: str
    outputs_dir: str
    reports_dir: str
    checks: list[HealthItem]


class ModelItem(BaseModel):
    name: str
    path: str
    type: str
    size_mb: float
    source: str
    modified_at: str | None = None


class ArtifactItem(BaseModel):
    name: str
    path: str
    kind: Literal["report", "package", "benchmark", "matrix", "other"]
    size_mb: float
    modified_at: str


class JobCreateRequest(BaseModel):
    action: str = Field(..., min_length=1)
    params: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class JobRecord(BaseModel):
    id: str
    action: str
    status: Literal["queued", "running", "success", "failed", "timeout", "cancelled"]
    command: list[str]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    code: int | None = None
    log_path: str | None = None
    error: str | None = None


class DashboardResponse(BaseModel):
    health: HealthResponse
    models: list[ModelItem]
    matrix: list[dict[str, Any]]
    artifacts: list[ArtifactItem]
    jobs: list[JobRecord]
