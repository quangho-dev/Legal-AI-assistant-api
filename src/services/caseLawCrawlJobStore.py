from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis

from src.config.index import appConfig

JOB_TTL_SECONDS = 6 * 60 * 60
JOB_KEY_PREFIX = "case_law_crawl_job:"
# If a job stays queued this long without a worker picking it up, mark failed.
STALE_QUEUED_SECONDS = 90

_text_client: redis.Redis | None = None


def _get_text_client() -> redis.Redis:
    global _text_client
    if _text_client is None:
        _text_client = redis.Redis.from_url(
            appConfig["redis_url"],
            decode_responses=True,
        )
    return _text_client


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_crawl_job(
    *,
    linh_vuc: int,
    max_pages: int,
    max_items: Optional[int],
) -> str:
    job_id = str(uuid.uuid4())
    now = _utcnow_iso()
    set_crawl_job(
        job_id,
        {
            "jobId": job_id,
            "status": "queued",
            "percent": 0,
            "message": "Đã xếp hàng crawl án lệ...",
            "linhVuc": linh_vuc,
            "maxPages": max_pages,
            "maxItems": max_items,
            "discovered": 0,
            "processed": 0,
            "queued": 0,
            "skipped": 0,
            "failed": 0,
            "currentCase": None,
            "recentItems": [],
            "error": None,
            "createdAt": now,
            "updatedAt": now,
        },
    )
    return job_id


def set_crawl_job(job_id: str, data: dict[str, Any]) -> None:
    data = {**data, "updatedAt": _utcnow_iso()}
    _get_text_client().setex(
        f"{JOB_KEY_PREFIX}{job_id}",
        JOB_TTL_SECONDS,
        json.dumps(data, ensure_ascii=False),
    )


def update_crawl_job(job_id: str, **fields: Any) -> dict[str, Any]:
    current = get_crawl_job(job_id, resolve_stale=False) or {"jobId": job_id}
    current.update(fields)
    set_crawl_job(job_id, current)
    return current


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fail_stale_queued_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") != "queued":
        return job

    created_at = _parse_iso(job.get("createdAt")) or _parse_iso(job.get("updatedAt"))
    if created_at is None:
        # Legacy jobs without timestamps: treat as stale immediately when still queued.
        age_seconds = STALE_QUEUED_SECONDS + 1
    else:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()

    if age_seconds < STALE_QUEUED_SECONDS:
        return job

    job_id = job.get("jobId")
    if not job_id:
        return job

    return update_crawl_job(
        job_id,
        status="failed",
        percent=100,
        message="Crawl bị treo ở hàng đợi (worker không nhận task).",
        error=(
            "Task Celery không được worker xử lý (Redis/worker có thể đã restart). "
            "Hãy đảm bảo Redis + Celery worker đang chạy, rồi bấm crawl lại."
        ),
    )


def get_crawl_job(
    job_id: str,
    *,
    resolve_stale: bool = True,
) -> dict[str, Any] | None:
    raw = _get_text_client().get(f"{JOB_KEY_PREFIX}{job_id}")
    if not raw:
        return None
    job = json.loads(raw)
    if resolve_stale:
        return _fail_stale_queued_job(job)
    return job


def append_recent_item(job_id: str, item: dict[str, Any], limit: int = 12) -> None:
    job = get_crawl_job(job_id, resolve_stale=False) or {
        "jobId": job_id,
        "recentItems": [],
    }
    recent = list(job.get("recentItems") or [])
    recent.insert(0, item)
    job["recentItems"] = recent[:limit]
    set_crawl_job(job_id, job)
