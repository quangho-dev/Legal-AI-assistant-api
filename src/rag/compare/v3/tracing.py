from __future__ import annotations

import os
from typing import Any, Optional

_TRUTHY = {"1", "true", "yes", "on"}
_CONFIGURED = False


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def configure_langsmith() -> bool:
    """Enable LangSmith tracing when env vars are present."""
    global _CONFIGURED
    if _CONFIGURED:
        return is_langsmith_enabled()

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    tracing_requested = _is_truthy(os.getenv("LANGSMITH_TRACING")) or _is_truthy(
        os.getenv("LANGCHAIN_TRACING_V2")
    )

    if not tracing_requested or not api_key:
        _CONFIGURED = True
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key

    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "legal-ai-compare-v3"
    )
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint

    _CONFIGURED = True
    return True


def is_langsmith_enabled() -> bool:
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = _is_truthy(os.getenv("LANGSMITH_TRACING")) or _is_truthy(
        os.getenv("LANGCHAIN_TRACING_V2")
    )
    return bool(tracing_enabled and api_key)


def build_workflow_run_config(
    *,
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: list[str],
    user_role: str,
    source_filename: str = "",
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    configure_langsmith()

    config: dict[str, Any] = {
        "run_name": run_name or f"compare-v3:{source_filename or source_document_id}",
        "tags": ["compare-v3", "langgraph", "legal-ai"],
        "metadata": {
            "workflow": "compare-v3",
            "clerk_id": clerk_id,
            "source_document_id": source_document_id,
            "source_filename": source_filename,
            "reference_document_ids": reference_document_ids,
            "reference_document_count": len(reference_document_ids),
            "user_role": user_role,
        },
    }

    if is_langsmith_enabled():
        config["metadata"]["langsmith_project"] = os.getenv("LANGSMITH_PROJECT")

    return config
