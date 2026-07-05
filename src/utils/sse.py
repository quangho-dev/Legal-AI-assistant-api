from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi.responses import StreamingResponse


def format_sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_status(message: str) -> str:
    return format_sse_event("status", {"message": message})


def sse_token(content: str) -> str:
    return format_sse_event("token", {"content": content})


def sse_outline_token(content: str) -> str:
    return format_sse_event("outline_token", {"content": content})


def sse_done(data: dict[str, Any]) -> str:
    return format_sse_event("done", data)


def sse_error(message: str) -> str:
    return format_sse_event("error", {"message": message})


def create_sse_response(generator: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
