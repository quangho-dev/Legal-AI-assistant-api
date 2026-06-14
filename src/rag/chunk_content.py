import json
import re
from html import unescape
from typing import Any, Dict, List


def _parse_original_content(original_content: Any) -> Dict[str, Any]:
    if isinstance(original_content, dict):
        return original_content

    if isinstance(original_content, str):
        try:
            parsed = json.loads(original_content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def _html_to_text(html: str) -> str:
    if not html:
        return ""

    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def build_chunk_display_text(chunk: Dict[str, Any]) -> str:
    original_content = _parse_original_content(chunk.get("original_content"))
    parts: List[str] = []

    text = _normalize_text(original_content.get("text"))
    if text:
        parts.append(text)

    tables = original_content.get("tables") or []
    if isinstance(tables, list):
        for index, table in enumerate(tables, start=1):
            table_text = _html_to_text(_normalize_text(table))
            if table_text:
                parts.append(f"Bảng {index}:\n{table_text}")

    images = original_content.get("images") or []
    if isinstance(images, list) and images and not parts:
        parts.append(
            "Đoạn này chủ yếu chứa hình ảnh. Nội dung văn bản không khả dụng."
        )

    if parts:
        return "\n\n".join(parts)

    return _normalize_text(chunk.get("content"))
