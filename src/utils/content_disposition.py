from __future__ import annotations

from urllib.parse import quote


def build_attachment_content_disposition(filename: str) -> str:
    ascii_fallback = (
        filename.encode("ascii", "ignore").decode("ascii").strip() or "contract.docx"
    )
    if not ascii_fallback.lower().endswith(".docx"):
        ascii_fallback = "contract.docx"

    encoded_filename = quote(filename, safe="")
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{encoded_filename}"
    )
