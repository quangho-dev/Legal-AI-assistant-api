from __future__ import annotations

import re
from io import BytesIO

from docx import Document


def _add_formatted_runs(paragraph, text: str) -> None:
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            paragraph.add_run(part)


def markdown_to_docx_bytes(markdown_text: str) -> bytes:
    doc = Document()

    for line in markdown_text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("### "):
            doc.add_heading(stripped[4:].strip(), level=3)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:].strip(), level=2)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:].strip(), level=1)
        elif stripped.startswith("> "):
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(stripped[2:].strip())
            run.italic = True
        elif stripped.startswith("- ") or stripped.startswith("* "):
            paragraph = doc.add_paragraph(style="List Bullet")
            _add_formatted_runs(paragraph, stripped[2:].strip())
        elif re.match(r"^\d+\.\s", stripped):
            paragraph = doc.add_paragraph(style="List Number")
            _add_formatted_runs(paragraph, re.sub(r"^\d+\.\s", "", stripped).strip())
        else:
            paragraph = doc.add_paragraph()
            _add_formatted_runs(paragraph, stripped)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
