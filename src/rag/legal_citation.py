import os
import re
from typing import Any, Dict, List, Optional


def _clean_law_name(filename: str) -> str:
    name = os.path.splitext(filename or "")[0]
    return name.replace("_", " ").replace("-", " ").strip() or "Văn bản pháp luật"


def extract_legal_citation_metadata(text: str, document_filename: str = "") -> Dict[str, Optional[str]]:
    law_name = _clean_law_name(document_filename)
    section = None
    section_name = None

    if not text:
        return {
            "law_name": law_name,
            "section": section,
            "section_name": section_name,
        }

    normalized = text.strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    section_patterns = [
        (r"Điều\s+(\d+[a-zA-Z]?)", "Điều"),
        (r"Chương\s+([IVXLCivxlc\d]+)", "Chương"),
        (r"Mục\s+(\d+)", "Mục"),
        (r"Phần\s+(\d+)", "Phần"),
        (r"Article\s+(\d+[a-zA-Z]?)", "Article"),
        (r"Section\s+(\d+[a-zA-Z]?)", "Section"),
    ]

    for line in lines[:8]:
        same_line = re.match(
            r"^(Điều|Chương|Mục|Phần)\s+([^\s.\-–:]+)\s*[.\-–:]\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if same_line:
            section = f"{same_line.group(1).capitalize()} {same_line.group(2)}"
            section_name = same_line.group(3).strip()
            break

        for pattern, label in section_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                section = f"{label} {match.group(1)}"
                remainder = re.sub(pattern, "", line, count=1, flags=re.IGNORECASE)
                remainder = re.sub(r"^[\s.\-–:]+", "", remainder).strip()
                if remainder and len(remainder) <= 200:
                    section_name = remainder
                break

        if section:
            if not section_name:
                line_index = lines.index(line)
                if line_index + 1 < len(lines):
                    candidate = lines[line_index + 1]
                    if len(candidate) <= 200 and not re.match(
                        r"^(Điều|Chương|Mục|Phần)\s+",
                        candidate,
                        re.IGNORECASE,
                    ):
                        section_name = candidate
            break

    return {
        "law_name": law_name,
        "section": section,
        "section_name": section_name,
    }


def extract_legal_metadata_from_chunk(
    chunk: Any,
    document_filename: str,
    text: str,
) -> Dict[str, Optional[str]]:
    result = extract_legal_citation_metadata(text, document_filename)

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        title_texts: List[str] = []
        for element in chunk.metadata.orig_elements:
            if type(element).__name__ in ("Title", "Header"):
                title = (getattr(element, "text", None) or "").strip()
                if title:
                    title_texts.append(title)

        for title in title_texts:
            if not result["section"]:
                match = re.search(
                    r"(Điều|Chương|Mục|Phần)\s+([^\s.\-–:]+)",
                    title,
                    re.IGNORECASE,
                )
                if match:
                    result["section"] = f"{match.group(1).capitalize()} {match.group(2)}"

            inline_title = re.match(
                r"^(Điều|Chương|Mục|Phần)\s+([^\s.\-–:]+)\s*[.\-–:]\s*(.+)$",
                title,
                re.IGNORECASE,
            )
            if inline_title:
                result["section"] = (
                    f"{inline_title.group(1).capitalize()} {inline_title.group(2)}"
                )
                result["section_name"] = inline_title.group(3).strip()

        if not result["section_name"] and title_texts:
            for title in reversed(title_texts):
                if re.match(r"^(Điều|Chương|Mục|Phần)\s+", title, re.IGNORECASE):
                    continue
                if len(title) <= 200:
                    result["section_name"] = title
                    break

    return result


def get_legal_citation_from_chunk_record(
    chunk: Dict[str, Any],
    document_filename: str,
) -> Dict[str, Optional[str]]:
    original_content = chunk.get("original_content") or {}
    if not isinstance(original_content, dict):
        original_content = {}

    stored = original_content.get("legal_citation")
    if isinstance(stored, dict) and any(stored.values()):
        return {
            "law_name": stored.get("law_name") or _clean_law_name(document_filename),
            "section": stored.get("section"),
            "section_name": stored.get("section_name"),
        }

    text = original_content.get("text") or chunk.get("content") or ""
    return extract_legal_citation_metadata(text, document_filename)


def format_legal_citation_for_client(legal: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    return {
        "lawName": legal.get("law_name"),
        "section": legal.get("section"),
        "sectionName": legal.get("section_name"),
    }
