import os
import tempfile
from typing import List

from fastapi import HTTPException, UploadFile
from unstructured.partition.html import partition_html

from src.rag.ingestion.utils import partition_document

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "txt",
    ".md": "md",
    ".pptx": "pptx",
    ".html": "html",
    ".htm": "html",
}

MAX_COMPARE_FILE_SIZE = 25 * 1024 * 1024
MAX_REFERENCE_FILES = 5
DOCUMENT_TEXT_LIMIT = 18000


def _resolve_file_type(filename: str) -> str:
    extension = os.path.splitext(filename or "")[1].lower()
    file_type = SUPPORTED_EXTENSIONS.get(extension)

    if not file_type:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=422,
            detail=f"Định dạng file không được hỗ trợ: {filename}. Hỗ trợ: {supported}",
        )

    return file_type


def _extract_text_from_elements(elements) -> str:
    parts: List[str] = []

    for element in elements:
        text = (getattr(element, "text", None) or "").strip()
        if not text:
            continue

        element_type = type(element).__name__
        if element_type == "Table":
            table_html = getattr(getattr(element, "metadata", None), "text_as_html", None)
            if table_html:
                parts.append(table_html)
            else:
                parts.append(text)
        else:
            parts.append(text)

    return "\n\n".join(parts)


def _partition_uploaded_file(temp_path: str, file_type: str):
    if file_type == "html":
        return partition_html(filename=temp_path)

    return partition_document(temp_path, file_type, source_type="file")


async def parse_uploaded_compare_document(upload: UploadFile) -> dict:
    if not upload.filename:
        raise HTTPException(status_code=422, detail="Tên file không hợp lệ")

    file_type = _resolve_file_type(upload.filename)
    content = await upload.read()

    if not content:
        raise HTTPException(
            status_code=422,
            detail=f"File '{upload.filename}' trống",
        )

    if len(content) > MAX_COMPARE_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File '{upload.filename}' vượt quá 25MB",
        )

    suffix = os.path.splitext(upload.filename)[1] or ""
    temp_file = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(content)
            temp_file = handle.name

        elements = _partition_uploaded_file(temp_file, file_type)
        full_text = _extract_text_from_elements(elements)

        if not full_text.strip():
            raise HTTPException(
                status_code=422,
                detail=f"Không thể trích xuất nội dung từ '{upload.filename}'",
            )

        trimmed_text = full_text[:DOCUMENT_TEXT_LIMIT]
        sections = [
            {
                "chunkIndex": index,
                "pageNumber": None,
                "text": part,
            }
            for index, part in enumerate(trimmed_text.split("\n\n"))
            if part.strip()
        ]

        return {
            "filename": upload.filename,
            "sections": sections,
            "fullText": trimmed_text,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Không thể đọc file '{upload.filename}': {error}",
        ) from error
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        await upload.close()


async def parse_compare_uploads(
    source_file: UploadFile,
    reference_files: List[UploadFile],
) -> tuple[dict, List[dict]]:
    if not reference_files:
        raise HTTPException(
            status_code=422,
            detail="Cần ít nhất một tài liệu tham khảo",
        )

    if len(reference_files) > MAX_REFERENCE_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"Chỉ được tải tối đa {MAX_REFERENCE_FILES} tài liệu tham khảo",
        )

    source_doc = await parse_uploaded_compare_document(source_file)
    reference_docs = []

    for reference_file in reference_files:
        parsed = await parse_uploaded_compare_document(reference_file)
        if parsed["filename"] == source_doc["filename"]:
            raise HTTPException(
                status_code=422,
                detail="Tài liệu tham khảo không được trùng tên với tài liệu gốc",
            )
        reference_docs.append(parsed)

    filenames = [doc["filename"] for doc in reference_docs]
    if len(filenames) != len(set(filenames)):
        raise HTTPException(
            status_code=422,
            detail="Các tài liệu tham khảo không được trùng tên nhau",
        )

    return source_doc, reference_docs
