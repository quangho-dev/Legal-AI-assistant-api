import json
from typing import List, Optional

from fastapi import HTTPException

from src.models.index import ProcessingStatus
from src.rag.chunk_content import build_chunk_display_text
from src.services.compareDocumentService import (
    MAX_REFERENCE_DOCUMENTS,
    get_user_compare_document,
)
from src.services.supabase import supabase


def _parse_chunk_embedding(raw_embedding) -> Optional[List[float]]:
    if raw_embedding is None:
        return None

    if isinstance(raw_embedding, list):
        return [float(value) for value in raw_embedding]

    if isinstance(raw_embedding, str):
        try:
            parsed = json.loads(raw_embedding)
            if isinstance(parsed, list):
                return [float(value) for value in parsed]
        except json.JSONDecodeError:
            cleaned = raw_embedding.strip("[]")
            if not cleaned:
                return None
            return [
                float(value.strip())
                for value in cleaned.split(",")
                if value.strip()
            ]

    return None


def _load_document_chunks_for_retrieval(document_id: str) -> List[dict]:
    chunks_result = (
        supabase.table("document_chunks")
        .select("chunk_index, page_number, content, original_content, embedding")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )

    chunks = []
    for chunk in chunks_result.data or []:
        text = build_chunk_display_text(chunk)
        if not text:
            continue

        chunks.append(
            {
                "chunkIndex": chunk.get("chunk_index"),
                "pageNumber": chunk.get("page_number"),
                "text": text,
                "embedding": _parse_chunk_embedding(chunk.get("embedding")),
            }
        )

    return chunks


def _ensure_document_ready(document: dict) -> None:
    if document.get("processing_status") != ProcessingStatus.COMPLETED.value:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Tài liệu '{document.get('filename')}' chưa sẵn sàng "
                f"(trạng thái: {document.get('processing_status')})"
            ),
        )


def fetch_compare_documents_for_retrieval(
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: List[str],
) -> List[dict]:
    if source_document_id in reference_document_ids:
        raise HTTPException(
            status_code=422,
            detail="Tài liệu gốc không được nằm trong danh sách tham khảo",
        )

    unique_reference_ids = list(dict.fromkeys(reference_document_ids))

    if not unique_reference_ids:
        raise HTTPException(
            status_code=422,
            detail="Cần ít nhất một tài liệu tham khảo",
        )

    if len(unique_reference_ids) > MAX_REFERENCE_DOCUMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"Chỉ được chọn tối đa {MAX_REFERENCE_DOCUMENTS} tài liệu tham khảo",
        )

    source_document = get_user_compare_document(clerk_id, source_document_id)
    _ensure_document_ready(source_document)

    source_chunks = _load_document_chunks_for_retrieval(source_document_id)
    if not source_chunks:
        raise HTTPException(
            status_code=422,
            detail=f"Tài liệu '{source_document.get('filename')}' chưa có chunk để tra cứu",
        )

    documents = [
        {
            "id": source_document_id,
            "filename": source_document.get("filename"),
            "role": "source",
            "chunks": source_chunks,
        }
    ]

    for reference_id in unique_reference_ids:
        reference_document = get_user_compare_document(clerk_id, reference_id)
        _ensure_document_ready(reference_document)

        reference_chunks = _load_document_chunks_for_retrieval(reference_id)
        if not reference_chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Tài liệu '{reference_document.get('filename')}' "
                    "chưa có chunk để tra cứu"
                ),
            )

        documents.append(
            {
                "id": reference_id,
                "filename": reference_document.get("filename"),
                "role": "reference",
                "chunks": reference_chunks,
            }
        )

    return documents
