from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException

from src.services.supabase import supabase


def _serialize_case_law(row: dict[str, Any], *, include_full_text: bool = False) -> dict:
    data = {
        "id": row.get("id"),
        "dDocName": row.get("d_doc_name"),
        "caseNumber": row.get("case_number"),
        "title": row.get("title"),
        "linhVuc": row.get("linh_vuc"),
        "linhVucLabel": row.get("linh_vuc_label"),
        "adoptedDate": row.get("adopted_date"),
        "publishedDate": row.get("published_date"),
        "effectiveDate": row.get("effective_date"),
        "status": row.get("status"),
        "sourceUrl": row.get("source_url"),
        "pdfUrl": row.get("pdf_url"),
        "processingStatus": row.get("processing_status"),
        "createdAt": row.get("created_at"),
        "rank": row.get("rank"),
    }
    if include_full_text:
        data["fullText"] = row.get("full_text") or ""
        data["attributesText"] = row.get("attributes_text")
        data["fileSize"] = row.get("file_size")
    return data


def search_case_laws(
    *,
    query: str = "",
    linh_vuc: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    try:
        result = supabase.rpc(
            "search_case_laws",
            {
                "query_text": query or "",
                "filter_linh_vuc": linh_vuc,
                "match_limit": limit,
                "match_offset": offset,
            },
        ).execute()
        rows = result.data or []
        return [_serialize_case_law(row) for row in rows]
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tra cứu án lệ: {error}",
        )


def get_case_law(case_law_id: str) -> dict:
    result = (
        supabase.table("case_laws")
        .select("*")
        .eq("id", case_law_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Không tìm thấy án lệ")
    return _serialize_case_law(result.data[0], include_full_text=True)


def list_case_laws(
    *,
    linh_vuc: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    query = (
        supabase.table("case_laws")
        .select(
            "id, d_doc_name, case_number, title, linh_vuc, linh_vuc_label, "
            "adopted_date, published_date, effective_date, status, source_url, "
            "pdf_url, processing_status, created_at"
        )
        .eq("processing_status", "completed")
        .order("created_at", desc=True)
        .range(offset, offset + max(limit, 1) - 1)
    )
    if linh_vuc is not None:
        query = query.eq("linh_vuc", linh_vuc)

    result = query.execute()
    return [_serialize_case_law(row) for row in (result.data or [])]
