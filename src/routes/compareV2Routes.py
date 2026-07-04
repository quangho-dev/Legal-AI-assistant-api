from fastapi import APIRouter, Depends, HTTPException

from src.models.index import CompareDocumentsRequest
from src.services.clerkAuth import get_current_user_clerk_id
from src.services.compareServiceV2 import run_document_comparison_v2_by_ids
from src.services.compareStreamService import iter_compare_v2_stream
from src.utils.sse import create_sse_response

router = APIRouter(tags=["compareV2Routes"])


@router.post("/stream")
async def compare_documents_v2_stream(
    payload: CompareDocumentsRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    trimmed_instruction = payload.instruction.strip()

    if len(trimmed_instruction) < 10:
        raise HTTPException(
            status_code=422,
            detail="Yêu cầu so sánh phải có ít nhất 10 ký tự",
        )

    return create_sse_response(
        iter_compare_v2_stream(
            current_user_clerk_id,
            payload.sourceDocumentId,
            payload.referenceDocumentIds,
            trimmed_instruction,
        )
    )


@router.post("")
async def compare_documents_v2(
    payload: CompareDocumentsRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    trimmed_instruction = payload.instruction.strip()

    if len(trimmed_instruction) < 10:
        raise HTTPException(
            status_code=422,
            detail="Yêu cầu so sánh phải có ít nhất 10 ký tự",
        )

    try:
        result = run_document_comparison_v2_by_ids(
            current_user_clerk_id,
            payload.sourceDocumentId,
            payload.referenceDocumentIds,
            trimmed_instruction,
        )
        return {
            "message": "Document comparison v2 completed successfully",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể so sánh tài liệu (v2): {error}",
        )
