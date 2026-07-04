from fastapi import APIRouter, Depends, HTTPException

from src.models.index import CompareDocumentsV3Request
from src.services.clerkAuth import get_current_user_clerk_id
from src.services.compareServiceV3 import run_document_comparison_v3_by_ids
from src.services.compareStreamService import iter_compare_v3_stream
from src.utils.sse import create_sse_response

router = APIRouter(tags=["compareV3Routes"])


@router.post("/stream")
async def compare_documents_v3_stream(
    payload: CompareDocumentsV3Request,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    trimmed_instruction = payload.instruction.strip()
    trimmed_role = payload.userRole.strip()

    if len(trimmed_instruction) < 10:
        raise HTTPException(
            status_code=422,
            detail="Yêu cầu so sánh phải có ít nhất 10 ký tự",
        )

    if len(trimmed_role) < 2:
        raise HTTPException(
            status_code=422,
            detail="Vai trò người dùng phải có ít nhất 2 ký tự",
        )

    return create_sse_response(
        iter_compare_v3_stream(
            current_user_clerk_id,
            payload.sourceDocumentId,
            payload.referenceDocumentIds,
            trimmed_instruction,
            trimmed_role,
        )
    )


@router.post("")
async def compare_documents_v3(
    payload: CompareDocumentsV3Request,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    trimmed_instruction = payload.instruction.strip()
    trimmed_role = payload.userRole.strip()

    if len(trimmed_instruction) < 10:
        raise HTTPException(
            status_code=422,
            detail="Yêu cầu so sánh phải có ít nhất 10 ký tự",
        )

    if len(trimmed_role) < 2:
        raise HTTPException(
            status_code=422,
            detail="Vai trò người dùng phải có ít nhất 2 ký tự",
        )

    try:
        result = run_document_comparison_v3_by_ids(
            current_user_clerk_id,
            payload.sourceDocumentId,
            payload.referenceDocumentIds,
            trimmed_instruction,
            trimmed_role,
        )
        return {
            "message": "Document comparison v3 workflow completed successfully",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể chạy workflow so sánh (v3): {error}",
        )
