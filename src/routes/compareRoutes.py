from fastapi import APIRouter, Depends, HTTPException

from src.models.index import (
    CompareDocumentsRequest,
    ConfirmUploadRequest,
    FileUploadRequest,
    RenameDocumentRequest,
)
from src.services.chatService import ensure_user_exists
from src.services.clerkAuth import get_current_user_clerk_id
from src.services.compareDocumentService import (
    confirm_compare_upload,
    create_compare_upload_url,
    delete_user_compare_document,
    get_user_compare_document,
    list_user_compare_documents,
    rename_user_compare_document,
)
from src.services.compareService import run_document_comparison_by_ids

router = APIRouter(tags=["compareRoutes"])


def _serialize_compare_document(document: dict) -> dict:
    return {
        "id": document["id"],
        "filename": document["filename"],
        "fileSize": document.get("file_size"),
        "fileType": document.get("file_type"),
        "processingStatus": document.get("processing_status"),
        "createdAt": document.get("created_at"),
    }


@router.get("/documents")
async def get_compare_documents(
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        ensure_user_exists(current_user_clerk_id)
        documents = list_user_compare_documents(current_user_clerk_id)
        return {
            "message": "Compare documents retrieved successfully",
            "data": [_serialize_compare_document(doc) for doc in documents],
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải danh sách tài liệu: {error}",
        )


@router.get("/documents/{document_id}")
async def get_compare_document_status(
    document_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document = get_user_compare_document(current_user_clerk_id, document_id)
        return {
            "message": "Compare document retrieved successfully",
            "data": _serialize_compare_document(document),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải trạng thái tài liệu: {error}",
        )


@router.post("/upload-url")
async def get_compare_upload_url(
    file_upload_request: FileUploadRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        ensure_user_exists(current_user_clerk_id)
        upload_data = create_compare_upload_url(
            current_user_clerk_id,
            file_upload_request,
        )
        return {
            "message": "Compare upload presigned url generated successfully",
            "data": {
                "uploadUrl": upload_data["upload_url"],
                "s3Key": upload_data["s3_key"],
                "document": _serialize_compare_document(upload_data["document"]),
            },
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tạo URL tải lên: {error}",
        )


@router.post("/confirm")
async def confirm_compare_document_upload(
    confirm_request: ConfirmUploadRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document = confirm_compare_upload(
            current_user_clerk_id,
            confirm_request.s3_key,
        )
        return {
            "message": "Compare file upload confirmed and embedding started",
            "data": _serialize_compare_document(document),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xác nhận tải lên: {error}",
        )


@router.patch("/documents/{document_id}")
async def rename_compare_document(
    document_id: str,
    rename_request: RenameDocumentRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document = rename_user_compare_document(
            current_user_clerk_id,
            document_id,
            rename_request.filename,
        )
        return {
            "message": "Compare document renamed successfully",
            "data": _serialize_compare_document(document),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể đổi tên tài liệu: {error}",
        )


@router.delete("/documents/{document_id}")
async def delete_compare_document(
    document_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document = delete_user_compare_document(
            current_user_clerk_id,
            document_id,
        )
        return {
            "message": "Compare document deleted successfully",
            "data": _serialize_compare_document(document),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xóa tài liệu: {error}",
        )


@router.post("")
async def compare_documents(
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
        result = run_document_comparison_by_ids(
            current_user_clerk_id,
            payload.sourceDocumentId,
            payload.referenceDocumentIds,
            trimmed_instruction,
        )
        return {
            "message": "Document comparison completed successfully",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể so sánh tài liệu: {error}",
        )
