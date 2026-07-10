from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from src.models.index import (
    ConfirmUploadRequest,
    ContractDraftRequest,
    ContractExportDocxRequest,
    FileUploadRequest,
    RenameDocumentRequest,
)
from src.services.chatService import ensure_user_exists
from src.services.clerkAuth import get_current_user_clerk_id
from src.services.contractDocumentService import (
    confirm_contract_upload as confirm_contract_upload_service,
    create_contract_upload_url,
    delete_user_contract_document,
    list_user_contract_documents,
    rename_user_contract_document,
)
from src.services.contractDraftService import (
    draft_contract,
    export_contract_docx,
    iter_contract_draft_stream,
    iter_contract_export_docx_stream,
)
from src.utils.content_disposition import build_attachment_content_disposition
from src.utils.sse import create_sse_response

router = APIRouter(tags=["contractRoutes"])


def _serialize_document(document: dict) -> dict:
    return {
        "id": document["id"],
        "filename": document["filename"],
        "fileSize": document.get("file_size"),
        "fileType": document.get("file_type"),
        "processingStatus": document.get("processing_status"),
        "createdAt": document.get("created_at"),
    }


@router.get("/documents")
async def list_contract_documents(
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        ensure_user_exists(current_user_clerk_id)
        documents = list_user_contract_documents(current_user_clerk_id)
        return {
            "message": "Contract documents retrieved successfully",
            "data": [_serialize_document(doc) for doc in documents],
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải danh sách tài liệu: {error}",
        )


@router.post("/upload-url")
async def get_contract_upload_url(
    file_upload_request: FileUploadRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        ensure_user_exists(current_user_clerk_id)
        upload_data = create_contract_upload_url(
            current_user_clerk_id,
            file_upload_request,
        )
        return {
            "message": "Contract upload presigned url generated successfully",
            "data": {
                "uploadUrl": upload_data["upload_url"],
                "s3Key": upload_data["s3_key"],
                "document": _serialize_document(upload_data["document"]),
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
async def confirm_contract_upload(
    confirm_request: ConfirmUploadRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document = confirm_contract_upload_service(
            current_user_clerk_id,
            confirm_request.s3_key,
        )
        return {
            "message": "Contract file upload confirmed and embedding started",
            "data": _serialize_document(document),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xác nhận tải lên: {error}",
        )


@router.patch("/documents/{document_id}")
async def rename_contract_document(
    document_id: str,
    rename_request: RenameDocumentRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document = rename_user_contract_document(
            current_user_clerk_id,
            document_id,
            rename_request.filename,
        )
        return {
            "message": "Contract document renamed successfully",
            "data": _serialize_document(document),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể đổi tên tài liệu: {error}",
        )


@router.delete("/documents/{document_id}")
async def delete_contract_document(
    document_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document = delete_user_contract_document(
            current_user_clerk_id,
            document_id,
        )
        return {
            "message": "Contract document deleted successfully",
            "data": _serialize_document(document),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xóa tài liệu: {error}",
        )


@router.post("/draft")
async def draft_contract_document(
    payload: ContractDraftRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    trimmed_requirements = payload.requirements.strip()
    trimmed_party_role = payload.partyRole.strip()

    if len(trimmed_requirements) < 10:
        raise HTTPException(
            status_code=422,
            detail="Mô tả yêu cầu phải có ít nhất 10 ký tự",
        )

    if len(trimmed_party_role) < 2:
        raise HTTPException(
            status_code=422,
            detail="Vai trò bên trong hợp đồng phải có ít nhất 2 ký tự",
        )

    normalized_payload = payload.model_copy(
        update={
            "requirements": trimmed_requirements,
            "partyRole": trimmed_party_role,
        }
    )

    try:
        result = draft_contract(current_user_clerk_id, normalized_payload)
        return {
            "message": "Contract outline drafted successfully",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể soạn thảo hợp đồng: {error}",
        )


@router.post("/draft/stream")
async def draft_contract_document_stream(
    payload: ContractDraftRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    trimmed_requirements = payload.requirements.strip()
    trimmed_party_role = payload.partyRole.strip()

    if len(trimmed_requirements) < 10:
        raise HTTPException(
            status_code=422,
            detail="Mô tả yêu cầu phải có ít nhất 10 ký tự",
        )

    if len(trimmed_party_role) < 2:
        raise HTTPException(
            status_code=422,
            detail="Vai trò bên trong hợp đồng phải có ít nhất 2 ký tự",
        )

    normalized_payload = payload.model_copy(
        update={
            "requirements": trimmed_requirements,
            "partyRole": trimmed_party_role,
        }
    )

    return create_sse_response(
        iter_contract_draft_stream(current_user_clerk_id, normalized_payload)
    )


@router.post("/export/docx")
async def export_contract_docx_document(
    payload: ContractExportDocxRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    trimmed_requirements = payload.requirements.strip()
    trimmed_party_role = payload.partyRole.strip()
    trimmed_outline = payload.outline.strip()

    if len(trimmed_requirements) < 10:
        raise HTTPException(
            status_code=422,
            detail="Mô tả yêu cầu phải có ít nhất 10 ký tự",
        )

    if len(trimmed_party_role) < 2:
        raise HTTPException(
            status_code=422,
            detail="Vai trò bên trong hợp đồng phải có ít nhất 2 ký tự",
        )

    if len(trimmed_outline) < 20:
        raise HTTPException(
            status_code=422,
            detail="Dàn ý phác thảo phải có ít nhất 20 ký tự",
        )

    normalized_payload = payload.model_copy(
        update={
            "requirements": trimmed_requirements,
            "partyRole": trimmed_party_role,
            "outline": trimmed_outline,
        }
    )

    try:
        docx_bytes, filename = export_contract_docx(
            current_user_clerk_id,
            normalized_payload,
        )
        return Response(
            content=docx_bytes,
            media_type=(
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            headers={
                "Content-Disposition": build_attachment_content_disposition(
                    filename
                ),
            },
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xuất file Word: {error}",
        )


@router.post("/export/docx/stream")
async def export_contract_docx_document_stream(
    payload: ContractExportDocxRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    ensure_user_exists(current_user_clerk_id)

    trimmed_requirements = payload.requirements.strip()
    trimmed_party_role = payload.partyRole.strip()
    trimmed_outline = payload.outline.strip()

    if len(trimmed_requirements) < 10:
        raise HTTPException(
            status_code=422,
            detail="Mô tả yêu cầu phải có ít nhất 10 ký tự",
        )

    if len(trimmed_party_role) < 2:
        raise HTTPException(
            status_code=422,
            detail="Vai trò bên trong hợp đồng phải có ít nhất 2 ký tự",
        )

    if len(trimmed_outline) < 20:
        raise HTTPException(
            status_code=422,
            detail="Dàn ý phác thảo phải có ít nhất 20 ký tự",
        )

    normalized_payload = payload.model_copy(
        update={
            "requirements": trimmed_requirements,
            "partyRole": trimmed_party_role,
            "outline": trimmed_outline,
        }
    )

    return create_sse_response(
        iter_contract_export_docx_stream(
            current_user_clerk_id,
            normalized_payload,
        )
    )
