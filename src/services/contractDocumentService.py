import uuid
from typing import List

from fastapi import HTTPException

from src.config.index import appConfig
from src.models.index import FileUploadRequest, ProcessingStatus
from src.rag.chunk_content import build_chunk_display_text
from src.services.awsS3 import s3_client
from src.services.celery import perform_rag_ingestion_task
from src.services.supabase import supabase

DOCUMENT_SCOPE_CONTRACT = "contract"
DOCUMENT_TEXT_LIMIT = 12000
MAX_DESCRIPTION_DOCUMENTS = 5


def _normalize_contract_filename(new_name: str, document: dict) -> str:
    trimmed = new_name.strip()
    if not trimmed:
        raise HTTPException(
            status_code=400,
            detail="Tên tài liệu không được để trống",
        )

    old_filename = document.get("filename") or ""
    if "." in old_filename:
        extension = old_filename.rsplit(".", 1)[-1]
        if extension and not trimmed.lower().endswith(f".{extension.lower()}"):
            return f"{trimmed}.{extension}"

    return trimmed


def _contract_document_query(clerk_id: str):
    return (
        supabase.table("documents")
        .select("*")
        .eq("clerk_id", clerk_id)
        .eq("document_scope", DOCUMENT_SCOPE_CONTRACT)
    )


def list_user_contract_documents(clerk_id: str) -> List[dict]:
    result = (
        _contract_document_query(clerk_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def get_user_contract_document(clerk_id: str, document_id: str) -> dict:
    result = (
        _contract_document_query(clerk_id)
        .eq("id", document_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy tài liệu hoặc bạn không có quyền truy cập",
        )

    return result.data[0]


def create_contract_upload_url(
    clerk_id: str,
    file_upload_request: FileUploadRequest,
) -> dict:
    file_extension = (
        file_upload_request.filename.split(".")[-1]
        if "." in file_upload_request.filename
        else ""
    )
    unique_file_id = uuid.uuid4()
    s3_key = (
        f"contract-documents/{unique_file_id}.{file_extension}"
        if file_extension
        else f"contract-documents/{unique_file_id}"
    )

    presigned_url = s3_client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": appConfig["s3_bucket_name"],
            "Key": s3_key,
            "ContentType": file_upload_request.file_type,
        },
        ExpiresIn=3600,
    )

    if not presigned_url:
        raise HTTPException(
            status_code=422,
            detail="Không thể tạo URL tải lên",
        )

    document_creation_result = (
        supabase.table("documents")
        .insert(
            {
                "filename": file_upload_request.filename,
                "s3_key": s3_key,
                "file_size": file_upload_request.file_size,
                "file_type": file_upload_request.file_type,
                "processing_status": ProcessingStatus.PENDING,
                "document_scope": DOCUMENT_SCOPE_CONTRACT,
                "clerk_id": clerk_id,
            }
        )
        .execute()
    )

    if not document_creation_result.data:
        raise HTTPException(
            status_code=422,
            detail="Không thể tạo bản ghi tài liệu",
        )

    return {
        "upload_url": presigned_url,
        "s3_key": s3_key,
        "document": document_creation_result.data[0],
    }


def confirm_contract_upload(clerk_id: str, s3_key: str) -> dict:
    document_verification_result = (
        supabase.table("documents")
        .select("id, clerk_id, document_scope")
        .eq("s3_key", s3_key)
        .execute()
    )

    if not document_verification_result.data:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy tài liệu với S3 key này",
        )

    document = document_verification_result.data[0]

    if document.get("clerk_id") != clerk_id:
        raise HTTPException(
            status_code=403,
            detail="Bạn không có quyền xác nhận tài liệu này",
        )

    if document.get("document_scope") != DOCUMENT_SCOPE_CONTRACT:
        raise HTTPException(
            status_code=422,
            detail="Tài liệu không thuộc kho soạn thảo hợp đồng",
        )

    try:
        s3_client.head_object(
            Bucket=appConfig["s3_bucket_name"],
            Key=s3_key,
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="File chưa được tải lên S3. Vui lòng thử lại.",
        )

    document_id = document["id"]

    document_update_result = (
        supabase.table("documents")
        .update(
            {
                "processing_status": ProcessingStatus.UPLOADED,
                "processing_details": {
                    "uploaded": {
                        "s3_key": s3_key,
                        "message": "Uploaded to S3 successfully",
                    }
                },
            }
        )
        .eq("id", document_id)
        .execute()
    )

    if not document_update_result.data:
        raise HTTPException(
            status_code=422,
            detail="Không thể cập nhật trạng thái tài liệu",
        )

    task_result = perform_rag_ingestion_task.delay(document_id)
    task_id = task_result.id

    document_update_result = (
        supabase.table("documents")
        .update({"task_id": task_id})
        .eq("id", document_id)
        .execute()
    )

    if not document_update_result.data:
        raise HTTPException(
            status_code=422,
            detail="Không thể ghi nhận tác vụ xử lý",
        )

    return document_update_result.data[0]


def delete_user_contract_document(clerk_id: str, document_id: str) -> dict:
    document = get_user_contract_document(clerk_id, document_id)
    s3_key = document.get("s3_key") or ""

    if s3_key and not s3_key.startswith("url://"):
        try:
            s3_client.delete_object(
                Bucket=appConfig["s3_bucket_name"],
                Key=s3_key,
            )
        except Exception:
            pass

    deletion_result = (
        supabase.table("documents")
        .delete()
        .eq("id", document_id)
        .eq("clerk_id", clerk_id)
        .eq("document_scope", DOCUMENT_SCOPE_CONTRACT)
        .execute()
    )

    if not deletion_result.data:
        raise HTTPException(
            status_code=404,
            detail="Không thể xóa tài liệu",
        )

    return deletion_result.data[0]


def rename_user_contract_document(
    clerk_id: str,
    document_id: str,
    new_filename: str,
) -> dict:
    document = get_user_contract_document(clerk_id, document_id)
    normalized_filename = _normalize_contract_filename(new_filename, document)

    update_result = (
        supabase.table("documents")
        .update({"filename": normalized_filename})
        .eq("id", document_id)
        .eq("clerk_id", clerk_id)
        .eq("document_scope", DOCUMENT_SCOPE_CONTRACT)
        .execute()
    )

    if not update_result.data:
        raise HTTPException(
            status_code=422,
            detail="Không thể đổi tên tài liệu",
        )

    return update_result.data[0]


def fetch_contract_document_content(clerk_id: str, document_id: str) -> dict:
    document = get_user_contract_document(clerk_id, document_id)

    if document.get("processing_status") != ProcessingStatus.COMPLETED.value:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Tài liệu '{document.get('filename')}' chưa sẵn sàng "
                f"(trạng thái: {document.get('processing_status')})"
            ),
        )

    chunks_result = (
        supabase.table("document_chunks")
        .select("chunk_index, page_number, content, original_content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )

    sections = []
    full_text_parts = []

    for chunk in chunks_result.data or []:
        text = build_chunk_display_text(chunk)
        if not text:
            continue

        sections.append(
            {
                "chunkIndex": chunk.get("chunk_index"),
                "pageNumber": chunk.get("page_number"),
                "text": text,
            }
        )
        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Tài liệu '{document.get('filename')}' không có nội dung",
        )

    return {
        "id": document_id,
        "filename": document.get("filename"),
        "sections": sections,
        "fullText": full_text[:DOCUMENT_TEXT_LIMIT],
    }
