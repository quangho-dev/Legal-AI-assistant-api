from fastapi import APIRouter, HTTPException, Depends
from src.services.supabase import supabase
from src.services.clerkAuth import get_current_user_clerk_id, require_admin_user
from src.models.index import FileUploadRequest, ProcessingStatus, UrlRequest, ConfirmUploadRequest, RenameDocumentRequest
from src.utils.index import validate_url
from src.config.index import appConfig
from src.services.awsS3 import s3_client
import uuid
from src.services.celery import perform_rag_ingestion_task
from src.rag.legal_citation import (
    format_legal_citation_for_client,
    get_legal_citation_from_chunk_record,
)
from src.rag.chunk_content import build_chunk_display_text


router = APIRouter(tags=["filesRoutes"])


def normalize_document_filename(new_name: str, document: dict) -> str:
    trimmed = new_name.strip()
    if not trimmed:
        raise HTTPException(
            status_code=400,
            detail="Tên tài liệu không được để trống",
        )

    if document.get("source_type") == "url":
        return trimmed

    old_filename = document.get("filename") or ""
    if "." in old_filename:
        extension = old_filename.rsplit(".", 1)[-1]
        if extension and not trimmed.lower().endswith(f".{extension.lower()}"):
            return f"{trimmed}.{extension}"

    return trimmed


"""
`/api/projects`

  - GET `/{project_id}/files` ~ List all project files
  - POST `/{project_id}/files/upload-url` ~ Generate presigned url for file upload for frontend
  - POST `/{project_id}/files/confirm` ~ Confirmation of file upload to S3
  - POST `/{project_id}/urls` ~ Add website URL to database
  - DELETE `/{project_id}/files/{file_id}` ~ Delete document from s3 and database
  - GET `/{project_id}/files/{file_id}/chunks` ~ Get project document chunks
"""


@router.get("/{project_id}/files")
async def get_project_files(
    current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Select all project documents from the project documents table for given project_id
    * 3. Return project documents data
    """
    try:
        project_files_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("clerk_id", current_user_clerk_id)
            .order("created_at", desc=True)
            .execute()
        )

        # * If there are no project documents for the project, return an empty list
        # * A User may or may not have any project files.

        return {
            "message": "Project files retrieved successfully",
            "data": project_files_result.data or [],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project {project_id} files: {str(e)}",
        )


@router.post("/files/upload-url")
async def get_upload_presigned_url(
    file_upload_request: FileUploadRequest,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        # Generate s3 key
        file_extension = (
            file_upload_request.filename.split(".")[-1]
            if "." in file_upload_request.filename
            else ""
        )
        unique_file_id = uuid.uuid4()
        s3_key = (
            f"documents/{unique_file_id}.{file_extension}"
            if file_extension
            else f"documents/{unique_file_id}"
        )

        # Generate upload presigned url (will expire in 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": appConfig["s3_bucket_name"],
                "Key": s3_key,
                "ContentType": file_upload_request.file_type,
            },
            ExpiresIn=3600,  # 1 hour
        )

        if not presigned_url:
            raise HTTPException(
                status_code=422,
                detail="Failed to generate upload presigned url",
            )

        # Generate database record with pending status
        document_creation_result = (
            supabase.table("documents")
            .insert(
                {
                    "filename": file_upload_request.filename,
                    "s3_key": s3_key,
                    "file_size": file_upload_request.file_size,
                    "file_type": file_upload_request.file_type,
                    "processing_status": ProcessingStatus.PENDING,
                    "document_scope": "corpus",
                }
            )
            .execute()
        )

        if not document_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create document - invalid data provided",
            )

        return {
            "message": "Upload presigned url generated successfully",
            "data": {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "document": document_creation_result.data[0],
            },
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while generating upload presigned url file {file_upload_request.filename}: {str(e)}",
        )


@router.get("/files")
async def list_documents(
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        documents_result = (
            supabase.table("documents")
            .select("*")
            .eq("document_scope", "corpus")
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "message": "Documents retrieved successfully",
            "data": documents_result.data or [],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving documents: {str(e)}",
        )


@router.get("/documents/{document_id}/content")
async def get_document_content(
    document_id: str,
    _current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        document_result = (
            supabase.table("documents")
            .select("id, filename, processing_status, document_scope, clerk_id")
            .eq("id", document_id)
            .execute()
        )

        if not document_result.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")

        document = document_result.data[0]

        if document.get("document_scope") in ("compare", "contract"):
            scope_label = (
                "so sánh"
                if document.get("document_scope") == "compare"
                else "soạn thảo hợp đồng"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Tài liệu {scope_label} chỉ truy cập qua API tương ứng",
            )

        if document.get("processing_status") != ProcessingStatus.COMPLETED.value:
            raise HTTPException(
                status_code=422,
                detail="Tài liệu chưa sẵn sàng để xem",
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

            sections.append(
                {
                    "chunkIndex": chunk.get("chunk_index"),
                    "pageNumber": chunk.get("page_number"),
                    "text": text,
                }
            )

            if text:
                full_text_parts.append(text)

        return {
            "message": "Document content retrieved successfully",
            "data": {
                "documentId": document_id,
                "filename": document.get("filename"),
                "sections": sections,
                "fullText": "\n\n".join(full_text_parts),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải nội dung tài liệu: {str(e)}",
        )


@router.get("/chunks/{chunk_id}")
async def get_chunk_content(
    chunk_id: str,
    _current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        chunk_result = (
            supabase.table("document_chunks")
            .select(
                "id, document_id, chunk_index, page_number, content, original_content"
            )
            .eq("id", chunk_id)
            .execute()
        )

        if not chunk_result.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy đoạn tài liệu")

        chunk = chunk_result.data[0]
        document_result = (
            supabase.table("documents")
            .select("id, filename, processing_status")
            .eq("id", chunk["document_id"])
            .execute()
        )

        if not document_result.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")

        document = document_result.data[0]

        if document.get("processing_status") != ProcessingStatus.COMPLETED.value:
            raise HTTPException(
                status_code=422,
                detail="Tài liệu chưa sẵn sàng để xem",
            )

        original_content = chunk.get("original_content") or {}
        text = build_chunk_display_text(chunk)
        legal = get_legal_citation_from_chunk_record(chunk, document.get("filename", ""))

        return {
            "message": "Chunk content retrieved successfully",
            "data": {
                "chunkId": chunk_id,
                "documentId": chunk["document_id"],
                "filename": document.get("filename"),
                "chunkIndex": chunk.get("chunk_index"),
                "pageNumber": chunk.get("page_number"),
                "text": text,
                **format_legal_citation_for_client(legal),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải nội dung đoạn tài liệu: {str(e)}",
        )


@router.delete("/files/{document_id}")
async def delete_document(
    document_id: str,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        document_result = (
            supabase.table("documents")
            .select("*")
            .eq("id", document_id)
            .execute()
        )

        if not document_result.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        document = document_result.data[0]
        s3_key = document.get("s3_key") or ""

        if s3_key and not s3_key.startswith("url://"):
            try:
                s3_client.delete_object(
                    Bucket=appConfig["s3_bucket_name"],
                    Key=s3_key,
                )
            except Exception:
                pass

        document_deletion_result = (
            supabase.table("documents")
            .delete()
            .eq("id", document_id)
            .execute()
        )

        if not document_deletion_result.data:
            raise HTTPException(
                status_code=404,
                detail="Failed to delete document",
            )

        return {
            "message": "Document deleted successfully",
            "data": document_deletion_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while deleting document {document_id}: {str(e)}",
        )


@router.patch("/files/{document_id}")
async def rename_document(
    document_id: str,
    rename_request: RenameDocumentRequest,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        document_result = (
            supabase.table("documents")
            .select("*")
            .eq("id", document_id)
            .execute()
        )

        if not document_result.data:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy tài liệu",
            )

        document = document_result.data[0]
        new_filename = normalize_document_filename(
            rename_request.filename,
            document,
        )

        document_update_result = (
            supabase.table("documents")
            .update({"filename": new_filename})
            .eq("id", document_id)
            .execute()
        )

        if not document_update_result.data:
            raise HTTPException(
                status_code=422,
                detail="Không thể đổi tên tài liệu",
            )

        return {
            "message": "Document renamed successfully",
            "data": document_update_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while renaming document {document_id}: {str(e)}",
        )


@router.get("/files/monitor")
async def get_documents_monitor(
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        documents_result = (
            supabase.table("documents")
            .select("*")
            .eq("document_scope", "corpus")
            .order("created_at", desc=True)
            .execute()
        )

        documents = documents_result.data or []
        monitor_data = []

        for document in documents:
            chunk_result = (
                supabase.table("document_chunks")
                .select("id", count="exact")
                .eq("document_id", document["id"])
                .execute()
            )

            monitor_data.append(
                {
                    **document,
                    "chunk_count": chunk_result.count or 0,
                }
            )

        return {
            "message": "Document monitor data retrieved successfully",
            "data": monitor_data,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving monitor data: {str(e)}",
        )


@router.post("/files/confirm")
async def confirm_file_upload(
    confirm_request: ConfirmUploadRequest,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        document_verification_result = (
            supabase.table("documents")
            .select("id")
            .eq("s3_key", confirm_request.s3_key)
            .execute()
        )

        if not document_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="File not found for the provided S3 key",
            )

        document_id = document_verification_result.data[0]["id"]
        s3_key = confirm_request.s3_key

        try:
            s3_client.head_object(
                Bucket=appConfig["s3_bucket_name"],
                Key=s3_key,
            )
        except Exception:
            raise HTTPException(
                status_code=404,
                detail="File not found in S3. Upload may have failed.",
            )

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
                detail="Failed to update document status",
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
                detail="Failed to update document record with task_id",
            )

        return {
            "message": "File upload confirmed and embedding process started",
            "data": document_update_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while confirming upload: {str(e)}",
        )


@router.post("/files/urls")
async def process_url_admin(
    url_request: UrlRequest,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        url = url_request.url
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        if not validate_url(url):
            raise HTTPException(status_code=400, detail="Invalid URL")

        document_creation_result = (
            supabase.table("documents")
            .insert(
                {
                    "filename": url,
                    "s3_key": f"url://{uuid.uuid4()}",
                    "file_size": 0,
                    "file_type": "text/html",
                    "processing_status": ProcessingStatus.UPLOADED,
                    "source_type": "url",
                    "source_url": url,
                    "document_scope": "corpus",
                    "processing_details": {
                        "uploaded": {
                            "source_url": url,
                            "message": "URL source registered successfully",
                        }
                    },
                }
            )
            .execute()
        )

        if not document_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create document record for URL",
            )

        document_id = document_creation_result.data[0]["id"]
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
                detail="Failed to update document record with task_id",
            )

        return {
            "message": "URL added and embedding process started",
            "data": document_update_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while processing URL: {str(e)}",
        )


@router.post("/{project_id}/files/confirm")
async def confirm_file_upload_to_s3(
    project_id: str,
    confirm_file_upload_request: dict,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Verify S3 key is provided
    * 2. Verify file exists in database
    * 3. Update file status to "queued"
    * 4. Perform Celery - RAG Ingestion Task
    * 5. Update the project document record with the task_id
    * 6. Return successfully confirmed file upload data
    """
    try:
        s3_key = confirm_file_upload_request.get("s3_key")
        if not s3_key:
            raise HTTPException(
                status_code=400,
                detail="S3 key is required",
            )

        # Verify file exists in database
        document_verification_result = (
            supabase.table("project_documents")
            .select("id")
            .eq("s3_key", s3_key)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="File not found or you don't have permission to confirm upload to S3 for this file",
            )

        # Update file status to "queued"
        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "processing_status": ProcessingStatus.QUEUED,
                }
            )
            .eq("s3_key", s3_key)
            .execute()
        )

        # ! Celery - Starts Background Processing - RAG Ingestion Task
        document_id = document_update_result.data[0]["id"]
        task_result = perform_rag_ingestion_task.delay(document_id)
        task_id = task_result.id

        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "task_id": task_id,
                }
            )
            .eq("id", document_id)
            .execute()
        )
        if not document_update_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to update project document record with task_id",
            )

        return {
            "message": "File upload to S3 confirmed successfully And Started Background Pre-Processing of this file",
            "data": document_update_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while confirming upload to S3 for {project_id}: {str(e)}",
        )


@router.post("/{project_id}/urls")
async def process_url(
    project_id: str,
    url: UrlRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Validate URL
    * 2. Add website URL to database
    * 3. Start background pre-processing of this URL
    * 4. Return successfully processed URL data
    """
    try:
        # Validate URL
        url = url.url
        if url.startswith("http://") or url.startswith("https://"):
            url = url
        else:
            url = f"https://{url}"

        if not validate_url(url):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL",
            )

        # Add website Url to database
        document_creation_result = (
            supabase.table("project_documents")
            .insert(
                {
                    "project_id": project_id,
                    "filename": url,
                    "s3_key": "",
                    "file_size": 0,
                    "file_type": "text/html",
                    "processing_status": ProcessingStatus.QUEUED,
                    "clerk_id": current_user_clerk_id,
                    "source_type": "url",
                    "source_url": url,
                }
            )
            .execute()
        )

        if not document_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create project document with URL Record - invalid data provided",
            )

        # ! Celery - Starts Background Processing - RAG Ingestion Task
        document_id = document_creation_result.data[0]["id"]
        task_result = perform_rag_ingestion_task.delay(document_id)
        task_id = task_result.id

        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "task_id": task_id,
                }
            )
            .eq("id", document_id)
            .execute()
        )

        if not document_update_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to update project document record with task_id",
            )

        return {
            "message": "Website URL added to database successfully And Started Background Pre-Processing of this URL",
            "data": document_creation_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while processing urls for {project_id}: {str(e)}",
        )


@router.delete("/{project_id}/files/{file_id}")
async def delete_project_document(
    project_id: str,
    file_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Verify document exists and belongs to the current user and take complete project document record
    * 2. Delete file from S3 (only for actual files, not for URLs)
    * 3. Delete document from database
    * 4. Return successfully deleted document data
    """
    try:
        # Verify document exists and belongs to the current user and Take complete project document record
        document_ownership_verification_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have permission to delete this document",
            )

        # Delete file from S3 (only for actual files, not for URLs)
        s3_key = document_ownership_verification_result.data[0]["s3_key"]
        if s3_key:
            s3_client.delete_object(Bucket=appConfig["s3_bucket_name"], Key=s3_key)

        # Delete document from database
        document_deletion_result = (
            supabase.table("project_documents")
            .delete()
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_deletion_result.data:
            raise HTTPException(
                status_code=404,
                detail="Failed to delete document",
            )

        return {
            "message": "Document deleted successfully",
            "data": document_deletion_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while deleting project document {file_id} for {project_id}: {str(e)}",
        )


@router.get("/{project_id}/files/{file_id}/chunks")
async def get_project_document_chunks(
    project_id: str,
    file_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Verify document exists and belongs to the current user and Take complete project document record
    * 2. Get project document chunks
    * 3. Return project document chunks data
    """
    try:
        # Verify document exists and belongs to the current user and Take complete project document record
        document_ownership_verification_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have permission to delete this document",
            )

        document_chunks_result = (
            supabase.table("document_chunks")
            .select("*")
            .eq("document_id", file_id)
            .order("chunk_index")
            .execute()
        )

        return {
            "message": "Project document chunks retrieved successfully",
            "data": document_chunks_result.data or [],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while getting project document chunks for {file_id} for {project_id}: {str(e)}",
        )
