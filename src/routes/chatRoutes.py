from fastapi import APIRouter, Depends, HTTPException

from src.models.index import ChatCreate, ProcessingStatus, SendChatMessageRequest
from src.services.chatService import (
    ensure_user_exists,
    format_chat_for_client,
    get_chat_for_user,
    process_chat_message,
)
from src.services.clerkAuth import get_current_user_clerk_id
from src.services.supabase import supabase

message_router = APIRouter(tags=["chatMessageRoutes"])
session_router = APIRouter(tags=["chatSessionRoutes"])


def _serialize_chat_document(document: dict) -> dict:
    return {
        "id": document["id"],
        "filename": document["filename"],
        "processingStatus": document.get("processing_status"),
        "createdAt": document.get("created_at"),
    }


@message_router.get("/documents")
async def list_chat_documents(
    _current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        documents_result = (
            supabase.table("documents")
            .select("id, filename, processing_status, created_at")
            .eq("document_scope", "corpus")
            .eq("processing_status", ProcessingStatus.COMPLETED.value)
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "message": "Chat documents retrieved successfully",
            "data": [
                _serialize_chat_document(document)
                for document in documents_result.data or []
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải danh sách tài liệu: {str(e)}",
        )


@message_router.post("")
async def send_chat_message(
    payload: SendChatMessageRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        result = process_chat_message(
            payload.chatId,
            current_user_clerk_id,
            payload.message.strip(),
            payload.documentIds,
        )
        return {
            "message": "Chat message processed successfully",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xử lý tin nhắn: {str(e)}",
        )


@session_router.get("")
async def list_chats(
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        chats_result = (
            supabase.table("chats")
            .select("*")
            .eq("clerk_id", current_user_clerk_id)
            .order("created_at", desc=True)
            .execute()
        )

        chats = chats_result.data or []
        formatted = []

        for chat in chats:
            messages_result = (
                supabase.table("messages")
                .select("*")
                .eq("chat_id", chat["id"])
                .order("created_at", desc=False)
                .execute()
            )
            chat["messages"] = messages_result.data or []
            formatted.append(format_chat_for_client(chat))

        return {
            "message": "Chats retrieved successfully",
            "data": formatted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải danh sách cuộc trò chuyện: {str(e)}",
        )


@session_router.post("")
async def create_chat(
    chat: ChatCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        ensure_user_exists(current_user_clerk_id)
        insert_data = {
            "title": chat.title,
            "clerk_id": current_user_clerk_id,
        }
        if chat.id:
            insert_data["id"] = chat.id

        chat_creation_result = (
            supabase.table("chats").insert(insert_data).execute()
        )

        if not chat_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Không thể tạo cuộc trò chuyện",
            )

        return {
            "message": "Chat created successfully",
            "data": format_chat_for_client(chat_creation_result.data[0]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tạo cuộc trò chuyện: {str(e)}",
        )


@session_router.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        chat = get_chat_for_user(chat_id, current_user_clerk_id)
        return {
            "message": "Chat retrieved successfully",
            "data": format_chat_for_client(chat),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải cuộc trò chuyện: {str(e)}",
        )


@session_router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        chat_deletion_result = (
            supabase.table("chats")
            .delete()
            .eq("id", chat_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not chat_deletion_result.data:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy cuộc trò chuyện hoặc bạn không có quyền xóa",
            )

        return {
            "message": "Chat deleted successfully",
            "data": chat_deletion_result.data[0],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xóa cuộc trò chuyện: {str(e)}",
        )
