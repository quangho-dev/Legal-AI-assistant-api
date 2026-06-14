import uuid

from fastapi import HTTPException

from src.rag.retrieval.index import retrieve_context
from src.rag.retrieval.utils import prepare_prompt_and_invoke_llm
from src.rag.legal_citation import (
    extract_legal_citation_metadata,
    format_legal_citation_for_client,
)
from src.services.supabase import supabase

DEFAULT_CHAT_TITLE = "Cuộc trò chuyện mới"


def ensure_user_exists(clerk_id: str) -> None:
    existing = (
        supabase.table("users").select("clerk_id").eq("clerk_id", clerk_id).execute()
    )
    if not existing.data:
        supabase.table("users").insert({"clerk_id": clerk_id}).execute()


def build_chat_title(message: str) -> str:
    trimmed = message.strip()
    if not trimmed:
        return DEFAULT_CHAT_TITLE
    return trimmed[:42] + "…" if len(trimmed) > 42 else trimmed


def format_citation_for_client(citation: dict) -> dict:
    law_name = citation.get("law_name")
    section = citation.get("section")
    section_name = citation.get("section_name")

    if not law_name and citation.get("filename"):
        law_name = extract_legal_citation_metadata("", citation["filename"])["law_name"]

    return {
        "chunkId": citation.get("chunk_id"),
        "documentId": citation.get("document_id"),
        **format_legal_citation_for_client(
            {
                "law_name": law_name,
                "section": section,
                "section_name": section_name,
            }
        ),
    }


def format_message_for_client(message: dict) -> dict:
    citations = message.get("citations") or []
    return {
        "id": message["id"],
        "role": message["role"],
        "content": message["content"],
        "citations": [format_citation_for_client(citation) for citation in citations],
        "createdAt": message.get("created_at"),
    }


def format_chat_for_client(chat: dict) -> dict:
    messages = chat.get("messages") or []
    return {
        "id": chat["id"],
        "title": chat.get("title") or DEFAULT_CHAT_TITLE,
        "messages": [format_message_for_client(message) for message in messages],
        "createdAt": chat.get("created_at"),
        "updatedAt": chat.get("updated_at") or chat.get("created_at"),
    }


def get_chat_for_user(chat_id: str, clerk_id: str) -> dict:
    chat_result = (
        supabase.table("chats")
        .select("*")
        .eq("id", chat_id)
        .eq("clerk_id", clerk_id)
        .execute()
    )

    if not chat_result.data:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy cuộc trò chuyện hoặc bạn không có quyền truy cập",
        )

    messages_result = (
        supabase.table("messages")
        .select("*")
        .eq("chat_id", chat_id)
        .order("created_at", desc=False)
        .execute()
    )

    chat = chat_result.data[0]
    chat["messages"] = messages_result.data or []
    return chat


def ensure_chat_for_user(chat_id: str, clerk_id: str, title: str) -> dict:
    existing = (
        supabase.table("chats")
        .select("*")
        .eq("id", chat_id)
        .eq("clerk_id", clerk_id)
        .execute()
    )

    if existing.data:
        return existing.data[0]

    insert_data = {
        "id": chat_id,
        "title": title,
        "clerk_id": clerk_id,
    }
    created = supabase.table("chats").insert(insert_data).execute()

    if not created.data:
        raise HTTPException(status_code=500, detail="Không thể tạo cuộc trò chuyện")

    return created.data[0]


def process_chat_message(chat_id: str, clerk_id: str, message: str) -> dict:
    ensure_user_exists(clerk_id)

    chat = ensure_chat_for_user(
        chat_id,
        clerk_id,
        build_chat_title(message),
    )

    existing_messages = (
        supabase.table("messages")
        .select("id")
        .eq("chat_id", chat_id)
        .execute()
    )
    is_first_message = not existing_messages.data

    user_message_result = (
        supabase.table("messages")
        .insert(
            {
                "chat_id": chat_id,
                "clerk_id": clerk_id,
                "content": message,
                "role": "user",
                "citations": [],
            }
        )
        .execute()
    )

    if not user_message_result.data:
        raise HTTPException(status_code=500, detail="Không thể lưu tin nhắn")

    try:
        texts, images, tables, citations = retrieve_context(message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tra cứu tài liệu: {str(e)}",
        )

    assistant_content = prepare_prompt_and_invoke_llm(
        message,
        texts,
        images,
        tables,
    )

    assistant_message_result = (
        supabase.table("messages")
        .insert(
            {
                "chat_id": chat_id,
                "clerk_id": clerk_id,
                "content": assistant_content,
                "role": "assistant",
                "citations": citations,
                "trace_id": str(uuid.uuid4()),
            }
        )
        .execute()
    )

    if not assistant_message_result.data:
        raise HTTPException(
            status_code=500, detail="Không thể lưu câu trả lời của trợ lý"
        )

    chat_title = chat.get("title") or DEFAULT_CHAT_TITLE
    if is_first_message:
        chat_title = build_chat_title(message)
        supabase.table("chats").update({"title": chat_title}).eq("id", chat_id).execute()

    return {
        "message": format_message_for_client(assistant_message_result.data[0]),
        "chatTitle": chat_title,
    }
