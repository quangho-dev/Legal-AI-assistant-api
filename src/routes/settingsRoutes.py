from fastapi import APIRouter, Depends, HTTPException
from src.models.index import (
    ChatSettingsCreate,
    ChatSettingsUpdate,
    OPENAI_CHAT_MODELS,
    RAG_STRATEGIES,
)
from src.services.clerkAuth import require_admin_user
from src.services.supabase import supabase

router = APIRouter(tags=["settingsRoutes"])


def _validate_settings_payload(
    payload: ChatSettingsCreate | ChatSettingsUpdate,
):
    if payload.rag_strategy not in RAG_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"rag_strategy must be one of: {', '.join(RAG_STRATEGIES)}",
        )

    if payload.chunks_per_search < 1 or payload.chunks_per_search > 100:
        raise HTTPException(
            status_code=422,
            detail="chunks_per_search must be between 1 and 100",
        )

    if payload.final_context_size < 1 or payload.final_context_size > 50:
        raise HTTPException(
            status_code=422,
            detail="final_context_size must be between 1 and 50",
        )

    if payload.similarity_threshold < 0 or payload.similarity_threshold > 1:
        raise HTTPException(
            status_code=422,
            detail="similarity_threshold must be between 0 and 1",
        )

    if payload.number_of_queries < 1 or payload.number_of_queries > 10:
        raise HTTPException(
            status_code=422,
            detail="number_of_queries must be between 1 and 10",
        )

    if payload.vector_weight < 0 or payload.keyword_weight < 0:
        raise HTTPException(
            status_code=422,
            detail="vector_weight and keyword_weight must be non-negative",
        )

    if payload.chat_model not in OPENAI_CHAT_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"chat_model must be one of: {', '.join(OPENAI_CHAT_MODELS)}",
        )


def _strip_singleton_field(row: dict | None) -> dict | None:
    if not row:
        return row
    return {key: value for key, value in row.items() if key != "singleton"}


@router.get("/retrieval-settings")
async def get_retrieval_settings(
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        result = supabase.table("chat_settings").select("*").limit(1).execute()

        if not result.data:
            return {
                "message": "Retrieval settings not found",
                "data": None,
            }

        return {
            "message": "Retrieval settings retrieved successfully",
            "data": _strip_singleton_field(result.data[0]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve settings: {str(e)}",
        )


@router.post("/retrieval-settings")
async def create_retrieval_settings(
    payload: ChatSettingsCreate,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    _validate_settings_payload(payload)

    try:
        existing = supabase.table("chat_settings").select("id").limit(1).execute()

        if existing.data:
            raise HTTPException(
                status_code=409,
                detail="Retrieval settings already exist",
            )

        insert_data = payload.model_dump()
        result = supabase.table("chat_settings").insert(insert_data).execute()

        if not result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to create retrieval settings",
            )

        return {
            "message": "Retrieval settings created successfully",
            "data": _strip_singleton_field(result.data[0]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create settings: {str(e)}",
        )


@router.put("/retrieval-settings")
async def update_retrieval_settings(
    payload: ChatSettingsUpdate,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    _validate_settings_payload(payload)

    try:
        existing = supabase.table("chat_settings").select("id").limit(1).execute()

        if not existing.data:
            raise HTTPException(
                status_code=404,
                detail="Retrieval settings not found",
            )

        settings_id = existing.data[0]["id"]
        update_data = payload.model_dump()

        result = (
            supabase.table("chat_settings")
            .update(update_data)
            .eq("id", settings_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update retrieval settings",
            )

        return {
            "message": "Retrieval settings updated successfully",
            "data": _strip_singleton_field(result.data[0]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update settings: {str(e)}",
        )
