from fastapi import APIRouter, Depends, HTTPException

from src.models.index import AdminUpdateUserUsageRequest
from src.services.clerkAuth import require_admin_user
from src.services.usageQuotaService import (
    get_available_plans_for_client,
    list_all_users_for_admin,
    update_user_usage_for_admin,
)

router = APIRouter(tags=["adminUserRoutes"])


@router.get("/plans")
async def list_available_plans(
    _admin_clerk_id: str = Depends(require_admin_user),
):
    return {
        "message": "Plans retrieved successfully",
        "data": get_available_plans_for_client(),
    }


@router.get("")
async def list_users_usage(
    _admin_clerk_id: str = Depends(require_admin_user),
):
    try:
        users = list_all_users_for_admin()
        return {
            "message": "Users usage retrieved successfully",
            "data": users,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải danh sách người dùng: {str(e)}",
        )


@router.patch("/{clerk_id}/usage")
async def update_user_usage(
    clerk_id: str,
    payload: AdminUpdateUserUsageRequest,
    _admin_clerk_id: str = Depends(require_admin_user),
):
    if (
        payload.plan is None
        and payload.questionLimit is None
        and payload.questionsUsed is None
    ):
        raise HTTPException(
            status_code=422,
            detail="Cần ít nhất một trường để cập nhật",
        )

    try:
        user = update_user_usage_for_admin(
            clerk_id,
            plan=payload.plan,
            question_limit=payload.questionLimit,
            questions_used=payload.questionsUsed,
        )
        return {
            "message": "User usage updated successfully",
            "data": user,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể cập nhật gói sử dụng: {str(e)}",
        )
