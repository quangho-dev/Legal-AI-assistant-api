from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.services.clerkAuth import get_current_user_clerk_id
from src.services.caseLawSearchService import (
    get_case_law,
    list_case_laws,
    search_case_laws,
)

router = APIRouter(tags=["caseLawRoutes"])


@router.get("/search")
async def search_case_laws_endpoint(
    q: str = Query(default="", description="Từ khóa tra cứu án lệ"),
    linhVuc: Optional[int] = Query(default=None, description="Mã lĩnh vực"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        data = search_case_laws(
            query=q.strip(),
            linh_vuc=linhVuc,
            limit=limit,
            offset=offset,
        )
        return {
            "message": "Case laws retrieved successfully",
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tra cứu án lệ: {error}",
        )


@router.get("")
async def list_case_laws_endpoint(
    linhVuc: Optional[int] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        data = list_case_laws(linh_vuc=linhVuc, limit=limit, offset=offset)
        return {
            "message": "Case laws listed successfully",
            "data": data,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải danh sách án lệ: {error}",
        )


@router.get("/{case_law_id}")
async def get_case_law_endpoint(
    case_law_id: str,
    _clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        return {
            "message": "Case law retrieved successfully",
            "data": get_case_law(case_law_id),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải chi tiết án lệ: {error}",
        )
