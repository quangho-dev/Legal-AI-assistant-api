from fastapi import HTTPException

from src.services.clerkAuth import _get_clerk_sdk
from src.services.supabase import supabase

PENDING_PLAN = "pending"
DEFAULT_PLAN = "basic"
DEFAULT_QUESTION_LIMIT = 50

PLAN_LABELS = {
    PENDING_PLAN: "Chưa kích hoạt",
    "basic": "Gói cơ bản",
}

PLAN_LIMITS = {
    "basic": DEFAULT_QUESTION_LIMIT,
}

AVAILABLE_PLANS = [plan for plan in PLAN_LABELS if plan != PENDING_PLAN]

PENDING_ACCOUNT_MESSAGE = (
    "Tài khoản chưa được kích hoạt. Vui lòng liên hệ quản trị viên "
    "để được cấp gói sử dụng."
)


def is_plan_active(plan: str | None) -> bool:
    return plan in AVAILABLE_PLANS


def format_usage_for_client(user: dict) -> dict:
    plan = user.get("plan") or PENDING_PLAN
    question_limit = int(user.get("question_limit") or 0)
    questions_used = int(user.get("questions_used") or 0)
    can_use_app = is_plan_active(plan) and question_limit > 0

    return {
        "plan": plan,
        "planLabel": PLAN_LABELS.get(plan, plan),
        "questionLimit": question_limit,
        "questionsUsed": questions_used,
        "questionsRemaining": max(question_limit - questions_used, 0) if can_use_app else 0,
        "canUseApp": can_use_app,
        "isPending": plan == PENDING_PLAN,
    }


def ensure_user_with_quota(clerk_id: str) -> dict:
    existing = (
        supabase.table("users")
        .select("clerk_id, plan, question_limit, questions_used")
        .eq("clerk_id", clerk_id)
        .execute()
    )

    if existing.data:
        return existing.data[0]

    created = (
        supabase.table("users")
        .insert(
            {
                "clerk_id": clerk_id,
                "plan": PENDING_PLAN,
                "question_limit": 0,
                "questions_used": 0,
            }
        )
        .execute()
    )

    if not created.data:
        raise HTTPException(status_code=500, detail="Không thể tạo tài khoản người dùng")

    return created.data[0]


def assert_user_can_use_app(clerk_id: str) -> dict:
    user = ensure_user_with_quota(clerk_id)
    if not is_plan_active(user.get("plan")):
        raise HTTPException(status_code=403, detail=PENDING_ACCOUNT_MESSAGE)
    return user


def get_user_usage(clerk_id: str) -> dict:
    user = ensure_user_with_quota(clerk_id)
    return format_usage_for_client(user)


def consume_question_quota(clerk_id: str) -> dict:
    user = assert_user_can_use_app(clerk_id)
    question_limit = int(user.get("question_limit") or 0)
    questions_used = int(user.get("questions_used") or 0)

    if questions_used >= question_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Bạn đã dùng hết {question_limit} lần hỏi của "
                f"{PLAN_LABELS.get(user.get('plan') or DEFAULT_PLAN, 'gói hiện tại')}. "
                "Vui lòng nâng cấp gói để tiếp tục."
            ),
        )

    updated = (
        supabase.table("users")
        .update({"questions_used": questions_used + 1})
        .eq("clerk_id", clerk_id)
        .eq("questions_used", questions_used)
        .execute()
    )

    if not updated.data:
        latest = get_user_usage(clerk_id)
        if latest["questionsRemaining"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Bạn đã dùng hết {latest['questionLimit']} lần hỏi của "
                    f"{latest['planLabel']}. Vui lòng nâng cấp gói để tiếp tục."
                ),
            )
        raise HTTPException(
            status_code=409,
            detail="Không thể cập nhật lượt hỏi. Vui lòng thử lại.",
        )

    return format_usage_for_client(updated.data[0])


def _fetch_clerk_profile(clerk_id: str) -> dict:
    try:
        user = _get_clerk_sdk().users.get(user_id=clerk_id)
        emails = user.email_addresses or []
        primary = next(
            (
                email
                for email in emails
                if email.id == user.primary_email_address_id
            ),
            emails[0] if emails else None,
        )
        name = " ".join(
            part for part in [user.first_name, user.last_name] if part
        ).strip()

        return {
            "email": primary.email_address if primary else None,
            "name": name or None,
        }
    except Exception:
        return {"email": None, "name": None}


def format_admin_user_for_client(user: dict) -> dict:
    profile = _fetch_clerk_profile(user["clerk_id"])
    usage = format_usage_for_client(user)

    return {
        "clerkId": user["clerk_id"],
        "email": profile["email"],
        "name": profile["name"],
        "createdAt": user.get("created_at"),
        **usage,
    }


def list_all_users_for_admin() -> list[dict]:
    result = (
        supabase.table("users")
        .select("clerk_id, plan, question_limit, questions_used, created_at")
        .order("created_at", desc=True)
        .execute()
    )

    return [format_admin_user_for_client(user) for user in result.data or []]


def update_user_usage_for_admin(
    clerk_id: str,
    plan: str | None = None,
    question_limit: int | None = None,
    questions_used: int | None = None,
) -> dict:
    existing = (
        supabase.table("users")
        .select("clerk_id, plan, question_limit, questions_used, created_at")
        .eq("clerk_id", clerk_id)
        .execute()
    )

    if not existing.data:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")

    current = existing.data[0]
    next_plan = plan if plan is not None else current.get("plan") or PENDING_PLAN

    if next_plan not in PLAN_LABELS:
        raise HTTPException(status_code=422, detail="Gói sử dụng không hợp lệ")

    if next_plan == PENDING_PLAN:
        next_limit = 0
        next_used = 0
    else:
        next_used = (
            questions_used
            if questions_used is not None
            else int(current.get("questions_used") or 0)
        )

        if question_limit is not None:
            next_limit = question_limit
        elif plan is not None:
            next_limit = PLAN_LIMITS.get(next_plan, DEFAULT_QUESTION_LIMIT)
        else:
            next_limit = int(
                current.get("question_limit") or PLAN_LIMITS.get(next_plan, 0)
            )

        if next_limit < 1:
            raise HTTPException(
                status_code=422,
                detail="Giới hạn lần hỏi phải lớn hơn 0 cho gói đang hoạt động",
            )

        if next_used > next_limit:
            raise HTTPException(
                status_code=422,
                detail="Số lần đã dùng không được lớn hơn giới hạn gói",
            )

    updated = (
        supabase.table("users")
        .update(
            {
                "plan": next_plan,
                "question_limit": next_limit,
                "questions_used": next_used,
            }
        )
        .eq("clerk_id", clerk_id)
        .execute()
    )

    if not updated.data:
        raise HTTPException(
            status_code=500,
            detail="Không thể cập nhật gói sử dụng của người dùng",
        )

    return format_admin_user_for_client(updated.data[0])


def get_available_plans_for_client() -> list[dict]:
    return [
        {"value": plan, "label": PLAN_LABELS[plan]}
        for plan in [PENDING_PLAN, *AVAILABLE_PLANS]
    ]
