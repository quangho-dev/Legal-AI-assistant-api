from __future__ import annotations

import re
from collections.abc import Iterator
from typing import List

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.index import (
    ContractDraftRequest,
    ContractExportDocxRequest,
    ContractLanguage,
)
from src.utils.markdown_docx import markdown_to_docx_bytes
from src.rag.retrieval.utils import get_chat_settings
from src.services.contractDocumentService import fetch_contract_document_content
from src.services.llm import get_chat_llm

DOCUMENT_TEXT_LIMIT = 12000
MAX_DESCRIPTION_DOCUMENTS = 5

LANGUAGE_LABELS = {
    ContractLanguage.VI: "tiếng Việt",
    ContractLanguage.EN: "English",
}


def _truncate_text(text: str, limit: int = DOCUMENT_TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... nội dung đã rút gọn ...]"


def _serialize_document_ref(document: dict) -> dict:
    return {
        "id": document["id"],
        "filename": document["filename"],
    }


def _resolve_draft_documents(
    clerk_id: str,
    template_document_id: str | None,
    description_document_ids: List[str],
) -> tuple[dict | None, List[dict]]:
    unique_description_ids = list(dict.fromkeys(description_document_ids))

    if len(unique_description_ids) > MAX_DESCRIPTION_DOCUMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"Chỉ được chọn tối đa {MAX_DESCRIPTION_DOCUMENTS} tài liệu mô tả",
        )

    if template_document_id and template_document_id in unique_description_ids:
        raise HTTPException(
            status_code=422,
            detail="Hợp đồng mẫu không được trùng với tài liệu mô tả",
        )

    template_document = None
    if template_document_id:
        template_document = fetch_contract_document_content(
            clerk_id,
            template_document_id,
        )

    description_documents = [
        fetch_contract_document_content(clerk_id, document_id)
        for document_id in unique_description_ids
    ]

    return template_document, description_documents


def _build_reference_context(
    template_document: dict | None,
    description_documents: List[dict],
) -> str:
    context_blocks: list[str] = []

    if template_document:
        context_blocks.append(
            "## Hợp đồng mẫu tham khảo\n"
            f"Tên file: {template_document['filename']}\n\n"
            f"{_truncate_text(template_document['fullText'])}"
        )

    if description_documents:
        for document in description_documents:
            context_blocks.append(
                f"## Tài liệu mô tả: {document['filename']}\n\n"
                f"{_truncate_text(document['fullText'])}"
            )

    return (
        "\n\n".join(context_blocks)
        if context_blocks
        else "Không có hợp đồng mẫu hoặc tài liệu mô tả đính kèm."
    )


def build_contract_outline_messages(
    request: ContractDraftRequest,
    template_document: dict | None,
    description_documents: List[dict],
) -> list:
    language_label = LANGUAGE_LABELS[request.language]
    reference_context = _build_reference_context(
        template_document,
        description_documents,
    )

    if request.language == ContractLanguage.VI:
        system_prompt = (
            "Bạn là luật sư soạn thảo hợp đồng chuyên nghiệp tại Việt Nam. "
            "Nhiệm vụ: lập DÀN Ý PHÁC THẢO hợp đồng (chưa viết văn bản đầy đủ).\n\n"
            "Quy tắc:\n"
            "- Bảo vệ lợi ích hợp lý của bên người dùng đại diện.\n"
            "- Dùng Markdown. Không dùng code block.\n"
            "- Cấu trúc dàn ý:\n"
            "  1. **Tên hợp đồng đề xuất**\n"
            "  2. **Các bên tham gia** (vai trò, thông tin cần điền)\n"
            "  3. **Mục đích và phạm vi**\n"
            "  4. **Cấu trúc điều khoản** — liệt kê từng mục/điều với gạch đầu dòng "
            "mô tả ngắn nội dung cần có\n"
            "  5. **Điểm cần chú ý / ưu tiên bảo vệ** cho bên người dùng\n"
            "  6. **Thông tin còn thiếu** cần người dùng bổ sung\n"
            "- Ngắn gọn, rõ ràng, dễ duyệt trước khi soạn chi tiết."
        )
        user_prompt = (
            f"Ngôn ngữ hợp đồng: {language_label}\n\n"
            f"Bạn đại diện cho bên: {request.partyRole}\n\n"
            f"Mô tả yêu cầu:\n{request.requirements}\n\n"
            f"Tài liệu tham khảo:\n{reference_context}\n\n"
            "Hãy lập dàn ý phác thảo hợp đồng."
        )
    else:
        system_prompt = (
            "You are a senior contract drafting lawyer. "
            "Task: produce a CONTRACT OUTLINE (not the full contract yet).\n\n"
            "Rules:\n"
            "- Protect the legitimate interests of the party the user represents.\n"
            "- Use Markdown. No code blocks.\n"
            "- Outline structure:\n"
            "  1. **Proposed contract title**\n"
            "  2. **Parties** (roles, placeholders needed)\n"
            "  3. **Purpose and scope**\n"
            "  4. **Clause structure** — list each section/article with bullet points "
            "describing what it should cover\n"
            "  5. **Key protections / priorities** for the user's party\n"
            "  6. **Missing information** the user should provide\n"
            "- Be concise and easy to review before full drafting."
        )
        user_prompt = (
            f"Contract language: {language_label}\n\n"
            f"You represent the party: {request.partyRole}\n\n"
            f"Requirements description:\n{request.requirements}\n\n"
            f"Reference materials:\n{reference_context}\n\n"
            "Produce the contract outline."
        )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def build_contract_full_messages(
    request: ContractDraftRequest,
    template_document: dict | None,
    description_documents: List[dict],
    outline: str,
) -> list:
    language_label = LANGUAGE_LABELS[request.language]
    reference_context = _build_reference_context(
        template_document,
        description_documents,
    )

    if request.language == ContractLanguage.VI:
        system_prompt = (
            "Bạn là luật sư soạn thảo hợp đồng chuyên nghiệp tại Việt Nam. "
            "Soạn HỢP ĐỒNG HOÀN CHỈNH bằng tiếng Việt dựa trên dàn ý đã duyệt.\n\n"
            "Quy tắc:\n"
            "- Tuân thủ đúng cấu trúc và nội dung trong dàn ý phác thảo.\n"
            "- Bảo vệ lợi ích hợp lý của bên người dùng đại diện, ngôn ngữ chuyên nghiệp.\n"
            "- Viết đầy đủ điều khoản chi tiết, không chỉ liệt kê tiêu đề.\n"
            "- Dùng Markdown. Không dùng code block. Điền [TÊN/ĐỊA CHỈ/...] cho thông tin thiếu.\n"
            "- Không bịa điều khoản ngoài dàn ý và yêu cầu."
        )
        user_prompt = (
            f"Ngôn ngữ hợp đồng: {language_label}\n\n"
            f"Bạn đại diện cho bên: {request.partyRole}\n\n"
            f"Mô tả yêu cầu:\n{request.requirements}\n\n"
            f"Tài liệu tham khảo:\n{reference_context}\n\n"
            f"Dàn ý phác thảo đã duyệt:\n{outline}\n\n"
            "Hãy soạn toàn bộ nội dung hợp đồng theo dàn ý trên."
        )
    else:
        system_prompt = (
            "You are a senior contract drafting lawyer. Draft the COMPLETE contract in English "
            "following the approved outline.\n\n"
            "Rules:\n"
            "- Follow the outline structure and substance exactly.\n"
            "- Protect the user's party with balanced, professional language.\n"
            "- Write full clause text, not just headings.\n"
            "- Use Markdown. No code blocks. Use [NAME/ADDRESS/...] placeholders.\n"
            "- Do not invent clauses outside the outline and requirements."
        )
        user_prompt = (
            f"Contract language: {language_label}\n\n"
            f"You represent the party: {request.partyRole}\n\n"
            f"Requirements description:\n{request.requirements}\n\n"
            f"Reference materials:\n{reference_context}\n\n"
            f"Approved outline:\n{outline}\n\n"
            "Draft the full contract text following this outline."
        )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def _build_draft_result(
    request: ContractDraftRequest,
    outline: str,
    template_document: dict | None,
    description_documents: List[dict],
) -> dict:
    return {
        "language": request.language.value,
        "partyRole": request.partyRole,
        "requirements": request.requirements,
        "outline": outline,
        "templateDocument": (
            _serialize_document_ref(template_document) if template_document else None
        ),
        "descriptionDocuments": [
            _serialize_document_ref(document) for document in description_documents
        ],
    }


def draft_contract(
    clerk_id: str,
    request: ContractDraftRequest,
) -> dict:
    template_document, description_documents = _resolve_draft_documents(
        clerk_id,
        request.templateDocumentId,
        request.descriptionDocumentIds,
    )

    settings = get_chat_settings()
    chat_model = settings.get("chat_model", "gpt-4o")
    chat_llm = get_chat_llm(chat_model)

    outline_response = chat_llm.invoke(
        build_contract_outline_messages(
            request,
            template_document,
            description_documents,
        )
    )
    outline = (
        outline_response.content
        if isinstance(outline_response.content, str)
        else str(outline_response.content)
    )

    return _build_draft_result(
        request,
        outline,
        template_document,
        description_documents,
    )


def iter_contract_draft_stream(
    clerk_id: str,
    request: ContractDraftRequest,
) -> Iterator[str]:
    from src.utils.sse import sse_done, sse_error, sse_outline_token, sse_status

    try:
        yield sse_status("Đang tải tài liệu tham khảo...")

        template_document, description_documents = _resolve_draft_documents(
            clerk_id,
            request.templateDocumentId,
            request.descriptionDocumentIds,
        )

        settings = get_chat_settings()
        chat_model = settings.get("chat_model", "gpt-4o")
        chat_llm = get_chat_llm(chat_model)

        yield sse_status("Đang lập dàn ý phác thảo...")

        outline_parts: list[str] = []
        for chunk in chat_llm.stream(
            build_contract_outline_messages(
                request,
                template_document,
                description_documents,
            )
        ):
            content = chunk.content
            if isinstance(content, str) and content:
                outline_parts.append(content)
                yield sse_outline_token(content)

        outline = "".join(outline_parts)
        yield sse_done(
            _build_draft_result(
                request,
                outline,
                template_document,
                description_documents,
            )
        )
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, str) else str(error.detail)
        yield sse_error(detail)
    except Exception as error:
        yield sse_error(str(error))


def _derive_docx_filename(outline: str, language: ContractLanguage) -> str:
    for line in outline.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        candidate = stripped.lstrip("#").strip()
        candidate = re.sub(r"^\*\*|\*\*$", "", candidate).strip()
        candidate = re.sub(r"^[\d.]+\s*", "", candidate).strip()
        if len(candidate) >= 3:
            safe = re.sub(r'[<>:"/\\|?*]+', "", candidate)
            safe = re.sub(r"\s+", "-", safe).strip("-")
            if safe:
                return f"{safe[:80]}.docx"

    prefix = "hop-dong" if language == ContractLanguage.VI else "contract"
    return f"{prefix}.docx"


def generate_full_contract_from_outline(
    clerk_id: str,
    request: ContractExportDocxRequest,
) -> str:
    draft_request = ContractDraftRequest(
        language=request.language,
        requirements=request.requirements,
        partyRole=request.partyRole,
        templateDocumentId=request.templateDocumentId,
        descriptionDocumentIds=request.descriptionDocumentIds,
    )

    template_document, description_documents = _resolve_draft_documents(
        clerk_id,
        request.templateDocumentId,
        request.descriptionDocumentIds,
    )

    settings = get_chat_settings()
    chat_model = settings.get("chat_model", "gpt-4o")
    chat_llm = get_chat_llm(chat_model)

    response = chat_llm.invoke(
        build_contract_full_messages(
            draft_request,
            template_document,
            description_documents,
            request.outline.strip(),
        )
    )

    return (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )


def export_contract_docx(
    clerk_id: str,
    request: ContractExportDocxRequest,
) -> tuple[bytes, str]:
    contract_text = generate_full_contract_from_outline(clerk_id, request)
    docx_bytes = markdown_to_docx_bytes(contract_text)
    filename = _derive_docx_filename(request.outline, request.language)
    return docx_bytes, filename
