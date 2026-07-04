"""
Document comparison v3 — LangGraph workflow.

Initial workflow: decompose user question into per-document retrieval questions
based on user role and document context.
"""

from __future__ import annotations

from fastapi import HTTPException

from src.models.index import ComparisonAgentStep
from src.rag.compare.v3.state import (
    build_initial_compare_state,
    serialize_workflow_state,
)
from src.rag.compare.v3.tracing import build_workflow_run_config
from src.rag.compare.v3.workflow import (
    run_compare_v3_retrieval_workflow,
    run_compare_v3_workflow,
)
from src.services.compareDocumentRetrievalService import (
    fetch_compare_documents_for_retrieval,
)
from src.services.compareDocumentService import resolve_compare_documents
from src.services.compareService import agent_identify_document_types


def prepare_compare_v3_initial_state(
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: list[str],
    instruction: str,
    user_role: str,
) -> tuple[dict, dict]:
    source_doc, reference_docs = resolve_compare_documents(
        clerk_id,
        source_document_id,
        reference_document_ids,
    )
    retrieval_documents = fetch_compare_documents_for_retrieval(
        clerk_id,
        source_document_id,
        reference_document_ids,
    )

    steps = [
        ComparisonAgentStep(
            agent="Parser",
            status="completed",
            summary=(
                f"Đã tải 1 tài liệu gốc và {len(reference_docs)} tài liệu quy chiếu."
            ),
        )
    ]

    identification_batch = agent_identify_document_types(
        source_doc,
        reference_docs,
    )
    document_context = [
        {
            "filename": item.filename,
            "role": item.role,
            "documentType": item.document_type,
            "titleOrSubject": item.title_or_subject,
            "summary": item.summary,
            "legalDomain": item.legal_domain,
        }
        for item in identification_batch.identifications
    ]
    steps.append(
        ComparisonAgentStep(
            agent="Classifier",
            status="completed",
            summary=(
                f"Đã nhận diện {len(document_context)} loại văn bản "
                "để hỗ trợ tách câu hỏi."
            ),
        )
    )

    initial_state = build_initial_compare_state(
        user_question=instruction,
        user_role=user_role,
        source_document=source_doc,
        reference_documents=reference_docs,
        document_context=document_context,
        retrieval_documents=retrieval_documents,
    )
    initial_state["steps"] = [step.model_dump() for step in steps]

    run_config = build_workflow_run_config(
        clerk_id=clerk_id,
        source_document_id=source_document_id,
        reference_document_ids=reference_document_ids,
        user_role=user_role,
        source_filename=source_doc.get("filename", ""),
    )

    return initial_state, run_config


def run_document_comparison_v3_by_ids(
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: list[str],
    instruction: str,
    user_role: str,
) -> dict:
    try:
        initial_state, run_config = prepare_compare_v3_initial_state(
            clerk_id,
            source_document_id,
            reference_document_ids,
            instruction,
            user_role,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể phân loại tài liệu (v3): {error}",
        )

    try:
        final_state = run_compare_v3_workflow(initial_state, run_config=run_config)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể chạy workflow so sánh (v3): {error}",
        )

    return serialize_workflow_state(final_state)
