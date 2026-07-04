"""
Document comparison v2 — SubQuestionQueryEngine retrieval pipeline.

Reuses planning agents from compareService (v1) but replaces full-text
comparison with LlamaIndex SubQuestionQueryEngine per-document retrieval.
"""

from __future__ import annotations

from typing import List, Tuple

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.index import ComparisonAgentStep, DocumentOutline, TopicComparisonBatch
from src.rag.compare.subquestion_engine import (
    build_document_query_engine_tools,
    retrieve_comparison_context,
)
from src.services.compareDocumentRetrievalService import (
    fetch_compare_documents_for_retrieval,
)
from src.services.compareDocumentService import resolve_compare_documents
from src.services.compareService import (
    TOPIC_BATCH_SIZE,
    _format_identifications_for_prompt,
    agent_extract_outline,
    agent_identify_document_types,
    agent_map_comparison_topics,
    agent_plan_comparison,
    agent_synthesize_report,
)
from src.services.llm import openAI

TOPIC_BATCH_SIZE_V2 = TOPIC_BATCH_SIZE


def agent_compare_topic_batch_v2(
    instruction: str,
    topics: List[str],
    source_filename: str,
    reference_filenames: List[str],
    retrieved_context: str,
) -> TopicComparisonBatch:
    topic_list = "\n".join(f"- {topic}" for topic in topics)
    reference_list = "\n".join(f"- {name}" for name in reference_filenames)

    messages = [
        SystemMessage(
            content=(
                "You are a legal document comparator. For each topic, compare the "
                "source document against all reference documents using ONLY the "
                "retrieved context below. "
                "Identify similarities, differences, and notable gaps. "
                "Use Vietnamese. Be specific and cite concrete legal points."
            )
        ),
        HumanMessage(
            content=(
                f"Yêu cầu người dùng: {instruction}\n\n"
                f"Chủ đề cần đối chiếu:\n{topic_list}\n\n"
                f"Tài liệu gốc: {source_filename}\n"
                f"Tài liệu tham khảo:\n{reference_list}\n\n"
                f"## Ngữ cảnh trích xuất (SubQuestionQueryEngine)\n"
                f"{retrieved_context}"
            )
        ),
    ]

    structured_llm = openAI["mini_llm"].with_structured_output(TopicComparisonBatch)
    return structured_llm.invoke(messages)


def run_document_comparison_v2(
    source_doc: dict,
    reference_docs: List[dict],
    instruction: str,
    retrieval_documents: List[dict],
    skip_report: bool = False,
) -> dict:
    steps: List[ComparisonAgentStep] = []

    if not reference_docs:
        raise HTTPException(
            status_code=422,
            detail="Cần ít nhất một tài liệu tham khảo",
        )

    steps.append(
        ComparisonAgentStep(
            agent="Parser",
            status="completed",
            summary=(
                f"Đã tải 1 tài liệu gốc và {len(reference_docs)} tài liệu tham khảo "
                "từ kho đã embed."
            ),
        )
    )

    identifications = []
    try:
        identification_batch = agent_identify_document_types(
            source_doc,
            reference_docs,
        )
        identifications = identification_batch.identifications
        type_labels = [item.document_type for item in identifications]
        steps.append(
            ComparisonAgentStep(
                agent="Classifier",
                status="completed",
                summary=(
                    f"Đã nhận diện {len(identifications)} loại văn bản từ đoạn mở đầu: "
                    f"{', '.join(type_labels)}."
                ),
            )
        )
    except Exception as error:
        steps.append(
            ComparisonAgentStep(
                agent="Classifier",
                status="failed",
                summary=str(error),
            )
        )
        raise HTTPException(
            status_code=500,
            detail=f"Không thể phân loại tài liệu: {error}",
        )

    document_context = _format_identifications_for_prompt(identifications)

    try:
        plan = agent_plan_comparison(
            instruction,
            source_doc["filename"],
            [doc["filename"] for doc in reference_docs],
            document_context,
        )
        steps.append(
            ComparisonAgentStep(
                agent="Planner",
                status="completed",
                summary=(
                    f"Đã lập kế hoạch với {len(plan.focus_areas)} trọng tâm "
                    f"và {len(plan.comparison_dimensions)} chiều đối chiếu."
                ),
            )
        )
    except Exception as error:
        steps.append(
            ComparisonAgentStep(
                agent="Planner",
                status="failed",
                summary=str(error),
            )
        )
        raise HTTPException(
            status_code=500,
            detail=f"Không thể lập kế hoạch so sánh: {error}",
        )

    source_outline = None
    reference_outlines: List[Tuple[str, DocumentOutline]] = []

    try:
        source_outline = agent_extract_outline(
            source_doc["filename"], source_doc["fullText"]
        )
        for doc in reference_docs:
            outline = agent_extract_outline(doc["filename"], doc["fullText"])
            reference_outlines.append((doc["filename"], outline))

        steps.append(
            ComparisonAgentStep(
                agent="Analyst",
                status="completed",
                summary=(
                    f"Đã phân tích cấu trúc {1 + len(reference_docs)} tài liệu "
                    f"({len(source_outline.sections)} mục gốc)."
                ),
            )
        )
    except Exception as error:
        steps.append(
            ComparisonAgentStep(
                agent="Analyst",
                status="failed",
                summary=str(error),
            )
        )
        raise HTTPException(
            status_code=500,
            detail=f"Không thể phân tích tài liệu: {error}",
        )

    try:
        topic_map = agent_map_comparison_topics(
            instruction,
            plan,
            source_doc["filename"],
            source_outline,
            reference_outlines,
        )
        steps.append(
            ComparisonAgentStep(
                agent="Mapper",
                status="completed",
                summary=f"Đã xác định {len(topic_map.topics)} chủ đề đối chiếu.",
            )
        )
    except Exception as error:
        steps.append(
            ComparisonAgentStep(
                agent="Mapper",
                status="failed",
                summary=str(error),
            )
        )
        raise HTTPException(
            status_code=500,
            detail=f"Không thể lập chủ đề đối chiếu: {error}",
        )

    all_comparisons = []
    topic_titles = [topic.title for topic in topic_map.topics]
    reference_filenames = [doc["filename"] for doc in reference_docs]
    query_engine_tools = None
    retrieved_batches = 0

    try:
        query_engine_tools = build_document_query_engine_tools(retrieval_documents)

        for start in range(0, len(topic_titles), TOPIC_BATCH_SIZE_V2):
            batch_topics = topic_titles[start : start + TOPIC_BATCH_SIZE_V2]
            retrieved_context = retrieve_comparison_context(
                instruction,
                batch_topics,
                query_engine_tools,
            )
            retrieved_batches += 1

            batch_result = agent_compare_topic_batch_v2(
                instruction,
                batch_topics,
                source_doc["filename"],
                reference_filenames,
                retrieved_context,
            )
            all_comparisons.extend(batch_result.comparisons)

        steps.append(
            ComparisonAgentStep(
                agent="Retriever",
                status="completed",
                summary=(
                    f"Đã tra cứu ngữ cảnh qua SubQuestionQueryEngine "
                    f"cho {retrieved_batches} nhóm chủ đề "
                    f"({len(query_engine_tools)} tài liệu)."
                ),
            )
        )
        steps.append(
            ComparisonAgentStep(
                agent="Comparator",
                status="completed",
                summary=f"Đã đối chiếu {len(all_comparisons)} chủ đề.",
            )
        )
    except Exception as error:
        failed_agent = "Retriever" if retrieved_batches == 0 else "Comparator"
        steps.append(
            ComparisonAgentStep(
                agent=failed_agent,
                status="failed",
                summary=str(error),
            )
        )
        raise HTTPException(
            status_code=500,
            detail=f"Không thể đối chiếu tài liệu (v2): {error}",
        )

    report_context = {
        "instruction": instruction,
        "plan": plan,
        "source_filename": source_doc["filename"],
        "reference_filenames": reference_filenames,
        "comparisons": all_comparisons,
    }

    if skip_report:
        return {
            "version": "v2",
            "report": "",
            "_reportContext": report_context,
            "steps": [step.model_dump() for step in steps],
            "plan": {
                "objectives": plan.objectives,
                "focusAreas": plan.focus_areas,
                "comparisonDimensions": plan.comparison_dimensions,
            },
            "documentContext": [
                {
                    "filename": item.filename,
                    "role": item.role,
                    "documentType": item.document_type,
                    "titleOrSubject": item.title_or_subject,
                    "summary": item.summary,
                    "legalDomain": item.legal_domain,
                }
                for item in identifications
            ],
            "sourceDocument": {
                "id": source_doc["id"],
                "filename": source_doc["filename"],
            },
            "referenceDocuments": [
                {"id": doc["id"], "filename": doc["filename"]} for doc in reference_docs
            ],
        }

    try:
        report = agent_synthesize_report(
            instruction,
            plan,
            source_doc["filename"],
            reference_filenames,
            all_comparisons,
        )
        steps.append(
            ComparisonAgentStep(
                agent="Reporter",
                status="completed",
                summary="Đã tổng hợp báo cáo Markdown.",
            )
        )
    except Exception as error:
        steps.append(
            ComparisonAgentStep(
                agent="Reporter",
                status="failed",
                summary=str(error),
            )
        )
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tạo báo cáo: {error}",
        )

    return {
        "version": "v2",
        "report": report,
        "steps": [step.model_dump() for step in steps],
        "plan": {
            "objectives": plan.objectives,
            "focusAreas": plan.focus_areas,
            "comparisonDimensions": plan.comparison_dimensions,
        },
        "documentContext": [
            {
                "filename": item.filename,
                "role": item.role,
                "documentType": item.document_type,
                "titleOrSubject": item.title_or_subject,
                "summary": item.summary,
                "legalDomain": item.legal_domain,
            }
            for item in identifications
        ],
        "sourceDocument": {
            "id": source_doc["id"],
            "filename": source_doc["filename"],
        },
        "referenceDocuments": [
            {"id": doc["id"], "filename": doc["filename"]} for doc in reference_docs
        ],
    }


def run_document_comparison_v2_by_ids(
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: List[str],
    instruction: str,
) -> dict:
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
    return run_document_comparison_v2(
        source_doc,
        reference_docs,
        instruction,
        retrieval_documents,
    )
