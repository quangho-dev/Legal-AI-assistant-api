"""
Agentic document comparison service.

Multi-agent pipeline:
0. Parser — load embedded documents from user library
1. Classifier — infer document types from opening excerpts
2. Planner — interpret user instruction and define comparison plan
3. Analyst — extract structured outline per document
4. Mapper — define comparison topics across documents
5. Comparator — diff source vs references per topic
6. Reporter — synthesize final Markdown report
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.index import (
    ComparisonAgentStep,
    ComparisonPlan,
    ComparisonTopicMap,
    DocumentIdentification,
    DocumentIdentificationBatch,
    DocumentOutline,
    TopicComparisonBatch,
)
from src.services.compareDocumentService import resolve_compare_documents
from src.rag.retrieval.utils import get_chat_settings
from src.services.llm import get_chat_llm, openAI

DOCUMENT_TEXT_LIMIT = 18000
OUTLINE_INPUT_LIMIT = 14000
OPENING_PREVIEW_LENGTH = 500
TOPIC_BATCH_SIZE = 4


def _get_opening_excerpt(full_text: str) -> str:
    text = full_text.strip()
    if len(text) <= OPENING_PREVIEW_LENGTH:
        return text
    return f"{text[:OPENING_PREVIEW_LENGTH]}…"


def _format_identifications_for_prompt(
    identifications: List[DocumentIdentification],
) -> str:
    blocks = []
    for item in identifications:
        role_label = "Tài liệu gốc" if item.role == "source" else "Tài liệu tham khảo"
        blocks.append(
            f"- {role_label}: {item.document_type}\n"
            f"  Chủ đề: {item.title_or_subject}\n"
            f"  Tóm tắt: {item.summary}"
            + (f"\n  Lĩnh vực: {item.legal_domain}" if item.legal_domain else "")
        )
    return "\n".join(blocks)


def agent_identify_document_types(
    source_doc: dict,
    reference_docs: List[dict],
) -> DocumentIdentificationBatch:
    blocks = [
        (
            f"### Tài liệu gốc\n"
            f"Tên file (tham khảo): {source_doc['filename']}\n"
            f"Đoạn mở đầu ({OPENING_PREVIEW_LENGTH} ký tự đầu):\n"
            f"{_get_opening_excerpt(source_doc['fullText'])}"
        )
    ]

    for index, doc in enumerate(reference_docs, start=1):
        blocks.append(
            (
                f"### Tài liệu tham khảo {index}\n"
                f"Tên file (tham khảo): {doc['filename']}\n"
                f"Đoạn mở đầu ({OPENING_PREVIEW_LENGTH} ký tự đầu):\n"
                f"{_get_opening_excerpt(doc['fullText'])}"
            )
        )

    messages = [
        SystemMessage(
            content=(
                "You are a legal document classifier. Infer each document's type, "
                "subject, and legal domain from the opening excerpt only. "
                "Do not rely primarily on filenames — use the actual opening text. "
                "Respond in Vietnamese. Return one identification per document with "
                "role set to 'source' or 'reference'."
            )
        ),
        HumanMessage(
            content=(
                "Phân loại các tài liệu sau dựa trên đoạn mở đầu:\n\n"
                f"{chr(10).join(blocks)}"
            )
        ),
    ]

    structured_llm = openAI["mini_llm"].with_structured_output(
        DocumentIdentificationBatch
    )
    return structured_llm.invoke(messages)


def agent_plan_comparison(
    instruction: str,
    source_filename: str,
    reference_filenames: List[str],
    document_context: str,
) -> ComparisonPlan:
    messages = [
        SystemMessage(
            content=(
                "You are a legal document comparison planner. "
                "Create a focused comparison plan based on the user's instruction "
                "and the inferred document types from opening excerpts. "
                "Respond in Vietnamese for objectives, focus_areas, and comparison_dimensions."
            )
        ),
        HumanMessage(
            content=(
                f"Yêu cầu người dùng:\n{instruction}\n\n"
                f"Ngữ cảnh loại văn bản (từ đoạn mở đầu, không dựa vào tên file):\n"
                f"{document_context}\n\n"
                f"Tài liệu gốc (tên file): {source_filename}\n"
                f"Tài liệu tham khảo (tên file): {', '.join(reference_filenames)}\n\n"
                "Lập kế hoạch so sánh gồm mục tiêu, trọng tâm và các chiều đối chiếu."
            )
        ),
    ]

    structured_llm = openAI["mini_llm"].with_structured_output(ComparisonPlan)
    return structured_llm.invoke(messages)


def _truncate_for_outline(text: str) -> str:
    if len(text) <= OUTLINE_INPUT_LIMIT:
        return text
    return f"{text[:OUTLINE_INPUT_LIMIT]}…"


def _format_outline_for_prompt(outline: DocumentOutline) -> str:
    blocks = []
    for section in outline.sections:
        blocks.append(
            f"- {section.title}: {section.summary}"
            + (f"\n  Trích: {section.excerpt[:400]}" if section.excerpt else "")
        )
    return "\n".join(blocks) if blocks else "(Không có cấu trúc)"


def agent_extract_outline(filename: str, full_text: str) -> DocumentOutline:
    messages = [
        SystemMessage(
            content=(
                "You are a legal document analyst. Extract a structured outline "
                "of the document with key sections, summaries, and short excerpts. "
                "Use Vietnamese for titles and summaries. "
                "Return 5-12 sections maximum."
            )
        ),
        HumanMessage(
            content=(
                f"Tài liệu: {filename}\n\n"
                f"Nội dung:\n{_truncate_for_outline(full_text)}"
            )
        ),
    ]

    structured_llm = openAI["mini_llm"].with_structured_output(DocumentOutline)
    return structured_llm.invoke(messages)


def agent_map_comparison_topics(
    instruction: str,
    plan: ComparisonPlan,
    source_filename: str,
    source_outline: DocumentOutline,
    reference_docs: List[Tuple[str, DocumentOutline]],
) -> ComparisonTopicMap:
    reference_blocks = []
    for filename, outline in reference_docs:
        reference_blocks.append(
            f"### {filename}\n{_format_outline_for_prompt(outline)}"
        )

    messages = [
        SystemMessage(
            content=(
                "You are a legal comparison mapper. Define 4-8 concrete comparison "
                "topics that align source and reference documents according to the plan. "
                "Use Vietnamese."
            )
        ),
        HumanMessage(
            content=(
                f"Yêu cầu: {instruction}\n\n"
                f"Mục tiêu: {', '.join(plan.objectives)}\n"
                f"Trọng tâm: {', '.join(plan.focus_areas)}\n"
                f"Chiều đối chiếu: {', '.join(plan.comparison_dimensions)}\n\n"
                f"## Tài liệu gốc: {source_filename}\n"
                f"{_format_outline_for_prompt(source_outline)}\n\n"
                f"## Tài liệu tham khảo\n"
                f"{chr(10).join(reference_blocks)}"
            )
        ),
    ]

    structured_llm = openAI["mini_llm"].with_structured_output(ComparisonTopicMap)
    return structured_llm.invoke(messages)


def agent_compare_topic_batch(
    instruction: str,
    topics: List[str],
    source_filename: str,
    source_text: str,
    reference_docs: List[dict],
) -> TopicComparisonBatch:
    reference_blocks = []
    for doc in reference_docs:
        reference_blocks.append(
            f"### {doc['filename']}\n{_truncate_for_outline(doc['fullText'])}"
        )

    topic_list = "\n".join(f"- {topic}" for topic in topics)

    messages = [
        SystemMessage(
            content=(
                "You are a legal document comparator. For each topic, compare the "
                "source document against all reference documents. "
                "Identify similarities, differences, and notable gaps. "
                "Use Vietnamese. Be specific and cite concrete legal points."
            )
        ),
        HumanMessage(
            content=(
                f"Yêu cầu người dùng: {instruction}\n\n"
                f"Chủ đề cần đối chiếu:\n{topic_list}\n\n"
                f"## Tài liệu gốc: {source_filename}\n"
                f"{_truncate_for_outline(source_text)}\n\n"
                f"## Tài liệu tham khảo\n"
                f"{chr(10).join(reference_blocks)}"
            )
        ),
    ]

    structured_llm = openAI["mini_llm"].with_structured_output(TopicComparisonBatch)
    return structured_llm.invoke(messages)


def agent_synthesize_report(
    instruction: str,
    plan: ComparisonPlan,
    source_filename: str,
    reference_filenames: List[str],
    comparisons: List,
) -> str:
    topic_titles = [item.topic for item in comparisons]
    topic_index = "\n".join(
        f"{index}. {title}" for index, title in enumerate(topic_titles, start=1)
    )

    comparison_blocks = []
    for index, item in enumerate(comparisons, start=1):
        refs = "\n".join(f"  - {pos}" for pos in item.reference_positions)
        sims = "\n".join(f"  - {s}" for s in item.similarities)
        diffs = "\n".join(f"  - {d}" for d in item.differences)
        gaps = "\n".join(f"  - {g}" for g in item.notable_gaps)
        comparison_blocks.append(
            f"### Chủ đề {index}: {item.topic}\n"
            f"**Văn bản gốc:** {item.source_position}\n"
            f"**Văn bản tham khảo:**\n{refs}\n"
            f"**Điểm giống:**\n{sims}\n"
            f"**Điểm khác:**\n{diffs}\n"
            f"**Khoảng trống / rủi ro:**\n{gaps}"
        )

    settings = get_chat_settings()
    chat_model = settings.get("chat_model", "gpt-4o")
    chat_llm = get_chat_llm(chat_model)

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal analyst writing a comparison report in Vietnamese. "
                "Structure the report EXACTLY as follows:\n"
                "1. Header table: document names, date, objectives\n"
                "2. Executive summary in a blockquote with emoji risk indicators (✅ ⚠️ 🔴)\n"
                "3. Section '## Đối chiếu chi tiết theo chủ đề':\n"
                "   - Begin with '### Mục lục chủ đề' and a numbered list of ALL topics\n"
                "   - For EACH topic, use this EXACT visual pattern (never merge topics):\n"
                "     ---\n"
                "     ### Chủ đề {n}: {tên chủ đề}\n"
                "     *{một dòng mô tả ngắn chủ đề đối chiếu}*\n"
                "     | Tiêu chí | Văn bản gốc | [Tên từng VB tham khảo] |\n"
                "     | ... | ... | ... |\n"
                "     **Nhận xét:** ...\n"
                "     **Khuyến nghị:** ...\n"
                "   - Put a horizontal rule '---' BEFORE every topic block (including the first)\n"
                "   - Keep topic numbers sequential and matching the table of contents\n"
                "   - Each topic must be visually self-contained and easy to scan\n"
                "4. Risk summary table with 🔴🟡🟢 severity levels\n"
                "5. Numbered action recommendations grouped by priority\n"
                "Use **bold** for emphasis. No code blocks. "
                "Emoji only in summary and risk table, not in topic sections."
            )
        ),
        HumanMessage(
            content=(
                f"Yêu cầu người dùng: {instruction}\n\n"
                f"Tài liệu gốc: {source_filename}\n"
                f"Tài liệu tham khảo: {', '.join(reference_filenames)}\n\n"
                f"Mục tiêu: {', '.join(plan.objectives)}\n"
                f"Trọng tâm: {', '.join(plan.focus_areas)}\n\n"
                f"Danh sách chủ đề ({len(topic_titles)} chủ đề):\n{topic_index}\n\n"
                f"Kết quả đối chiếu chi tiết:\n\n"
                f"{chr(10).join(comparison_blocks)}"
            )
        ),
    ]

    response = chat_llm.invoke(messages)
    return response.content


def run_document_comparison(
    source_doc: dict,
    reference_docs: List[dict],
    instruction: str,
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

    # Agent 1: Classifier — infer document types from opening excerpts
    identifications: List[DocumentIdentification] = []
    try:
        identification_batch = agent_identify_document_types(
            source_doc,
            reference_docs,
        )
        identifications = identification_batch.identifications
        type_labels = [
            f"{item.document_type}" for item in identifications
        ]
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

    # Agent 2: Planner
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

    # Agent 3: Analyst — extract outlines
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

    # Agent 4: Mapper
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

    # Agent 5: Comparator — batch by topic
    all_comparisons = []
    topic_titles = [topic.title for topic in topic_map.topics]

    try:
        for start in range(0, len(topic_titles), TOPIC_BATCH_SIZE):
            batch_topics = topic_titles[start : start + TOPIC_BATCH_SIZE]
            batch_result = agent_compare_topic_batch(
                instruction,
                batch_topics,
                source_doc["filename"],
                source_doc["fullText"],
                reference_docs,
            )
            all_comparisons.extend(batch_result.comparisons)

        steps.append(
            ComparisonAgentStep(
                agent="Comparator",
                status="completed",
                summary=f"Đã đối chiếu {len(all_comparisons)} chủ đề.",
            )
        )
    except Exception as error:
        steps.append(
            ComparisonAgentStep(
                agent="Comparator",
                status="failed",
                summary=str(error),
            )
        )
        raise HTTPException(
            status_code=500,
            detail=f"Không thể đối chiếu tài liệu: {error}",
        )

    # Agent 6: Reporter
    try:
        report = agent_synthesize_report(
            instruction,
            plan,
            source_doc["filename"],
            [doc["filename"] for doc in reference_docs],
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


def run_document_comparison_by_ids(
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
    return run_document_comparison(source_doc, reference_docs, instruction)
