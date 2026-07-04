from __future__ import annotations

from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from src.models.index import ComparisonAgentStep, QuestionDecompositionResult
from src.rag.compare.v3.retrieval import retrieve_document_by_questions
from src.rag.compare.v3.state import CompareWorkflowState
from src.rag.retrieval.utils import get_chat_settings
from src.services.llm import get_chat_llm, openAI

MAX_QUESTIONS_PER_DOCUMENT = 5


def _format_documents_for_decomposition(state: CompareWorkflowState) -> str:
    context_by_filename = {
        item.get("filename"): item for item in state["document_context"]
    }
    blocks = []

    source = state["source_document"]
    source_ctx = context_by_filename.get(source["filename"], {})
    blocks.append(
        _format_document_block(
            document_id=source["id"],
            filename=source["filename"],
            role="source",
            document_type=source_ctx.get("documentType", ""),
            title_or_subject=source_ctx.get("titleOrSubject", ""),
            summary=source_ctx.get("summary", ""),
        )
    )

    for document in state["reference_documents"]:
        doc_ctx = context_by_filename.get(document["filename"], {})
        blocks.append(
            _format_document_block(
                document_id=document["id"],
                filename=document["filename"],
                role="reference",
                document_type=doc_ctx.get("documentType", ""),
                title_or_subject=doc_ctx.get("titleOrSubject", ""),
                summary=doc_ctx.get("summary", ""),
            )
        )

    return "\n\n".join(blocks)


def _format_document_block(
    document_id: str,
    filename: str,
    role: str,
    document_type: str,
    title_or_subject: str,
    summary: str,
) -> str:
    role_label = "Văn bản gốc" if role == "source" else "Văn bản quy chiếu"
    return (
        f"### {role_label}\n"
        f"- ID: {document_id}\n"
        f"- Tên file: {filename}\n"
        f"- Vai trò: {role}\n"
        f"- Loại văn bản: {document_type or 'Chưa xác định'}\n"
        f"- Chủ đề: {title_or_subject or 'Chưa xác định'}\n"
        f"- Tóm tắt: {summary or 'Chưa xác định'}"
    )


def decompose_questions_node(state: CompareWorkflowState) -> CompareWorkflowState:
    documents_block = _format_documents_for_decomposition(state)

    messages = [
        SystemMessage(
            content=(
                "You are a legal document comparison question planner. "
                "Given a user role, a comparison question, and a list of documents "
                "(source + references), decompose the user's question into specific "
                "retrieval questions for EACH document.\n\n"
                "Rules:\n"
                "- Tailor questions to the document's role (source vs reference) "
                "and inferred document type.\n"
                "- Questions must help retrieve precise legal clauses relevant "
                "to the user's comparison intent.\n"
                "- Use Vietnamese for all questions.\n"
                f"- Generate 2-{MAX_QUESTIONS_PER_DOCUMENT} focused questions per document.\n"
                "- Source document questions: what does THIS document say about "
                "the user's concern?\n"
                "- Reference document questions: what does THIS reference say "
                "that can be compared against the source?\n"
                "- Return document_id, filename, role, document_type, and questions "
                "for every document listed."
            )
        ),
        HumanMessage(
            content=(
                f"Vai trò người dùng: {state['user_role']}\n\n"
                f"Câu hỏi / yêu cầu so sánh:\n{state['user_question']}\n\n"
                f"Danh sách tài liệu:\n{documents_block}\n\n"
                "Hãy tách thành các câu hỏi truy vấn cụ thể cho từng tài liệu."
            )
        ),
    ]

    structured_llm = openAI["mini_llm"].with_structured_output(
        QuestionDecompositionResult
    )
    result: QuestionDecompositionResult = structured_llm.invoke(messages)

    document_questions = [
        {
            "documentId": item.document_id,
            "filename": item.filename,
            "role": item.role,
            "documentType": item.document_type,
            "questions": item.questions,
        }
        for item in result.document_questions
    ]

    steps = list(state.get("steps", []))
    total_questions = sum(len(item["questions"]) for item in document_questions)
    steps.append(
        ComparisonAgentStep(
            agent="QuestionDecomposer",
            status="completed",
            summary=(
                f"Đã tách {total_questions} câu hỏi truy vấn "
                f"cho {len(document_questions)} tài liệu."
            ),
        ).model_dump()
    )

    return {
        **state,
        "document_questions": document_questions,
        "reasoning": result.reasoning,
        "steps": steps,
    }


def fan_out_document_retrieval(state: CompareWorkflowState):
    retrieval_by_id = {
        document["id"]: document for document in state["retrieval_documents"]
    }
    sends = []

    for item in state["document_questions"]:
        document_id = item["documentId"]
        retrieval_document = retrieval_by_id.get(document_id)
        if not retrieval_document:
            continue

        sends.append(
            Send(
                "retrieve_document",
                {
                    "retrieval_document": retrieval_document,
                    "active_document_questions": item,
                    "user_question": state["user_question"],
                },
            )
        )

    if not sends:
        raise ValueError("Không có tài liệu hợp lệ để truy xuất")

    return sends


def retrieve_document_node(state: CompareWorkflowState) -> dict:
    retrieval_document = state["retrieval_document"]
    document_questions = state["active_document_questions"]
    questions = document_questions.get("questions", [])

    if not questions:
        raise ValueError(
            f"Tài liệu '{document_questions.get('filename')}' không có câu hỏi truy vấn"
        )

    result = retrieve_document_by_questions(
        retrieval_document,
        questions,
        state["user_question"],
    )

    return {"retrieval_results": [result]}


def aggregate_retrieval_node(state: CompareWorkflowState) -> CompareWorkflowState:
    retrieval_results = state.get("retrieval_results", [])
    total_answers = sum(len(item.get("retrievedQa", [])) for item in retrieval_results)

    steps = list(state.get("steps", []))
    steps.append(
        ComparisonAgentStep(
            agent="DocumentRetriever",
            status="completed",
            summary=(
                f"Đã truy xuất {total_answers} câu trả lời từ "
                f"{len(retrieval_results)} tài liệu (song song)."
            ),
        ).model_dump()
    )

    return {
        **state,
        "steps": steps,
    }


def _format_retrieval_results_for_synthesis(
    retrieval_results: list[dict],
) -> str:
    blocks = []
    for item in retrieval_results:
        role_label = "Văn bản gốc" if item.get("role") == "source" else "Văn bản quy chiếu"
        blocks.append(
            f"## {role_label}: {item.get('filename')}\n"
            f"{item.get('combinedContext', '')}"
        )
    return "\n\n".join(blocks)


def build_synthesis_messages(state: CompareWorkflowState) -> list:
    retrieval_results = state.get("retrieval_results", [])
    if not retrieval_results:
        raise ValueError("Không có kết quả truy xuất để tổng hợp")

    retrieval_context = _format_retrieval_results_for_synthesis(retrieval_results)
    source_filename = state["source_document"]["filename"]
    reference_filenames = [
        document["filename"] for document in state["reference_documents"]
    ]

    return [
        SystemMessage(
            content=(
                "You are a senior legal analyst. Answer the user's comparison question "
                "in Vietnamese using ONLY the retrieved document context below.\n\n"
                "Structure the response in Markdown:\n"
                "1. Brief header with document names\n"
                "2. Executive summary (blockquote, emoji risk indicators ✅ ⚠️ 🔴)\n"
                "3. Section '## Đối chiếu chi tiết' comparing source vs references\n"
                "4. Section '## Điểm giống và khác biệt' with bullet lists\n"
                "5. Section '## Rủi ro và khoảng trống pháp lý'\n"
                "6. Section '## Khuyến nghị' with numbered actions\n\n"
                "Be specific, cite concrete legal points from the retrieved context. "
                "Do not invent content outside the provided context. No code blocks."
            )
        ),
        HumanMessage(
            content=(
                f"Vai trò người dùng: {state['user_role']}\n\n"
                f"Câu hỏi / yêu cầu so sánh:\n{state['user_question']}\n\n"
                f"Tài liệu gốc: {source_filename}\n"
                f"Tài liệu quy chiếu: {', '.join(reference_filenames)}\n\n"
                f"Ngữ cảnh đã truy xuất từ các tài liệu:\n\n"
                f"{retrieval_context}"
            )
        ),
    ]


def iter_synthesis_report_tokens(state: CompareWorkflowState) -> Iterator[str]:
    settings = get_chat_settings()
    chat_model = settings.get("chat_model", "gpt-4o")
    chat_llm = get_chat_llm(chat_model)

    for chunk in chat_llm.stream(build_synthesis_messages(state)):
        content = chunk.content
        if isinstance(content, str) and content:
            yield content


def synthesize_answer_node(state: CompareWorkflowState) -> CompareWorkflowState:
    settings = get_chat_settings()
    chat_model = settings.get("chat_model", "gpt-4o")
    chat_llm = get_chat_llm(chat_model)

    response = chat_llm.invoke(build_synthesis_messages(state))
    report = response.content

    steps = list(state.get("steps", []))
    steps.append(
        ComparisonAgentStep(
            agent="Synthesizer",
            status="completed",
            summary="Đã tổng hợp câu trả lời so sánh từ ngữ cảnh truy xuất.",
        ).model_dump()
    )

    return {
        **state,
        "report": report,
        "steps": steps,
    }
