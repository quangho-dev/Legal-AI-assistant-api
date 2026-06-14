"""
Corrective RAG (CRAG) pipeline.

Based on the CRAG framework:
1. Retrieve candidate chunks
2. Evaluate retrieval quality per chunk (correct / incorrect / ambiguous)
3. Apply corrective action when retrieval is poor (re-retrieve with rewritten query)
4. Refine accepted knowledge into query-focused strips before generation
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.models.index import (
    ChunkEvaluation,
    ChunkRelevanceLabel,
    RefinedKnowledgeStrip,
    RetrievalEvaluationResult,
    RewrittenRetrievalQuery,
)
from src.rag.chunk_content import build_chunk_display_text
from src.services.llm import openAI

RetrievalVerdict = Literal["correct", "ambiguous", "incorrect"]

EVALUATION_PREVIEW_CHARS = 1200
REFINEMENT_INPUT_CHARS = 6000


def _preview_chunk_text(chunk: Dict) -> str:
    text = build_chunk_display_text(chunk)
    if len(text) <= EVALUATION_PREVIEW_CHARS:
        return text
    return f"{text[:EVALUATION_PREVIEW_CHARS]}…"


def _build_evaluation_prompt(user_query: str, chunks: List[Dict]) -> str:
    chunk_blocks = []
    for index, chunk in enumerate(chunks):
        chunk_blocks.append(
            f"[Chunk {index}]\n{_preview_chunk_text(chunk)}"
        )

    return (
        "Đánh giá mức độ liên quan của từng đoạn tài liệu với câu hỏi người dùng.\n\n"
        f"Câu hỏi: {user_query}\n\n"
        "Nhãn:\n"
        "- correct: đoạn chứa thông tin trực tiếp trả lời hoặc hỗ trợ rõ ràng câu hỏi\n"
        "- ambiguous: đoạn có thể liên quan một phần hoặc cần thêm ngữ cảnh\n"
        "- incorrect: đoạn không liên quan hoặc không giúp trả lời câu hỏi\n\n"
        "Các đoạn tài liệu:\n"
        f"{chr(10).join(chunk_blocks)}"
    )


def evaluate_retrieved_chunks(
    user_query: str, chunks: List[Dict]
) -> List[ChunkEvaluation]:
    if not chunks:
        return []

    try:
        messages = [
            SystemMessage(
                content=(
                    "You are a retrieval evaluator for a legal document assistant. "
                    "Classify each chunk's relevance to the user query. "
                    "Return one evaluation per chunk index."
                )
            ),
            HumanMessage(content=_build_evaluation_prompt(user_query, chunks)),
        ]

        structured_llm = openAI["mini_llm"].with_structured_output(
            RetrievalEvaluationResult
        )
        result: RetrievalEvaluationResult = structured_llm.invoke(messages)

        valid_evaluations = [
            evaluation
            for evaluation in result.evaluations
            if 0 <= evaluation.chunk_index < len(chunks)
        ]

        evaluated_indices = {evaluation.chunk_index for evaluation in valid_evaluations}
        for index in range(len(chunks)):
            if index not in evaluated_indices:
                valid_evaluations.append(
                    ChunkEvaluation(
                        chunk_index=index,
                        label=ChunkRelevanceLabel.AMBIGUOUS,
                        reason="Không có đánh giá từ mô hình, mặc định ambiguous.",
                    )
                )

        print(
            "🔍 CRAG evaluations: "
            + ", ".join(
                f"{item.chunk_index}={item.label.value}"
                for item in sorted(valid_evaluations, key=lambda x: x.chunk_index)
            )
        )
        return valid_evaluations
    except Exception as error:
        print(f"❌ CRAG evaluation failed, defaulting to ambiguous: {error}")
        return [
            ChunkEvaluation(
                chunk_index=index,
                label=ChunkRelevanceLabel.AMBIGUOUS,
                reason="Đánh giá thất bại, giữ lại để tránh mất ngữ cảnh.",
            )
            for index in range(len(chunks))
        ]


def compute_retrieval_verdict(evaluations: List[ChunkEvaluation]) -> RetrievalVerdict:
    if not evaluations:
        return "incorrect"

    labels = {evaluation.label for evaluation in evaluations}

    if ChunkRelevanceLabel.CORRECT in labels:
        return "correct"
    if ChunkRelevanceLabel.AMBIGUOUS in labels:
        return "ambiguous"
    return "incorrect"


def get_accepted_chunk_indices(
    evaluations: List[ChunkEvaluation],
    verdict: RetrievalVerdict,
) -> List[int]:
    if verdict == "incorrect":
        return []

    accepted_labels = {
        ChunkRelevanceLabel.CORRECT,
        ChunkRelevanceLabel.AMBIGUOUS,
    }

    return sorted(
        {
            evaluation.chunk_index
            for evaluation in evaluations
            if evaluation.label in accepted_labels
        }
    )


def rewrite_query_for_corrective_retrieval(
    user_query: str,
    evaluations: List[ChunkEvaluation],
) -> str:
    incorrect_reasons = [
        evaluation.reason
        for evaluation in evaluations
        if evaluation.label == ChunkRelevanceLabel.INCORRECT and evaluation.reason
    ]

    feedback = (
        "\n".join(f"- {reason}" for reason in incorrect_reasons[:5])
        if incorrect_reasons
        else "- Các đoạn tài liệu hiện tại không liên quan đến câu hỏi."
    )

    try:
        messages = [
            SystemMessage(
                content=(
                    "Rewrite the user's question into a better search query for "
                    "retrieving relevant legal documents. Keep the same intent, "
                    "use clearer legal keywords, and improve recall."
                )
            ),
            HumanMessage(
                content=(
                    f"Original question:\n{user_query}\n\n"
                    f"Why previous retrieval failed:\n{feedback}\n\n"
                    "Return a single improved search query."
                )
            ),
        ]

        structured_llm = openAI["mini_llm"].with_structured_output(
            RewrittenRetrievalQuery
        )
        result: RewrittenRetrievalQuery = structured_llm.invoke(messages)
        rewritten = result.query.strip() or user_query
        print(f"🔁 CRAG rewritten query: {rewritten}")
        return rewritten
    except Exception as error:
        print(f"❌ CRAG query rewrite failed, using original query: {error}")
        return user_query


def refine_chunk_knowledge(user_query: str, chunk: Dict) -> str:
    source_text = build_chunk_display_text(chunk)
    if not source_text:
        return ""

    if len(source_text) <= 400:
        return source_text

    input_text = source_text
    if len(input_text) > REFINEMENT_INPUT_CHARS:
        input_text = f"{input_text[:REFINEMENT_INPUT_CHARS]}…"

    try:
        messages = [
            SystemMessage(
                content=(
                    "Extract only the knowledge strips from the document chunk that "
                    "are relevant to answering the user's legal question. "
                    "Remove unrelated sentences. Keep Vietnamese legal wording accurate. "
                    "If nothing is relevant, return an empty string."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{user_query}\n\n"
                    f"Document chunk:\n{input_text}"
                )
            ),
        ]

        structured_llm = openAI["mini_llm"].with_structured_output(
            RefinedKnowledgeStrip
        )
        result: RefinedKnowledgeStrip = structured_llm.invoke(messages)
        refined = result.refined_text.strip()
        return refined or source_text
    except Exception as error:
        print(f"❌ CRAG knowledge refinement failed, using original chunk: {error}")
        return source_text


def apply_knowledge_refinement(
    user_query: str,
    chunk_eval_pairs: List[Tuple[Dict, ChunkEvaluation]],
) -> List[Dict]:
    refined_chunks: List[Dict] = []

    for chunk, evaluation in chunk_eval_pairs:
        refined_chunk = dict(chunk)
        original_content = dict(refined_chunk.get("original_content") or {})

        if evaluation.label == ChunkRelevanceLabel.CORRECT:
            refined_text = build_chunk_display_text(chunk)
        else:
            refined_text = refine_chunk_knowledge(user_query, chunk)

        if not refined_text.strip():
            continue

        original_content["text"] = refined_text
        original_content["crag_label"] = evaluation.label.value
        refined_chunk["original_content"] = original_content
        refined_chunks.append(refined_chunk)

    print(f"✨ CRAG refined {len(refined_chunks)} chunks for generation")
    return refined_chunks


def run_corrective_rag_pipeline(
    user_query: str,
    initial_chunks: List[Dict],
    corrective_search_fn,
    final_context_size: int,
) -> Tuple[List[Dict], RetrievalVerdict]:
    chunks = initial_chunks

    if not chunks:
        print("⚠️ CRAG: no chunks retrieved")
        return [], "incorrect"

    evaluations = evaluate_retrieved_chunks(user_query, chunks)
    verdict = compute_retrieval_verdict(evaluations)

    if verdict == "incorrect":
        rewritten_query = rewrite_query_for_corrective_retrieval(
            user_query, evaluations
        )
        chunks = corrective_search_fn(rewritten_query)
        print(f"🔁 CRAG corrective retrieval returned {len(chunks)} chunks")

        if not chunks:
            return [], "incorrect"

        evaluations = evaluate_retrieved_chunks(user_query, chunks)
        verdict = compute_retrieval_verdict(evaluations)

        if verdict == "incorrect":
            print("⛔ CRAG: retrieval still incorrect after corrective action")
            return [], "incorrect"

    accepted_indices = get_accepted_chunk_indices(evaluations, verdict)
    evaluation_by_index = {
        evaluation.chunk_index: evaluation for evaluation in evaluations
    }
    chunk_eval_pairs = [
        (chunks[index], evaluation_by_index[index])
        for index in accepted_indices
        if index < len(chunks) and index in evaluation_by_index
    ]

    if not chunk_eval_pairs:
        print("⚠️ CRAG: no accepted chunks after evaluation")
        return [], verdict

    refined_chunks = apply_knowledge_refinement(user_query, chunk_eval_pairs)

    return refined_chunks[:final_context_size], verdict
