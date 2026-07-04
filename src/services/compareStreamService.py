from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException

from src.models.index import ComparisonAgentStep
from src.rag.compare.v3.nodes import iter_synthesis_report_tokens
from src.rag.compare.v3.state import serialize_workflow_state
from src.rag.compare.v3.workflow import run_compare_v3_retrieval_workflow
from src.services.compareDocumentRetrievalService import (
    fetch_compare_documents_for_retrieval,
)
from src.services.compareDocumentService import resolve_compare_documents
from src.services.compareService import iter_report_tokens, run_document_comparison
from src.services.compareServiceV2 import run_document_comparison_v2
from src.services.compareServiceV3 import prepare_compare_v3_initial_state
from src.utils.sse import sse_done, sse_error, sse_status, sse_token


def _finalize_v1_v2_result(pipeline_result: dict, report: str) -> dict:
    result = {key: value for key, value in pipeline_result.items() if key != "_reportContext"}
    result["report"] = report
    steps = list(result.get("steps", []))
    steps.append(
        ComparisonAgentStep(
            agent="Reporter",
            status="completed",
            summary="Đã tổng hợp báo cáo Markdown.",
        ).model_dump()
    )
    result["steps"] = steps
    return result


def iter_compare_v1_stream(
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: list[str],
    instruction: str,
) -> Iterator[str]:
    try:
        yield sse_status("Đang phân tích và đối chiếu tài liệu...")

        source_doc, reference_docs = resolve_compare_documents(
            clerk_id,
            source_document_id,
            reference_document_ids,
        )
        pipeline_result = run_document_comparison(
            source_doc,
            reference_docs,
            instruction,
            skip_report=True,
        )
        report_context = pipeline_result["_reportContext"]

        yield sse_status("Đang tổng hợp báo cáo...")

        report_parts: list[str] = []
        for token in iter_report_tokens(
            report_context["instruction"],
            report_context["plan"],
            report_context["source_filename"],
            report_context["reference_filenames"],
            report_context["comparisons"],
        ):
            report_parts.append(token)
            yield sse_token(token)

        report = "".join(report_parts)
        yield sse_done(_finalize_v1_v2_result(pipeline_result, report))
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, str) else str(error.detail)
        yield sse_error(detail)
    except Exception as error:
        yield sse_error(str(error))


def iter_compare_v2_stream(
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: list[str],
    instruction: str,
) -> Iterator[str]:
    try:
        yield sse_status("Đang phân tích và tra cứu ngữ cảnh...")

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
        pipeline_result = run_document_comparison_v2(
            source_doc,
            reference_docs,
            instruction,
            retrieval_documents,
            skip_report=True,
        )
        report_context = pipeline_result["_reportContext"]

        yield sse_status("Đang tổng hợp báo cáo...")

        report_parts: list[str] = []
        for token in iter_report_tokens(
            report_context["instruction"],
            report_context["plan"],
            report_context["source_filename"],
            report_context["reference_filenames"],
            report_context["comparisons"],
        ):
            report_parts.append(token)
            yield sse_token(token)

        report = "".join(report_parts)
        yield sse_done(_finalize_v1_v2_result(pipeline_result, report))
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, str) else str(error.detail)
        yield sse_error(detail)
    except Exception as error:
        yield sse_error(str(error))


def iter_compare_v3_stream(
    clerk_id: str,
    source_document_id: str,
    reference_document_ids: list[str],
    instruction: str,
    user_role: str,
) -> Iterator[str]:
    try:
        yield sse_status("Đang tải và phân loại tài liệu...")

        initial_state, run_config = prepare_compare_v3_initial_state(
            clerk_id,
            source_document_id,
            reference_document_ids,
            instruction,
            user_role,
        )

        yield sse_status("Đang tách câu hỏi và truy xuất song song...")

        retrieval_state = run_compare_v3_retrieval_workflow(
            initial_state,
            run_config=run_config,
        )

        yield sse_status("Đang tổng hợp báo cáo...")

        report_parts: list[str] = []
        for token in iter_synthesis_report_tokens(retrieval_state):
            report_parts.append(token)
            yield sse_token(token)

        report = "".join(report_parts)
        steps = list(retrieval_state.get("steps", []))
        steps.append(
            ComparisonAgentStep(
                agent="Synthesizer",
                status="completed",
                summary="Đã tổng hợp câu trả lời so sánh từ ngữ cảnh truy xuất.",
            ).model_dump()
        )

        final_state = {
            **retrieval_state,
            "report": report,
            "steps": steps,
        }
        yield sse_done(serialize_workflow_state(final_state))
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, str) else str(error.detail)
        yield sse_error(detail)
    except Exception as error:
        yield sse_error(str(error))
