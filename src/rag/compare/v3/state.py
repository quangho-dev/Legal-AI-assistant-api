from __future__ import annotations

import operator
from typing import Annotated, Any, List, TypedDict


class CompareWorkflowState(TypedDict):
    user_question: str
    user_role: str
    source_document: dict
    reference_documents: List[dict]
    document_context: List[dict]
    retrieval_documents: List[dict]
    document_questions: List[dict]
    retrieval_results: Annotated[List[dict], operator.add]
    report: str
    steps: List[dict]
    reasoning: str
    retrieval_document: dict
    active_document_questions: dict


def build_initial_compare_state(
    user_question: str,
    user_role: str,
    source_document: dict,
    reference_documents: List[dict],
    document_context: List[dict],
    retrieval_documents: List[dict],
) -> CompareWorkflowState:
    return {
        "user_question": user_question,
        "user_role": user_role,
        "source_document": source_document,
        "reference_documents": reference_documents,
        "document_context": document_context,
        "retrieval_documents": retrieval_documents,
        "document_questions": [],
        "retrieval_results": [],
        "report": "",
        "steps": [],
        "reasoning": "",
        "retrieval_document": {},
        "active_document_questions": {},
    }


def serialize_workflow_state(state: CompareWorkflowState) -> dict[str, Any]:
    return {
        "version": "v3",
        "userQuestion": state["user_question"],
        "userRole": state["user_role"],
        "documentQuestions": state["document_questions"],
        "retrievalResults": state.get("retrieval_results", []),
        "report": state.get("report", ""),
        "reasoning": state["reasoning"],
        "steps": state["steps"],
        "documentContext": state["document_context"],
        "sourceDocument": {
            "id": state["source_document"]["id"],
            "filename": state["source_document"]["filename"],
        },
        "referenceDocuments": [
            {"id": doc["id"], "filename": doc["filename"]}
            for doc in state["reference_documents"]
        ],
    }
