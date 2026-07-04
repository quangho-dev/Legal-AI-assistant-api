from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.rag.compare.v3.nodes import (
    aggregate_retrieval_node,
    decompose_questions_node,
    fan_out_document_retrieval,
    retrieve_document_node,
    synthesize_answer_node,
)
from src.rag.compare.v3.state import CompareWorkflowState
from src.rag.compare.v3.tracing import build_workflow_run_config, configure_langsmith

_compiled_workflow = None
_compiled_retrieval_workflow = None

configure_langsmith()


def build_compare_v3_retrieval_workflow():
    graph = StateGraph(CompareWorkflowState)
    graph.add_node("decompose_questions", decompose_questions_node)
    graph.add_node("retrieve_document", retrieve_document_node)
    graph.add_node("aggregate_retrieval", aggregate_retrieval_node)

    graph.add_edge(START, "decompose_questions")
    graph.add_conditional_edges(
        "decompose_questions",
        fan_out_document_retrieval,
        ["retrieve_document"],
    )
    graph.add_edge("retrieve_document", "aggregate_retrieval")
    graph.add_edge("aggregate_retrieval", END)
    return graph.compile()


def get_compare_v3_retrieval_workflow():
    global _compiled_retrieval_workflow
    if _compiled_retrieval_workflow is None:
        _compiled_retrieval_workflow = build_compare_v3_retrieval_workflow()
    return _compiled_retrieval_workflow


def run_compare_v3_retrieval_workflow(
    initial_state: CompareWorkflowState,
    run_config: dict | None = None,
) -> CompareWorkflowState:
    workflow = get_compare_v3_retrieval_workflow()
    if run_config:
        return workflow.invoke(initial_state, config=run_config)
    return workflow.invoke(initial_state)


def build_compare_v3_workflow():
    graph = StateGraph(CompareWorkflowState)
    graph.add_node("decompose_questions", decompose_questions_node)
    graph.add_node("retrieve_document", retrieve_document_node)
    graph.add_node("aggregate_retrieval", aggregate_retrieval_node)
    graph.add_node("synthesize_answer", synthesize_answer_node)

    graph.add_edge(START, "decompose_questions")
    graph.add_conditional_edges(
        "decompose_questions",
        fan_out_document_retrieval,
        ["retrieve_document"],
    )
    graph.add_edge("retrieve_document", "aggregate_retrieval")
    graph.add_edge("aggregate_retrieval", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)
    return graph.compile()


def get_compare_v3_workflow():
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_compare_v3_workflow()
    return _compiled_workflow


def run_compare_v3_workflow(
    initial_state: CompareWorkflowState,
    run_config: dict | None = None,
) -> CompareWorkflowState:
    workflow = get_compare_v3_workflow()
    if run_config:
        return workflow.invoke(initial_state, config=run_config)
    return workflow.invoke(initial_state)
