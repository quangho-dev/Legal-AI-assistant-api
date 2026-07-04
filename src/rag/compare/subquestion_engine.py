from __future__ import annotations

import re
from typing import List

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.schema import TextNode
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from src.config.index import appConfig

TOP_K_CHUNKS = 6
_llama_configured = False


def _configure_llama_index() -> None:
    global _llama_configured
    if _llama_configured:
        return

    Settings.llm = LlamaOpenAI(
        model="gpt-4o-mini",
        api_key=appConfig["openai_api_key"],
        temperature=0,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=appConfig["openai_api_key"],
        dimensions=1536,
    )
    _llama_configured = True


def _safe_tool_name(filename: str, role: str, index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", filename.lower()).strip("_")
    slug = slug[:40] or f"document_{index}"
    prefix = "source" if role == "source" else "reference"
    return f"{prefix}_{slug}"[:60]


def build_document_query_engine_tools(documents: List[dict]) -> List[QueryEngineTool]:
    _configure_llama_index()
    tools: List[QueryEngineTool] = []

    for doc_index, document in enumerate(documents, start=1):
        nodes: List[TextNode] = []

        for chunk in document["chunks"]:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            node_kwargs = {
                "text": text,
                "metadata": {
                    "document_id": document["id"],
                    "filename": document["filename"],
                    "role": document["role"],
                    "chunk_index": chunk.get("chunkIndex"),
                    "page_number": chunk.get("pageNumber"),
                },
            }
            embedding = chunk.get("embedding")
            if embedding:
                node_kwargs["embedding"] = embedding

            nodes.append(TextNode(**node_kwargs))

        if not nodes:
            continue

        vector_index = VectorStoreIndex(nodes)
        query_engine = vector_index.as_query_engine(similarity_top_k=TOP_K_CHUNKS)

        role_label = "gốc" if document["role"] == "source" else "tham khảo"
        tools.append(
            QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=_safe_tool_name(
                    document["filename"], document["role"], doc_index
                ),
                description=(
                    f"Tài liệu {role_label}: {document['filename']}. "
                    "Tra cứu các điều khoản liên quan trong tài liệu này."
                ),
            )
        )

    if not tools:
        raise ValueError("Không thể tạo query engine cho tài liệu so sánh")

    return tools


def retrieve_comparison_context(
    instruction: str,
    topics: List[str],
    query_engine_tools: List[QueryEngineTool],
) -> str:
    _configure_llama_index()

    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),
        use_async=False,
        verbose=False,
    )

    topic_list = "\n".join(f"- {topic}" for topic in topics)
    query = (
        f"Yêu cầu so sánh của người dùng: {instruction}\n\n"
        "Hãy tạo sub-question cho TỪNG tài liệu (tool) để trích xuất nội dung "
        "liên quan các chủ đề sau:\n"
        f"{topic_list}\n\n"
        "Trả lời bằng tiếng Việt. Ghi rõ tên tài liệu, trích các điểm pháp lý cụ thể "
        "và phân tách theo từng chủ đề."
    )

    response = sub_question_engine.query(query)
    return str(response)
