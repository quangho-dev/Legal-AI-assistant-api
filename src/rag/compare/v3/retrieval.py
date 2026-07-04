from __future__ import annotations

from typing import List

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
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


def _build_query_engine_for_document(document: dict):
    _configure_llama_index()
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
        raise ValueError(
            f"Tài liệu '{document.get('filename')}' không có chunk để truy xuất"
        )

    vector_index = VectorStoreIndex(nodes)
    return vector_index.as_query_engine(similarity_top_k=TOP_K_CHUNKS)


def retrieve_document_by_questions(
    document: dict,
    questions: List[str],
    user_question: str,
) -> dict:
    query_engine = _build_query_engine_for_document(document)
    retrieved_qa = []

    for question in questions:
        query = (
            f"Yêu cầu so sánh gốc của người dùng: {user_question}\n"
            f"Câu hỏi truy vấn cho tài liệu này: {question}\n\n"
            "Trả lời bằng tiếng Việt. Trích các điều khoản pháp lý cụ thể "
            "có trong tài liệu, không suy diễn ngoài nội dung được truy xuất."
        )
        response = query_engine.query(query)
        retrieved_qa.append({"question": question, "answer": str(response)})

    combined_blocks = []
    for index, item in enumerate(retrieved_qa, start=1):
        combined_blocks.append(
            f"### Câu hỏi {index}: {item['question']}\n{item['answer']}"
        )

    return {
        "documentId": document["id"],
        "filename": document["filename"],
        "role": document["role"],
        "retrievedQa": retrieved_qa,
        "combinedContext": "\n\n".join(combined_blocks),
    }
