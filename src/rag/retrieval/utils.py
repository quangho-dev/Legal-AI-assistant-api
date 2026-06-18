from src.services.supabase import supabase
from fastapi import HTTPException
from typing import List, Dict, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from src.services.llm import get_chat_llm, openAI
from src.models.index import ChatSettings, QueryVariations
from src.rag.legal_citation import (
    format_legal_citation_for_client,
    get_legal_citation_from_chunk_record,
)


def get_chat_settings():
    try:
        settings_result = (
            supabase.table("chat_settings").select("*").limit(1).execute()
        )

        if not settings_result.data:
            return ChatSettings().model_dump()

        row = settings_result.data[0]
        return {
            key: row[key] for key in ChatSettings.model_fields if key in row
        }
    except Exception as e:
        raise Exception(f"Failed to get chat settings: {str(e)}")


def get_document_ids():
    try:
        document_ids_result = (
            supabase.table("documents")
            .select("id")
            .eq("processing_status", "completed")
            .eq("document_scope", "corpus")
            .execute()
        )

        if not document_ids_result.data:
            return []

        return [document["id"] for document in document_ids_result.data]
    except Exception as e:
        raise Exception(f"Failed to get document IDs: {str(e)}")


def build_context_from_retrieved_chunks(
    chunks: List[Dict],
) -> Tuple[List[str], List[str], List[str], List[Dict]]:
    """
    Build the context from the retrieved chunks and format them into a structured context with citations.
    Citations are the entries in the citations list that contain the information about the document and the page number of the chunk.
    """
    if not chunks:
        return [], [], [], []

    texts = []
    images = []
    tables = []
    citations = []

    # Batch fetch all filenames of chunks in ONE query
    doc_ids = [chunk["document_id"] for chunk in chunks if chunk.get("document_id")]
    # Get the unique document IDs from the doc_ids list.
    unique_doc_ids = list(set(doc_ids))

    # Create a dictionary to store the filenames for the documents in the unique_doc_ids list.
    filename_map = {}

    # Fetch the filenames for the documents in the unique_doc_ids list.
    if unique_doc_ids:
        result = (
            supabase.table("documents")
            .select("id, filename")
            .in_("id", unique_doc_ids)
            .execute()
        )
        filename_map = {doc["id"]: doc["filename"] for doc in result.data}

    # Process each chunk
    for chunk in chunks:
        original_content = chunk.get("original_content", {})

        # Extract content from chunk
        chunk_text = original_content.get("text", "")
        chunk_images = original_content.get("images", [])
        chunk_tables = original_content.get("tables", [])

        if (
            chunk_text
        ):  # Since chunk_text is not going to be an array, Thus we will append it
            texts.append(chunk_text)
        # Meanwhile, chunk_images and chunk_tables are going to be arrays, Thus we will extend them to the images and tables lists.
        images.extend(chunk_images)
        tables.extend(chunk_tables)

        # * Add citation for every chunk
        doc_id = chunk.get("document_id")
        if doc_id:
            filename = filename_map.get(doc_id, "Unknown Document")
            legal = get_legal_citation_from_chunk_record(chunk, filename)
            citations.append(
                {
                    "chunk_id": chunk.get("id"),
                    "document_id": doc_id,
                    "filename": filename,
                    "page": chunk.get("page_number", "Unknown"),
                    "law_name": legal.get("law_name"),
                    "section": legal.get("section"),
                    "section_name": legal.get("section_name"),
                }
            )

    return texts, images, tables, citations


def validate_context_from_retrieved_chunks(
    texts: List[str], images: List[str], tables: List[str], citations: List[Dict]
) -> None:
    """Validate and print context data from retrieved chunks in a readable format"""
    print("\n" + "=" * 80)
    print("📦 CONTEXT VALIDATION")
    print("=" * 80)

    # Texts - SHOW FULL TEXT
    print(f"\n📝 TEXTS: {len(texts)} chunks")
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*80}")
        print(f"CHUNK [{i}] - {len(text)} characters")
        print(f"{'='*80}")
        print(text)
        print(f"{'='*80}\n")

    # Images
    print(f"\n🖼️  IMAGES: {len(images)}")
    for i, img in enumerate(images, 1):
        img_preview = str(img)[:60] + ("..." if len(str(img)) > 60 else "")
        print(f"  [{i}] {img_preview}")

    # Tables
    print(f"\n📊 TABLES: {len(tables)}")
    for i, table in enumerate(tables, 1):
        if isinstance(table, dict):
            rows = len(table.get("rows", []))
            cols = len(table.get("headers", []))
            print(f"  [{i}] {rows} rows × {cols} cols")
        else:
            print(f"  [{i}] Type: {type(table).__name__}")

    # Citations
    print(f"\n📚 CITATIONS: {len(citations)}")
    for i, cite in enumerate(citations, 1):
        chunk_id = cite["chunk_id"][:8] if cite.get("chunk_id") else "N/A"
        print(f"  [{i}] {cite['filename']} (pg.{cite['page']}) | chunk: {chunk_id}...")

    # Summary
    total_chars = sum(len(text) for text in texts)
    print(f"\n{'='*80}")
    print(
        f"✅ Total: {len(texts)} texts ({total_chars:,} chars), {len(images)} images, {len(tables)} tables, {len(citations)} citations"
    )
    print("=" * 80 + "\n")


def prepare_prompt_and_invoke_llm(
    user_query: str, texts: List[str], images: List[str], tables: List[str]
) -> str:
    """
    Builds system prompt with context and invokes LLM with multi-modal support.
    """
    # Build system prompt parts
    prompt_parts = []

    # Main instruction
    prompt_parts.append(
        "Bạn là trợ lý AI pháp lý. Trả lời bằng tiếng Việt, dựa hoàn toàn trên ngữ cảnh được cung cấp. "
        "Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, rõ ràng chỉ từ thông tin trong ngữ cảnh bên dưới.\n\n"
        "QUY TẮC QUAN TRỌNG:\n"
        "- Chỉ trả lời dựa trên ngữ cảnh đã cung cấp (văn bản, bảng và hình ảnh)\n"
        "- Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy trả lời: "
        "'Tôi không có đủ thông tin trong tài liệu đã cung cấp để trả lời câu hỏi này.'\n"
        "- Không dùng kiến thức bên ngoài hoặc suy đoán vượt quá nội dung được nêu\n"
        "- Khi trích dẫn, hãy cụ thể và chỉ rõ phần liên quan trong ngữ cảnh\n"
        "- Tổng hợp thông tin từ văn bản, bảng và hình ảnh để trả lời đầy đủ\n"
        "- Định dạng câu trả lời bằng Markdown (tiêu đề ##, danh sách -, **in đậm**, "
        "bảng khi cần). Không bọc toàn bộ câu trả lời trong khối mã ```markdown\n\n"
    )

    # Add text contexts
    if texts:
        prompt_parts.append("=" * 80)
        prompt_parts.append("CONTEXT DOCUMENTS")
        prompt_parts.append("=" * 80 + "\n")

        for i, text in enumerate(texts, 1):
            prompt_parts.append(f"--- Document Chunk {i} ---")
            prompt_parts.append(text.strip())
            prompt_parts.append("")

    # Add tables if present
    if tables:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED TABLES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            "The following tables contain structured data that may be relevant to your answer. "
            "Analyze the table contents carefully.\n"
        )

        for i, table_html in enumerate(tables, 1):
            prompt_parts.append(f"--- Table {i} ---")
            prompt_parts.append(table_html)
            prompt_parts.append("")

    # Reference images if present
    if images:
        prompt_parts.append("\n" + "=" * 80)
        prompt_parts.append("RELATED IMAGES")
        prompt_parts.append("=" * 80)
        prompt_parts.append(
            f"{len(images)} image(s) will be provided alongside the user's question. "
            "These images may contain diagrams, charts, figures, formulas, or other visual information. "
            "Carefully analyze the visual content when formulating your response. "
            "The images are part of the retrieved context and should be used to answer the question.\n"
        )

    # Final instruction
    prompt_parts.append("=" * 80)
    prompt_parts.append(
        "Based on all the context provided above (documents, tables, and images), "
        "please answer the user's question accurately and comprehensively in Markdown format."
    )
    prompt_parts.append("=" * 80)

    system_prompt = "\n".join(prompt_parts)

    # Build messages for LLM
    messages = [SystemMessage(content=system_prompt)]

    # Create human message with user query and images
    if images:
        # Multi-modal message: text + images
        content_parts = [{"type": "text", "text": user_query}]

        # Add each image to the content array
        for img_base64 in images:
            # Clean base64 string if it has data URI prefix
            if img_base64.startswith("data:image"):
                img_base64 = img_base64.split(",", 1)[1]

            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                }
            )

        messages.append(HumanMessage(content=content_parts))
    else:
        # Text-only message
        messages.append(HumanMessage(content=user_query))

    # Invoke LLM and return response
    settings = get_chat_settings()
    chat_model = settings.get("chat_model", "gpt-4o")
    chat_llm = get_chat_llm(chat_model)

    print(
        f"🤖 Invoking {chat_model} with {len(messages)} messages ({len(texts)} texts, {len(tables)} tables, {len(images)} images)..."
    )
    response = chat_llm.invoke(messages)

    return response.content


def rrf_rank_and_fuse(search_results_list, weights=None, k=60):
    """RRF (Reciprocal Rank Fusion) ranking"""
    if not search_results_list or not any(search_results_list):
        return []

    if weights is None:
        weights = [1.0 / len(search_results_list)] * len(search_results_list)

    chunk_scores = {}
    all_chunks = {}

    for search_idx, results in enumerate(search_results_list):
        weight = weights[search_idx]

        for rank, chunk in enumerate(results):
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue

            rrf_score = weight * (1.0 / (k + rank + 1))

            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += rrf_score
            else:
                chunk_scores[chunk_id] = rrf_score
                all_chunks[chunk_id] = chunk

    sorted_chunk_ids = sorted(
        chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True
    )
    return [all_chunks[chunk_id] for chunk_id in sorted_chunk_ids]


def generate_query_variations(original_query: str, num_queries: int = 3) -> List[str]:
    """Generate query variations using LLM"""
    system_prompt = f"""Generate {num_queries-1} alternative ways to phrase this question for document search. Use different keywords and synonyms while maintaining the same intent. Return exactly {num_queries-1} variations."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original query: {original_query}"),
        ]

        structured_llm = openAI["chat_llm"].with_structured_output(QueryVariations)
        result = structured_llm.invoke(messages)

        print(f"✅ Generated {len(result.queries)} query variations")  # ✅ Debug
        print(f"Queries: {result.queries}")  # ✅ Debug

        return [original_query] + result.queries[: num_queries - 1]
    except Exception as e:
        print(f"❌ Query variation generation failed: {str(e)}")  # ✅ Better error
        import traceback

        traceback.print_exc()  # ✅ Full stack trace
        return [original_query]
