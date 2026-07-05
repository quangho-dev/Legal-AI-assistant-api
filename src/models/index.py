from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ProjectCreate(BaseModel):
    name: str = Field(..., description="The name of the project")
    description: Optional[str] = Field(None, description="Project description")


class ChatCreate(BaseModel):
    title: str = Field(default="Cuộc trò chuyện mới", description="The title of the chat")
    id: Optional[str] = Field(None, description="Optional client-provided chat ID")


class SendChatMessageRequest(BaseModel):
    chatId: str = Field(..., description="The chat session ID")
    message: str = Field(..., min_length=1, description="The user message content")
    documentIds: Optional[List[str]] = Field(
        None,
        description="Optional corpus document IDs to limit retrieval scope",
    )


RAG_STRATEGIES = (
    "basic",
    "hybrid",
    "multi-query-vector",
    "multi-query-hybrid",
    "corrective-rag",
)

OPENAI_CHAT_MODELS = (
    # GPT-5.5 frontier
    "gpt-5.5",
    "gpt-5.5-pro",
    # GPT-5.4
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    # GPT-5 family
    "gpt-5.3-codex",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    # o-series reasoning
    "o3-pro",
    "o3",
    "o4-mini",
    "o3-mini",
    "o1",
    "o1-mini",
    # GPT-4 family
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
)

OPENAI_CHAT_MODELS_WITH_TEMPERATURE = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
}

DEFAULT_CHAT_MODEL = "gpt-4o"


class ChatSettings(BaseModel):
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="The embedding model to use",
    )
    chat_model: str = Field(
        default=DEFAULT_CHAT_MODEL,
        description="The OpenAI model used to generate the final chat answer",
    )
    rag_strategy: str = Field(default="hybrid", description="The RAG strategy to use")
    agent_type: str = Field(default="default", description="The agent type to use")
    chunks_per_search: int = Field(
        default=20, description="The number of chunks per search"
    )
    final_context_size: int = Field(default=8, description="The final context size")
    similarity_threshold: float = Field(
        default=0.3, description="The similarity threshold"
    )
    number_of_queries: int = Field(default=3, description="The number of queries")
    reranking_enabled: bool = Field(
        default=False, description="Whether reranking is enabled"
    )
    reranking_model: str = Field(
        default="cohere-rerank-3", description="The reranking model to use"
    )
    vector_weight: float = Field(default=0.7, description="The vector weight")
    keyword_weight: float = Field(default=0.3, description="The keyword weight")


class ChatSettingsCreate(ChatSettings):
    pass


class ChatSettingsUpdate(ChatSettings):
    pass


class FileUploadRequest(BaseModel):
    filename: str = Field(..., description="The name of the file")
    file_type: str = Field(..., description="The type of the file")
    file_size: int = Field(..., description="The size of the file")


class ProcessingStatus(str, Enum):
    UPLOADING = "uploading"
    PENDING = "pending"
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    PARTITIONING = "partitioning"
    CHUNKING = "chunking"
    SUMMARISING = "summarising"
    VECTORIZATION = "vectorization"
    COMPLETED = "completed"


class ConfirmUploadRequest(BaseModel):
    s3_key: str = Field(..., description="The S3 key of the uploaded file")


class RenameDocumentRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=500, description="The new document name")


class UrlRequest(BaseModel):
    url: str = Field(..., description="The URL to process")


class MessageCreate(BaseModel):
    content: str = Field(..., description="The content of the message")


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class QueryVariations(BaseModel):
    queries: List[str] = Field(..., description="The variations of the query")


class ChunkRelevanceLabel(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


class ChunkEvaluation(BaseModel):
    chunk_index: int = Field(
        ..., description="Zero-based index of the chunk in the evaluation batch"
    )
    label: ChunkRelevanceLabel = Field(
        ..., description="Relevance of the chunk to the user query"
    )
    reason: str = Field(
        default="",
        description="Brief explanation for the relevance label",
    )


class RetrievalEvaluationResult(BaseModel):
    evaluations: List[ChunkEvaluation] = Field(
        ..., description="Per-chunk relevance evaluations"
    )


class RewrittenRetrievalQuery(BaseModel):
    query: str = Field(
        ..., description="Rewritten query optimized for document retrieval"
    )
    reason: str = Field(
        default="",
        description="Why the query was rewritten",
    )


class RefinedKnowledgeStrip(BaseModel):
    refined_text: str = Field(
        ...,
        description="Query-relevant knowledge extracted from the chunk",
    )


class InputGuardrailCheck(BaseModel):
    """Schema for input safety check"""
    is_safe: bool = Field(description="Whether the input is safe to process")
    is_toxic: bool = Field(description="Contains toxic or harmful content")
    is_prompt_injection: bool = Field(description="Appears to be a prompt injection attempt")
    contains_pii: bool = Field(description="Contains personal identifiable information")
    reason: str = Field(description="Brief explanation if unsafe, empty string if safe")


class CompareDocumentsRequest(BaseModel):
    sourceDocumentId: str = Field(..., description="ID of the user's compare source document")
    referenceDocumentIds: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="IDs of user's compare reference documents",
    )
    instruction: str = Field(
        ...,
        min_length=10,
        max_length=4000,
        description="Custom comparison instruction from the user",
    )


class CompareDocumentsV3Request(BaseModel):
    sourceDocumentId: str = Field(..., description="ID of the user's compare source document")
    referenceDocumentIds: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="IDs of user's compare reference documents",
    )
    instruction: str = Field(
        ...,
        min_length=10,
        max_length=4000,
        description="User comparison question or instruction",
    )
    userRole: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="User role or perspective, e.g. luật sư tư vấn, nhân sự",
    )


class ContractLanguage(str, Enum):
    VI = "vi"
    EN = "en"


class ContractDraftRequest(BaseModel):
    language: ContractLanguage = Field(
        ...,
        description="Contract language: vi (Vietnamese) or en (English)",
    )
    requirements: str = Field(
        ...,
        min_length=10,
        max_length=8000,
        description="General description of contract requirements",
    )
    partyRole: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Which party the user represents in the contract",
    )
    templateDocumentId: Optional[str] = Field(
        default=None,
        description="Optional uploaded template contract document ID",
    )
    descriptionDocumentIds: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Optional uploaded description documents (MOU, key terms, etc.)",
    )


class ContractExportDocxRequest(BaseModel):
    language: ContractLanguage = Field(
        ...,
        description="Contract language: vi (Vietnamese) or en (English)",
    )
    requirements: str = Field(
        ...,
        min_length=10,
        max_length=8000,
        description="General description of contract requirements",
    )
    partyRole: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Which party the user represents in the contract",
    )
    outline: str = Field(
        ...,
        min_length=20,
        max_length=20000,
        description="Approved contract outline used to draft the full document",
    )
    templateDocumentId: Optional[str] = Field(
        default=None,
        description="Optional uploaded template contract document ID",
    )
    descriptionDocumentIds: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Optional uploaded description documents",
    )


class DocumentQueryQuestion(BaseModel):
    document_id: str
    filename: str
    role: str = Field(description="source or reference")
    document_type: str = Field(default="", description="Inferred document type")
    questions: List[str] = Field(
        ...,
        min_length=1,
        description="Retrieval questions tailored for this document",
    )


class QuestionDecompositionResult(BaseModel):
    document_questions: List[DocumentQueryQuestion]
    reasoning: str = Field(
        default="",
        description="Brief explanation of how questions were decomposed",
    )


class ComparisonPlan(BaseModel):
    objectives: List[str] = Field(..., description="Goals of the comparison")
    focus_areas: List[str] = Field(..., description="Areas to focus on")
    comparison_dimensions: List[str] = Field(
        ..., description="Dimensions used to compare documents"
    )


class DocumentIdentification(BaseModel):
    filename: str
    role: str = Field(
        description="Document role in comparison: source or reference",
    )
    document_type: str = Field(
        description="Inferred legal document type in Vietnamese",
    )
    title_or_subject: str = Field(
        description="Inferred title or subject from opening text",
    )
    summary: str = Field(
        description="Brief summary of what the document appears to be about",
    )
    legal_domain: str = Field(
        default="",
        description="Legal domain or field, e.g. lao động, dân sự",
    )


class DocumentIdentificationBatch(BaseModel):
    identifications: List[DocumentIdentification]


class OutlineSection(BaseModel):
    title: str
    summary: str
    excerpt: str = Field(
        default="",
        description="Representative excerpt from this section",
    )


class DocumentOutline(BaseModel):
    sections: List[OutlineSection]


class ComparisonTopic(BaseModel):
    title: str
    rationale: str


class ComparisonTopicMap(BaseModel):
    topics: List[ComparisonTopic]


class TopicComparisonItem(BaseModel):
    topic: str
    source_position: str
    reference_positions: List[str]
    similarities: List[str]
    differences: List[str]
    notable_gaps: List[str]


class TopicComparisonBatch(BaseModel):
    comparisons: List[TopicComparisonItem]


class ComparisonAgentStep(BaseModel):
    agent: str
    status: str
    summary: str
