from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config.index import appConfig
from src.models.index import (
    DEFAULT_CHAT_MODEL,
    OPENAI_CHAT_MODELS,
    OPENAI_CHAT_MODELS_WITH_TEMPERATURE,
)

openAI = {
    "embeddings_llm": ChatOpenAI(
        model="gpt-4-turbo", api_key=appConfig["openai_api_key"], temperature=0
    ),
    "embeddings": OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=appConfig["openai_api_key"],
        dimensions=1536,  # ! Do not changes this value. It is used in the document_chunks embedding vector.
    ),
    "chat_llm": ChatOpenAI(
        model=DEFAULT_CHAT_MODEL,
        api_key=appConfig["openai_api_key"],
        temperature=0,
    ),
    "mini_llm": ChatOpenAI(
        model="gpt-4o-mini", api_key=appConfig["openai_api_key"], temperature=0
    ),
}


def get_chat_llm(model: str | None = None) -> ChatOpenAI:
    selected_model = model if model in OPENAI_CHAT_MODELS else DEFAULT_CHAT_MODEL
    llm_kwargs = {
        "model": selected_model,
        "api_key": appConfig["openai_api_key"],
    }

    if selected_model in OPENAI_CHAT_MODELS_WITH_TEMPERATURE:
        llm_kwargs["temperature"] = 0

    return ChatOpenAI(**llm_kwargs)
