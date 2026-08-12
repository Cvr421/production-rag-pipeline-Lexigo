from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cerebras_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CEREBRAS_API_KEY", "OPENAI_API_KEY"),
    )
    pinecone_api_key: str = Field(default="", validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(
        default="lexigo-legal-corpus", validation_alias="PINECONE_INDEX_NAME"
    )
    pinecone_cloud: str = Field(default="aws", validation_alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", validation_alias="PINECONE_REGION")
    pinecone_namespace: str = Field(default="sample-corpus", validation_alias="PINECONE_NAMESPACE")
    embedding_model: str = Field(
        default="llama-text-embed-v2", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=1024, validation_alias="EMBEDDING_DIMENSION")
    chat_model: str = Field(default="llama-3.3-70b", validation_alias="CHAT_MODEL")
    cerebras_base_url: str = Field(
        default="https://api.cerebras.ai/v1",
        validation_alias="CEREBRAS_BASE_URL",
    )
    chat_temperature: float = Field(default=0.0, validation_alias="CHAT_TEMPERATURE")
    corpus_dir: str = Field(default="gen_ai_takehome_sample_corpus", validation_alias="CORPUS_DIR")
    top_k: int = Field(default=5, ge=1, le=20, validation_alias="TOP_K")
    min_relevance_score: float = Field(
        default=0.4, ge=-1.0, le=1.0, validation_alias="MIN_RELEVANCE_SCORE"
    )
    min_query_term_overlap: float = Field(
        default=0.12, ge=0.0, le=1.0, validation_alias="MIN_QUERY_TERM_OVERLAP"
    )
    max_graph_steps: int = Field(default=6, ge=2, le=20, validation_alias="MAX_GRAPH_STEPS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
