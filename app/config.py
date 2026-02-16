# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Ollama settings (generation only)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    # local embedding model name (sentence-transformers)
    embedding_model_name: str = "all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10
    judge_timeout_s: float = 8.0
    synthesis_timeout_s: float = 12.0
    retriever_timeout_s: float = 12.0
    judge_max_tokens: int = 150
    synthesis_max_tokens: int = 250
    max_evidence_snippets: int = 8

    # Service URLs (Docker service hostnames)
    qdrant_url: str = "http://qdrant:6333"
    elastic_url: str = "http://elastic:9200"
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "mysecurepassword"
    graph_enabled: bool = False
    audit_db_path: str = "/data/audit/sag_rag_audit.db"
    graph_max_claims: int = 5
    graph_contradiction_window: int = 200
    graph_contradiction_overlap: float = 0.6
    judge_contradiction_confidence_cap: float = 0.6
    judge_contradiction_penalty_per_claim: float = 0.1
    judge_contradiction_penalty_max: float = 0.5
    judge_relation_boost_per_relation: float = 0.05
    judge_relation_boost_max: float = 0.2
    judge_relation_conflict_penalty: float = 0.15
    otel_enabled: bool = False
    otel_service_name: str = "sag-rag-backend"
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    feedback_db_path: str = "/data/feedback/feedback.db"
    llm_max_retries: int = 3
    llm_retry_base_s: float = 0.5
    enable_judge: bool = True
    enable_synthesis: bool = True
    rate_limit_per_minute: int = 60
    llm_max_concurrent: int = 2
    tenant_isolation: bool = False
    learning_export_path: str = "/data/learning/train.jsonl"
    learning_min_rating: int = 4
    default_freshness_days: int | None = None
    policy_blocklist: str = ""
    policy_allowlist: str = ""
    policy_source_types_allow: list[str] = []
    policy_source_types_block: list[str] = []
    policy_domains_allow: list[str] = []
    policy_domains_block: list[str] = []
    policy_rules: list[dict] = []
    domain_index_map: dict[str, str] = {}
    domain_keywords: dict[str, list[str]] = {}
    domain_aliases: dict[str, list[str]] = {}
    domain_fallbacks: list[str] = []
    domain_min_keyword_hits: int = 2
    query_term_synonyms: dict[str, list[str]] = {}
    author_index_path: str = "/data/author_index.json"
    min_results_count: int = 3
    min_top_rerank_score: float = -12.0

    class Config:
        # set this to the path of your .env as mounted in the container.
        env_file = "../.env"
        env_file_encoding = "utf-8"

settings = Settings()
