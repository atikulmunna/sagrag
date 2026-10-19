# app/schemas.py
"""Request/response models. These drive validation and the OpenAPI docs at
/docs; anything not declared here is rejected or ignored, never trusted."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_LIST_FIELDS = (
    "blocklist", "allowlist", "source_types_allow", "source_types_block", "domains_allow", "domains_block",
)


def _as_str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip().lower() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    raise ValueError("expected a list of strings or a comma-separated string")


def _strip_required(value):
    if isinstance(value, str):
        value = value.strip()
    if not value:
        raise ValueError("must not be empty")
    return value


class Preferences(BaseModel):
    """Per-request preferences. They can only narrow what the caller sees."""
    model_config = ConfigDict(extra="ignore")

    domain: str | None = Field(default=None, max_length=48, description="Restrict to one known domain.")
    freshness_days: int | None = Field(default=None, ge=1, le=36500)
    blocklist: list[str] = Field(default_factory=list, max_length=50)
    allowlist: list[str] = Field(default_factory=list, max_length=50)
    source_types_allow: list[str] = Field(default_factory=list, max_length=20)
    source_types_block: list[str] = Field(default_factory=list, max_length=20)
    domains_allow: list[str] = Field(default_factory=list, max_length=50)
    domains_block: list[str] = Field(default_factory=list, max_length=50)

    _lists = field_validator(*_LIST_FIELDS, mode="before")(_as_str_list)

    @field_validator("domain", mode="before")
    @classmethod
    def _blank_domain(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v.strip().lower() if isinstance(v, str) else v


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)
    tenant: str | None = Field(
        default=None, max_length=128,
        description="Only used when tenant isolation is on and the request is not authenticated.",
    )
    preferences: Preferences = Field(default_factory=Preferences)

    _required = field_validator("user_id", "query", mode="before")(_strip_required)

    @field_validator("preferences", mode="before")
    @classmethod
    def _null_prefs(cls, v):
        return v if isinstance(v, (dict, Preferences)) else {}


class Claim(BaseModel):
    text: str
    status: str = Field(description="supported, unsupported, or contradicted")
    support_id: str | int | None = None
    support_text: str | None = Field(default=None, description="The cited passage text that supports the sentence.")
    entailment: float | None = None
    contradiction: float | None = None


class Citation(BaseModel):
    id: str | int | None = None
    source: str
    offset_start: int
    offset_end: int


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    query_id: str
    user_id: str
    query: str
    answer: str
    provenance: list[Citation]
    confidence: float
    explain_trace: str | None = None
    domain: str | None = None
    domain_source: str
    author_terms: list[str]
    author_gap: bool
    retrieval_stats: dict[str, Any]
    retrieval_failures: list[str]
    hallucination_risk: float
    evidence_coverage: float
    intent: str | None = None
    plan: dict[str, Any]
    results: list[dict[str, Any]]
    graph_signals: list[Any] | None = None
    graph_subgraph: dict[str, Any] | None = None
    graph_reasoning: dict[str, Any] | None = None
    judge: dict[str, Any]
    timings_ms: dict[str, int] = Field(default_factory=dict)
    hallucination_risk_method: str = Field(
        default="heuristic", description="nli when every sentence was verified, else heuristic"
    )
    claims: list[Claim] | None = None
    supported_ratio: float | None = None
    abstained: bool = False
    unverified_answer: str | None = None
    cache: str | None = None


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tenant: str | None = Field(default=None, max_length=128)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=2000)
    query_id: str | None = Field(default=None, max_length=64, description="query_id from the /query response.")
    tenant: str | None = Field(default=None, max_length=128)

    _required = field_validator("user_id", "query", mode="before")(_strip_required)


class AuditExportRequest(BaseModel):
    # extra="forbid": the old client-chosen "path" field is rejected outright.
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=1000, ge=1, le=10000)
    user_id: str | None = Field(default=None, max_length=128)
    tenant: str | None = Field(default=None, max_length=128)
    save: bool = Field(default=False, description="Write to the server's configured export path instead of downloading.")


class LearningExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=1000, ge=1, le=10000)
    min_rating: int | None = Field(default=None, ge=1, le=5)
    tenant: str | None = Field(default=None, max_length=128)
    save: bool = Field(default=False, description="Write to the server's configured export path instead of downloading.")


class GraphSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chunk_ids: list[str] = Field(max_length=100)
    tenant: str | None = Field(default=None, max_length=128)
