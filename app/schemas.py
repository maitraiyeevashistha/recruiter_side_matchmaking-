from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StudentProfileRequest(BaseModel):
    student_id: str = Field(min_length=1)

    name: str = ""

    resume_text: str = Field(min_length=1)

    skills: list[str] = Field(default_factory=list)

    projects: list[str] = Field(default_factory=list)

    tools: list[str] = Field(default_factory=list)

    target_roles: list[str] = Field(default_factory=list)

    assignment_score: float = 0.0

    cgpa: float = 0.0

    metadata: dict[str, Any] = Field(default_factory=dict)


class JobPostingRequest(BaseModel):
    job_id: str = Field(min_length=1)

    title: str = Field(min_length=1)

    company: str = ""

    industry: str = ""

    description: str = Field(min_length=1)

    skills: list[str] = Field(default_factory=list)

    tools: list[str] = Field(default_factory=list)

    location: str = ""

    job_type: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)


class RecruiterMatchRequest(BaseModel):
    job_id: str = Field(min_length=1)

    top_k: int = Field(
        default=25,
        ge=1,
        le=100,
    )

    use_reranker: bool = True


class RecruiterMatchResult(BaseModel):
    student_id: str

    name: str = ""

    semantic_score: float = 0.0

    lexical_score: float = 0.0

    assignment_score: float = 0.0

    rerank_score: float | None = None

    final_score: float = 0.0

    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str

    id: str


class MatchResponse(BaseModel):
    status: str

    total_candidates: int

    results: list[RecruiterMatchResult]

class ApplicationRequest(BaseModel):
    application_id: str = Field(min_length=1)

    student_id: str = Field(min_length=1)

    job_id: str = Field(min_length=1)

    assignment_score: float = 0.0

    status: str = "applied"

    metadata: dict[str, Any] = Field(default_factory=dict)


class ApplicationResponse(BaseModel):
    status: str
    application_id: str 

class RankCandidatesRequest(BaseModel):
    job_id: str = Field(min_length=1)


class RankedCandidate(BaseModel):
    student_id: str
    name: str
    assignment_score: float
    final_score: float


class RankCandidatesResponse(BaseModel):
    job_id: str
    total_candidates: int
    results: list[RankedCandidate]