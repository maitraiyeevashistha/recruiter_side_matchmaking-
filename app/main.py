from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.config import load_settings
from app.schemas import (
    StudentProfileRequest,
    JobPostingRequest,
    RecruiterMatchResult,
    ApplicationRequest,
    ApplicationResponse,
    RankCandidatesRequest,
    IngestResponse,
)
from app.services.embedding import get_embedding_service
from app.services.hybrid_search import RecruiterHybridSearchEngine
from app.services.store import ChromaRepository
from app.services.application_service import (
    ApplicationService
)
from app.services.ranking_service import (
    RankingService
)

settings = load_settings()

repo = ChromaRepository(settings)

embedder = get_embedding_service(
    settings.embedding_model_name
)

engine = RecruiterHybridSearchEngine(
    settings=settings,
    repo=repo,
    embedder=embedder,
)

application_service = ApplicationService(
    repo
)

ranking_service = RankingService(
    repo
)

app = FastAPI(
    title="Recruiter Matchmaking Service"
)


def require_api_key(
    x_api_key: str = Header(default="")
):
    if (
        settings.api_key
        and x_api_key != settings.api_key
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/v1/students",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
def ingest_student(
    payload: StudentProfileRequest,
):
    student_id = engine.ingest_student(
        payload
    )

    return IngestResponse(
        status="stored",
        id=student_id,
    )


@app.post(
    "/v1/jobs",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
def ingest_job(
    payload: JobPostingRequest,
):
    job_id = engine.ingest_job(
        payload
    )

    return IngestResponse(
        status="stored",
        id=job_id,
    )


@app.post(
    "/v1/recruiter-match",
    dependencies=[Depends(require_api_key)],
)
def recruiter_match(
    payload: JobPostingRequest,
):
    results = engine.match(
        payload,
        top_k=25,
    )

    return {
        "status": "ok",
        "total_candidates": len(results),
        "results": results,
    }

@app.post(
    "/v1/applications",
    response_model=ApplicationResponse,
)
def apply_job(
    payload: ApplicationRequest,
):

    application_id = (
        application_service.apply(
            payload.application_id,
            payload.student_id,
            payload.job_id,
            payload.assignment_score,
        )
    )

    return ApplicationResponse(
        status="applied",
        application_id=application_id,
    )

@app.post(
    "/v1/rank-candidates",
)
def rank_candidates(
    payload: RankCandidatesRequest,
):

    results = (
        ranking_service.rank_job_candidates(
            payload.job_id
        )
    )

    return {
        "job_id": payload.job_id,
        "total_candidates": len(results),
        "results": results,
    }