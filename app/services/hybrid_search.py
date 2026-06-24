from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import Settings
from app.schemas import (
    StudentProfileRequest,
    JobPostingRequest,
    RecruiterMatchRequest,
    RecruiterMatchResult,
)
from app.services.embedding import EmbeddingService
from app.services.store import ChromaRepository, CollectionRecord
from app.services.text_utils import (
    build_student_search_text,
    build_job_search_text,
    safe_json,
    tokenize,
)


@dataclass(slots=True)
class CachedStudent:
    student_id: str
    document: str
    metadata: dict[str, Any]


class RecruiterHybridSearchEngine:
    def __init__(
        self,
        settings: Settings,
        repo: ChromaRepository,
        embedder: EmbeddingService,
    ):
        self.settings = settings
        self.repo = repo
        self.embedder = embedder

        self.cached_students = []
        self.student_tokens = []

        self.bm25 = None
        self.reranker = None

        if settings.enable_reranker:
            try:
                self.reranker = CrossEncoder(
                    settings.reranker_model_name
                )
            except Exception:
                self.reranker = None

        self.refresh_student_cache()

    def refresh_student_cache(self):
        raw = self.repo.fetch_all_students()

        ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        self.cached_students = []
        self.student_tokens = []

        for sid, doc, meta in zip(
            ids,
            documents,
            metadatas,
        ):
            self.cached_students.append(
                CachedStudent(
                    student_id=sid,
                    document=doc,
                    metadata=meta,
                )
            )

            self.student_tokens.append(
                tokenize(doc)
            )

        if self.student_tokens:
            self.bm25 = BM25Okapi(
                self.student_tokens
            )

    # -------------------------------------------------
    # STUDENT INGESTION
    # -------------------------------------------------

    def ingest_student(
        self,
        payload: StudentProfileRequest,
    ):
        document = build_student_search_text(
            name=payload.name,
            resume_text=payload.resume_text,
            skills=payload.skills,
            projects=payload.projects,
            tools=payload.tools,
            target_roles=payload.target_roles,
        )

        embedding = self.embedder.embed_query(
            document
        )

        metadata = {
            "student_id": payload.student_id,
            "name": payload.name,
            "resume_text": payload.resume_text,
            "skills_text": ", ".join(payload.skills),
            "projects_text": ", ".join(payload.projects),
            "assignment_score": payload.assignment_score,
            "cgpa": payload.cgpa,
            "raw_json": safe_json(
                payload.model_dump()
            ),
        }

        self.repo.upsert_student(
            CollectionRecord(
                record_id=payload.student_id,
                document=document,
                metadata=metadata,
                embedding=embedding,
            )
        )

        self.refresh_student_cache()

        return payload.student_id

    # -------------------------------------------------
    # JOB INGESTION
    # -------------------------------------------------

    def ingest_job(
        self,
        payload: JobPostingRequest,
    ):
        document = build_job_search_text(
            title=payload.title,
            company=payload.company,
            description=payload.description,
            skills=payload.skills,
            tools=payload.tools,
            location=payload.location,
            job_type=payload.job_type,
        )

        embedding = self.embedder.embed_query(
            document
        )

        metadata = {
            "job_id": payload.job_id,
            "title": payload.title,
            "company": payload.company,
            "description": payload.description,
            "skills_text": ", ".join(payload.skills),
            "tools_text": ", ".join(payload.tools),
            "raw_json": safe_json(
                payload.model_dump()
            ),
        }

        self.repo.upsert_job(
            CollectionRecord(
                record_id=payload.job_id,
                document=document,
                metadata=metadata,
                embedding=embedding,
            )
        )

        return payload.job_id

    # -------------------------------------------------
    # MATCHING
    # -------------------------------------------------

    def match(
        self,
        job_payload: JobPostingRequest,
        top_k: int = 25,
    ):
        query_text = build_job_search_text(
            title=job_payload.title,
            company=job_payload.company,
            description=job_payload.description,
            skills=job_payload.skills,
            tools=job_payload.tools,
            location=job_payload.location,
            job_type=job_payload.job_type,
        )

        query_embedding = (
            self.embedder.embed_query(
                query_text
            )
        )

        semantic_results = (
            self.repo.query_students(
                query_embedding=query_embedding,
                n_results=self.settings.semantic_top_k,
            )
        )

        candidates = {}

        ids = semantic_results.get(
            "ids",
            [[]],
        )[0]

        metas = semantic_results.get(
            "metadatas",
            [[]],
        )[0]

        distances = semantic_results.get(
            "distances",
            [[]],
        )[0]

        for sid, meta, dist in zip(
            ids,
            metas,
            distances,
        ):
            semantic_score = (
                1.0 - float(dist)
            )

            assignment_score = (
                float(
                    meta.get(
                        "assignment_score",
                        0,
                    )
                )
                / 100
            )

            final_score = (
                0.50 * semantic_score
                + 0.30 * assignment_score
            )

            candidates[sid] = RecruiterMatchResult(
                student_id=sid,
                name=meta.get(
                    "name",
                    "",
                ),
                semantic_score=semantic_score,
                assignment_score=assignment_score
                * 100,
                final_score=final_score,
                metadata=dict(meta),
            )

        # -------------------------------------
        # BM25
        # -------------------------------------

        if (
            self.bm25
            and self.cached_students
        ):
            query_tokens = tokenize(
                query_text
            )

            scores = np.asarray(
                self.bm25.get_scores(
                    query_tokens
                )
            )

            max_score = (
                float(scores.max())
                if scores.size > 0
                else 1.0
            )

            for idx, score in enumerate(
                scores
            ):
                student = (
                    self.cached_students[
                        idx
                    ]
                )

                if (
                    student.student_id
                    not in candidates
                ):
                    continue

                lexical_score = (
                    float(score)
                    / max_score
                    if max_score > 0
                    else 0
                )

                candidates[
                    student.student_id
                ].lexical_score = (
                    lexical_score
                )

                candidates[
                    student.student_id
                ].final_score += (
                    0.20
                    * lexical_score
                )

        results = sorted(
            candidates.values(),
            key=lambda x: x.final_score,
            reverse=True,
        )

        return results[:top_k]